#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
score_hyperlora_forum_sentiment.py

Purpose
-------
Post-hoc sentiment evaluation for simulated forum outputs.

Given a forum.parquet (from build_hyperlora_forum.py), compute a comment-level
sentiment polarity score using the same SST-2 sentiment probe used by the
feature builder, then:

1) Derive 5-bin thresholds from a reference distribution (global_features_10000)
2) Assign each comment a sentiment_label in {rage, negative, neutral, positive, empath}
3) Aggregate per-user and produce confusion metrics against author_type (if present)

This script is designed to run both:
  - locally (CPU default if no CUDA)
  - in-cluster (GPU if available)

Outputs (written to --out_dir)
------------------------------
- forum_scored.parquet
- user_sentiment_eval.csv
- confusion_author_type.csv (if author_type exists)
- score_metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import shared bootstrap utility
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hypergames_utils import (  # type: ignore  # noqa: E402
    bootstrap_ci,
    has_repetition_loop,
    score_texts_vader,
    score_texts_goemo,
    # Canonical coherence gate (with C1 patent-keyword safety net) lives
    # in hypergames_utils so every scoring pipeline applies the same
    # filter. See the 2026-05-14 patent-leak audit.
    _is_coherent,
)

# ---------------------------------------------------------------------
# Local module import robustness (match train_hyperlora.py conventions)
# ---------------------------------------------------------------------
_CANDIDATE_SCRIPT_DIRS = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parents[1] / "data_scripts",
    Path("/workspace/hypernets/data_scripts"),
]
for _p in _CANDIDATE_SCRIPT_DIRS:
    if _p.exists():
        sys.path.append(str(_p))

# sentiment helpers from feature builder (keeps behavior aligned)
try:
    from hypernetwork_feature_builder_10000 import (  # type: ignore
        polarity_batch,
        DEFAULT_SENT_MODEL as _DEFAULT_SENT_MODEL,
    )
except Exception as e:
    raise RuntimeError(
        "Could not import hypernetwork_feature_builder_10000. "
        "Ensure data_scripts is on PYTHONPATH or co-located with this script. "
        f"Import error: {e}"
    )


def _default_device_id() -> int:
    """Return 0 if CUDA is available; otherwise return -1 (CPU)."""
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _build_sentiment_pipe(
    *,
    device_id: int,
    model_id: str,
    batch_size: int,
):
    """Build a transformers sentiment-analysis pipeline compatible with polarity_batch."""
    try:
        from transformers import pipeline
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()

        dev = int(device_id)
        if dev < 0:
            dev = -1

        # NOTE: return_all_scores=True yields [{'label': 'NEGATIVE', 'score': ...}, {'label':'POSITIVE', ...}]
        # 2026-05-19 fix: framework="pt" prevents Keras 3 / tf-keras compatibility
        # crash on some local Python envs where the pipeline auto-falls-back to TF
        # if no torch model is registered for the model_id. Same fix already in
        # evaluate_user_reconstruction.py.
        return pipeline(
            task="sentiment-analysis",
            model=model_id,
            tokenizer=model_id,
            device=dev,
            top_k=None,
            truncation=True,
            batch_size=int(max(1, batch_size)),
            framework="pt",
        )
    except Exception:
        return None


def load_threshold_source(path: Path, col: str) -> np.ndarray:
    if path.suffix.lower() == ".parquet":
        gdf = pd.read_parquet(path)
    else:
        gdf = pd.read_csv(path)
    if col not in gdf.columns:
        raise KeyError(f"threshold_source missing column {col!r}")
    x = pd.to_numeric(gdf[col], errors="coerce").dropna().astype(np.float32).values
    if x.size < 1000:
        raise RuntimeError(f"threshold_source too small after dropna: n={x.size}")
    return x


def compute_thresholds(
    x: np.ndarray,
    quantile_breaks: Optional[List[float]] = None,
) -> Dict[str, float]:
    if quantile_breaks is None:
        quantile_breaks = [0.2, 0.4, 0.6, 0.8]
    qs = np.quantile(x, quantile_breaks).tolist()
    # Build keys from percentile values (e.g. 0.2 -> "q20", 0.333 -> "q33")
    keys = [f"q{int(round(q * 100))}" for q in quantile_breaks]
    return {k: float(v) for k, v in zip(keys, qs)}


# Label names for 5-bin (default) and variable-bin configurations.
# Endpoints are cohort labels (label_a/label_b); middle bins are fixed.
_LABEL_ORDER_5_DEFAULT = ["rage", "grumpy", "mellow", "calm", "empath"]
_LABEL_ORDER_3_DEFAULT = ["rage", "neutral", "empath"]


def _make_label_order(n_bins: int, label_a: str = "rage", label_b: str = "empath") -> List[str]:
    """Build label order with cohort labels at the extremes."""
    if n_bins == 5:
        return [label_a, "grumpy", "mellow", "calm", label_b]
    elif n_bins == 3:
        return [label_a, "neutral", label_b]
    else:
        return [f"bin_{i}" for i in range(n_bins)]


def label_from_thresholds(score: float, thr: Dict[str, float],
                          label_a: str = "rage", label_b: str = "empath") -> str:
    """Assign a bin label based on threshold dict.

    Works with any number of thresholds (3-bin, 5-bin, etc.).
    Endpoint bins use label_a (most negative) and label_b (most positive).
    """
    sorted_keys = sorted(thr.keys(), key=lambda k: thr[k])
    n_bins = len(sorted_keys) + 1
    labels = _make_label_order(n_bins, label_a, label_b)

    for i, key in enumerate(sorted_keys):
        if score <= thr[key]:
            return labels[i]
    return labels[-1]


# _is_coherent is imported from hypergames_utils above. The duplicate
# definition that lived here was removed during the 2026-05-14 patent-leak
# audit so the coherence gate (now with patent-keyword regex) stays in
# lock-step with build_forum_pythia_v6.py and
# score_persona_signature.py.


def score_comments(
    df: pd.DataFrame,
    *,
    pipe,
    text_col: str,
    chunk_size: int,
) -> List[float]:
    texts = df[text_col].astype(str).fillna("").tolist()
    out: List[float] = []
    step = max(1, int(chunk_size))
    total_chunks = (len(texts) + step - 1) // step
    for chunk_idx, i in enumerate(range(0, len(texts), step), start=1):
        batch = texts[i : i + step]
        out.extend(polarity_batch(pipe, batch))
        print(f"[score] chunk {chunk_idx}/{total_chunks} ({len(out)}/{len(texts)} texts scored)", flush=True)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Score forum outputs with sentiment probe and produce eval tables.")
    ap.add_argument("--input_parquet", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--threshold_source", type=str, required=True, help="Parquet/CSV containing threshold_col.")
    ap.add_argument("--threshold_col", type=str, default="gstat_user_sent_mean")

    # Column overrides for compatibility with different forum schemas
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--user_col", type=str, default="author_user_id")
    ap.add_argument("--author_type_col", type=str, default="author_type")

    # Sentiment probe settings
    ap.add_argument(
        "--sentiment_model",
        type=str,
        default="",
        help="Optional: override sentiment model path/id. If empty, uses SENTIMENT_MODEL env or feature-builder default.",
    )
    ap.add_argument(
        "--device_id",
        type=int,
        default=None,
        help="CUDA device id (0..). Use -1 for CPU. Default: auto-detect.",
    )
    ap.add_argument(
        "--pipe_batch_size",
        type=int,
        default=128,
        help="Transformers pipeline batch_size (not the outer chunk size).",
    )

    # Throughput controls (useful for local quick runs)
    ap.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="If >0, only score first N rows (useful for local quick checks).",
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
        help="How many texts to pass per polarity_batch call.",
    )

    # Quantile binning
    ap.add_argument(
        "--quantile_breaks",
        type=float,
        nargs="+",
        default=None,
        help="Custom quantile breaks (e.g. 0.2 0.4 0.6 0.8 for 5 bins, 0.333 0.667 for 3 bins). "
             "Default: [0.2, 0.4, 0.6, 0.8].",
    )
    ap.add_argument(
        "--n_bins",
        type=int,
        default=0,
        help="Alternative to --quantile_breaks: number of equal-width quantile bins "
             "(e.g. 3 produces breaks at [0.333, 0.667]). Ignored if --quantile_breaks is set.",
    )

    ap.add_argument("--sentiment_backend", type=str, default="goemo",
                    choices=["sst2", "vader", "goemo", "both", "all"],
                    help="Sentiment backend: goemo (GoEmotions, Reddit-trained, default), "
                         "vader (rule-based), sst2 (transformer), both (SST-2 + VADER), "
                         "or all (GoEmotions + SST-2 + VADER).")
    ap.add_argument("--seed", type=int, default=142)
    ap.add_argument("--n_bootstrap", type=int, default=1000,
                    help="Number of bootstrap resamples for confidence intervals.")
    ap.add_argument("--ci_level", type=float, default=0.95,
                    help="Confidence interval level (default: 0.95).")
    ap.add_argument("--label_a", type=str, default="rage",
                    help="Name of cohort A (most negative, default: rage)")
    ap.add_argument("--label_b", type=str, default="empath",
                    help="Name of cohort B (most positive, default: empath)")
    ap.add_argument("--score_toxicity", action="store_true", default=True,
                    help="Score posts for toxicity using s-nlp/roberta_toxicity_classifier")
    ap.add_argument("--no_toxicity", dest="score_toxicity", action="store_false",
                    help="Skip toxicity scoring")
    args = ap.parse_args()
    return args


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    label_a = args.label_a
    label_b = args.label_b
    cohort_labels = [label_a, label_b]

    in_path = Path(args.input_parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)

    # Optional local quick-run truncation
    if int(args.max_rows) > 0 and len(df) > int(args.max_rows):
        df = df.head(int(args.max_rows)).copy()

    for c in (args.text_col, args.user_col):
        if c not in df.columns:
            raise KeyError(f"input_parquet missing required column {c!r}")

    # thresholds — resolve quantile breaks from CLI
    qbreaks = args.quantile_breaks
    if qbreaks is None and args.n_bins > 1:
        qbreaks = [i / args.n_bins for i in range(1, args.n_bins)]
    # default to 5-bin if nothing specified
    if qbreaks is None:
        qbreaks = [0.2, 0.4, 0.6, 0.8]

    thr_x = load_threshold_source(Path(args.threshold_source), args.threshold_col)
    thr = compute_thresholds(thr_x, quantile_breaks=qbreaks)
    print(f"[score] quantile breaks: {qbreaks} -> thresholds: {thr}", flush=True)

    # sentiment model selection
    model_id = (args.sentiment_model or "").strip()
    if not model_id:
        model_id = (os.environ.get("SENTIMENT_MODEL", "") or "").strip()
    if not model_id:
        model_id = str(_DEFAULT_SENT_MODEL)

    # device selection
    dev = args.device_id
    if dev is None:
        dev = _default_device_id()
    dev = int(dev)

    use_sst2 = args.sentiment_backend in ("sst2", "both", "all")
    use_vader = args.sentiment_backend in ("vader", "both", "all")
    use_goemo = args.sentiment_backend in ("goemo", "all")

    # Filter garbled text (shared across backends)
    coherent = df[args.text_col].astype(str).apply(_is_coherent).to_numpy()
    n_garbled = int((~coherent).sum())
    if n_garbled:
        print(f"[score] {n_garbled}/{len(df)} comments flagged as incoherent — excluded from scoring.", flush=True)

    df = df.copy()

    # --- SST-2 scoring ---
    if use_sst2:
        pipe = _build_sentiment_pipe(device_id=dev, model_id=model_id, batch_size=int(args.pipe_batch_size))
        if pipe is None:
            raise RuntimeError(
                "Failed to create sentiment pipeline. "
                "Check that transformers is installed and that --sentiment_model points to a valid model id/path."
            )
        pol = score_comments(df, pipe=pipe, text_col=args.text_col, chunk_size=int(args.chunk_size))
        if len(pol) != len(df):
            raise RuntimeError(f"polarity_batch length mismatch: got {len(pol)} for df rows {len(df)}")
        pol_arr = np.asarray(pol, dtype=np.float32)
        pol_arr = np.where(np.isfinite(pol_arr), pol_arr, 0.0)
        pol_arr = np.where(coherent, pol_arr, np.nan)
        df["sent_polarity"] = pol_arr.astype(float)
        df["sentiment_label"] = [
            label_from_thresholds(float(s), thr, label_a, label_b) if np.isfinite(s) else "unknown"
            for s in df["sent_polarity"].tolist()
        ]

    # --- VADER scoring ---
    if use_vader:
        print("[score] Computing VADER sentiment...", flush=True)
        texts_for_vader = df[args.text_col].astype(str).fillna("").tolist()
        vader_pol = score_texts_vader(texts_for_vader)
        vader_arr = np.asarray(vader_pol, dtype=np.float32)
        vader_arr = np.where(coherent, vader_arr, np.nan)
        col_name = "sent_polarity_vader" if use_sst2 else "sent_polarity"
        label_col = "sentiment_label_vader" if use_sst2 else "sentiment_label"
        df[col_name] = vader_arr.astype(float)
        df[label_col] = [
            label_from_thresholds(float(s), thr, label_a, label_b) if np.isfinite(s) else "unknown"
            for s in df[col_name].tolist()
        ]
        print(f"[score] VADER done: {int(np.isfinite(vader_arr).sum())} scored", flush=True)

    # --- GoEmotions scoring ---
    if use_goemo:
        print("[score] Computing GoEmotions sentiment (simple polarity)...", flush=True)
        texts_for_goemo = df[args.text_col].astype(str).fillna("").tolist()
        goemo_pol = score_texts_goemo(texts_for_goemo, device_id=dev, batch_size=int(args.pipe_batch_size), mode="simple")
        goemo_arr = np.asarray(goemo_pol, dtype=np.float32)
        goemo_arr = np.where(coherent, goemo_arr, np.nan)
        # Column naming: suffix when other backends also active, otherwise primary
        has_other = use_sst2 or use_vader
        col_name = "sent_polarity_goemo" if has_other else "sent_polarity"
        label_col = "sentiment_label_goemo" if has_other else "sentiment_label"
        df[col_name] = goemo_arr.astype(float)
        df[label_col] = [
            label_from_thresholds(float(s), thr, label_a, label_b) if np.isfinite(s) else "unknown"
            for s in df[col_name].tolist()
        ]
        print(f"[score] GoEmotions done: {int(np.isfinite(goemo_arr).sum())} scored", flush=True)

    # --- Toxicity scoring ---
    score_toxicity = getattr(args, "score_toxicity", True)
    if score_toxicity:
        try:
            from hypergames_utils import score_texts_toxicity
            print("[score] Computing toxicity (s-nlp/roberta_toxicity_classifier)...", flush=True)
            texts_for_tox = df[args.text_col].astype(str).fillna("").tolist()
            tox_scores = score_texts_toxicity(texts_for_tox, device_id=dev,
                                              batch_size=int(args.pipe_batch_size))
            tox_arr = np.asarray(tox_scores, dtype=np.float32)
            tox_arr = np.where(coherent, tox_arr, np.nan)
            df["toxicity_score"] = tox_arr.astype(float)
            print(f"[score] Toxicity done: {int(np.isfinite(tox_arr).sum())} scored, "
                  f"mean={float(np.nanmean(tox_arr)):.4f}", flush=True)
        except Exception as exc:
            print(f"[score] WARNING: toxicity scoring failed: {exc}", flush=True)
            df["toxicity_score"] = np.nan

    # If no backend produced the primary "sent_polarity" column, error out
    if "sent_polarity" not in df.columns:
        raise RuntimeError("No sentiment backend produced results")

    # Topic-relative scoring: subtract per-thread mean polarity
    gid_col = "gid" if "gid" in df.columns else None
    if gid_col:
        thread_mean = df.groupby(gid_col)["sent_polarity"].transform("mean")
        df["sent_polarity_relative"] = df["sent_polarity"] - thread_mean
    else:
        df["sent_polarity_relative"] = df["sent_polarity"]

    # per-user aggregation (pandas .mean()/.var() skip NaN, so garbled
    # comments are excluded from user-level sentiment mean)
    g = df.groupby(args.user_col, dropna=False)
    user_eval = g["sent_polarity"].agg(["count", "mean", "var"]).reset_index()
    user_eval.rename(columns={"count": "n_comments", "mean": "sent_mean", "var": "sent_var"}, inplace=True)

    # Topic-relative user-level aggregation
    user_rel = g["sent_polarity_relative"].agg(sent_mean_rel="mean").reset_index()
    user_eval = user_eval.merge(user_rel, on=args.user_col, how="left")

    if args.author_type_col in df.columns:
        # mean-then-label: matches build_hyperlora_forum.py _score_and_save_sentiment
        user_eval["mode_label"] = user_eval["sent_mean"].apply(
            lambda x: label_from_thresholds(float(x), thr, label_a, label_b) if pd.notna(x) else "unknown"
        )

        # Topic-relative label (uses same quantile breaks as absolute scoring)
        rel_valid = user_eval["sent_mean_rel"].dropna()
        if len(rel_valid) >= 10:
            rel_thr = compute_thresholds(rel_valid.values, quantile_breaks=qbreaks)
            user_eval["mode_label_rel"] = user_eval["sent_mean_rel"].apply(
                lambda x: label_from_thresholds(float(x), rel_thr, label_a, label_b) if pd.notna(x) else "unknown"
            )
            _rel_thr_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(rel_thr.items()))
            print(f"[score] topic-relative thresholds | {_rel_thr_str}", flush=True)
        else:
            user_eval["mode_label_rel"] = "unknown"

        author_type = g[args.author_type_col].agg(lambda x: x.value_counts().index[0]).reset_index()
        author_type.rename(columns={args.author_type_col: "author_type"}, inplace=True)

        user_eval = user_eval.merge(author_type, on=args.user_col, how="left")

        # match_extreme: 1 when author_type == mode_label AND author_type in {rage, empath}
        user_eval["match_extreme"] = (
            (user_eval["author_type"] == user_eval["mode_label"])
            & (user_eval["author_type"].isin(cohort_labels))
        ).astype(int)
        user_eval["match_extreme_rel"] = (
            (user_eval["author_type"] == user_eval.get("mode_label_rel", ""))
            & (user_eval["author_type"].isin(cohort_labels))
        ).astype(int)

        # --- Ordinal distance metric ---
        # How many bins off is the predicted label from the true author_type?
        _label_order = _make_label_order(5, label_a, label_b)
        _label_to_ord = {lbl: i for i, lbl in enumerate(_label_order)}
        _n_labels = len(_label_order) - 1  # max distance

        def _ordinal_dist(row):
            at = _label_to_ord.get(row["author_type"])
            ml = _label_to_ord.get(row["mode_label"])
            if at is None or ml is None:
                return np.nan
            return float(abs(at - ml))

        user_eval["ordinal_distance"] = user_eval.apply(_ordinal_dist, axis=1)
        user_eval["ordinal_distance_norm"] = user_eval["ordinal_distance"] / _n_labels

        # --- VADER user-level aggregation (parallel to SST-2) ---
        if "sent_polarity_vader" in df.columns:
            vader_user = g["sent_polarity_vader"].agg(vader_sent_mean="mean").reset_index()
            user_eval = user_eval.merge(vader_user, on=args.user_col, how="left")
            # VADER mode label + match_extreme
            user_eval["vader_mode_label"] = user_eval["vader_sent_mean"].apply(
                lambda x: label_from_thresholds(float(x), thr, label_a, label_b) if pd.notna(x) else "unknown"
            )
            user_eval["vader_match_extreme"] = (
                (user_eval["author_type"] == user_eval["vader_mode_label"])
                & (user_eval["author_type"].isin(cohort_labels))
            ).astype(int)

        # --- Confidence-weighted match rate ---
        # Users with more comments provide more reliable signal
        user_eval["match_extreme_confident"] = np.where(
            user_eval["n_comments"] >= 5,
            user_eval["match_extreme"].astype(float),
            np.nan,
        )

        # per-type summary stats
        summary: Dict[str, Any] = {}
        for t in user_eval["author_type"].dropna().unique():
            sub = user_eval[user_eval["author_type"] == t]
            if sub.empty:
                continue
            # Bootstrap CIs for key metrics
            sent_vals = sub["sent_mean"].dropna().values
            match_vals = sub["match_extreme"].dropna().values.astype(float)

            mean_sent_pt, mean_sent_lo, mean_sent_hi = bootstrap_ci(
                sent_vals, np.mean, n_boot=args.n_bootstrap,
                ci=args.ci_level, seed=args.seed)
            match_pt, match_lo, match_hi = bootstrap_ci(
                match_vals, np.mean, n_boot=args.n_bootstrap,
                ci=args.ci_level, seed=args.seed)

            # Ordinal distance bootstrap
            ord_vals = sub["ordinal_distance"].dropna().values
            ord_pt, ord_lo, ord_hi = bootstrap_ci(
                ord_vals, np.mean, n_boot=args.n_bootstrap,
                ci=args.ci_level, seed=args.seed)

            # Confidence-weighted match rate (users with >= 5 comments)
            conf_vals = sub["match_extreme_confident"].dropna().values.astype(float)
            match_confident_rate = float(np.nanmean(conf_vals)) if len(conf_vals) > 0 else np.nan

            # Comment-weighted match rate
            sub_extreme = sub[sub["author_type"].isin(cohort_labels)]
            if len(sub_extreme) > 0 and sub_extreme["n_comments"].sum() > 0:
                match_weighted = float(
                    (sub_extreme["match_extreme"] * sub_extreme["n_comments"]).sum()
                    / sub_extreme["n_comments"].sum()
                )
            else:
                match_weighted = np.nan

            d = {
                "users": int(len(sub)),
                "avg_comments": float(sub["n_comments"].mean()),
                "match_extreme_rate": float(sub["match_extreme"].mean()),
                "match_extreme_rate_ci95": [float(match_lo), float(match_hi)],
                "match_extreme_rate_weighted": float(match_weighted),
                "match_extreme_rate_confident": float(match_confident_rate),
                "mean_ordinal_distance": float(ord_pt),
                "mean_ordinal_distance_ci95": [float(ord_lo), float(ord_hi)],
                "mean_sent": float(sub["sent_mean"].mean()),
                "mean_sent_ci95": [float(mean_sent_lo), float(mean_sent_hi)],
            }
            if "match_extreme_rel" in sub.columns:
                d["match_extreme_rel_rate"] = float(sub["match_extreme_rel"].mean())

            # VADER parallel metrics (when running both backends)
            if "sent_polarity_vader" in user_eval.columns:
                vader_vals = sub["vader_sent_mean"].dropna().values if "vader_sent_mean" in sub.columns else np.array([])
                if len(vader_vals) > 0:
                    v_pt, v_lo, v_hi = bootstrap_ci(
                        vader_vals, np.mean, n_boot=args.n_bootstrap,
                        ci=args.ci_level, seed=args.seed)
                    d["vader_mean_sent"] = float(v_pt)
                    d["vader_mean_sent_ci95"] = [float(v_lo), float(v_hi)]
                if "vader_match_extreme" in sub.columns:
                    d["vader_match_extreme_rate"] = float(sub["vader_match_extreme"].mean())

            # Toxicity per cohort (toxicity_score is in df, not user_eval)
            if "toxicity_score" in df.columns:
                user_tox = df.loc[
                    df[args.user_col].isin(sub[args.user_col]),
                    "toxicity_score"
                ].dropna()
                if len(user_tox) > 0:
                    d["toxicity_rate"] = float((user_tox > 0.5).mean())
                    d["mean_toxicity"] = float(user_tox.mean())

            summary[str(t)] = d

        # confusion vs author_type if author_type looks like labels
        conf = (
            user_eval.dropna(subset=["author_type", "mode_label"])
            .groupby(["author_type", "mode_label"], dropna=False)
            .size()
            .reset_index(name="n_users")
        )
        conf.to_csv(out_dir / "confusion_author_type.csv", index=False)

        # topic-relative confusion
        if "mode_label_rel" in user_eval.columns:
            try:
                conf_rel = (
                    user_eval.dropna(subset=["author_type", "mode_label_rel"])
                    .groupby(["author_type", "mode_label_rel"], dropna=False)
                    .size()
                    .reset_index(name="n_users")
                )
                conf_rel.to_csv(out_dir / "confusion_author_type_relative.csv", index=False)
            except Exception:
                pass

    user_eval.to_csv(out_dir / "user_sentiment_eval.csv", index=False)
    df.to_parquet(out_dir / "forum_scored.parquet", index=False)

    meta: Dict[str, Any] = {
        "input": str(in_path),
        "out_dir": str(out_dir),
        "n_rows": int(len(df)),
        "n_garbled": int(n_garbled),
        "threshold_source": str(args.threshold_source),
        "threshold_col": str(args.threshold_col),
        "quantile_breaks": [float(q) for q in qbreaks],
        "thresholds": thr,
        "sentiment_backend": str(args.sentiment_backend),
        "sentiment_model": str(model_id),
        "device_id": int(dev),
        "pipe_batch_size": int(args.pipe_batch_size),
        "chunk_size": int(args.chunk_size),
        "max_rows": int(args.max_rows),
        "seed": int(args.seed),
    }

    if args.author_type_col in df.columns:
        meta["summary"] = summary

    # Fix 6: real-vs-generated sentiment correlation
    try:
        thr_path = Path(args.threshold_source)
        if thr_path.suffix.lower() == ".parquet":
            thr_df = pd.read_parquet(thr_path)
        else:
            thr_df = pd.read_csv(thr_path)
        if "target_user_id" in thr_df.columns and "gstat_user_sent_mean" in thr_df.columns:
            rs = thr_df[["target_user_id", "gstat_user_sent_mean"]].copy()
            rs["target_user_id"] = pd.to_numeric(rs["target_user_id"], errors="coerce").astype("Int64")
            rs = rs.dropna(subset=["target_user_id"])
            rs["target_user_id"] = rs["target_user_id"].astype(int)
            rs["gstat_user_sent_mean"] = pd.to_numeric(rs["gstat_user_sent_mean"], errors="coerce")
            rs = rs.rename(columns={"target_user_id": args.user_col, "gstat_user_sent_mean": "real_sent_mean"})
            cmp = user_eval.merge(rs, on=args.user_col, how="inner")
            vv = cmp[["real_sent_mean", "sent_mean"]].dropna()
            if len(vv) >= 5:
                corr = float(np.corrcoef(
                    vv["real_sent_mean"].astype(float).values,
                    vv["sent_mean"].astype(float).values,
                )[0, 1])
                if np.isfinite(corr):
                    meta["real_vs_gen_sent_corr"] = corr
    except Exception:
        pass

    (out_dir / "score_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"[done] wrote forum_scored.parquet + user_sentiment_eval.csv to {out_dir}")


if __name__ == "__main__":
    main()