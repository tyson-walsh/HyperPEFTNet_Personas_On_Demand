"""
label_synthetic_personas.py

One-shot CPU-only offline step that sits between Phase 1 (synthesis) and
Phase 2 of the GPU node pipeline.  Takes the `synthetic_personas.parquet`
emitted by `synthesize_personas.py` and projects every synth user onto
all nine behavioral label dimensions used for real users:

    Held-out probes (primary fidelity axes):
        politeness, curiosity, tempo, self_focus, expressiveness,
        anxiety, warmth, hostility
    Held-in sentiment (cohort routing + diagnostic only; NOT in fidelity cosine):
        sentiment_goemo

Projection methods (decision tree per-dim x per-stratum):

  (A) Direct gstat replay  - 4 binary z-scored probes + 1 raw-harmonic probe
      use per-synth gstat columns + `labels_manifest.json` boundaries.  Works
      for all three strata because the synth g-vector already lives in the
      same z-score space (Dirichlet mixtures or PCA extrapolation preserve
      feature axes).

  (B) Anchor-weighted projection  - the three GoEmo composites and
      sentiment_goemo need per-anchor real-user label lookups.  For in_hull
      / near_hull we use the Dirichlet mixing weights stored in
      `mixing_weights_json` with anchor UIDs from `source_anchor_uids`:

          y_synth = sum_i w_i * y_{anchor_i}

      This is mathematically exact for convex combinations of continuous
      labels.  Confidence = 1 - H(w)/H_max (high-entropy Dirichlet draws
      spread across many anchors and are less decisive).

  (C) kNN-g-space regression fallback  - required for far_from_hull (no
      anchors) AND computed as a second opinion for every synth so we can
      flag ambiguous cases where (B) and (C) disagree sharply.  k=10
      Gaussian kernel regression over author_static_10000.parquet (the
      same g-vectors that conditioned training).

Outputs
-------
    synthetic_personas_labeled.parquet
        All original columns + per-dim pol_{dim} continuous scores,
        cohort_goemo (rage/grumpy/mellow/calm/empath from quintile
        boundaries), ambiguity_score (max |anchor - knn| over dims),
        label_profile_json (full dict per user).

    label_coverage.csv
        Rows: (stratum, cohort_goemo, dim).
        Columns: n, mean, std, q25, q50, q75, n_ambiguous.

The coverage CSV is the pre-forum sanity check: a missing cell warns us
before Phase 2d burns wall-clock on a degenerate synth sub-population.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("label_synth")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

USER_COL = "target_user_id"

# --- Held-out probes derived directly from synth z-scored gstat features ----
# Each entry points at a column on the synth parquet + the manifest key that
# carries the (low_max_score, high_min_score) rank boundaries for that probe.
GSTAT_PROBES: Dict[str, Dict[str, str]] = {
    "politeness":      {"col": "gstat_profanity_ratio",   "low": "vulgar",      "high": "polite",      "manifest": "labels_politeness.csv"},
    "curiosity":       {"col": "gstat_question_ratio",    "low": "declarative", "high": "inquisitive", "manifest": "labels_curiosity.csv"},
    "tempo":           {"col": "gstat_reply_delay_mean",  "low": "reactive",    "high": "deliberate",  "manifest": "labels_tempo.csv"},
    "self_focus":      {"col": "gstat_firstperson_ratio", "low": "selfless",    "high": "egocentric",  "manifest": "labels_self_focus.csv"},
}

# Expressiveness is special: raw harmonic mean of caps & punct ratios.
EXPRESSIVENESS = {
    "caps":  "gstat_caps_ratio",
    "punct": "gstat_punct_ratio",
    "manifest": "labels_expressiveness.csv",
    "low_label":  "reserved",
    "high_label": "emphatic",
}

# GoEmo composites (anxiety / warmth / hostility) read emotion-mean columns
# from goemo_user_emotions.csv keyed by real-user UID.  The composite formulae
# and cut thresholds come from `goemo_emotion_labels_metadata.json`.
GOEMO_COMPOSITES = {
    "anxiety":   {"emotions": ["fear", "nervousness"],                 "low_label": "composed",  "high_label": "anxious"},
    "warmth":    {"emotions": ["caring", "love", "gratitude"],         "low_label": "detached",  "high_label": "warm"},
    "hostility": {"emotions": ["anger", "disgust", "disapproval"],     "low_label": "agreeable", "high_label": "hostile"},
}

# Held-in sentiment (cohort assignment + one-way diagnostic; excluded from
# the held-out-only fidelity cosine per plan Item 2.5).
SENT_QUINTILE_LABELS = ("rage", "grumpy", "mellow", "calm", "empath")


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _load_manifests(data_dir: Path) -> Tuple[Dict, Dict, Dict]:
    labels_manifest = _load_json(data_dir / "labels_manifest.json")
    goemo_sent_meta = _load_json(data_dir / "goemo_labels_metadata.json")
    goemo_emo_meta  = _load_json(data_dir / "goemo_emotion_labels_metadata.json")
    return labels_manifest, goemo_sent_meta, goemo_emo_meta


# --------------------------------------------------------------------------- #
# Direct gstat replay
# --------------------------------------------------------------------------- #
def _score_gstat_binary(synth: pd.DataFrame,
                         dim: str,
                         probe: Dict[str, str],
                         manifest: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (continuous_score, binary_label_index) for a gstat rank probe.
    binary_label_index: 0 = low_label, 1 = high_label, -1 = middle (unlabeled).
    """
    spec = manifest["outputs"][probe["manifest"]]
    low_max  = float(spec["boundary_low_max_score"])
    high_min = float(spec["boundary_high_min_score"])
    col = probe["col"]
    if col not in synth.columns:
        raise KeyError(
            f"synth parquet is missing expected column {col!r} for probe {dim!r}. "
            "Re-run synthesize_personas.py with a feature set that "
            "includes this gstat."
        )
    scores = synth[col].to_numpy(dtype=np.float32)
    labels = np.full(len(scores), -1, dtype=np.int8)
    labels[scores <= low_max] = 0
    labels[scores >= high_min] = 1
    return scores, labels


def _score_expressiveness(synth: pd.DataFrame,
                           manifest: Dict,
                           norm_stats: Dict[str, Dict[str, float]],
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expressiveness = harmonic mean of RAW caps_ratio * punct_ratio.  Synth
    values are z-scored, so invert via feature_norm_stats before thresholding.
    """
    caps_col  = EXPRESSIVENESS["caps"]
    punct_col = EXPRESSIVENESS["punct"]
    for c in (caps_col, punct_col):
        if c not in synth.columns:
            raise KeyError(f"synth parquet missing {c!r} for expressiveness probe")

    def _zinv(z: np.ndarray, col: str) -> np.ndarray:
        s = norm_stats.get(col, {})
        mu = float(s.get("mean", 0.0))
        sd = float(s.get("std", 1.0))
        if not np.isfinite(sd) or sd <= 0.0:
            return z
        return z.astype(np.float64) * sd + mu

    caps_raw  = _zinv(synth[caps_col].to_numpy(),  caps_col)
    punct_raw = _zinv(synth[punct_col].to_numpy(), punct_col)
    eps = 1e-12
    score = 2.0 * (caps_raw * punct_raw) / (caps_raw + punct_raw + eps)

    spec = manifest["outputs"][EXPRESSIVENESS["manifest"]]
    low_max  = float(spec["boundary_low_max_score"])
    high_min = float(spec["boundary_high_min_score"])
    labels = np.full(len(score), -1, dtype=np.int8)
    labels[score <= low_max]  = 0
    labels[score >= high_min] = 1
    return score.astype(np.float32), labels


# --------------------------------------------------------------------------- #
# Anchor-weighted projection for GoEmo dims + sentiment_goemo
# --------------------------------------------------------------------------- #
def _anchor_project(synth: pd.DataFrame,
                    values_by_uid: pd.Series,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each synth row with non-empty source_anchor_uids + mixing_weights_json,
    compute y_hat = sum_i w_i * y_{anchor_i}.  Missing anchors (UID not in
    values_by_uid) drop out and `coverage` records the surviving weight mass.

    Returns:
        values     : (N,) float32, NaN where no anchor info or zero coverage
        coverage   : (N,) float32 in [0, 1], fraction of weight with a label
        confidence : (N,) float32, 1 - H(w_covered) / log(k_eff)
    """
    n = len(synth)
    values = np.full(n, np.nan, dtype=np.float32)
    coverage = np.zeros(n, dtype=np.float32)
    confidence = np.zeros(n, dtype=np.float32)

    uid2val = values_by_uid.to_dict()
    anchor_uid_col = synth["source_anchor_uids"].tolist()
    weights_col    = synth["mixing_weights_json"].tolist()

    for i in range(n):
        # source_anchor_uids is a list-typed parquet column: pyarrow returns
        # each row as a numpy ndarray, so `arr or []` raises the ambiguous
        # truth-value error. Normalise to a Python list of ints first.
        raw_uids = anchor_uid_col[i]
        if raw_uids is None:
            continue
        if hasattr(raw_uids, "tolist"):
            uids = list(raw_uids.tolist())
        else:
            uids = list(raw_uids)
        if len(uids) == 0:
            continue
        w_raw = weights_col[i]
        if not isinstance(w_raw, str) or not w_raw:
            continue
        try:
            w = np.asarray(json.loads(w_raw), dtype=np.float64)
        except Exception:
            continue
        if len(w) != len(uids):
            continue

        mask = np.array([int(u) in uid2val for u in uids], dtype=bool)
        if not mask.any():
            continue
        w_eff = w[mask]
        vals  = np.asarray([uid2val[int(u)] for u in np.asarray(uids)[mask]],
                           dtype=np.float64)
        w_sum = float(w_eff.sum())
        if w_sum <= 0.0:
            continue
        w_norm = w_eff / w_sum
        values[i]   = float((w_norm * vals).sum())
        coverage[i] = float(w_sum)
        ent = -float((w_norm * np.log(np.clip(w_norm, 1e-12, 1.0))).sum())
        ent_max = float(np.log(len(w_norm))) if len(w_norm) > 1 else 1.0
        confidence[i] = 1.0 - (ent / ent_max if ent_max > 0 else 0.0)

    return values, coverage, confidence


# --------------------------------------------------------------------------- #
# kNN-g-space fallback
# --------------------------------------------------------------------------- #
@dataclass
class KNNContext:
    X_real: np.ndarray            # (10000, K) z-scored g-vectors
    uids_real: np.ndarray         # (10000,) int64
    uid2row: Dict[int, int]

    @classmethod
    def build(cls, author_parq: Path, feature_cols: List[str]) -> "KNNContext":
        df = pd.read_parquet(author_parq)
        df = df.sort_values(USER_COL).reset_index(drop=True)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"author_static parquet is missing required feature columns: {missing}. "
                "Did the synth parquet use a different K=18 feature set?"
            )
        X = df[feature_cols].to_numpy(dtype=np.float32)
        uids = df[USER_COL].to_numpy(dtype=np.int64)
        return cls(X_real=X, uids_real=uids,
                   uid2row={int(u): i for i, u in enumerate(uids)})

    def predict(self, X_query: np.ndarray, y: np.ndarray,
                mask_valid: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Gaussian-kernel k-NN regression.  y is aligned to self.uids_real;
        mask_valid marks real users that actually have a label (1 = usable).
        """
        X_real = self.X_real[mask_valid]
        y_real = y[mask_valid]
        if len(X_real) == 0:
            return np.full(len(X_query), np.nan, dtype=np.float32)
        # Pairwise squared distances (Nq, Nr); K=18 makes this cheap on CPU.
        d2 = (
            (X_query ** 2).sum(axis=1, keepdims=True)
            + (X_real ** 2).sum(axis=1)[None, :]
            - 2.0 * X_query @ X_real.T
        )
        k_eff = int(min(k, len(X_real)))
        idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(len(X_query))[:, None]
        d2_k = d2[rows, idx]
        # Median bandwidth per query for scale invariance.
        h = np.median(d2_k, axis=1, keepdims=True)
        h = np.where(h > 1e-12, h, 1.0)
        w = np.exp(-d2_k / (2.0 * h))
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum = np.where(w_sum > 1e-12, w_sum, 1.0)
        y_k = y_real[idx]
        y_hat = (w * y_k).sum(axis=1) / w_sum.squeeze(1)
        return y_hat.astype(np.float32)


# --------------------------------------------------------------------------- #
# Cohort assignment from sentiment_goemo score
# --------------------------------------------------------------------------- #
def _cohort_from_sent(sent_score: np.ndarray, quint: Dict[str, float]) -> List[str]:
    # quint keys are "0.2" ... "0.8".  Assign quintile label per synth.
    b20 = float(quint["0.2"])
    b40 = float(quint["0.4"])
    b60 = float(quint["0.6"])
    b80 = float(quint["0.8"])
    out: List[str] = []
    for s in sent_score:
        if np.isnan(s):
            out.append("unknown")
        elif s <= b20:
            out.append("rage")
        elif s <= b40:
            out.append("grumpy")
        elif s <= b60:
            out.append("mellow")
        elif s <= b80:
            out.append("calm")
        else:
            out.append("empath")
    return out


# --------------------------------------------------------------------------- #
# Coverage report
# --------------------------------------------------------------------------- #
def _build_coverage(df: pd.DataFrame, dims: List[str]) -> pd.DataFrame:
    rows = []
    for stratum, cohort in df.groupby(["stratum", "cohort_goemo"]).groups.keys():
        sub = df[(df["stratum"] == stratum) & (df["cohort_goemo"] == cohort)]
        for d in dims:
            col = f"pol_{d}"
            if col not in sub.columns:
                continue
            vals = sub[col].to_numpy(dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            n_amb = int((sub["ambiguity_score"].fillna(0) > 0.5).sum())
            rows.append({
                "stratum": stratum,
                "cohort_goemo": cohort,
                "dim": d,
                "n": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "std":  float(vals.std(ddof=0)) if len(vals) else np.nan,
                "q25":  float(np.quantile(vals, 0.25)) if len(vals) else np.nan,
                "q50":  float(np.quantile(vals, 0.50)) if len(vals) else np.nan,
                "q75":  float(np.quantile(vals, 0.75)) if len(vals) else np.nan,
                "n_ambiguous": n_amb,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth_parquet", required=True,
                    help="Phase 1 output: synthetic_personas.parquet")
    ap.add_argument("--data_dir", default="/workspace/hypernets/data",
                    help="Directory with labels_manifest.json, author_static, etc.")
    ap.add_argument("--author_parquet", default="",
                    help="Override path for author_static_10000.parquet")
    ap.add_argument("--feature_names_json", default="",
                    help="Optional path to hypernet feature_names.json. If absent, "
                         "we read `feature_names` from the sibling synth manifest.")
    ap.add_argument("--synth_meta_json", default="",
                    help="Optional path to the meta JSON written next to the synth parquet.")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write synthetic_personas_labeled.parquet + coverage CSV")
    ap.add_argument("--k_knn", type=int, default=10,
                    help="k for Gaussian-kernel g-space regression (default 10)")
    ap.add_argument("--ambiguity_threshold", type=float, default=0.5,
                    help="|anchor - knn| threshold to flag a dim as ambiguous")
    args = ap.parse_args()

    synth_path = Path(args.synth_parquet)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load synth + manifests ----
    LOG.info("Loading synth parquet: %s", synth_path)
    synth = pd.read_parquet(synth_path)
    LOG.info("Synth rows=%d cols=%d; strata=%s",
             len(synth), len(synth.columns),
             synth["stratum"].value_counts().to_dict())

    labels_manifest, goemo_sent_meta, goemo_emo_meta = _load_manifests(data_dir)
    norm_stats = _load_json(data_dir / "feature_norm_stats_10000.json")

    # Feature column order for kNN (must match the synth's K gstat columns).
    feat_cols: List[str]
    if args.feature_names_json and Path(args.feature_names_json).exists():
        raw = _load_json(Path(args.feature_names_json))
        feat_cols = list(raw.get("feature_names") or raw.get("features") or [])
    else:
        meta_path = Path(args.synth_meta_json) if args.synth_meta_json \
            else synth_path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = _load_json(meta_path)
            feat_cols = list(meta.get("feature_names") or [])
        else:
            feat_cols = [c for c in synth.columns if c.startswith("gstat_")]
    feat_cols = [c for c in feat_cols if c in synth.columns]
    if not feat_cols:
        raise RuntimeError("No gstat feature columns found on synth parquet.")
    LOG.info("K=%d feature columns for kNN: %s%s",
             len(feat_cols), feat_cols[:4], " ..." if len(feat_cols) > 4 else "")

    author_parq = Path(args.author_parquet) if args.author_parquet \
        else data_dir / "author_static_10000.parquet"
    knn = KNNContext.build(author_parq, feat_cols)
    LOG.info("kNN context built: X_real=%s", knn.X_real.shape)

    X_synth = synth[feat_cols].to_numpy(dtype=np.float32)

    # ---- Anchor-weighted + kNN lookup for GoEmo composites ----
    emo_df = pd.read_csv(data_dir / "goemo_user_emotions.csv")
    emo_df = emo_df.set_index(USER_COL)
    sent_df = pd.read_csv(data_dir / "goemo_user_sentiment.csv")
    sent_df = sent_df.set_index(USER_COL)

    # Values aligned to real-user row order (knn.uids_real).
    def _align_values(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        y = np.full(len(knn.uids_real), np.nan, dtype=np.float32)
        valid = np.zeros(len(knn.uids_real), dtype=bool)
        for u, v in series.items():
            row = knn.uid2row.get(int(u))
            if row is None:
                continue
            y[row] = float(v)
            valid[row] = True
        return y, valid

    # Container for per-dim results.
    per_dim: Dict[str, Dict[str, np.ndarray]] = {}

    # --- sentiment_goemo (held-in; cohort + diagnostic only) ---
    sent_series = sent_df["goemo_sent_mean"].astype(float)
    y_sent_real, m_sent = _align_values(sent_series)
    sent_by_uid = pd.Series(dict(zip(knn.uids_real, y_sent_real)))
    anchor_sent, cov_sent, conf_sent = _anchor_project(synth, sent_by_uid)
    knn_sent = knn.predict(X_synth, y_sent_real, m_sent, k=args.k_knn)
    combined_sent = np.where(np.isnan(anchor_sent), knn_sent, anchor_sent)
    per_dim["sentiment_goemo"] = {
        "anchor": anchor_sent, "knn": knn_sent,
        "value": combined_sent,
        "coverage": cov_sent, "confidence": conf_sent,
    }

    # --- GoEmo composites (anxiety / warmth / hostility) ---
    for dim, spec in GOEMO_COMPOSITES.items():
        cols = [f"{e}_mean" for e in spec["emotions"]]
        missing = [c for c in cols if c not in emo_df.columns]
        if missing:
            raise KeyError(f"{dim}: goemo_user_emotions.csv missing {missing}")
        y_series = emo_df[cols].mean(axis=1)
        y_real, m_real = _align_values(y_series)
        by_uid = pd.Series(dict(zip(knn.uids_real, y_real)))
        anc, cov, conf = _anchor_project(synth, by_uid)
        knn_y = knn.predict(X_synth, y_real, m_real, k=args.k_knn)
        combined = np.where(np.isnan(anc), knn_y, anc)
        per_dim[dim] = {
            "anchor": anc, "knn": knn_y, "value": combined,
            "coverage": cov, "confidence": conf,
        }

    # --- Direct-gstat probes (politeness / curiosity / tempo / self_focus) ---
    for dim, probe in GSTAT_PROBES.items():
        scores, bin_idx = _score_gstat_binary(synth, dim, probe, labels_manifest)
        per_dim[dim] = {
            "anchor": scores.astype(np.float32),
            "knn":    scores.astype(np.float32),
            "value":  scores.astype(np.float32),
            "coverage":   np.ones(len(synth), dtype=np.float32),
            "confidence": np.ones(len(synth), dtype=np.float32),
            "binary":  bin_idx,
        }

    # --- Expressiveness (raw harmonic mean) ---
    expr_scores, expr_bin = _score_expressiveness(synth, labels_manifest, norm_stats)
    per_dim["expressiveness"] = {
        "anchor": expr_scores, "knn": expr_scores, "value": expr_scores,
        "coverage": np.ones(len(synth), dtype=np.float32),
        "confidence": np.ones(len(synth), dtype=np.float32),
        "binary": expr_bin,
    }

    # ---- Cohort assignment from sentiment_goemo ----
    quint = goemo_sent_meta["rank_boundaries_score_at_fraction"]
    cohort = _cohort_from_sent(per_dim["sentiment_goemo"]["value"], quint)

    # ---- Assemble output frame ----
    out = synth.copy()
    all_dims = list(GSTAT_PROBES.keys()) + ["expressiveness"] + \
        list(GOEMO_COMPOSITES.keys()) + ["sentiment_goemo"]

    profile_rows: List[str] = []
    ambig = np.zeros(len(synth), dtype=np.float32)
    held_out_dims = [d for d in all_dims if d != "sentiment_goemo"]

    for d in all_dims:
        out[f"pol_{d}"] = per_dim[d]["value"].astype(np.float32)
        out[f"pol_{d}_anchor"] = per_dim[d]["anchor"].astype(np.float32)
        out[f"pol_{d}_knn"] = per_dim[d]["knn"].astype(np.float32)
        out[f"pol_{d}_confidence"] = per_dim[d]["confidence"].astype(np.float32)
        out[f"pol_{d}_coverage"] = per_dim[d]["coverage"].astype(np.float32)
        diff = np.abs(per_dim[d]["anchor"] - per_dim[d]["knn"])
        diff = np.nan_to_num(diff, nan=0.0)
        ambig = np.maximum(ambig, diff)

    out["cohort_goemo"] = cohort
    out["ambiguity_score"] = ambig.astype(np.float32)

    for i in range(len(out)):
        prof = {}
        for d in all_dims:
            prof[d] = {
                "value": float(per_dim[d]["value"][i])
                    if not np.isnan(per_dim[d]["value"][i]) else None,
                "anchor": float(per_dim[d]["anchor"][i])
                    if not np.isnan(per_dim[d]["anchor"][i]) else None,
                "knn": float(per_dim[d]["knn"][i])
                    if not np.isnan(per_dim[d]["knn"][i]) else None,
                "confidence": float(per_dim[d]["confidence"][i]),
                "coverage":   float(per_dim[d]["coverage"][i]),
                "method": ("gstat_direct"
                           if d in GSTAT_PROBES or d == "expressiveness"
                           else ("anchor_dirichlet"
                                 if not np.isnan(per_dim[d]["anchor"][i])
                                 else "knn_gspace")),
                "policy_role": ("cohort_diagnostic_only"
                                if d == "sentiment_goemo" else "held_out_fidelity"),
            }
        profile_rows.append(json.dumps(prof))
    out["label_profile_json"] = profile_rows

    # ---- Write outputs ----
    out_parq = out_dir / "synthetic_personas_labeled.parquet"
    out.to_parquet(out_parq, index=False)
    LOG.info("wrote %s rows=%d cols=%d", out_parq, len(out), len(out.columns))

    cov_df = _build_coverage(out, all_dims)
    cov_csv = out_dir / "label_coverage.csv"
    cov_df.to_csv(cov_csv, index=False)
    LOG.info("wrote %s rows=%d", cov_csv, len(cov_df))

    # ---- Sanity banner ----
    LOG.info("=== per-dim summary (primary held-out fidelity axes) ===")
    for d in held_out_dims:
        v = out[f"pol_{d}"].to_numpy(dtype=np.float64)
        v = v[~np.isnan(v)]
        LOG.info(" %-16s  n=%5d  mean=%+.4f  std=%.4f",
                 d, len(v), v.mean() if len(v) else np.nan,
                 v.std(ddof=0) if len(v) else np.nan)
    cohort_counts = out["cohort_goemo"].value_counts().to_dict()
    LOG.info("cohort_goemo (held-in, cohort routing only): %s", cohort_counts)
    LOG.info("ambiguity_score: median=%.4f  p95=%.4f  max=%.4f",
             float(np.median(out["ambiguity_score"])),
             float(np.quantile(out["ambiguity_score"], 0.95)),
             float(out["ambiguity_score"].max()))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOG.exception("label_synthetic_personas failed: %s", exc)
        sys.exit(1)
