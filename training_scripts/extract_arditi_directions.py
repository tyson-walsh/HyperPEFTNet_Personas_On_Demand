#!/usr/bin/env python3
"""
extract_arditi_directions.py

Direction-extraction protocol for the Arditi Patch (Appendix V.5b in
zzzzz_COMPLETE_TECHNICAL_REFERENCE.md).

Given a trained HyperPEFT-LoRA checkpoint (M1 production), compute the
per-layer dominant residual-stream direction by contrasting:
    delta-OFF  : forward pass with force_zero_delta=True (Pile-mode)
    delta-ON   : forward pass with the user's hypernetwork delta applied

The direction at each transformer block is

    r_ell = mean_{prompt, user} h_off[ell] - mean_{prompt, user} h_on[ell]

normalized to unit length. At inference (build_hyperlora_forum.py with
--arditi_patch DIRECTIONS_PATH) we project this direction out of the
residual stream at every late layer:

    h' = h - alpha * (h . r_hat) * r_hat

This file produces arditi_directions.safetensors with keys "layer_0" ...
"layer_L" where L is the number of transformer blocks. Layer 0 is the
embedding output (kept for diagnostics, not used by the inference patch).

Inspired by Arditi et al. 2024 NeurIPS, *Refusal in Language Models Is
Mediated by a Single Direction*. Their protocol contrasts harmful vs
harmless prompts; ours contrasts persona-on vs persona-off forward passes
on the same prompts. Both produce a residual-stream direction recoverable
by simple difference-of-means.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# HyperPEFTLoRAEngine is heavy; we import it lazily so --help is fast.
LOG = logging.getLogger("extract_arditi_directions")


# ----------------------------------------------------------------------------
# Default forum-reply prompts. Twenty short, generic, topic-diverse prompts
# that the M1 forum simulator would plausibly receive. Direction extraction
# is robust to prompt-set composition (Arditi 2024 §4); the only requirement
# is the same prompts are used for delta-on and delta-off, so the difference
# of means isolates the per-user delta's residual-stream effect rather than
# any prompt-specific artifact.
# ----------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "I just got back from the store and I want to talk about ",
    "Honestly, my opinion on this whole thing is that ",
    "What really gets me about today is ",
    "Let me tell you what happened this morning. ",
    "Reading the news this week, I keep thinking about ",
    "I've been working on a new project and the hardest part is ",
    "My weekend plans got changed and now I'm dealing with ",
    "Talking with my coworker yesterday, we ended up discussing ",
    "The thing nobody talks about with this is ",
    "Walking home tonight, I started thinking about ",
    "I tried that new restaurant downtown and ",
    "Looking back at the past month, the part that stands out is ",
    "Something happened at the meeting today that I want to share. ",
    "I keep going back and forth on whether ",
    "If I'm being completely honest, ",
    "The most surprising thing about this week was ",
    "I had a long conversation with my neighbor about ",
    "Reading through the comments on that thread, I noticed ",
    "Three things I learned from this experience: ",
    "What I keep coming back to is the question of whether ",
]


def _extract_layers_from_engine(engine):
    """Walk PEFT wrapper to find the underlying transformer block stack."""
    obj = engine.backbone
    # PEFT wrap: backbone -> base_model -> model -> gpt_neox -> layers
    for attr in ("base_model", "model"):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
    if hasattr(obj, "gpt_neox"):
        return obj.gpt_neox.layers
    if hasattr(obj, "transformer") and hasattr(obj.transformer, "h"):
        return obj.transformer.h  # GPT-2 family
    if hasattr(obj, "layers"):
        return obj.layers  # decoder layers at this depth
    raise RuntimeError("Could not locate transformer block stack on backbone.")


def _sample_users_cohort(
    *,
    labels_csv: Path,
    cohorts: List[str],
    n_users_per_cohort: int,
    seed: int,
) -> List[int]:
    """Legacy sentiment-cohort sampling. BIASED: only controls sentiment axis;
    other 8 label dimensions are uncontrolled. Use --sampling balanced or
    --sampling random instead for cleanest Pile-mode extraction."""
    rng = np.random.default_rng(seed)
    ldf = pd.read_csv(labels_csv)
    if "sentiment_label" in ldf.columns:
        ldf["label"] = ldf["sentiment_label"].astype(str).str.lower()
    elif "label" in ldf.columns:
        ldf["label"] = ldf["label"].astype(str).str.lower()
    else:
        raise KeyError(
            f"labels_csv {labels_csv} must have 'label' or 'sentiment_label' column"
        )
    user_ids: List[int] = []
    for cohort in cohorts:
        pool = ldf[ldf["label"] == cohort.lower()]["target_user_id"].astype(int).tolist()
        if not pool:
            LOG.warning("[direction] no users for cohort=%s; skipping", cohort)
            continue
        take = min(n_users_per_cohort, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        sampled = [int(pool[i]) for i in idx]
        user_ids.extend(sampled)
        LOG.info("[direction] cohort=%s sampled=%d (pool=%d)", cohort, take, len(pool))
    return user_ids


# Polar tail-pair convention per known label CSV (low_label, high_label).
# For sentiment with 5 classes (rage/grumpy/calm/mellow/empath), we pick the
# two extremes (rage / empath) so the perturbation cancels along the
# sentiment axis. For binary CSVs, the two classes ARE the tails.
_POLAR_PAIRS = {
    "labels_sentiment_goemo.csv": ("rage", "empath"),
    "labels_politeness.csv":      ("vulgar", "polite"),
    "labels_self_focus.csv":      ("egocentric", "selfless"),
    "labels_curiosity.csv":       ("declarative", "inquisitive"),
    "labels_expressiveness.csv":  ("reserved", "emphatic"),
    "labels_tempo.csv":           ("reactive", "deliberate"),
    "labels_anxiety.csv":         ("anxious", "composed"),
    "labels_warmth.csv":          ("detached", "warm"),
    "labels_hostility.csv":       ("hostile", "agreeable"),
}


def _sample_users_balanced(
    *,
    label_csv_dir: Path,
    label_dim_files: List[str],
    n_per_tail: int,
    seed: int,
    **_unused,  # absorb legacy quintile_low/high args
) -> List[int]:
    """Stratified balanced sampling across N label dimensions. Each label CSV
    has a (target_user_id, label) schema with categorical labels. For each
    CSV, sample n_per_tail users from each of two polar classes (the tails
    of that behavioral dimension). By construction the average per-user
    perturbation is ~= 0 along every dimension, since the two polar classes
    in that dim are equally represented.

    Polar pairs are taken from the _POLAR_PAIRS table when the CSV is known;
    for unknown CSVs we fall back to the two MOST FREQUENT distinct labels.
    """
    rng = np.random.default_rng(seed)
    user_ids: List[int] = []
    for fname in label_dim_files:
        path = label_csv_dir / fname
        if not path.exists():
            LOG.warning("[direction] balanced: label file missing %s; skipping", path)
            continue
        ldf = pd.read_csv(path)
        if "target_user_id" not in ldf.columns or "label" not in ldf.columns:
            LOG.warning("[direction] balanced: %s missing target_user_id/label; skipping",
                        fname)
            continue
        label_str = ldf["label"].astype(str).str.strip().str.lower()
        # Resolve polar pair: prefer hardcoded mapping, else top-2 most frequent
        pair = _POLAR_PAIRS.get(fname)
        if pair is None:
            counts = label_str.value_counts()
            top2 = counts.head(2).index.tolist()
            if len(top2) < 2:
                LOG.warning("[direction] balanced: %s has < 2 distinct labels; skipping", fname)
                continue
            pair = (str(top2[0]).lower(), str(top2[1]).lower())
            LOG.info("[direction] balanced: %s polar-pair auto-detected = %s", fname, pair)
        lo_label, hi_label = (pair[0].lower(), pair[1].lower())
        lo_pool = ldf[label_str == lo_label]["target_user_id"].astype(int).tolist()
        hi_pool = ldf[label_str == hi_label]["target_user_id"].astype(int).tolist()
        take_lo = min(n_per_tail, len(lo_pool))
        take_hi = min(n_per_tail, len(hi_pool))
        if take_lo:
            idx = rng.choice(len(lo_pool), size=take_lo, replace=False)
            user_ids.extend([int(lo_pool[i]) for i in idx])
        if take_hi:
            idx = rng.choice(len(hi_pool), size=take_hi, replace=False)
            user_ids.extend([int(hi_pool[i]) for i in idx])
        LOG.info("[direction] balanced: %s polar=%s/%s lo_take=%d hi_take=%d (pools=%d/%d)",
                 fname, lo_label, hi_label, take_lo, take_hi, len(lo_pool), len(hi_pool))
    # Deduplicate while preserving order
    seen = set()
    out: List[int] = []
    for u in user_ids:
        if u not in seen:
            seen.add(u)
            out.append(u)
    LOG.info("[direction] balanced: %d unique users across %d label dimensions",
             len(out), len(label_dim_files))
    return out


def _sample_users_random(
    *,
    author_parquet: Path,
    n_users: int,
    seed: int,
) -> List[int]:
    """Uniform random sample from the entire training population. Relies on
    LLN: with n_users >= 500, mean per-user perturbation ~= 0 in every
    dimension because the hypernet was trained on a centered descriptor."""
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(author_parquet)
    if "target_user_id" not in df.columns:
        raise KeyError(f"{author_parquet} missing 'target_user_id' column")
    pool = df["target_user_id"].astype(int).tolist()
    take = min(n_users, len(pool))
    idx = rng.choice(len(pool), size=take, replace=False)
    out = [int(pool[i]) for i in idx]
    LOG.info("[direction] random: %d users sampled from pool=%d", take, len(pool))
    return out


def load_user_g_vectors(
    *,
    engine,
    author_parquet: Path,
    labels_csv: Path,
    label_csv_dir: Optional[Path],
    label_dim_files: List[str],
    feature_names: List[str],
    sampling: str,
    n_users_per_cohort: int,
    n_per_tail: int,
    n_random: int,
    cohorts: List[str],
    quintile_low: float,
    quintile_high: float,
    seed: int,
) -> Dict[int, torch.Tensor]:
    """Build user_id -> g vector tensor map, with sampling chosen by `sampling`.

    sampling:
        'cohort'   - legacy sentiment-cohort sampling (BIASED, only controls 1
                     of 9 dims; left in for backward compatibility).
        'balanced' - stratified across all label dims in label_dim_files;
                     top + bottom quintile per dim cancels each dim.
        'random'   - uniform random from entire training pool; LLN.
    """
    K = len(feature_names)

    if sampling == "cohort":
        user_ids = _sample_users_cohort(
            labels_csv=labels_csv, cohorts=cohorts,
            n_users_per_cohort=n_users_per_cohort, seed=seed)
    elif sampling == "balanced":
        if label_csv_dir is None:
            raise ValueError("--sampling balanced requires --label_csv_dir")
        user_ids = _sample_users_balanced(
            label_csv_dir=label_csv_dir, label_dim_files=label_dim_files,
            n_per_tail=n_per_tail,
            quintile_low=quintile_low, quintile_high=quintile_high, seed=seed)
    elif sampling == "random":
        user_ids = _sample_users_random(
            author_parquet=author_parquet, n_users=n_random, seed=seed)
    else:
        raise ValueError(f"unknown --sampling={sampling!r}")

    author_df = pd.read_parquet(author_parquet).set_index("target_user_id")
    gvec_tensors: Dict[int, torch.Tensor] = {}
    n_zeroed = 0
    for uid in user_ids:
        if uid not in author_df.index:
            gvec_tensors[uid] = torch.zeros(
                (1, K), dtype=torch.float32, device=engine.device
            )
            n_zeroed += 1
            continue
        row = author_df.loc[uid]
        vec = np.zeros(K, dtype=np.float32)
        for i, fname in enumerate(feature_names[:K]):
            v = row.get(fname, 0.0)
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    fv = 0.0
            except Exception:
                fv = 0.0
            vec[i] = fv
        gvec_tensors[uid] = torch.tensor(
            [vec], dtype=torch.float32, device=engine.device
        )
    if n_zeroed:
        LOG.warning(
            "[direction] %d/%d users not in author_parquet; using zero-g fallback",
            n_zeroed, len(user_ids),
        )
    return gvec_tensors


def collect_per_user_per_layer_activations(
    *,
    engine,
    user_ids: List[int],
    gvec_tensors: Dict[int, torch.Tensor],
    prompts: List[str],
    max_len: int,
    force_zero: bool,
    aggregator: str,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """Run forward passes and return per-user, per-layer mean activation.

    Returns {user_id: {layer_idx: torch.Tensor[H]}} averaged across prompts.
    Used by extract_multi_axis_directions() so we can compute many different
    contrastives (main / per-dim / per-cohort / persona) from the same set of
    forward passes.
    """
    per_user_sums: Dict[int, Dict[int, torch.Tensor]] = {uid: {} for uid in user_ids}
    per_user_count: Dict[int, int] = {uid: 0 for uid in user_ids}

    for prompt in prompts:
        enc = engine.tok(prompt, return_tensors="pt", truncation=True,
                          max_length=int(max_len)).to(engine.device)
        for uid in user_ids:
            g = gvec_tensors[uid]
            ids_exp = enc["input_ids"].contiguous()
            mask_exp = enc["attention_mask"].contiguous()
            with torch.no_grad():
                outs = engine.model(
                    input_ids=ids_exp,
                    attention_mask=mask_exp,
                    global_features=g,
                    force_zero_delta=force_zero,
                    return_hidden_only=False,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden = getattr(outs, "hidden_states", None)
            if hidden is None and isinstance(outs, dict):
                hidden = outs.get("hidden_states", None)
            if hidden is None:
                raise RuntimeError("engine.model did not return hidden_states")
            for ell, h in enumerate(hidden):
                if aggregator == "last_token":
                    vec = h[:, -1, :]
                else:
                    vec = h.mean(dim=1)
                vec = vec.detach().float().cpu().sum(dim=0)  # [H]
                if ell not in per_user_sums[uid]:
                    per_user_sums[uid][ell] = vec
                else:
                    per_user_sums[uid][ell] = per_user_sums[uid][ell] + vec
            per_user_count[uid] += 1

    out: Dict[int, Dict[int, torch.Tensor]] = {}
    for uid in user_ids:
        c = max(1, per_user_count[uid])
        out[uid] = {ell: per_user_sums[uid][ell] / float(c)
                    for ell in per_user_sums[uid]}
    return out


def _normalize_per_layer(
    diff: Dict[int, torch.Tensor],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, float]]:
    """Normalize a per-layer direction dict to unit vectors. Returns (unit_dirs, raw_norms)."""
    norms: Dict[int, float] = {}
    units: Dict[int, torch.Tensor] = {}
    for ell, v in diff.items():
        n = float(torch.linalg.norm(v).item())
        norms[ell] = n
        if n < 1e-8:
            units[ell] = v.float().contiguous()
        else:
            units[ell] = (v / n).float().contiguous()
    return units, norms


def extract_multi_axis_directions(
    *,
    engine,
    label_csv_dir: Path,
    label_dim_files: List[str],
    author_parquet: Path,
    feature_names: List[str],
    n_per_tail: int,
    n_per_cohort: int,
    prompts: List[str],
    max_len: int,
    aggregator: str,
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Comprehensive extraction that produces all 4-Patch direction families
    in a single forward-pass batch.

    Outputs (safetensors keys):
        main/layer_<L>          balanced Pile-mode direction
        dim/<dimname>/layer_<L> 9 polar-axis directions (one per label CSV)
        cohort/<c>/layer_<L>    rage / empath / calm cohort delta-OFF minus delta-ON
        persona/<c>/layer_<L>   negation of cohort/<c> (cohort persona direction)

    Procedure:
      1. Sample users to cover (a) polar pairs of all 9 label dims and
         (b) the three sentiment cohorts {rage, empath, calm}.
      2. Run BOTH delta-OFF and delta-ON forward passes on each user, on
         every prompt; cache per-user-per-layer mean activations.
      3. Compute the four direction families by averaging the appropriate
         user subsets.
      4. Normalize each per-layer direction and emit a structured dict.

    Returns (directions_flat_dict, summary_meta) where directions_flat_dict
    has keys like 'main/layer_5' or 'dim/sentiment/layer_5' mapping to
    [H]-shaped float32 tensors.
    """
    # --- 1. sample users for each polar pair + sentiment cohorts ---
    rng = np.random.default_rng(seed)
    polar_low: Dict[str, List[int]] = {}
    polar_high: Dict[str, List[int]] = {}

    for fname in label_dim_files:
        path = label_csv_dir / fname
        if not path.exists():
            LOG.warning("[multi-axis] missing %s; skipping dim", path)
            continue
        ldf = pd.read_csv(path)
        if "target_user_id" not in ldf.columns or "label" not in ldf.columns:
            LOG.warning("[multi-axis] %s missing target_user_id/label", fname)
            continue
        label_str = ldf["label"].astype(str).str.strip().str.lower()
        pair = _POLAR_PAIRS.get(fname)
        if pair is None:
            counts = label_str.value_counts()
            top2 = counts.head(2).index.tolist()
            if len(top2) < 2:
                continue
            pair = (str(top2[0]).lower(), str(top2[1]).lower())
            LOG.info("[multi-axis] %s polar-pair auto-detected = %s", fname, pair)
        lo_label, hi_label = pair[0].lower(), pair[1].lower()
        lo_pool = ldf[label_str == lo_label]["target_user_id"].astype(int).tolist()
        hi_pool = ldf[label_str == hi_label]["target_user_id"].astype(int).tolist()
        take_lo = min(n_per_tail, len(lo_pool))
        take_hi = min(n_per_tail, len(hi_pool))
        if take_lo:
            idx = rng.choice(len(lo_pool), size=take_lo, replace=False)
            polar_low[fname] = [int(lo_pool[i]) for i in idx]
        if take_hi:
            idx = rng.choice(len(hi_pool), size=take_hi, replace=False)
            polar_high[fname] = [int(hi_pool[i]) for i in idx]
        LOG.info("[multi-axis] %s polar=%s/%s lo=%d hi=%d (pools=%d/%d)",
                 fname, lo_label, hi_label, take_lo, take_hi, len(lo_pool), len(hi_pool))

    # Sentiment cohorts (rage, empath, calm) for cohort/<c> direction extraction
    sentiment_path = label_csv_dir / "labels_sentiment_goemo.csv"
    cohort_users: Dict[str, List[int]] = {"rage": [], "empath": [], "calm": []}
    if sentiment_path.exists():
        sdf = pd.read_csv(sentiment_path)
        sdf_lab = sdf["label"].astype(str).str.strip().str.lower()
        for c in cohort_users.keys():
            pool = sdf[sdf_lab == c]["target_user_id"].astype(int).tolist()
            take = min(n_per_cohort, len(pool))
            if take:
                idx = rng.choice(len(pool), size=take, replace=False)
                cohort_users[c] = [int(pool[i]) for i in idx]
            LOG.info("[multi-axis] cohort %s: sampled %d (pool=%d)", c, take, len(pool))

    # Union of all needed users (deduplicate)
    all_users = set()
    for users in polar_low.values():
        all_users.update(users)
    for users in polar_high.values():
        all_users.update(users)
    for users in cohort_users.values():
        all_users.update(users)
    user_ids = sorted(all_users)
    LOG.info("[multi-axis] total unique users: %d", len(user_ids))

    if not user_ids:
        raise RuntimeError("multi-axis extraction sampled zero users; check label CSVs")

    # --- 2. build g vectors and run BOTH delta-OFF and delta-ON forward passes ---
    K = len(feature_names)
    author_df = pd.read_parquet(author_parquet).set_index("target_user_id")
    gvec_tensors: Dict[int, torch.Tensor] = {}
    for uid in user_ids:
        if uid not in author_df.index:
            gvec_tensors[uid] = torch.zeros((1, K), dtype=torch.float32,
                                             device=engine.device)
            continue
        row = author_df.loc[uid]
        vec = np.zeros(K, dtype=np.float32)
        for i, fname in enumerate(feature_names[:K]):
            try:
                fv = float(row.get(fname, 0.0))
                if not np.isfinite(fv):
                    fv = 0.0
            except Exception:
                fv = 0.0
            vec[i] = fv
        gvec_tensors[uid] = torch.tensor([vec], dtype=torch.float32,
                                          device=engine.device)

    LOG.info("[multi-axis] starting forward passes (n_users=%d x n_prompts=%d x 2 modes = %d)",
             len(user_ids), len(prompts), len(user_ids) * len(prompts) * 2)

    LOG.info("[multi-axis] pass A: force_zero_delta=True (delta-OFF)")
    h_off = collect_per_user_per_layer_activations(
        engine=engine, user_ids=user_ids, gvec_tensors=gvec_tensors,
        prompts=prompts, max_len=max_len, force_zero=True, aggregator=aggregator,
    )
    LOG.info("[multi-axis] pass B: force_zero_delta=False (delta-ON)")
    h_on = collect_per_user_per_layer_activations(
        engine=engine, user_ids=user_ids, gvec_tensors=gvec_tensors,
        prompts=prompts, max_len=max_len, force_zero=False, aggregator=aggregator,
    )

    # --- 3. compute directions ---
    layer_idxs = sorted(next(iter(h_off.values())).keys())

    def _avg_users(table: Dict[int, Dict[int, torch.Tensor]],
                   users: List[int]) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for ell in layer_idxs:
            out[ell] = torch.zeros_like(table[users[0]][ell])
        for u in users:
            for ell in layer_idxs:
                out[ell] = out[ell] + table[u][ell]
        for ell in layer_idxs:
            out[ell] = out[ell] / float(len(users))
        return out

    flat_dirs: Dict[str, torch.Tensor] = {}
    summary_norms: Dict[str, Dict[int, float]] = {}

    # MAIN: balanced across all polar-pair users
    main_users = sorted({u for users in polar_low.values() for u in users}
                         | {u for users in polar_high.values() for u in users})
    if main_users:
        main_off = _avg_users(h_off, main_users)
        main_on = _avg_users(h_on, main_users)
        diff = {ell: main_off[ell] - main_on[ell] for ell in layer_idxs}
        units, norms = _normalize_per_layer(diff)
        for ell, v in units.items():
            flat_dirs[f"main/layer_{ell}"] = v
        summary_norms["main"] = norms

    # DIM/<name>: high-tail mean(on) MINUS low-tail mean(on); points toward high
    for fname in label_dim_files:
        if fname not in polar_low or fname not in polar_high:
            continue
        lo_users, hi_users = polar_low[fname], polar_high[fname]
        if not lo_users or not hi_users:
            continue
        on_lo = _avg_users(h_on, lo_users)
        on_hi = _avg_users(h_on, hi_users)
        diff = {ell: on_hi[ell] - on_lo[ell] for ell in layer_idxs}
        units, norms = _normalize_per_layer(diff)
        dim_name = fname.replace("labels_", "").replace(".csv", "").replace("_goemo", "")
        for ell, v in units.items():
            flat_dirs[f"dim/{dim_name}/layer_{ell}"] = v
        summary_norms[f"dim/{dim_name}"] = norms

    # COHORT/<c> + PERSONA/<c>: per-cohort delta-OFF MINUS delta-ON, and its negative
    for c, users in cohort_users.items():
        if not users:
            continue
        c_off = _avg_users(h_off, users)
        c_on = _avg_users(h_on, users)
        diff_cohort = {ell: c_off[ell] - c_on[ell] for ell in layer_idxs}
        units_c, norms_c = _normalize_per_layer(diff_cohort)
        for ell, v in units_c.items():
            flat_dirs[f"cohort/{c}/layer_{ell}"] = v
        summary_norms[f"cohort/{c}"] = norms_c
        # persona = -cohort (already unit-normed per layer)
        for ell, v in units_c.items():
            flat_dirs[f"persona/{c}/layer_{ell}"] = (-v).contiguous()

    LOG.info("[multi-axis] extracted %d direction groups, %d flat keys",
             len(summary_norms), len(flat_dirs))

    return flat_dirs, {
        "n_users": len(user_ids),
        "n_prompts": len(prompts),
        "polar_pairs_used": {f: _POLAR_PAIRS.get(f) for f in label_dim_files
                              if f in polar_low and f in polar_high},
        "cohorts_used": {c: len(u) for c, u in cohort_users.items() if u},
        "per_layer_norms": summary_norms,
    }


def collect_per_layer_means(
    *,
    engine,
    user_ids: List[int],
    gvec_tensors: Dict[int, torch.Tensor],
    prompts: List[str],
    max_len: int,
    force_zero: bool,
    aggregator: str,
) -> Dict[int, torch.Tensor]:
    """Run forward passes and return per-layer mean residual-stream activation.

    aggregator: 'last_token' | 'mean'
        last_token: take the residual stream at the last prompt token
                    (Arditi 2024 protocol).
        mean:       average over all prompt tokens.

    Returns {layer_idx: torch.Tensor[H]} on CPU. layer_idx 0 is the embedding
    output; 1..L are after each transformer block.
    """
    sums: Dict[int, torch.Tensor] = {}
    counts = 0

    for prompt in prompts:
        enc = engine.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_len),
        ).to(engine.device)
        for uid in user_ids:
            g = gvec_tensors[uid]
            B = 1
            ids_exp = enc["input_ids"].expand(B, -1).contiguous()
            mask_exp = enc["attention_mask"].expand(B, -1).contiguous()

            with torch.no_grad():
                outs = engine.model(
                    input_ids=ids_exp,
                    attention_mask=mask_exp,
                    global_features=g,
                    force_zero_delta=force_zero,
                    return_hidden_only=False,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden = getattr(outs, "hidden_states", None)
            if hidden is None and isinstance(outs, dict):
                hidden = outs.get("hidden_states", None)
            if hidden is None:
                raise RuntimeError(
                    "engine.model did not return hidden_states; "
                    "confirm output_hidden_states=True propagates."
                )

            for ell, h in enumerate(hidden):
                # h: [B, T, H]
                if aggregator == "last_token":
                    vec = h[:, -1, :]              # [B, H]
                else:
                    vec = h.mean(dim=1)            # [B, H]
                vec = vec.detach().float().cpu()
                vec = vec.sum(dim=0)               # [H]
                if ell not in sums:
                    sums[ell] = vec
                else:
                    sums[ell] = sums[ell] + vec

            counts += 1

    if counts == 0:
        raise RuntimeError("No (prompt, user) pairs processed.")
    return {ell: (sums[ell] / float(counts)) for ell in sums}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hyper_dir", type=str, required=True,
                    help="Path to HyperPEFT-LoRA checkpoint dir (M1 hyperlora_multi/)")
    ap.add_argument("--base_model", type=str, default="EleutherAI/pythia-1.4b")
    ap.add_argument("--target_modules", type=str,
                    default="query_key_value,dense,dense_h_to_4h,dense_4h_to_h")
    ap.add_argument("--lora_r", type=int, default=24)
    ap.add_argument("--lora_alpha", type=float, default=48.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--inject_clamp", type=float, default=0.020)
    ap.add_argument("--delta_gain", type=float, default=8.0)
    ap.add_argument("--use_best_ckpt", action="store_true", default=False)
    ap.add_argument("--online", action="store_true", default=False)
    ap.add_argument("--qlora", action="store_true", default=False)
    ap.add_argument("--emit_both", action="store_true", default=False)

    ap.add_argument("--author_parquet", type=str, required=True,
                    help="Path to author_static_*.parquet (per-user gstat features)")
    ap.add_argument("--labels_csv", type=str, required=True,
                    help="Path to labels_sentiment_goemo.csv (user -> cohort)")
    ap.add_argument("--norm_stats_json", type=str, default="",
                    help="Optional feature_norm_stats_*.json (used to read feature_names)")
    ap.add_argument("--feature_manifest_json", type=str, default="",
                    help="If --norm_stats_json absent, read feature_names from "
                         "<hyper_dir>/feature_manifest.json")

    ap.add_argument("--sampling", choices=("cohort", "balanced", "random", "multi_axis"),
                    default="balanced",
                    help="User-sampling strategy. 'cohort' is legacy (BIASED, only controls "
                         "sentiment axis). 'balanced' stratifies top + bottom quintile across "
                         "all 9 label dims so per-user perturbations cancel in every dim. "
                         "'random' samples uniformly. 'multi_axis' (NEW) extracts a comprehensive "
                         "multi-direction file: main + per-dim polar (9 dirs) + per-cohort "
                         "(rage/empath/calm) + persona; supports all four downstream Patch modes "
                         "from a single forward-pass batch.")
    ap.add_argument("--n_per_cohort_for_multi_axis", type=int, default=60,
                    help="Per-cohort sample size for sentiment cohort directions when "
                         "--sampling multi_axis (rage / empath / calm).")
    ap.add_argument("--label_csv_dir", type=str, default="/tmp",
                    help="Directory containing per-dimension label CSVs (only used for "
                         "--sampling balanced).")
    ap.add_argument("--label_dim_files", type=str,
                    default="labels_sentiment_goemo.csv,labels_politeness.csv,labels_self_focus.csv,"
                            "labels_curiosity.csv,labels_expressiveness.csv,labels_tempo.csv,"
                            "labels_anxiety.csv,labels_warmth.csv,labels_hostility.csv",
                    help="Comma-separated list of per-dim label CSV filenames in label_csv_dir.")
    ap.add_argument("--n_per_tail", type=int, default=30,
                    help="Users per quintile tail per dimension (--sampling balanced).")
    ap.add_argument("--quintile_low", type=float, default=0.20)
    ap.add_argument("--quintile_high", type=float, default=0.80)
    ap.add_argument("--n_random", type=int, default=600,
                    help="Pool size for --sampling random.")
    ap.add_argument("--cohorts", type=str, default="rage,empath,neutral",
                    help="Legacy: only used when --sampling cohort.")
    ap.add_argument("--n_users_per_cohort", type=int, default=20,
                    help="Legacy: only used when --sampling cohort.")
    ap.add_argument("--n_prompts", type=int, default=20,
                    help="Number of forum-reply-style prompts to use per pass")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--aggregator", choices=("last_token", "mean"), default="last_token")
    ap.add_argument("--seed", type=int, default=142)

    ap.add_argument("--out_path", type=str, required=True,
                    help="Output safetensors path (e.g., <hyper_dir>/arditi_directions.safetensors)")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load feature_names ----
    feature_names: Optional[List[str]] = None
    if args.norm_stats_json and Path(args.norm_stats_json).exists():
        with open(args.norm_stats_json, "r") as f:
            ns = json.load(f)
        if isinstance(ns, dict) and "feature_names" in ns:
            feature_names = list(ns["feature_names"])
        elif isinstance(ns, dict):
            feature_names = list(ns.keys())
    # The hypernet's global_input_dim ("gdim" / "g_dim" in feature_manifest.json)
    # is the only authoritative source for K. norm_stats_json may list ALL
    # gstat features (66+); the hypernet was trained on a subset (18 in M1).
    # Always prefer feature_manifest.json's global_columns + gdim.
    manifest_features: Optional[List[str]] = None
    manifest_K: Optional[int] = None
    manifest_path = Path(args.feature_manifest_json) if args.feature_manifest_json else \
                    Path(args.hyper_dir) / "feature_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            fm = json.load(f)
        if isinstance(fm, dict):
            cols = fm.get("global_columns") or fm.get("feature_names")
            if isinstance(cols, list) and cols:
                manifest_features = [str(c) for c in cols]
            kk = fm.get("gdim") or fm.get("g_dim") or fm.get("global_dim")
            if isinstance(kk, int) and kk > 0:
                manifest_K = int(kk)
        elif isinstance(fm, list):
            manifest_features = [str(c) for c in fm]
    # Manifest is authoritative when present; norm_stats_json is fallback only.
    if manifest_features is not None:
        feature_names = manifest_features
    if feature_names is None:
        raise RuntimeError(
            "Could not locate feature_names. Pass --norm_stats_json or "
            "--feature_manifest_json, or place feature_manifest.json under "
            f"--hyper_dir ({args.hyper_dir})."
        )
    # Truncate / pad feature_names to manifest_K if we know it (handles the case
    # where norm_stats lists the full feature catalog but the hypernet only
    # consumes the first K subset).
    if manifest_K is not None and manifest_K != len(feature_names):
        if len(feature_names) > manifest_K:
            feature_names = feature_names[:manifest_K]
            LOG.info("[direction] truncating feature_names to gdim=%d (manifest)", manifest_K)
        else:
            raise RuntimeError(
                f"feature_names has {len(feature_names)} entries but manifest "
                f"gdim={manifest_K}; cannot pad."
            )
    LOG.info("[direction] using K=%d global features", len(feature_names))

    # ---- Build engine ----
    from build_hyperlora_forum import HyperPEFTLoRAEngine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("[direction] device=%s base_model=%s hyper_dir=%s",
             device, args.base_model, args.hyper_dir)
    target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
    engine = HyperPEFTLoRAEngine(
        base_model=args.base_model,
        hyper_dir=args.hyper_dir,
        target_modules=target_modules,
        lora_r=int(args.lora_r),
        lora_alpha=float(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        inject_clamp=float(args.inject_clamp),
        delta_gain=float(args.delta_gain),
        use_best_ckpt=bool(args.use_best_ckpt),
        online=bool(args.online),
        qlora=bool(args.qlora),
        device=device,
        emit_both=bool(args.emit_both),
    )
    LOG.info("[direction] engine ready | vocab=%d", int(len(engine.tok)))

    # Diagnostic: confirm we can reach the layer stack
    try:
        layers = _extract_layers_from_engine(engine)
        LOG.info("[direction] located %d transformer blocks", len(layers))
    except Exception as e:
        LOG.warning("[direction] layer-stack diagnostic failed (%s); "
                    "extraction proceeds via output_hidden_states", e)

    # ---- Load user g vectors ----
    cohorts = [s.strip() for s in args.cohorts.split(",") if s.strip()]
    label_dim_files = [s.strip() for s in args.label_dim_files.split(",") if s.strip()]
    LOG.info("[direction] sampling=%s | n_per_tail=%d | n_random=%d | label_dims=%d",
             args.sampling, int(args.n_per_tail), int(args.n_random), len(label_dim_files))

    prompts = list(DEFAULT_PROMPTS)[: int(args.n_prompts)]
    if len(prompts) < int(args.n_prompts):
        LOG.warning("[direction] requested %d prompts, only %d available; using %d",
                    int(args.n_prompts), len(prompts), len(prompts))

    if str(args.sampling) == "multi_axis":
        # Comprehensive multi-direction extraction (main + per-dim + per-cohort + persona)
        flat_dirs, multi_meta = extract_multi_axis_directions(
            engine=engine,
            label_csv_dir=Path(args.label_csv_dir),
            label_dim_files=label_dim_files,
            author_parquet=Path(args.author_parquet),
            feature_names=feature_names,
            n_per_tail=int(args.n_per_tail),
            n_per_cohort=int(args.n_per_cohort_for_multi_axis),
            prompts=prompts,
            max_len=int(args.max_len),
            aggregator=args.aggregator,
            seed=int(args.seed),
        )
        save_file(flat_dirs, str(out_path))
        LOG.info("[direction] saved %d multi-axis keys -> %s", len(flat_dirs), out_path)
        norms = {}  # populated below from multi_meta for backward-compatible JSON
        directions = flat_dirs  # for downstream count print
        # Skip the legacy single-direction path entirely
    else:
        gvec_tensors = load_user_g_vectors(
            engine=engine,
            author_parquet=Path(args.author_parquet),
            labels_csv=Path(args.labels_csv),
            label_csv_dir=Path(args.label_csv_dir) if args.label_csv_dir else None,
            label_dim_files=label_dim_files,
            feature_names=feature_names,
            sampling=str(args.sampling),
            n_users_per_cohort=int(args.n_users_per_cohort),
            n_per_tail=int(args.n_per_tail),
            n_random=int(args.n_random),
            cohorts=cohorts,
            quintile_low=float(args.quintile_low),
            quintile_high=float(args.quintile_high),
            seed=int(args.seed),
        )
        user_ids = sorted(gvec_tensors.keys())
        if not user_ids:
            raise RuntimeError("No users sampled; check --labels_csv and --cohorts.")

        LOG.info("[direction] starting extraction | users=%d prompts=%d aggregator=%s",
                 len(user_ids), len(prompts), args.aggregator)

        # ---- Pass A: delta-OFF (Pile-mode) ----
        LOG.info("[direction] pass A: force_zero_delta=True")
        means_off = collect_per_layer_means(
            engine=engine,
            user_ids=user_ids,
            gvec_tensors=gvec_tensors,
            prompts=prompts,
            max_len=int(args.max_len),
            force_zero=True,
            aggregator=args.aggregator,
        )

        # ---- Pass B: delta-ON (persona-mode) ----
        LOG.info("[direction] pass B: force_zero_delta=False")
        means_on = collect_per_layer_means(
            engine=engine,
            user_ids=user_ids,
            gvec_tensors=gvec_tensors,
            prompts=prompts,
            max_len=int(args.max_len),
            force_zero=False,
            aggregator=args.aggregator,
        )

        # ---- Direction = off - on, normalize per layer ----
        directions: Dict[str, torch.Tensor] = {}
        norms: Dict[int, float] = {}
        for ell in sorted(means_off.keys()):
            if ell not in means_on:
                continue
            diff = means_off[ell] - means_on[ell]                # [H]
            n = float(torch.linalg.norm(diff).item())
            norms[ell] = n
            if n < 1e-8:
                LOG.warning("[direction] layer %d: |off - on| ~ 0; saving zero direction", ell)
                directions[f"layer_{ell}"] = diff.float().contiguous()
            else:
                directions[f"layer_{ell}"] = (diff / n).float().contiguous()

        # ---- Save ----
        save_file(directions, str(out_path))
        LOG.info("[direction] saved %d layer directions -> %s", len(directions), out_path)

    # ---- Companion JSON metadata ----
    if str(args.sampling) == "multi_axis":
        meta = {
            "hyper_dir": str(args.hyper_dir),
            "base_model": args.base_model,
            "sampling": "multi_axis",
            "n_users": int(multi_meta.get("n_users", 0)),
            "n_prompts": int(multi_meta.get("n_prompts", 0)),
            "polar_pairs_used": multi_meta.get("polar_pairs_used", {}),
            "cohorts_used": multi_meta.get("cohorts_used", {}),
            "max_len": int(args.max_len),
            "aggregator": args.aggregator,
            "seed": int(args.seed),
            "per_layer_norms_by_group": multi_meta.get("per_layer_norms", {}),
            "feature_names_n": len(feature_names),
            "key_families": sorted(set(k.rsplit("/layer_", 1)[0] for k in directions.keys())),
        }
    else:
        meta = {
            "hyper_dir": str(args.hyper_dir),
            "base_model": args.base_model,
            "sampling": str(args.sampling),
            "n_users": len(user_ids),
            "n_prompts": len(prompts),
            "cohorts": cohorts if args.sampling == "cohort" else None,
            "label_dim_files": label_dim_files if args.sampling == "balanced" else None,
            "n_per_tail": int(args.n_per_tail) if args.sampling == "balanced" else None,
            "quintile_low": float(args.quintile_low) if args.sampling == "balanced" else None,
            "quintile_high": float(args.quintile_high) if args.sampling == "balanced" else None,
            "n_random": int(args.n_random) if args.sampling == "random" else None,
            "max_len": int(args.max_len),
            "aggregator": args.aggregator,
            "seed": int(args.seed),
            "per_layer_norm": {str(k): v for k, v in norms.items()},
            "feature_names_n": len(feature_names),
        }
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=float)
    LOG.info("[direction] metadata -> %s", meta_path)

    if str(args.sampling) != "multi_axis" and norms:
        # Quick diagnostic: print per-layer norm so user sees magnitudes
        LOG.info("[direction] per-layer raw |off - on| norm:")
        for ell in sorted(norms.keys()):
            LOG.info("  layer_%-2d  |diff|=%.4f", ell, norms[ell])

    return 0


if __name__ == "__main__":
    sys.exit(main())
