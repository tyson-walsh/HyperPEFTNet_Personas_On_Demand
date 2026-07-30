"""
score_persona_signature.py

Per-turn persona-signature scorer for the HyperPEFT-LoRA pipeline.  Runs
after Phase 2d (synth forums) and Phase 5 (games) to extend text scoring from
"did this post match a rage/empath label?" (single axis) to "did this post
match the author's expected 9-dim persona profile?" (multi-axis).

Inputs
------
  --forum_parquet         forum.parquet (Phase 2d) or *_results.parquet (Phase 5)
  --author_profile_parquet author_label_profile.parquet sidecar (Phase 2d) or
                           a game-specific synthetic version that joins
                           synthetic_personas_labeled.parquet onto participant
                           UIDs.  Carries `label_profile_json` + pol_{dim} per
                           author_user_id.
  --norm_stats_json       data/feature_norm_stats_10000.json (for z-scoring
                           per-post ratios against the real-user population).
  --output_parquet        persona_signature.parquet destination.

Outputs (persona_signature.parquet)
-----------------------------------
  Keys:      gid, comment_id, author_user_id, turn_idx
  Realized:  realized_pol_{dim}  x 9 dims, z-scored against norm_stats.
  Expected:  expected_pol_{dim}  x 9 dims, from author_label_profile sidecar.
  Metrics:
      signature_cosine_heldout  - PRIMARY: cosine over 8 held-out dims
                                  (politeness / curiosity / tempo / self_focus
                                   / expressiveness / anxiety / warmth /
                                   hostility).  Excludes sentiment_goemo to
                                  avoid circular validation of the axis the
                                  hypernet was trained on.
      signature_cosine_all9     - supplementary cosine including sentiment_goemo.
      signature_L1_heldout      - mean |realized - expected| over the 8 held-out dims.
      cohort_agreement          - 1 if expected cohort_goemo == realized cohort_goemo else 0.
      realized_profile_json     - full dict per-row.
      expected_profile_json     - full dict per-row.

Metric rationale (plan Item 2.5)
--------------------------------
The hypernet was trained on held-in GoEmo sentiment.  Using that axis as a
fidelity metric is circular.  The 8 held-out probes were NEVER conditioned
during training, so realized-vs-expected on those axes is a clean transfer
test.  Cohort agreement on sentiment_goemo remains as a one-way diagnostic.

Realized text scoring
---------------------
  politeness  : z(profanity_ratio(text)) ; lower raw = more polite (convention)
  curiosity   : z(int(text.strip().endswith('?')))
  tempo       : NaN for per-post (reply_delay is inter-post; marked unmeasured)
  self_focus  : z(firstperson_ratio(text))
  expressive  : raw harmonic_mean(caps_ratio, punct_ratio) ; no z-score
                because expected profile also uses raw harmonic mean.
  anxiety     : mean(fear, nervousness) from GoEmotions 28-dim output
  warmth      : mean(caring, love, gratitude)
  hostility   : mean(anger, disgust, disapproval)
  sentiment_goemo : mean(positive 12) - mean(negative 11)  [simple polarity]

Cohort agreement uses goemo_labels_metadata.json quintile boundaries so both
sides live on the same scale.

This script is written so it can also be pointed at game parquets:
consensus_results.parquet, moderation_results.parquet, diffusion_results.parquet.
The row schema for all four sources carries (gid|round|turn, author_user_id,
text) which is sufficient for per-turn scoring.  Users supply the correct
column names via --gid_col / --turn_col if they depart from forum.parquet
defaults.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("score_sig")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Held-in vs held-out split (plan Item 2.5).
HELD_OUT_DIMS = [
    "politeness", "curiosity", "tempo", "self_focus",
    "expressiveness", "anxiety", "warmth", "hostility",
]
HELD_IN_DIMS = ["sentiment_goemo"]
ALL_DIMS = HELD_OUT_DIMS + HELD_IN_DIMS

# Quintile labels in ascending sentiment order.
QUINTILE_LABELS = ("rage", "grumpy", "mellow", "calm", "empath")


# --------------------------------------------------------------------------- #
# Per-post feature extraction (mirrors data_scripts/hypernetwork_feature_builder)
# --------------------------------------------------------------------------- #
_EMOJI_RE = re.compile(
    "["                   # rough emoji coverage, matches feature builder
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]",
    flags=re.UNICODE,
)
_PROFANITY = {
    "fuck", "fucking", "fucked", "fucker", "shit", "shitty", "bullshit",
    "damn", "asshole", "ass", "bitch", "dick", "cunt", "bastard", "piss",
    "crap", "cock", "pussy", "twat", "hell", "wank", "douche", "prick",
}
_FIRST_PERSON = {"i", "me", "my", "mine", "myself"}


def _toklist(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z']+", text.lower())


def _punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    denom = max(1, len(text))
    emoji_chars = set("".join(_EMOJI_RE.findall(text)))
    punct = 0
    for ch in text:
        if ch in emoji_chars:
            continue
        if ch.isspace() or ch.isalnum():
            continue
        punct += 1
    return punct / denom


def _caps_ratio(text: str) -> float:
    toks = (text or "").split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if t.isupper() and len(t) > 1) / len(toks)


def _profanity_ratio(text: str) -> float:
    toks = _toklist(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in _PROFANITY) / len(toks)


def _firstperson_ratio(text: str) -> float:
    toks = _toklist(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in _FIRST_PERSON) / len(toks)


def _question_indicator(text: str) -> float:
    return 1.0 if (text or "").strip().endswith("?") else 0.0


def _expressiveness(text: str) -> float:
    c = _caps_ratio(text)
    p = _punct_ratio(text)
    return 2.0 * (c * p) / (c + p + 1e-12)


# --------------------------------------------------------------------------- #
# GoEmotions scoring (lazy import of the shared helper)
# --------------------------------------------------------------------------- #
def _score_goemo(texts: List[str], device_id: int, batch_size: int = 64
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run GoEmotions over every post once and return:
        anxiety, warmth, hostility, sentiment_goemo
    all as (N,) float32 arrays.
    """
    # Defer import so CPU smoke tests of the rest of the script don't pull
    # in the whole transformers stack.
    try:
        from hypergames_utils import score_texts_goemo_full
    except Exception as exc:
        LOG.warning("hypergames_utils not importable: %s. Returning NaN GoEmo.", exc)
        z = np.full(len(texts), np.nan, dtype=np.float32)
        return z.copy(), z.copy(), z.copy(), z.copy()

    simple_pol, _re_pol, emotion_dicts = score_texts_goemo_full(
        texts=texts, device_id=device_id, batch_size=batch_size,
    )
    n = len(texts)
    anx = np.zeros(n, dtype=np.float32)
    war = np.zeros(n, dtype=np.float32)
    hos = np.zeros(n, dtype=np.float32)
    sent = np.asarray(simple_pol, dtype=np.float32)
    # 2026-05-18: import composite class lists from signed_hedges_g as single
    # source of truth so realized vs expected composites stay in lockstep.
    try:
        from signed_hedges_g import WARMTH_GOEMO_CLASSES, ANXIETY_GOEMO_CLASSES
    except Exception:
        WARMTH_GOEMO_CLASSES = ("caring", "love")
        ANXIETY_GOEMO_CLASSES = ("fear", "nervousness")
    _w_n = float(len(WARMTH_GOEMO_CLASSES))
    _a_n = float(len(ANXIETY_GOEMO_CLASSES))
    for i, ed in enumerate(emotion_dicts):
        anx[i]  = sum(float(ed.get(c, 0.0)) for c in ANXIETY_GOEMO_CLASSES) / _a_n
        war[i]  = sum(float(ed.get(c, 0.0)) for c in WARMTH_GOEMO_CLASSES) / _w_n
        hos[i]  = (float(ed.get("anger", 0.0))
                   + float(ed.get("disgust", 0.0))
                   + float(ed.get("disapproval", 0.0))) / 3.0
    return anx, war, hos, sent


# --------------------------------------------------------------------------- #
# Z-scoring against 10K real-user population stats
# --------------------------------------------------------------------------- #
def _zs(x: np.ndarray, norm: Dict[str, Dict[str, float]], col: str) -> np.ndarray:
    s = norm.get(col, {})
    mu = float(s.get("mean", 0.0))
    sd = float(s.get("std", 1.0))
    if not np.isfinite(sd) or sd <= 0.0:
        return x.astype(np.float32)
    return ((x.astype(np.float64) - mu) / sd).astype(np.float32)


# --------------------------------------------------------------------------- #
# Cosine / L1 helpers (NaN-safe)
# --------------------------------------------------------------------------- #
def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a, b: (N, D); NaN entries dropped pairwise.
    out = np.full(len(a), np.nan, dtype=np.float32)
    for i in range(len(a)):
        av = a[i]
        bv = b[i]
        mask = np.isfinite(av) & np.isfinite(bv)
        if not mask.any():
            continue
        x = av[mask].astype(np.float64)
        y = bv[mask].astype(np.float64)
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        if nx < 1e-12 or ny < 1e-12:
            continue
        out[i] = float((x * y).sum() / (nx * ny))
    return out


def _row_l1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """DEPRECATED scale-mixed L1: averages raw |a-b| across dims that have
    radically different scales (z-scored gstat dims at ~unit SD vs raw
    GoEmotions probs at SD ~ 0.01). Curiosity dominates the off-manifold sum.
    Kept for backward-compatibility with old parquets; prefer _row_l1_sdnorm.
    See FIGURE34_FIX_19MAY2026.md for the diagnosis."""
    out = np.full(len(a), np.nan, dtype=np.float32)
    for i in range(len(a)):
        mask = np.isfinite(a[i]) & np.isfinite(b[i])
        if not mask.any():
            continue
        out[i] = float(np.mean(np.abs(a[i][mask] - b[i][mask])))
    return out


# Per-dim real-user expected SD denominator for the SD-normalized L1.
# Computed once from the production 2c_real_user_{rage,empath,neutral} pool
# (n_users = 500, coherent replies only). Used to put every held-out polar
# dim on the same SD-unit scale before averaging the L1 row.
# Source: psi_fix_v4_sd_normalized.py output 2026-05-19. Recompute and update
# if the real-user pool or the polar-feature scaling ever changes.
EXPECTED_POLAR_SD_REAL_USER = {
    "politeness":      0.9261,
    "curiosity":       1.0213,
    "tempo":           1.0364,
    "self_focus":      1.2340,
    "expressiveness":  0.0168,
    "anxiety":         0.0031,
    "warmth":          0.0183,
    "hostility":       0.0126,
    "sentiment_goemo": 0.5000,  # not actually used in heldout L1; placeholder
}


def _row_l1_sdnorm(a: np.ndarray, b: np.ndarray, dim_names: List[str]) -> np.ndarray:
    """SD-normalized mean L1 per row. Each dim's |a-b| is divided by the
    real-user expected population SD on that dim, then averaged across dims.
    Range [0, +inf); 0 SDs off = 0, 1 SD off on average = 1. NaN-safe.
    dim_names: list of D dim names corresponding to columns of a, b. Used to
    look up EXPECTED_POLAR_SD_REAL_USER[dim]."""
    out = np.full(len(a), np.nan, dtype=np.float32)
    sd = np.asarray(
        [max(EXPECTED_POLAR_SD_REAL_USER.get(d, 1.0), 1e-6) for d in dim_names],
        dtype=np.float64,
    )
    for i in range(len(a)):
        mask = np.isfinite(a[i]) & np.isfinite(b[i])
        if not mask.any():
            continue
        diff = np.abs(a[i][mask].astype(np.float64) - b[i][mask].astype(np.float64))
        diff = diff / sd[mask]
        out[i] = float(np.mean(diff))
    return out


# --------------------------------------------------------------------------- #
# Cohort assignment
# --------------------------------------------------------------------------- #
def _cohort_from_score(scores: np.ndarray, quint: Dict[str, float]) -> List[str]:
    b20 = float(quint["0.2"]); b40 = float(quint["0.4"])
    b60 = float(quint["0.6"]); b80 = float(quint["0.8"])
    out: List[str] = []
    for s in scores:
        if not np.isfinite(s):
            out.append("unknown"); continue
        if   s <= b20: out.append("rage")
        elif s <= b40: out.append("grumpy")
        elif s <= b60: out.append("mellow")
        elif s <= b80: out.append("calm")
        else:          out.append("empath")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forum_parquet", required=True,
                    help="Phase 2d forum.parquet or Phase 5 *_results.parquet")
    ap.add_argument("--author_profile_parquet", required=True,
                    help="author_label_profile.parquet sidecar from "
                         "build_hyperlora_forum.py (or synthesized for games).")
    ap.add_argument("--norm_stats_json", required=True,
                    help="data/feature_norm_stats_10000.json")
    ap.add_argument("--goemo_meta_json", default="",
                    help="Optional goemo_labels_metadata.json for quintile cuts.")
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--author_col", default="author_user_id")
    ap.add_argument("--gid_col", default="gid")
    ap.add_argument("--turn_col", default="",
                    help="Optional per-thread order column (round/turn). If "
                         "absent, turn_idx is derived from sort order within gid.")
    ap.add_argument("--comment_id_col", default="comment_id")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device_id", type=int,
                    default=(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1))
    ap.add_argument("--skip_goemo", action="store_true",
                    help="Debug: skip GoEmotions scoring (4 GoEmo dims → NaN).")
    args = ap.parse_args()

    # ---- Load everything ----
    LOG.info("Loading forum parquet: %s", args.forum_parquet)
    fdf = pd.read_parquet(args.forum_parquet)
    LOG.info("Forum parquet loaded rows=%d cols=%d", len(fdf), len(fdf.columns))

    for c in (args.text_col, args.author_col):
        if c not in fdf.columns:
            raise KeyError(f"forum parquet missing required column {c!r}")
    fdf[args.author_col] = pd.to_numeric(fdf[args.author_col], errors="coerce") \
        .astype("Int64").dropna().astype(int)

    LOG.info("Loading author profile sidecar: %s", args.author_profile_parquet)
    pdf = pd.read_parquet(args.author_profile_parquet)
    if args.author_col not in pdf.columns:
        if "target_user_id" in pdf.columns:
            pdf = pdf.rename(columns={"target_user_id": args.author_col})
        else:
            raise KeyError("author_profile_parquet needs author_user_id or target_user_id")
    pdf[args.author_col] = pd.to_numeric(pdf[args.author_col], errors="coerce") \
        .astype("Int64").dropna().astype(int)

    norm = json.loads(Path(args.norm_stats_json).read_text())
    goemo_meta = (json.loads(Path(args.goemo_meta_json).read_text())
                  if args.goemo_meta_json and Path(args.goemo_meta_json).exists()
                  else None)

    # ---- Turn index derivation ----
    if args.turn_col and args.turn_col in fdf.columns:
        fdf["__turn_idx"] = pd.to_numeric(fdf[args.turn_col], errors="coerce").fillna(-1).astype(int)
    elif "created_min" in fdf.columns:
        fdf = fdf.sort_values([args.gid_col, "created_min", args.comment_id_col]) \
            if args.gid_col in fdf.columns and args.comment_id_col in fdf.columns \
            else fdf.sort_values(["created_min"])
        fdf["__turn_idx"] = fdf.groupby(args.gid_col).cumcount() \
            if args.gid_col in fdf.columns else np.arange(len(fdf))
    else:
        fdf["__turn_idx"] = np.arange(len(fdf))

    texts = [str(t) if t is not None else "" for t in fdf[args.text_col].tolist()]
    n = len(texts)
    LOG.info("Scoring %d posts", n)

    # I1 coherence gate (2026-05-14 patent-leak audit). Garbled / patent /
    # Stack-Overflow-template rows pollute every downstream metric: GoEmo
    # assigns extreme polarities to gibberish, profanity_ratio collapses to
    # zero, and cohort_agreement gets flipped for the wrong reason. Build
    # the mask once on raw text and NaN-mask the affected outputs after
    # they are computed, so the row is preserved (for audit) but excluded
    # from any cohort / signature aggregate.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from hypergames_utils import _is_coherent  # type: ignore  # noqa: E402
        coherent_mask = np.asarray([bool(_is_coherent(t)) for t in texts], dtype=bool)
    except Exception as exc:
        LOG.warning(
            "hypergames_utils._is_coherent not importable (%s); skipping I1 mask",
            exc,
        )
        coherent_mask = np.ones(n, dtype=bool)
    incoherent_n = int(np.sum(~coherent_mask))
    if incoherent_n:
        LOG.info("I1 coherence gate: %d / %d posts (%.1f%%) flagged incoherent",
                 incoherent_n, n, 100.0 * incoherent_n / max(1, n))

    # ---- Gstat-derived realized (per-post) values ----
    raw_punct = np.asarray([_punct_ratio(t) for t in texts], dtype=np.float32)
    raw_caps  = np.asarray([_caps_ratio(t) for t in texts], dtype=np.float32)
    raw_profanity = np.asarray([_profanity_ratio(t) for t in texts], dtype=np.float32)
    raw_firstperson = np.asarray([_firstperson_ratio(t) for t in texts], dtype=np.float32)
    raw_question = np.asarray([_question_indicator(t) for t in texts], dtype=np.float32)
    raw_expr = np.asarray([_expressiveness(t) for t in texts], dtype=np.float32)
    # Word count per post: tempo reactivity proxy (deliberate users write longer posts).
    word_counts = np.asarray([len(_toklist(t)) for t in texts], dtype=np.float32)

    # ---- GoEmo-derived realized values (moved up so politeness composite
    #      can use hostility as a safety-prior-robust signal). ----
    if args.skip_goemo:
        anx = war = hos = sent = np.full(n, np.nan, dtype=np.float32)
    else:
        anx, war, hos, sent = _score_goemo(
            texts, device_id=int(args.device_id), batch_size=int(args.batch_size),
        )

    def _cohort_z(x: np.ndarray) -> np.ndarray:
        """NaN-safe cohort-internal z-score; returns zeros if variance vanishes."""
        x = np.asarray(x, dtype=np.float32)
        mu = np.nanmean(x) if np.any(np.isfinite(x)) else 0.0
        sd = np.nanstd(x)  if np.any(np.isfinite(x)) else 0.0
        if not np.isfinite(sd) or sd < 1e-9:
            return np.zeros_like(x)
        return ((x - mu) / sd).astype(np.float32)

    # Politeness composite: 0.5 * z(profanity) + 0.5 * z(hostility). Profanity
    # alone z-scores to a near-constant on safety-priored backbone output
    # (variance ~ 0 -> Cohen's d NaN). GoEmo hostility carries residual anger /
    # disgust signal even when explicit profane tokens are suppressed. Sign
    # convention matches labels_manifest.json: HIGH = vulgar cohort.
    _prof_z = _zs(raw_profanity, norm, "gstat_profanity_ratio")
    _host_z = _cohort_z(hos)
    politeness_composite = (0.5 * _prof_z + 0.5 * _host_z).astype(np.float32)

    # Tempo proxy: z(word_count_per_post). Single-post rollouts carry no
    # inter-post reply-delay signal, but post length is a clean reactivity
    # proxy: reactive users write shorter posts, deliberate users write
    # longer ones. Sign matches gstat_reply_delay_mean convention: HIGH =
    # deliberate cohort.
    tempo_proxy = _cohort_z(word_counts)

    realized = {
        "politeness":      politeness_composite,
        "curiosity":       _zs(raw_question,     norm, "gstat_question_ratio"),
        "tempo":           tempo_proxy,
        "self_focus":      _zs(raw_firstperson,  norm, "gstat_firstperson_ratio"),
        # Expressiveness stays raw to match label_synthetic_personas.py.
        "expressiveness":  raw_expr,
    }
    realized["anxiety"]         = anx
    realized["warmth"]           = war
    realized["hostility"]        = hos
    realized["sentiment_goemo"]  = sent

    # ---- Expected profile lookup ----
    expected: Dict[str, np.ndarray] = {d: np.full(n, np.nan, dtype=np.float32)
                                        for d in ALL_DIMS}
    expected_cohort = np.array(["unknown"] * n, dtype=object)

    pol_cols = [f"pol_{d}" for d in ALL_DIMS]
    have_pol = all(c in pdf.columns for c in pol_cols)
    prof_lookup: Dict[int, Dict] = {}
    if "label_profile_json" in pdf.columns:
        for _, r in pdf.iterrows():
            try:
                prof_lookup[int(r[args.author_col])] = json.loads(r["label_profile_json"])
            except Exception:
                continue
    cohort_lookup: Dict[int, str] = {}
    if "cohort_goemo" in pdf.columns:
        cohort_lookup = dict(zip(
            pdf[args.author_col].astype(int).tolist(),
            pdf["cohort_goemo"].astype(str).tolist(),
        ))

    pol_lookup: Dict[str, Dict[int, float]] = {}
    if have_pol:
        for d in ALL_DIMS:
            pol_lookup[d] = dict(zip(
                pdf[args.author_col].astype(int).tolist(),
                pdf[f"pol_{d}"].astype(float).tolist(),
            ))

    authors = fdf[args.author_col].astype(int).tolist()
    for i, uid in enumerate(authors):
        expected_cohort[i] = cohort_lookup.get(int(uid), "unknown")
        if have_pol:
            for d in ALL_DIMS:
                v = pol_lookup.get(d, {}).get(int(uid), np.nan)
                expected[d][i] = float(v) if v is not None else np.nan
        else:
            prof = prof_lookup.get(int(uid), {})
            for d in ALL_DIMS:
                v = (prof.get(d) or {}).get("value")
                expected[d][i] = float(v) if v is not None else np.nan

    # ---- Metric computation ----
    Rh = np.stack([realized[d] for d in HELD_OUT_DIMS], axis=1)
    Eh = np.stack([expected[d] for d in HELD_OUT_DIMS], axis=1)
    Ra = np.stack([realized[d] for d in ALL_DIMS], axis=1)
    Ea = np.stack([expected[d] for d in ALL_DIMS], axis=1)

    signature_cosine_heldout = _row_cosine(Rh, Eh)
    signature_cosine_all9    = _row_cosine(Ra, Ea)
    signature_L1_heldout     = _row_l1(Rh, Eh)             # DEPRECATED: scale-mixed
    signature_L1_heldout_sdnorm = _row_l1_sdnorm(Rh, Eh, HELD_OUT_DIMS)
    # SD-normalized match score in [0, 1]: 0 SDs off = 1.0, 2+ SDs off = 0.0.
    # Per FIGURE34_FIX_19MAY2026.md the 2-SD cap is the natural break point
    # for the real-user population.
    bullseye_match_sdnorm = np.clip(
        1.0 - signature_L1_heldout_sdnorm / 2.0, 0.0, 1.0
    ).astype(np.float32)

    realized_cohort: List[str]
    if goemo_meta is not None and "rank_boundaries_score_at_fraction" in goemo_meta:
        realized_cohort = _cohort_from_score(
            realized["sentiment_goemo"],
            goemo_meta["rank_boundaries_score_at_fraction"],
        )
    else:
        realized_cohort = ["unknown"] * n
    cohort_agreement = np.array(
        [1 if (rc != "unknown" and rc == ec) else 0
         for rc, ec in zip(realized_cohort, expected_cohort.tolist())],
        dtype=np.int8,
    )

    # I1 NaN-mask: replace realized signals and per-row signature metrics
    # with NaN on incoherent rows so downstream Hedges' g / cohort-rate
    # aggregates skip them. cohort_agreement is promoted to float so NaN
    # can ride alongside 0/1 values without dtype gymnastics. Done AFTER
    # the row cosines so the cosine doesn't see partially-NaN vectors.
    if incoherent_n:
        bad = ~coherent_mask
        for d in ALL_DIMS:
            realized[d] = realized[d].astype(np.float32, copy=True)
            realized[d][bad] = np.nan
        signature_cosine_heldout = signature_cosine_heldout.astype(np.float32, copy=True)
        signature_cosine_all9    = signature_cosine_all9.astype(np.float32, copy=True)
        signature_L1_heldout     = signature_L1_heldout.astype(np.float32, copy=True)
        signature_L1_heldout_sdnorm = signature_L1_heldout_sdnorm.astype(np.float32, copy=True)
        bullseye_match_sdnorm    = bullseye_match_sdnorm.astype(np.float32, copy=True)
        signature_cosine_heldout[bad] = np.nan
        signature_cosine_all9[bad]    = np.nan
        signature_L1_heldout[bad]     = np.nan
        signature_L1_heldout_sdnorm[bad] = np.nan
        bullseye_match_sdnorm[bad]    = np.nan
        cohort_agreement = cohort_agreement.astype(np.float32, copy=True)
        cohort_agreement[bad] = np.nan

    # ---- Profile JSON columns ----
    realized_json: List[str] = []
    expected_json: List[str] = []
    for i in range(n):
        realized_json.append(json.dumps(
            {d: (float(realized[d][i]) if np.isfinite(realized[d][i]) else None)
             for d in ALL_DIMS}
        ))
        expected_json.append(json.dumps(
            {d: (float(expected[d][i]) if np.isfinite(expected[d][i]) else None)
             for d in ALL_DIMS}
        ))

    # ---- Assemble output frame ----
    out_cols: Dict[str, np.ndarray] = {}
    for k in (args.gid_col, args.comment_id_col):
        if k in fdf.columns:
            out_cols[k] = fdf[k].to_numpy()
    out_cols[args.author_col] = fdf[args.author_col].to_numpy()
    out_cols["turn_idx"] = fdf["__turn_idx"].to_numpy()
    for d in ALL_DIMS:
        out_cols[f"realized_pol_{d}"] = realized[d]
        out_cols[f"expected_pol_{d}"] = expected[d]
    out_cols["signature_cosine_heldout"] = signature_cosine_heldout
    out_cols["signature_cosine_all9"]    = signature_cosine_all9
    out_cols["signature_L1_heldout"]     = signature_L1_heldout  # DEPRECATED: scale-mixed
    out_cols["signature_L1_heldout_sdnorm"] = signature_L1_heldout_sdnorm
    out_cols["bullseye_match_sdnorm"]       = bullseye_match_sdnorm
    out_cols["expected_cohort_goemo"]    = expected_cohort
    out_cols["realized_cohort_goemo"]    = np.asarray(realized_cohort, dtype=object)
    out_cols["cohort_agreement"]         = cohort_agreement
    out_cols["realized_profile_json"]    = np.asarray(realized_json, dtype=object)
    out_cols["expected_profile_json"]    = np.asarray(expected_json, dtype=object)
    # Preserve the I1 coherence flag so downstream audits can inspect the
    # rows that were NaN-masked above.
    out_cols["is_coherent"]              = coherent_mask.astype(np.bool_)

    out_df = pd.DataFrame(out_cols)
    out_path = Path(args.output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    LOG.info("wrote %s rows=%d cols=%d", out_path, len(out_df), len(out_df.columns))

    # ---- Summary banner ----
    def _mean_safe(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        return float(a.mean()) if len(a) else float("nan")

    LOG.info("sig_cosine_heldout mean=%.4f (n=%d)",
             _mean_safe(signature_cosine_heldout),
             int(np.sum(np.isfinite(signature_cosine_heldout))))
    LOG.info("sig_cosine_all9    mean=%.4f", _mean_safe(signature_cosine_all9))
    LOG.info("sig_L1_heldout     mean=%.4f (DEPRECATED scale-mixed)",
             _mean_safe(signature_L1_heldout))
    LOG.info("sig_L1_heldout_sdnorm mean=%.4f", _mean_safe(signature_L1_heldout_sdnorm))
    LOG.info("bullseye_match_sdnorm mean=%.4f (range [0, 1], 1=perfect)",
             _mean_safe(bullseye_match_sdnorm))
    # cohort_agreement is float32 with NaN on incoherent rows (post-I1
    # mask), so use a NaN-aware mean for the summary banner.
    _coh_arr = np.asarray(cohort_agreement, dtype=np.float32)
    _coh_finite = _coh_arr[np.isfinite(_coh_arr)]
    _coh_rate = float(_coh_finite.mean()) if _coh_finite.size else float("nan")
    LOG.info("cohort_agreement   rate=%.3f (n_finite=%d / %d)",
             _coh_rate, int(_coh_finite.size), n)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOG.exception("score_persona_signature failed: %s", exc)
        sys.exit(1)
