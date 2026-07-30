"""
cohort_centroid_bon.py — Cohort-centroid best-of-N sampling helper.

Used by build_hyperlora_forum.py to lift surface cohort signal in generated
forum text. The trained M1 hypernet's per-user delta produces hidden states
that are cohort-organized (phase 4 layerwise Hedges' g = 1.06 sentiment),
but the lm_head + sampling collapses that signal to near-neutral surface
text. Best-of-N with cohort-centroid selection recovers cohort-aligned
candidates that the model already produces in 1-of-8 stochastic draws.

Validated on M1 Pythia-1.4B (gu03 diag_27 multi-pool, 2026-05-04):
  baseline N=1 cohort-d Hedges' g: ~+0.26
  best-of-8 cohort-centroid sel:    ~+2.30 (fixed-centroid: ~+4.50)
  bootstrap 95% CI on lift:        [+1.50, +2.56]
  coherence:                       100% across 3 disjoint pools

Mechanism: cohort centroids are precomputed once from training-set text
features in author_static_10000.parquet (gstat_*). At inference, per-user
delta produces N candidates; we compute each candidate's 8-dim text-style
vector, take cosine to the user's-cohort centroid, and select the highest.

This is purely inference-time. No model change, no training.

The user's cohort label is ALREADY known at inference (it determines which
forum file the user is written into for phase 2c/2d). The centroid table is
shared across all users of the same cohort.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# 8 stylometric features used for centroid scoring. Order matters and must
# match between centroid construction and per-text scoring.
COHORT_FEATURE_KEYS = [
    "gstat_caps_ratio",
    "gstat_question_ratio",
    "gstat_negation_ratio",
    "gstat_firstperson_ratio",
    "gstat_subjectivity_ratio",
    "gstat_avg_word_len",
    "gstat_long_word_ratio",
    "gstat_stopword_ratio",
]

# Word lists for matching gen_text features to cohort-feature dimensions.
_NEG_WORDS = {"not", "no", "never", "none", "nothing", "cant", "can't",
               "won't", "wouldn't", "shouldn't"}
_FIRSTPERSON = {"i", "me", "my", "mine", "myself"}
_STOPWORDS = {"the", "a", "an", "and", "or", "but", "of", "to", "in",
               "on", "at", "for", "with", "by"}


def build_cohort_centroids(
    author_static_path: str,
    labels_csv_path: str,
    cohort_to_label: Dict[str, str],
    max_users_per_cohort: int = 2000,
) -> Dict[str, np.ndarray]:
    """Build per-cohort centroid vectors in 8-dim text-style space.

    Args:
        author_static_path: parquet with `target_user_id` + gstat_* columns.
        labels_csv_path: CSV with columns target_user_id, label.
        cohort_to_label: e.g. {"rage": "rage", "empath": "empath", "neutral": "calm"}.
            Maps the cohort name used by the forum builder to the label string in the CSV.
            Multiple labels can be mapped to one cohort by passing a list, but the
            common case is 1:1.
        max_users_per_cohort: cap for centroid stability vs runtime.

    Returns:
        dict: cohort_name -> [8] float32 centroid in stylometric feature space.
    """
    ast = pd.read_parquet(author_static_path)
    ast = ast.drop_duplicates("target_user_id", keep="last").reset_index(drop=True)
    ast_idx = {int(r.target_user_id): i for i, r in ast.iterrows()}

    lbl = pd.read_csv(labels_csv_path)
    centroids: Dict[str, np.ndarray] = {}
    for cohort_name, label_str in cohort_to_label.items():
        if isinstance(label_str, str):
            label_str_list = [label_str]
        else:
            label_str_list = list(label_str)
        uids = [int(u) for u in lbl[lbl.label.isin(label_str_list)]["target_user_id"]
                 if int(u) in ast_idx]
        uids = uids[:int(max_users_per_cohort)]
        if not uids:
            continue
        rows = ast.iloc[[ast_idx[u] for u in uids]]
        # Pull the 8 stylometric features, NaN -> 0
        vec = np.zeros((len(uids), len(COHORT_FEATURE_KEYS)), dtype=np.float32)
        for k_idx, key in enumerate(COHORT_FEATURE_KEYS):
            if key not in rows.columns:
                continue
            col = pd.to_numeric(rows[key], errors="coerce").fillna(0.0).astype(np.float32).values
            vec[:, k_idx] = col
        centroids[cohort_name] = vec.mean(axis=0)
    return centroids


def text_style_vec(text: str) -> np.ndarray:
    """Extract 8-dim text-style vector from generated text. Same dims as centroid."""
    if not text:
        return np.zeros(len(COHORT_FEATURE_KEYS), dtype=np.float32)
    n = max(1, len(text))
    words = re.findall(r"[a-zA-Z']+", text)
    nw = max(1, len(words))
    return np.array([
        sum(1 for c in text if c.isupper()) / n,                     # caps_ratio
        text.count("?") / n,                                          # question_ratio
        sum(1 for w in words if w.lower() in _NEG_WORDS) / nw,        # negation_ratio
        sum(1 for w in words if w.lower() in _FIRSTPERSON) / nw,      # firstperson_ratio
        0.0,                                                          # subjectivity placeholder
        sum(len(w) for w in words) / nw,                              # avg_word_len
        sum(1 for w in words if len(w) > 6) / nw,                     # long_word_ratio
        sum(1 for w in words if w.lower() in _STOPWORDS) / nw,        # stopword_ratio
    ], dtype=np.float32)


def cohort_score(text: str, cohort: str, centroids: Dict[str, np.ndarray]) -> float:
    """Cosine similarity of text style vector to the user's cohort centroid."""
    if cohort not in centroids:
        return 0.0
    cent = centroids[cohort]
    g = text_style_vec(text)
    n_g = float(np.linalg.norm(g)) + 1e-9
    n_c = float(np.linalg.norm(cent)) + 1e-9
    return float(np.dot(g, cent) / (n_g * n_c))


def select_best_of_n(
    candidates: List[Dict],
    cohort: str,
    centroids: Dict[str, np.ndarray],
    coherence_filter_key: Optional[str] = "coherent",
) -> Dict:
    """Given a list of candidate dicts (each with `text` and optional `coherent`),
    pick the candidate whose text style is closest to the cohort centroid.
    Coherent candidates are preferred; if all are incoherent, pick highest-scoring.
    """
    if not candidates:
        return {}
    scored = []
    for c in candidates:
        s = cohort_score(c["text"], cohort, centroids)
        scored.append((s, c))
    coh = [(s, c) for s, c in scored if not coherence_filter_key
            or c.get(coherence_filter_key, True)]
    pool = coh if coh else scored
    pool.sort(key=lambda sc: sc[0], reverse=True)
    best_score, best_cand = pool[0]
    best_cand = dict(best_cand)
    best_cand["cohort_score"] = best_score
    return best_cand
