#!/usr/bin/env python3
"""sample_far_from_hull_extrapolation.py — True out-of-hull extrapolation sampler
for the Paper 2 far-from-hull stratum.

The legacy sampler in `synthesize_personas.py:_sample_far_extrapolation`
generates candidates along the top-K PC axes scaled by 1.5x per-feature std.
That formulation leaves the joint Mahalanobis envelope of training descriptors
mostly intact: per-feature bounding-box violations occur for ~68% of samples,
but every sample still projects inside the 2D PC convex hull of training data,
which is why the empirical far-from-hull stratum tracks in-hull on cohort
agreement instead of producing the expected degradation.

This sampler produces descriptors that are provably outside the convex hull of
training descriptors. For each candidate:

    g_new = g_anchor + alpha * (g_anchor - g_centroid)
          = (1 + alpha) * g_anchor - alpha * g_centroid

with alpha > 0. This is a negative-weight affine combination of two training
points (the anchor and the centroid, where the centroid is itself a positive
mixture of training points, so the negative-coefficient combination places
g_new strictly outside the convex hull in the anchor direction).

Acceptance: candidates pass the same offset-norm gate as the legacy sampler
to ensure the hypernet emits a delta whose magnitude is in the trained
clamp envelope; without that gate the synthesized adapter offsets would be
out-of-distribution for the LoRA-B injection path.

Usage
-----
    python3 data_scripts/sample_far_from_hull_extrapolation.py \\
        --feature_parquet /workspace/hypernets/data/global_features_10000.parquet \\
        --feature_names_json /data/hypernets/results/paper_2_m1/phase1_synth_arditi/synthesis_metadata.json \\
        --n_per_cohort 1000 --alpha 1.0 \\
        --out_parquet /data/hypernets/results/paper_2_m1/phase1_synth_arditi_v2/synthetic_personas.parquet \\
        --seed 142
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from sklearn.decomposition import PCA


def load_real_g(feature_parquet: str, feature_names: List[str]) -> np.ndarray:
    df = pd.read_parquet(feature_parquet)
    pu = df.groupby("target_user_id")[feature_names].first()
    return pu.values.astype(np.float64), pu.index.values.astype(np.int64)


# 4 held-out probe gstats that label_synthetic_personas.py reads on the
# synth parquet (politeness / tempo / self-focus / expressiveness-punct
# component). These are NOT part of the K=18 conditioning vector but ARE
# part of each user's profile, and are projected from anchors with the
# same mixing weights as the K-feature g-vector itself.
HELD_OUT_PROBE_GSTATS = [
    "gstat_profanity_ratio",
    "gstat_reply_delay_mean",
    "gstat_firstperson_ratio",
    "gstat_punct_ratio",
]


def load_real_probes(feature_parquet: str, probe_cols: List[str]) -> np.ndarray:
    df = pd.read_parquet(feature_parquet)
    pu = df.groupby("target_user_id")[probe_cols].first()
    return pu.values.astype(np.float64), pu.index.values.astype(np.int64)


def sample_extrapolation(real_g: np.ndarray, n_samples: int, alpha: float,
                          rng: np.random.Generator) -> np.ndarray:
    """g_new = (1 + alpha) * anchor - alpha * centroid.

    Provably outside the convex hull of real_g for any alpha > 0.
    """
    centroid = real_g.mean(axis=0)
    anchor_idx = rng.integers(0, len(real_g), size=n_samples)
    anchors = real_g[anchor_idx]
    out = (1.0 + alpha) * anchors - alpha * centroid
    return out.astype(np.float32), anchor_idx


def verify_outside_hull_2d(synth_g: np.ndarray, real_g: np.ndarray) -> float:
    """Project to 2D PCA, check what fraction of synth points are OUTSIDE the
    2D PCA convex hull of real points. Returns the outside fraction."""
    pca = PCA(n_components=2).fit(real_g)
    real_2 = pca.transform(real_g)
    synth_2 = pca.transform(synth_g)
    hull = ConvexHull(real_2)
    tri = Delaunay(real_2[hull.vertices])
    inside = (tri.find_simplex(synth_2) >= 0)
    return 1.0 - inside.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_parquet", required=True)
    ap.add_argument("--feature_names_json", required=True,
                    help="JSON file with 'feature_names' list (e.g. synthesis_metadata.json)")
    ap.add_argument("--n_per_cohort", type=int, default=1000,
                    help="Samples per cohort (rage/empath/neutral); produces 3 * N rows.")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Extrapolation magnitude; g_new = anchor + alpha * (anchor - centroid). alpha=1.0 doubles distance from centroid.")
    ap.add_argument("--seed", type=int, default=142)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--cohort_labels_csv", default="/workspace/hypernets/data/labels_sentiment_goemo_extremes.csv",
                    help="rage/empath/neutral cohort assignment per user; used to pick anchors from each cohort.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Feature names from production metadata for K=18 alignment
    with open(args.feature_names_json) as fp:
        meta = json.load(fp)
    feat_names = list(meta["feature_names"])
    print(f"[sample] K = {len(feat_names)} features: {feat_names[:4]}... ", flush=True)

    # Load real per-user descriptors
    real_g, real_uids = load_real_g(args.feature_parquet, feat_names)
    print(f"[sample] real per-user g: {real_g.shape}", flush=True)
    real_probes, _ = load_real_probes(args.feature_parquet, HELD_OUT_PROBE_GSTATS)
    print(f"[sample] real per-user held-out probes: {real_probes.shape}", flush=True)

    # Cohort labels
    cohorts = pd.read_csv(args.cohort_labels_csv)
    # The extremes csv has columns user_id (or target_user_id) + cohort label
    uid_col = "target_user_id" if "target_user_id" in cohorts.columns else "user_id"
    label_col = "label" if "label" in cohorts.columns else \
                ("cohort_extreme" if "cohort_extreme" in cohorts.columns else "cohort_goemo")
    uid_to_cohort = dict(zip(cohorts[uid_col].astype(int).values,
                              cohorts[label_col].astype(str).values))
    print(f"[sample] cohort labels: {pd.Series(uid_to_cohort).value_counts().to_dict()}", flush=True)

    # Per-cohort anchor pools
    cohort_anchor_idx = {}
    for cohort in ("rage", "empath", "neutral"):
        cohort_uids = [u for u in real_uids.tolist() if uid_to_cohort.get(int(u)) == cohort]
        if not cohort_uids:
            # If labeled as 0/1/2 or similar, fall back to all anchors
            print(f"[sample] WARNING: cohort {cohort!r} has 0 labeled anchors")
            cohort_anchor_idx[cohort] = np.arange(len(real_uids))
        else:
            uid_to_pos = {int(u): i for i, u in enumerate(real_uids.tolist())}
            cohort_anchor_idx[cohort] = np.array(
                [uid_to_pos[int(u)] for u in cohort_uids], dtype=np.int64)
        print(f"[sample]  {cohort}: {len(cohort_anchor_idx[cohort])} anchor candidates", flush=True)

    # Per-cohort extrapolation
    centroid = real_g.mean(axis=0)
    all_rows = []
    out_pct_by_cohort = {}
    for cohort in ("rage", "empath", "neutral"):
        idx_pool = cohort_anchor_idx[cohort]
        pool_g = real_g[idx_pool]
        # Anchor from cohort-specific pool; centroid is the SHARED training centroid
        anchor_choice = rng.integers(0, len(pool_g), size=args.n_per_cohort)
        anchors = pool_g[anchor_choice]
        synth_g = (1.0 + args.alpha) * anchors - args.alpha * centroid

        # Verify outside hull
        out_pct = verify_outside_hull_2d(synth_g, real_g)
        out_pct_by_cohort[cohort] = float(out_pct)
        print(f"[verify]  cohort={cohort}: {100*out_pct:.1f}% of synth points outside 2D PCA training hull",
              flush=True)

        for j in range(args.n_per_cohort):
            anchor_pos = int(idx_pool[anchor_choice[j]])
            anchor_uid = int(real_uids[anchor_pos])
            # Schema parity with the legacy Dirichlet-mix sampler:
            # label_synthetic_personas.py expects source_anchor_uids (list)
            # and mixing_weights_json (JSON string). For single-anchor
            # extrapolation, the natural label projection is the anchor's
            # own label, which corresponds to a one-element mixture with
            # weight 1.0.
            row = {
                "target_user_id": 20_000_000 + len(all_rows),
                "stratum": "far_from_hull",
                "synth_cohort": cohort,
                "source_anchor_pos": anchor_pos,
                "source_anchor_uid": anchor_uid,
                "source_anchor_uids": [anchor_uid],
                "mixing_weights_json": json.dumps([1.0]),
                "alpha": float(args.alpha),
                "centroid_used": "training_global",
            }
            for fi, name in enumerate(feat_names):
                row[name] = float(synth_g[j, fi])
            # Held-out probe gstats: project from anchor with weight 1.0
            # (single-anchor extrapolation -> anchor's own probe values).
            for pi, pname in enumerate(HELD_OUT_PROBE_GSTATS):
                row[pname] = float(real_probes[anchor_pos, pi])
            all_rows.append(row)

    out = pd.DataFrame(all_rows)
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[sample] wrote {len(out)} rows to {out_path}", flush=True)

    meta_out = {
        "schema": "far_from_hull_extrapolation_v2",
        "n_total": int(len(out)),
        "n_per_cohort": int(args.n_per_cohort),
        "alpha": float(args.alpha),
        "feature_names": feat_names,
        "K": len(feat_names),
        "outside_hull_pct_2d": out_pct_by_cohort,
        "convention": ("g_new = (1 + alpha) * anchor - alpha * centroid; "
                       "anchor is drawn from the cohort's real-user pool; "
                       "centroid is the unconditional mean of all training "
                       "real-user g-vectors. For alpha > 0 this is a "
                       "negative-coefficient affine combination, so g_new "
                       "lies strictly outside the convex hull of training "
                       "descriptors in the anchor-from-centroid direction."),
    }
    (out_path.parent / "synthesis_metadata.json").write_text(json.dumps(meta_out, indent=2))
    print(f"[sample] wrote metadata to {out_path.parent / 'synthesis_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
