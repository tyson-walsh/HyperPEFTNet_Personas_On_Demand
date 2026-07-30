#!/usr/bin/env python3
"""sample_midpoint_baseline.py — Naive linear-midpoint synthesis baseline
for Paper 2 ablation against Dirichlet-NN mixture.

For each synthetic baseline user we draw two random training anchors
(g_i, g_j) from the SAME cohort and form the simple linear midpoint:

    g_baseline = 0.5 * g_i + 0.5 * g_j

This is the trivial interpolation baseline: no k-NN retrieval, no
Dirichlet weighting, no per-anchor selection. Any cohort signal that
the production Dirichlet-NN mixture produces over this midpoint baseline
is attributable to the k-NN-plus-Dirichlet structure rather than to
linear descriptor combination per se.

Output schema matches sample_far_from_hull_extrapolation.py: a parquet
of synthetic users with stratum="midpoint_baseline", per-cohort tag,
the K-feature g-vector, and the two source anchors used for the midpoint.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_real_g(feature_parquet: str, feature_names: list[str]):
    df = pd.read_parquet(feature_parquet)
    pu = df.groupby("target_user_id")[feature_names].first()
    return pu.values.astype(np.float64), pu.index.values.astype(np.int64)


# 4 held-out probe gstats that label_synthetic_personas.py expects on the
# synth parquet (politeness / tempo / self-focus / expressiveness-punct).
HELD_OUT_PROBE_GSTATS = [
    "gstat_profanity_ratio",
    "gstat_reply_delay_mean",
    "gstat_firstperson_ratio",
    "gstat_punct_ratio",
]


def load_real_probes(feature_parquet: str, probe_cols: list[str]):
    df = pd.read_parquet(feature_parquet)
    pu = df.groupby("target_user_id")[probe_cols].first()
    return pu.values.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_parquet", required=True)
    ap.add_argument("--feature_names_json", required=True)
    ap.add_argument("--cohort_labels_csv", required=True)
    ap.add_argument("--n_per_cohort", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=142)
    ap.add_argument("--out_parquet", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    with open(args.feature_names_json) as fp:
        meta = json.load(fp)
    feat_names = list(meta["feature_names"])

    real_g, real_uids = load_real_g(args.feature_parquet, feat_names)
    real_probes = load_real_probes(args.feature_parquet, HELD_OUT_PROBE_GSTATS)
    cohorts = pd.read_csv(args.cohort_labels_csv)
    uid_col = "target_user_id" if "target_user_id" in cohorts.columns else "user_id"
    label_col = "label"
    uid_to_cohort = dict(zip(cohorts[uid_col].astype(int).values,
                              cohorts[label_col].astype(str).values))
    print(f"[midpoint] cohort labels: {pd.Series(uid_to_cohort).value_counts().to_dict()}", flush=True)

    uid_to_pos = {int(u): i for i, u in enumerate(real_uids.tolist())}
    cohort_pool = {}
    for cohort in ("rage", "empath", "neutral"):
        pool = [uid_to_pos[int(u)] for u, c in uid_to_cohort.items()
                if c == cohort and int(u) in uid_to_pos]
        if not pool:
            print(f"[midpoint] WARNING: cohort {cohort!r} has 0 labeled anchors; using full pool")
            pool = list(range(len(real_uids)))
        cohort_pool[cohort] = np.array(pool, dtype=np.int64)
        print(f"[midpoint]  {cohort}: {len(pool)} anchor candidates", flush=True)

    all_rows = []
    for cohort in ("rage", "empath", "neutral"):
        pool = cohort_pool[cohort]
        # Draw 2 distinct anchors per synthetic user
        for _ in range(args.n_per_cohort):
            i, j = rng.choice(pool, size=2, replace=False)
            g_mid = 0.5 * real_g[i] + 0.5 * real_g[j]
            probes_mid = 0.5 * real_probes[i] + 0.5 * real_probes[j]
            uid_i, uid_j = int(real_uids[i]), int(real_uids[j])
            # Schema parity with the legacy Dirichlet-mix sampler:
            # label_synthetic_personas.py reads source_anchor_uids (list)
            # and mixing_weights_json. For a 2-anchor midpoint baseline
            # the natural projection is the equal-weight mean over the
            # two anchors' labels, i.e., weights [0.5, 0.5].
            row = {
                "target_user_id": 30_000_000 + len(all_rows),
                "stratum": "midpoint_baseline",
                "synth_cohort": cohort,
                "source_anchor_pos_i": int(i),
                "source_anchor_pos_j": int(j),
                "source_anchor_uid_i": uid_i,
                "source_anchor_uid_j": uid_j,
                "source_anchor_uids": [uid_i, uid_j],
                "mixing_weights_json": json.dumps([0.5, 0.5]),
            }
            for fi, name in enumerate(feat_names):
                row[name] = float(g_mid[fi])
            for pi, pname in enumerate(HELD_OUT_PROBE_GSTATS):
                row[pname] = float(probes_mid[pi])
            all_rows.append(row)

    out = pd.DataFrame(all_rows)
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[midpoint] wrote {len(out)} rows to {out_path}", flush=True)

    meta_out = {
        "schema": "midpoint_baseline_v1",
        "n_total": int(len(out)),
        "n_per_cohort": int(args.n_per_cohort),
        "feature_names": feat_names,
        "K": len(feat_names),
        "convention": ("g_baseline = 0.5 * g_i + 0.5 * g_j for two random "
                       "training anchors drawn from the same cohort. No "
                       "k-NN retrieval, no Dirichlet weighting; the simplest "
                       "non-trivial linear-interpolation baseline against "
                       "the production Dirichlet-NN mixture sampler."),
    }
    (out_path.parent / "synthesis_metadata.json").write_text(json.dumps(meta_out, indent=2))
    print(f"[midpoint] wrote metadata to {out_path.parent / 'synthesis_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
