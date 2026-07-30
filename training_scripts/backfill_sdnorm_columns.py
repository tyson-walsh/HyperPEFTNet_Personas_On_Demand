"""Backfill signature_L1_heldout_sdnorm + bullseye_match_sdnorm columns on
existing persona_signature.parquet files WITHOUT re-running the scorer.

Why this exists: the 2026-05-19 fix added two SD-normalized columns to the
output of score_persona_signature.py. Re-running the full scorer on
every existing parquet means re-running GoEmotions inference on every reply,
which is GPU-expensive. The realized + expected polar columns already exist
on disk; we just need to recompute the SD-normalized L1 and the bullseye
match from those columns.

This script is RE-ENTRANT: running it twice does the same thing twice. It
overwrites the two new columns each run from the up-to-date polar columns.

Use:
    python backfill_sdnorm_columns.py /workspace/data10/.../phase2_forums_REFRESH/

It will walk that directory, find every persona_signature/persona_signature.parquet,
add or refresh the two columns, and write the file back in place.

ON_DISK MUTATION WARNING: this updates parquets in place. Back up first if
the existing parquets are still needed in their current schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Stay in sync with score_persona_signature.EXPECTED_POLAR_SD_REAL_USER
EXPECTED_POLAR_SD_REAL_USER = {
    "politeness":      0.9261,
    "curiosity":       1.0213,
    "tempo":           1.0364,
    "self_focus":      1.2340,
    "expressiveness":  0.0168,
    "anxiety":         0.0031,
    "warmth":          0.0183,
    "hostility":       0.0126,
}
HELD_OUT_DIMS = list(EXPECTED_POLAR_SD_REAL_USER.keys())


def _compute_sdnorm_columns(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (signature_L1_heldout_sdnorm, bullseye_match_sdnorm) arrays
    matching df row order. NaN-safe; replies with no valid dims emit NaN."""
    n = len(df)
    sd_l1 = np.full(n, np.nan, dtype=np.float32)
    bullseye = np.full(n, np.nan, dtype=np.float32)
    sd_denoms = np.asarray(
        [max(EXPECTED_POLAR_SD_REAL_USER[d], 1e-6) for d in HELD_OUT_DIMS],
        dtype=np.float64,
    )
    realized_cols = [f"realized_pol_{d}" for d in HELD_OUT_DIMS]
    expected_cols = [f"expected_pol_{d}" for d in HELD_OUT_DIMS]
    missing = [c for c in realized_cols + expected_cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing polar columns: {missing[:5]}...")
    R = df[realized_cols].to_numpy(dtype=np.float64)
    E = df[expected_cols].to_numpy(dtype=np.float64)
    for i in range(n):
        rmask = np.isfinite(R[i]) & np.isfinite(E[i])
        if not rmask.any():
            continue
        diff = np.abs(R[i][rmask] - E[i][rmask]) / sd_denoms[rmask]
        v = float(np.mean(diff))
        sd_l1[i] = v
        bullseye[i] = float(np.clip(1.0 - v / 2.0, 0.0, 1.0))
    return sd_l1, bullseye


def backfill_one(parquet_path: Path) -> dict:
    df = pd.read_parquet(parquet_path)
    sd_l1, bullseye = _compute_sdnorm_columns(df)
    df["signature_L1_heldout_sdnorm"] = sd_l1
    df["bullseye_match_sdnorm"] = bullseye
    df.to_parquet(parquet_path, index=False)
    n_finite = int(np.isfinite(bullseye).sum())
    return {
        "path": str(parquet_path),
        "n_rows": int(len(df)),
        "n_finite_bullseye": n_finite,
        "mean_bullseye": float(np.nanmean(bullseye)) if n_finite else float("nan"),
    }


def main(root: Path) -> None:
    parquets = sorted(root.glob("**/persona_signature/persona_signature.parquet"))
    if not parquets:
        print(f"[backfill] no persona_signature.parquet found under {root}")
        sys.exit(1)
    print(f"[backfill] found {len(parquets)} parquets under {root}")
    for p in parquets:
        try:
            info = backfill_one(p)
            print(f"[backfill] {p.parent.parent.name:<35} n={info['n_rows']:>5d} "
                  f"finite={info['n_finite_bullseye']:>5d} "
                  f"mean_bullseye={info['mean_bullseye']:.4f}")
        except Exception as exc:
            print(f"[backfill] {p}: FAILED ({exc})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: backfill_sdnorm_columns.py <forum_root_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]))
