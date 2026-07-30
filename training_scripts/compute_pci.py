#!/usr/bin/env python3
"""Compute the observational Persona Compositionality Index (PCI).

Inputs: one or more persona_signature.parquet files (output of
score_persona_signature.py). Each row is one reply with:
  - author_user_id, gid
  - signature_cosine_heldout: cosine of realized 8-dim polar vector vs expected
  - realized_pol_<d>, expected_pol_<d> for the 8 held-out probe dims:
    politeness, curiosity, tempo, self_focus, expressiveness, anxiety, warmth, hostility

Steps:
  1. Per (author_user_id, dim): mean realized vs mean expected over the user's
     replies. Determines per-user, per-dim agreement: 1 if sign matches AND
     |realized_mean| >= MAG_THRESHOLD (default 0.05), else 0.
  2. Per user: k = sum over 8 held-out dims of agreement.
  3. Partition users by k. Compute mean signature_cosine_heldout (per-user mean)
     inside each partition. Bootstrap percentile 95% CI by resampling users.

Output: prints per-k mean + 95% CI + n; optionally writes a JSON summary.

Usage:
  compute_pci.py PERSONA_SIG_PARQUET [PERSONA_SIG_PARQUET ...] \
      [--mag_threshold 0.05] [--n_boot 1000] [--seed 142] \
      [--out_json out.json]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HELD_OUT = ["politeness", "curiosity", "tempo", "self_focus",
            "expressiveness", "anxiety", "warmth", "hostility"]


def load(parquets):
    parts = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(parts, ignore_index=True)
    need = {"author_user_id", "signature_cosine_heldout"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"missing required columns: {miss}")
    return df


def per_user_agreement(df: pd.DataFrame, mag_threshold: float):
    grouped = df.groupby("author_user_id")
    # Mean signature cosine per user (the response variable we partition).
    mean_sig = grouped["signature_cosine_heldout"].mean().rename("mean_signature_cosine")
    # Per-dim per-user mean realized + expected
    out = pd.DataFrame(mean_sig)
    k = pd.Series(0, index=mean_sig.index, dtype=int)
    for d in HELD_OUT:
        rcol = f"realized_pol_{d}"
        ecol = f"expected_pol_{d}"
        if rcol not in df.columns or ecol not in df.columns:
            continue
        r = grouped[rcol].mean()
        e = grouped[ecol].mean()
        sign_match = (np.sign(r) == np.sign(e)) & (np.sign(e) != 0)
        magnitude_ok = r.abs() >= mag_threshold
        agree = (sign_match & magnitude_ok).astype(int)
        out[f"agree_{d}"] = agree
        k = k + agree
    out["k"] = k
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int,
                 alpha: float = 0.05):
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = values[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+", help="persona_signature.parquet file(s)")
    ap.add_argument("--mag_threshold", type=float, default=0.05)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=142)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    df = load(args.parquets)
    print(f"[pci] loaded {len(df)} replies from {len(args.parquets)} parquet(s)")
    print(f"[pci] unique authors: {df['author_user_id'].nunique()}")

    per_user = per_user_agreement(df, args.mag_threshold)
    print(f"[pci] per-user rows: {len(per_user)}")
    print(f"[pci] k distribution: {dict(per_user['k'].value_counts().sort_index())}")

    # Bucket k values
    buckets = [
        ("k=0",       lambda k: k == 0),
        ("k=1",       lambda k: k == 1),
        ("k=2",       lambda k: k == 2),
        ("k=3",       lambda k: k == 3),
        ("k>=4",      lambda k: k >= 4),
    ]
    results = []
    for label, predicate in buckets:
        mask = per_user["k"].map(predicate)
        sub = per_user.loc[mask, "mean_signature_cosine"].dropna()
        if len(sub) == 0:
            results.append({"bucket": label, "n": 0, "mean": None,
                            "ci_lo": None, "ci_hi": None})
            continue
        m = float(sub.mean())
        lo, hi = bootstrap_ci(sub.values, args.n_boot, args.seed)
        results.append({"bucket": label, "n": int(len(sub)),
                        "mean": m, "ci_lo": lo, "ci_hi": hi})

    print()
    print(f"  {'bucket':<8} {'n':>5} {'mean':>9} {'CI low':>9} {'CI high':>9}")
    print(f"  {'-'*8} {'-'*5} {'-'*9} {'-'*9} {'-'*9}")
    for r in results:
        if r["mean"] is None:
            print(f"  {r['bucket']:<8} {r['n']:>5}     --        --        --")
        else:
            print(f"  {r['bucket']:<8} {r['n']:>5} {r['mean']:>+9.4f} {r['ci_lo']:>+9.4f} {r['ci_hi']:>+9.4f}")

    if args.out_json:
        summary = {
            "schema": "observational_pci_v1",
            "mag_threshold": args.mag_threshold,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "n_users_total": int(len(per_user)),
            "k_distribution": {int(k): int(v) for k, v
                               in per_user["k"].value_counts().sort_index().items()},
            "buckets": results,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"\n[pci] wrote {args.out_json}")


if __name__ == "__main__":
    main()
