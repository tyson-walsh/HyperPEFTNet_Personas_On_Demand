"""Phase 3d persona-signature aggregation across forum dirs.

Lifted from m1_gpu-node_p2.yaml so the standard GPU phase-3 fan-out job can call
it as a script. Reads every <forum>/persona_signature/persona_signature.parquet
under FORUM_ROOT and writes one row per forum to SIGNATURE_ROOT/persona_signature_summary.csv.
"""
from __future__ import annotations
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

FORUM_ROOT = Path(os.environ["FORUM_ROOT"])
OUT_DIR = Path(os.environ["SIGNATURE_ROOT"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_RE = re.compile(
    r"^(2[abcd])_(vanilla|zero_delta|real_user|synth)_"
    r"(rage|empath|neutral)"
    r"(?:_(in_hull|near_hull|far_from_hull|far_kappa\d+|midpoint_baseline))?$"
)
HELD_OUT = ["politeness", "curiosity", "tempo", "self_focus",
            "expressiveness", "anxiety", "warmth", "hostility"]

rows: list[dict] = []
for fdir in sorted(FORUM_ROOT.iterdir()):
    if not fdir.is_dir():
        continue
    m = VARIANT_RE.match(fdir.name)
    if not m:
        continue
    code, variant, cohort, stratum = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    sig_p = fdir / "persona_signature" / "persona_signature.parquet"
    if not sig_p.exists():
        continue
    try:
        sig = pd.read_parquet(sig_p)
    except Exception as e:
        print(f"[phase3d-agg] {fdir.name}: read failed: {e}")
        continue
    row_base = dict(
        variant_code=code,
        variant=variant,
        cohort=cohort,
        stratum=stratum,
        n_rows=int(len(sig)),
    )
    row_base["signature_cosine_heldout_mean"] = float(
        pd.to_numeric(sig.get("signature_cosine_heldout"), errors="coerce").mean())
    row_base["signature_cosine_all9_mean"] = float(
        pd.to_numeric(sig.get("signature_cosine_all9"), errors="coerce").mean())
    row_base["signature_L1_heldout_mean"] = float(
        pd.to_numeric(sig.get("signature_L1_heldout"), errors="coerce").mean())
    # SD-normalized variants (2026-05-19 fix). Use these for the paper figures;
    # the cosine/L1 above are kept for backward-compat with old parquets.
    row_base["signature_L1_heldout_sdnorm_mean"] = float(
        pd.to_numeric(sig.get("signature_L1_heldout_sdnorm"), errors="coerce").mean())
    row_base["bullseye_match_sdnorm_mean"] = float(
        pd.to_numeric(sig.get("bullseye_match_sdnorm"), errors="coerce").mean())
    row_base["cohort_agreement_rate"] = float(
        pd.to_numeric(sig.get("cohort_agreement"), errors="coerce").mean())
    for d in HELD_OUT:
        rcol = f"realized_pol_{d}"
        ecol = f"expected_pol_{d}"
        if rcol in sig.columns and ecol in sig.columns:
            rv = pd.to_numeric(sig[rcol], errors="coerce")
            ev = pd.to_numeric(sig[ecol], errors="coerce")
            row_base[f"realized_{d}_mean"] = float(rv.mean())
            row_base[f"expected_{d}_mean"] = float(ev.mean())
            row_base[f"bias_{d}"] = float((rv - ev).mean())
            if "turn_idx" in sig.columns and "author_user_id" in sig.columns:
                slopes = []
                for _, g in sig.groupby("author_user_id"):
                    if len(g) < 3:
                        continue
                    x = pd.to_numeric(g["turn_idx"], errors="coerce").to_numpy()
                    y = pd.to_numeric(g[rcol], errors="coerce").to_numpy()
                    mask = np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 3:
                        continue
                    slopes.append(float(np.polyfit(x[mask], y[mask], 1)[0]))
                row_base[f"drift_slope_{d}_mean"] = float(np.mean(slopes)) if slopes else float("nan")
    rows.append(row_base)

if rows:
    df = pd.DataFrame(rows).sort_values(["variant_code", "cohort", "stratum"])
    df.to_csv(OUT_DIR / "persona_signature_summary.csv", index=False)
    print(f"[phase3d-agg] wrote {len(df)} rows -> persona_signature_summary.csv")
else:
    print("[phase3d-agg] no rows produced")
