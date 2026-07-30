#!/usr/bin/env python3
"""Render the kappa-sweep figure for Paper 2 App P.

Two-panel figure summarizing Experiment D + E from
deep_dive_stratum_geometry.py:

  A. Mahalanobis q50 vs kappa (log-log) with horizontal lines marking
     the real-user q90 and q99 envelopes. Shows that kappa=3 is the
     point where the off-manifold envelope is solidly cleared while
     kappa=10, 25, 50, 100 sit progressively further out without
     changing the underlying direction.
  B. Per-element clamp-saturation fraction vs kappa, and the
     corresponding median ||Delta theta||_2 (on a secondary y-axis).
     Shows that clamp saturation already covers 88% of entries at
     kappa=3 and asymptotes near 99.5% at kappa=100; the per-element
     magnitude floor at the production clamp (+/- 0.020) means
     increasing kappa beyond ~3 saturates more entries without adding
     per-user differentiation.

Reads the deep_dive_results.json produced by
deep_dive_stratum_geometry.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Prospectus palette
HullGreen     = "#115740"
BandGold      = "#B9975B"
RageRed     = "#C0392B"
EmpathBlue  = "#2980B9"
INBlue      = "#6CACE4"

ROOT = Path(__file__).parent
IN_JSON = ROOT / "deep_dive_results.json"
OUT_PDF = ROOT.parent / "figures" / "kappa_sweep.pdf"

COHORT_COLOR = {"rage": RageRed, "empath": HullGreen, "neutral": "#7F7F7F"}
COHORT_LS = {"rage": "-", "empath": "-", "neutral": "--"}


def main() -> None:
    data = json.loads(IN_JSON.read_text())
    desc = data["D_kappa_sweep_descriptor"]["kappa_sweep_descriptor"]
    real_q99 = float(data["D_kappa_sweep_descriptor"]["real_q99"])
    hyp = data["E_kappa_sweep_hypernet"]["kappa_sweep_hypernet"]

    kappas_sorted = sorted({float(r["kappa"]) for r in desc})
    cohorts = ["rage", "empath", "neutral"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.0), dpi=200)
    fig.patch.set_facecolor("white")

    # ---------- Panel A: Mahalanobis q50 vs kappa ----------
    for c in cohorts:
        xs = []
        ys = []
        for k in kappas_sorted:
            rows = [r for r in desc if float(r["kappa"]) == k and r["cohort"] == c]
            if rows:
                xs.append(k)
                ys.append(rows[0]["M_q50"])
        axA.plot(xs, ys, marker="o", color=COHORT_COLOR[c], linestyle=COHORT_LS[c],
                 lw=1.6, markersize=5, label=c)
    axA.axhline(real_q99, color="black", lw=0.8, ls=":", alpha=0.6,
                label=f"real $q_{{99}}$ = {real_q99:.1f}")
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel(r"$\kappa$ (anchor-pushout strength)")
    axA.set_ylabel(r"Mahalanobis $q_{50}$ to training centroid")
    axA.set_title("A. Off-manifold geometry vs $\\kappa$",
                  fontsize=10, fontweight="bold")
    axA.grid(True, which="both", alpha=0.18, ls=":")
    axA.legend(loc="lower right", fontsize=8, frameon=False)
    # Mark the deployment-mode kappa=3
    axA.axvline(3.0, color=BandGold, lw=0.8, ls="--", alpha=0.6)
    axA.text(3.0, axA.get_ylim()[1] * 0.4, "  paper's $\\kappa=3$",
             fontsize=8, color=BandGold, fontweight="bold",
             rotation=90, va="top", ha="left")

    # ---------- Panel B: clamp saturation + ||Delta theta||_2 vs kappa ----------
    axB2 = axB.twinx()
    # primary: clamp saturation
    for c in cohorts:
        xs, ys = [], []
        for k in kappas_sorted:
            rows = [r for r in hyp if float(r["kappa"]) == k and r["cohort"] == c]
            if rows:
                xs.append(k)
                ys.append(rows[0]["clamp_sat_mean"])
        axB.plot(xs, ys, marker="o", color=COHORT_COLOR[c], linestyle=COHORT_LS[c],
                 lw=1.6, markersize=5, label=f"clamp sat. ({c})" if c == "rage" else c)
    # secondary: ||Delta theta||_2 median (rage only for clarity)
    xs, ys = [], []
    for k in kappas_sorted:
        rows = [r for r in hyp if float(r["kappa"]) == k and r["cohort"] == "rage"]
        if rows:
            xs.append(k)
            ys.append(rows[0]["delta_norm_q50"])
    axB2.plot(xs, ys, marker="s", color=EmpathBlue, lw=1.0, markersize=4,
              ls="--", alpha=0.85, label=r"$\|\Delta\theta\|_2$ median (rage)")

    axB.set_xscale("log")
    axB.set_xlabel(r"$\kappa$ (anchor-pushout strength)")
    axB.set_ylabel("clamp-saturated fraction")
    axB2.set_ylabel(r"$\|\Delta\theta\|_2$ median", color=EmpathBlue)
    axB2.set_yscale("log")
    axB2.tick_params(axis="y", labelcolor=EmpathBlue)
    axB.set_ylim(0.65, 1.01)
    axB.set_title("B. Hypernet output vs $\\kappa$",
                  fontsize=10, fontweight="bold")
    axB.grid(True, which="both", alpha=0.18, ls=":")
    axB.axvline(3.0, color=BandGold, lw=0.8, ls="--", alpha=0.6)
    # Combined legend
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axB2.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=7,
               frameon=False, ncol=1)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[kappa-sweep] wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
