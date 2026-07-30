"""Re-render fig:psi (Figure 3) — v4 FIX 2026-05-19.

DEEPER BUG FIX: signature_L1_heldout in score_persona_signature.py
computes mean |realized - expected| across 8 held-out dims that are on
RADICALLY different scales:
  - politeness, curiosity, tempo, self_focus: z-scored gstat (range ~ [-3, 3])
  - expressiveness: z-scored ratio (range ~ [0, 0.05])
  - anxiety, warmth, hostility: raw GoEmotions probs (range ~ [0, 0.1])

Curiosity (question-rate z-score) alone goes from residual 1.78 (real-user)
to 61.5 (far_kappa100); it dominates the L1 sum. All previous PSI/PFI
versions inherited this scale-mix bug.

V4 FIX: per-dim SD-normalize the residual before averaging. Each dim's
residual divided by the dim's REAL-USER EXPECTED SD. Mean across the 8
dims gives SD-units of error per dim. Then:
    match = max(0, 1 - SD_units_of_error / 2)
2 SDs off = match 0, 0 SDs off = match 1, capped to [0, 1].

Per-user MEAN match across all coherent replies; bootstrap 95% CI on the
across-user mean per stratum.

Split far-from-hull into FOUR kappa-specific bars: 3, 10, 25, 100.

Source CSV: production_findings_18MAY/psi_per_stratum_PFI_V4_SDNORM.csv
Threshold reference: production_findings_18MAY/psi_threshold_derivation.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HullGreen    = "#115740"
BandGold     = "#B9975B"
RageRed    = "#B41A1F"
EmpathBlue = "#2566A8"
FarBrown   = "#7B4B26"

# label, mean_match, ci95_lo, ci95_hi, p25, p75, color
DATA = [
    ("real-user",   0.226, 0.214, 0.238, 0.133, 0.313, EmpathBlue),
    ("synth\nin-hull",  0.265, 0.256, 0.274, 0.179, 0.361, HullGreen),
    ("synth\nnear-hull",0.257, 0.248, 0.267, 0.185, 0.334, BandGold),
    ("synth\nfar κ=3",   0.104, 0.096, 0.113, 0.000, 0.181, FarBrown),
    ("synth\nfar κ=10",  0.033, 0.028, 0.039, 0.000, 0.000, "#A65A2C"),
    ("synth\nfar κ=25",  0.013, 0.011, 0.017, 0.000, 0.000, "#C0623B"),
    ("synth\nfar κ=100", 0.001, 0.000, 0.001, 0.000, 0.000, RageRed),
]
# Empirically-derived threshold (decile D3-D4 inflection on cohort_agreement).
THRESHOLD = 0.10  # "carries persona signal" floor; on-manifold strata above, far-hull below


def render():
    # 2-column / \textwidth target: ACL textwidth ≈ 6.3" wide page, so 2-col
    # figure renders at ~7" x 3.2" in print. Render larger for sharpness, let
    # LaTeX scale to \textwidth.
    fig, ax = plt.subplots(figsize=(11.0, 4.2), dpi=220, facecolor="white")
    ax.set_facecolor("white")

    xs = np.arange(len(DATA))
    means  = [d[1] for d in DATA]
    cis_lo = [d[2] for d in DATA]
    cis_hi = [d[3] for d in DATA]
    p25s   = [d[4] for d in DATA]
    p75s   = [d[5] for d in DATA]
    colors = [d[6] for d in DATA]
    labels = [d[0] for d in DATA]

    # Pass-band tint (above empirical threshold).
    ax.axhspan(THRESHOLD, 0.50, color=HullGreen, alpha=0.06, zorder=1)
    # Collapse-band tint.
    ax.axhspan(-0.01, 0.05, color=RageRed, alpha=0.07, zorder=1)

    # On-manifold separator: visual dotted line between near-hull and far_κ=3
    ax.axvline(2.5, color="black", linestyle=":", lw=0.7, alpha=0.5, zorder=2)
    ax.text(0.97, 0.48, "on-manifold", ha="right", va="top", fontsize=9,
            transform=ax.transData, fontstyle="italic", color="black", alpha=0.7)
    ax.text(5.6, 0.48, "off-manifold (κ pushout)", ha="center", va="top", fontsize=9,
            transform=ax.transData, fontstyle="italic", color="black", alpha=0.7)

    bar_width = 0.66
    for i, (m, lo, hi, q25, q75, c) in enumerate(
            zip(means, cis_lo, cis_hi, p25s, p75s, colors)):
        ax.bar(i, m, width=bar_width, color=c, edgecolor="black",
               linewidth=0.6, zorder=3)
        # IQR whiskers
        ax.plot([i, i], [q25, q75], color="black", lw=1.0, alpha=0.7, zorder=4)
        ax.plot([i - 0.08, i + 0.08], [q25, q25], color="black", lw=1.0,
                alpha=0.7, zorder=4)
        ax.plot([i - 0.08, i + 0.08], [q75, q75], color="black", lw=1.0,
                alpha=0.7, zorder=4)
        # 95% CI on mean (thicker red bracket)
        ax.plot([i, i], [lo, hi], color="black", lw=2.4, zorder=5)
        # value label
        y_label = max(m, q75) + 0.012
        ax.text(i, y_label, f"{m:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", zorder=6)

    # Threshold line — annotation pinned to the RIGHT side of the plot, in the
    # empty space to the right of the far-kappa=100 bar (which sits at ~0.001
    # so the annotation has clear vertical space at the 0.10 threshold level).
    ax.axhline(THRESHOLD, ls="--", color=HullGreen, lw=1.2, zorder=2)
    ax.text(len(DATA) - 0.55, THRESHOLD + 0.008,
            f"empirical threshold $\\approx$ {THRESHOLD:.2f}",
            ha="right", va="bottom", fontsize=9, color=HullGreen,
            fontweight="bold", zorder=7)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("per-user persona fidelity\n(SD-normalized bullseye match)",
                  fontsize=10)
    ax.set_ylim(-0.01, 0.50)
    ax.set_xlim(-0.55, len(DATA) - 0.45)
    ax.grid(axis="y", alpha=0.18, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out_pdf = Path("/workspace/hypernets/PROSPECTUS/"
                   "HyperPEFTNet_RQ2/paper2/figures/psi_distribution.pdf")
    fig.tight_layout()
    fig.savefig(out_pdf, facecolor="white", edgecolor="none")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=220, facecolor="white",
                edgecolor="none")
    plt.close(fig)
    print(f"[psi-production-v4] -> {out_pdf}")


if __name__ == "__main__":
    render()
