"""Render the production fig:bullseye for the paper.

Two-panel (rage + empath), 6 markers each: recon, in-hull, near-hull, and
three kappa stars on the SW ray with the kappa value inside each star.

Production data sourced from
`paper2/production_findings_18MAY/bullseye_match_with_ci.csv` (recon,
in-hull, near-hull, kappa=3, kappa=10) plus
`kappa_sweep_bullseye_match.csv` (kappa=25).

Output writes to `paper2/figures/bullseye.pdf` so the paper picks it up
without changing the IfFileExists path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

RageRed = "#B41A1F"
HullGreen = "#115740"
BandGold = "#B9975B"
EmpathBlue = "#2566A8"

PRODUCTION = {
    "rage": {
        "recon":     {"match": 0.278, "ci": (0.257, 0.298)},
        "in_hull":   {"match": 0.262, "ci": (0.244, 0.281)},
        "near_hull": {"match": 0.228, "ci": (0.207, 0.250)},
        "far_k3":    {"match": 0.174, "ci": (0.152, 0.196)},
        "far_k10":   {"match": 0.086, "ci": (0.070, 0.106)},
        "far_k25":   {"match": 0.037, "ci": (0.026, 0.049)},
        "far_k100":  {"match": 0.000, "ci": (0.000, 0.001)},
    },
    "empath": {
        "recon":     {"match": 0.235, "ci": (0.216, 0.254)},
        "in_hull":   {"match": 0.314, "ci": (0.297, 0.332)},
        "near_hull": {"match": 0.322, "ci": (0.301, 0.342)},
        "far_k3":    {"match": 0.188, "ci": (0.168, 0.209)},
        "far_k10":   {"match": 0.075, "ci": (0.060, 0.091)},
        "far_k25":   {"match": 0.036, "ci": (0.025, 0.048)},
        "far_k100":  {"match": 0.002, "ci": (0.000, 0.004)},
    },
}

MARKER_ANGLES = {
    "recon":     np.deg2rad(45),
    "in_hull":   np.deg2rad(135),
    "near_hull": np.deg2rad(-45),
}
KAPPA_ANGLE = np.deg2rad(-135)

MARKER_STYLE = {
    "recon":     dict(marker="o", color=EmpathBlue, size=240),
    "in_hull":   dict(marker="D", color=HullGreen,    size=240),
    "near_hull": dict(marker="p", color=BandGold,     size=300),
    "far_k":     dict(marker="*", color=RageRed,    size=720),
}
# Spacing 0.85 between consecutive kappa markers; star radius at size=720
# is ~0.27 data units in this axis, giving ~0.3 gap between star edges.
KAPPA_RADII = {"far_k3": 1.10, "far_k10": 1.95, "far_k25": 2.80, "far_k100": 3.65}
INSIDE_R_MIN = 0.40
INSIDE_R_MAX = 1.30


def _inside_r(match: float, vals: list[float]) -> float:
    best_m = max(vals)
    worst_m = min(vals)
    if best_m - worst_m < 1e-6:
        return 0.5 * (INSIDE_R_MIN + INSIDE_R_MAX)
    frac = (match - worst_m) / (best_m - worst_m)
    return INSIDE_R_MIN + (INSIDE_R_MAX - INSIDE_R_MIN) * (1.0 - frac)


def draw_panel(ax, cx, cy, ring_color, cohort_name, data):
    for radius_units, alpha in [(2.8, 0.10), (2.2, 0.20), (1.5, 0.30),
                                (0.9, 0.45), (0.35, 0.65)]:
        ax.add_patch(plt.Circle((cx, cy), radius_units, color=ring_color,
                                alpha=alpha, ec=ring_color, linewidth=0.3,
                                zorder=2))
    ax.plot([cx - 3.0, cx + 3.0], [cy, cy], ls="--", color="black",
            alpha=0.18, lw=0.4, zorder=3)
    ax.plot([cx, cx], [cy - 3.0, cy + 3.0], ls="--", color="black",
            alpha=0.18, lw=0.4, zorder=3)
    ax.text(cx, cy + 3.05, f"true {cohort_name}", color=ring_color,
            fontsize=11, fontweight="bold", ha="center", va="bottom",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none",
                      boxstyle="round,pad=0.18"), zorder=11)

    inside_vals = [data[k]["match"] for k in ("recon", "in_hull", "near_hull")]

    for name in ("recon", "in_hull", "near_hull"):
        r_disp = _inside_r(data[name]["match"], inside_vals)
        ang = MARKER_ANGLES[name]
        x = cx + r_disp * np.cos(ang)
        y = cy + r_disp * np.sin(ang)
        st = MARKER_STYLE[name]
        ax.scatter([x], [y], marker=st["marker"], s=st["size"],
                   color=st["color"], edgecolor="white",
                   linewidths=1.4, zorder=10)
        label_r = min(2.80, max(1.55, r_disp + 0.45))
        lx = cx + label_r * np.cos(ang)
        ly = cy + label_r * np.sin(ang)
        ax.annotate(f"{data[name]['match']:.2f}", xy=(lx, ly),
                    ha="center", va="center", fontsize=8.0,
                    color="black", zorder=11, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.95,
                              edgecolor=st["color"], linewidth=0.7,
                              boxstyle="round,pad=0.20"))

    kappa_labels = {"far_k3": "3", "far_k10": "10", "far_k25": "25",
                    "far_k100": "100"}
    perp_side = {"far_k3": +1, "far_k10": -1, "far_k25": +1, "far_k100": -1}
    for name in ("far_k3", "far_k10", "far_k25", "far_k100"):
        r_disp = KAPPA_RADII[name]
        x = cx + r_disp * np.cos(KAPPA_ANGLE)
        y = cy + r_disp * np.sin(KAPPA_ANGLE)
        st = MARKER_STYLE["far_k"]
        ax.scatter([x], [y], marker=st["marker"], s=st["size"],
                   color=st["color"], edgecolor="white",
                   linewidths=1.6, zorder=10)
        # Digit sizes scale to fit inside star marker without overflow.
        # size=720 -> radius ~0.27 data units; 1-2 char digits at 5-7pt.
        if kappa_labels[name] == "100":
            kappa_label_size = 5
        elif kappa_labels[name] == "25":
            kappa_label_size = 6.5
        else:
            kappa_label_size = 7.5
        ax.text(x, y, kappa_labels[name], ha="center", va="center",
                fontsize=kappa_label_size, color="white",
                fontweight="bold", zorder=12)
        perp_ang = KAPPA_ANGLE + np.deg2rad(90 * perp_side[name])
        lx = x + 0.42 * np.cos(perp_ang)
        ly = y + 0.42 * np.sin(perp_ang)
        ax.annotate(f"{data[name]['match']:.2f}", xy=(lx, ly),
                    ha="center", va="center", fontsize=8.0,
                    color="black", zorder=11, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.95,
                              edgecolor=RageRed, linewidth=0.7,
                              boxstyle="round,pad=0.20"))


def render():
    fig = plt.figure(figsize=(13.0, 6.5), dpi=220, facecolor="white")
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.84])
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(-5.4, 5.2)
    ax.axis("off")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(plt.Rectangle((-9.5, -5.4), 19.0, 10.6,
                               facecolor="white", edgecolor="none",
                               zorder=0))

    draw_panel(ax, cx=-4.6, cy=0.4, ring_color=RageRed,
               cohort_name="rage", data=PRODUCTION["rage"])
    draw_panel(ax, cx=4.6, cy=0.4, ring_color=HullGreen,
               cohort_name="empath", data=PRODUCTION["empath"])

    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=EmpathBlue,
               markersize=11, markeredgecolor="white", markeredgewidth=1.4,
               label="reconstructed real user"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=HullGreen,
               markersize=11, markeredgecolor="white", markeredgewidth=1.4,
               label="synth in-hull"),
        Line2D([0], [0], marker="p", color="w", markerfacecolor=BandGold,
               markersize=13, markeredgecolor="white", markeredgewidth=1.4,
               label="synth near-hull"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=RageRed,
               markersize=22, markeredgecolor="white", markeredgewidth=1.4,
               label=r"synth far-from-hull (digit $=\kappa$)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.02))

    out_pdf = Path("/workspace/hypernets/PROSPECTUS/"
                   "HyperPEFTNet_RQ2/paper2/figures/bullseye.pdf")
    fig.savefig(out_pdf, facecolor="white", edgecolor="none",
                transparent=False)
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=200, facecolor="white", edgecolor="none",
                transparent=False)
    plt.close(fig)
    print(f"[bullseye-production] -> {out_pdf}")
    print(f"[bullseye-production] -> {out_png}")


if __name__ == "__main__":
    render()
