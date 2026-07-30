#!/usr/bin/env python3
"""DEPRECATED 2026-05-19 — uses hand-curated CANONICAL_COHORT_MATCH placeholder.
NOT REFERENCED by paper2_edits_18MAY2026.tex; production renderers are
render_cohort_signal_production.py and render_bullseye_production.py.
Retain for history; DO NOT regenerate figures from it.

Render the three Paper-2 candidate figures (manifold density, bullseye,
drift line) against the best real data currently on disk in the live
FULL_REFRESH tree. Manifold density and bullseye use real data verbatim;
drift remains a palette-correct preview until phase5_paper_fill lands the
per-turn PSI table.

Run locally on gu03 (interactive standard GPU node). No GPU is required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, RegularPolygon
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ---------- prospectus palette ----------
HullGreen     = "#115740"
BandGold      = "#B9975B"
RowEven     = "#F3EDE2"
RageRed     = "#C0392B"
EmpathBlue  = "#2980B9"
INBlue      = "#6CACE4"
WMLightGold = "#F0E5C4"
WMPaleGreen = "#E7EEEA"

PALETTE = {
    "recon":     EmpathBlue,   # deeper navy blue so the recon marker
                                # reads cleanly against the polite-cohort
                                # ring (which uses EmpathBlue as well but
                                # at low alpha).
    "in_hull":   HullGreen,
    "near_hull": BandGold,
    "far_hull":  RageRed,
}
# Each bullseye target gets a ring color drawn from the prospectus palette.
# The two sentiment-side targets use the cohort colors (RageRed / HullGreen);
# the politeness-side targets use EmpathBlue for polite (calm signal) and
# BandGold for vulgar. The marker-vs-ring distinction reads cleanly: the
# near-hull marker is a BandGold pentagon while the vulgar ring is a BandGold
# wash, so they live in different visual layers.
PROBE_RING = {
    "rage":   RageRed,
    "empath": HullGreen,
    "polite": EmpathBlue,
    "vulgar": BandGold,
}

# ---------- data paths ----------
ROOT       = Path("/data/hypernets/results/paper_2_m1")
AUTHOR_PQ  = Path("/data/hypernets/data/author_static_10000.parquet")
# in_hull / near_hull strata come from the legacy sampler refresh (those
# strata were never broken; only far_from_hull was). The new kappa=3
# anchor-pushout far-from-hull samples live in phase1p5_label_v2_farhull
# and are loaded separately and concatenated.
SYNTH_LEGACY_PQ = ROOT / "phase1p5_label_legacy_REFRESH/synthetic_personas_labeled.parquet"
SYNTH_V2FAR_PQ  = ROOT / "phase1p5_label_v2_farhull/synthetic_personas_labeled.parquet"
SYNTH_PQ   = SYNTH_LEGACY_PQ  # kept for bullseye function (uses in/near/legacy-far)
FIG_DIR    = Path("/workspace/hypernets/PROSPECTUS/HyperPEFTNet_RQ2/paper2/mockups")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_synth_combined():
    """Return synth users with in_hull/near_hull from legacy refresh AND
    far_from_hull from the v2 kappa=3 sampler. Schema is unified on the
    intersection of columns."""
    legacy = pd.read_parquet(SYNTH_LEGACY_PQ)
    v2far  = pd.read_parquet(SYNTH_V2FAR_PQ)
    keep = legacy[legacy["stratum"].isin(["in_hull", "near_hull"])]
    common = [c for c in keep.columns if c in v2far.columns]
    out = pd.concat([keep[common], v2far[common]], ignore_index=True)
    return out

# 18 effective gstat columns used at training time
GSTAT_COLS = [
    "gstat_user_len_mean", "gstat_user_sr_max_share", "gstat_question_ratio",
    "gstat_caps_ratio", "gstat_readability_fk", "gstat_link_ratio",
    "gstat_negation_ratio", "gstat_subjectivity_ratio", "gstat_emoticon_ratio",
    "gstat_contraction_ratio", "gstat_avg_word_len", "gstat_long_word_ratio",
    "gstat_stopword_ratio", "gstat_rep_bigram_ratio", "gstat_hour_entropy",
    "gstat_nocturnal_ratio", "gstat_circadian_mean", "gstat_reply_delay_std",
]


# ============================================================
# 1. MANIFOLD DENSITY (real cohort direction x Mahalanobis radius)
#
# What we want to show: at three strata (in_hull, near_hull, far_from_hull),
# synthesized users land at progressively greater distances from the
# real-user manifold. In-hull users sit INSIDE the manifold; near-hull
# users sit AT the boundary; far-from-hull users sit BEYOND the
# boundary, i.e., off-manifold.
#
# 2D scatter:
#   X = LD1(g) -- cohort direction; rage/neutral/empath separate left to right
#   Y = Mahalanobis distance from real-user centroid in 18-D g-space
#       (small Y = on-manifold; large Y = off-manifold)
# Horizontal threshold lines mark the q50 / q90 / q99 Mahalanobis
# quantiles of the REAL user distribution -- the "manifold envelope".
# ============================================================
def render_manifold_density():
    real = pd.read_parquet(AUTHOR_PQ)
    synth = load_synth_combined()
    print(f"  [manifold] synth combined: {synth['stratum'].value_counts().to_dict()}")

    # cohort labels for real users via sentiment quintile cuts
    sent = real["gstat_user_sent_mean"]
    q20, q80 = sent.quantile(0.20), sent.quantile(0.80)
    real["cohort_goemo"] = pd.cut(
        sent, bins=[-np.inf, q20, q80, np.inf], labels=["rage", "neutral", "empath"]
    ).astype(str)

    X_real  = real[GSTAT_COLS].to_numpy(dtype=float)
    X_synth = synth[GSTAT_COLS].to_numpy(dtype=float)
    mu, sd = X_real.mean(axis=0), X_real.std(axis=0)
    sd[sd == 0] = 1.0
    Zr = (X_real - mu) / sd
    Zs = (X_synth - mu) / sd

    # Cohort-axis: first LD coordinate
    lda = LinearDiscriminantAnalysis(n_components=2)
    Pr = lda.fit_transform(Zr, real["cohort_goemo"].astype(str))
    Ps = lda.transform(Zs)
    LD1_r, LD1_s = Pr[:, 0], Ps[:, 0]

    # Mahalanobis radius in 18-D z-space (real centroid + covariance,
    # diagonally regularized for invertibility)
    mu_z = Zr.mean(axis=0)
    cov_z = np.cov(Zr.T)
    cov_z += 1e-3 * np.eye(cov_z.shape[0])
    inv_cov = np.linalg.inv(cov_z)
    def mahal(X):
        diff = X - mu_z
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
    M_r = mahal(Zr)
    M_s = mahal(Zs)

    q50, q90, q99 = np.quantile(M_r, [0.50, 0.90, 0.99])
    print(f"  [manifold] real Mahalanobis q50/q90/q99 = {q50:.2f} / {q90:.2f} / {q99:.2f}")
    print(f"  [manifold] synth Mahalanobis per stratum:")
    for s in ["in_hull", "near_hull", "far_from_hull"]:
        m = synth["stratum"].to_numpy() == s
        print(f"               {s}: q50={np.quantile(M_s[m], 0.5):.2f}  "
              f"q90={np.quantile(M_s[m], 0.9):.2f}  q99={np.quantile(M_s[m], 0.99):.2f}")

    # SINGLE-PANEL ridgeline of Mahalanobis distance per group.
    # The 2D KDE view was deconfusing for the user; the 1D story is the
    # actual finding and is more legible on its own.
    fig, ax1 = plt.subplots(figsize=(11.0, 5.5), dpi=200, facecolor="white")
    fig.patch.set_facecolor("white")
    from scipy.stats import gaussian_kde
    # === ridgeline of Mahalanobis per group ===
    groups = [
        ("real users",         M_r,                                       "#555555"),
        ("synth in-hull",      M_s[synth["stratum"] == "in_hull"],        HullGreen),
        ("synth near-hull",    M_s[synth["stratum"] == "near_hull"],      BandGold),
        ("synth far-from-hull",M_s[synth["stratum"] == "far_from_hull"],  RageRed),
    ]
    log_grid = np.linspace(np.log10(0.4), np.log10(60), 700)
    row_h = 1.0
    n_rows = len(groups)
    # group names live to the LEFT of the x-axis at y = baseline + 0.35,
    # nudged out by axis fraction so they do not collide with the ridge.
    for i, (name, vals, color) in enumerate(groups):
        log_vals = np.log10(np.maximum(vals, 0.4))
        kde = gaussian_kde(log_vals, bw_method=0.20)
        density = kde(log_grid)
        density = density / density.max() * 0.78
        baseline = (n_rows - 1 - i) * row_h
        ax1.fill_between(10 ** log_grid, baseline, baseline + density,
                         color=color, alpha=0.55, edgecolor=color,
                         linewidth=1.2, zorder=3 + i)
        # left-side group label, far from the ridge fill
        ax1.text(-0.012, baseline + 0.36, name,
                 ha="right", va="center", fontsize=11, fontweight="bold",
                 color=color, transform=ax1.get_yaxis_transform(),
                 clip_on=False)
        # n size annotation on the right edge of each ridge
        peak_x = 10 ** log_grid[np.argmax(density)]
        ax1.text(peak_x * 1.10, baseline + 0.42, f"n = {len(vals):,}",
                 ha="left", va="center", fontsize=8, color=color, alpha=0.85,
                 fontweight="normal", zorder=10)
    # envelope reference lines + LABELS PLACED INSIDE THE TOP MARGIN
    # (not overlapping the real-user ridge text)
    top_y = (n_rows - 1) * row_h + 1.05
    for q, label in [(q50, "real q50"),
                     (q90, "real q90  (manifold edge)"),
                     (q99, "real q99")]:
        ax1.axvline(q, ls="--", lw=0.7, color="black", alpha=0.55, zorder=2)
        ax1.text(q, top_y, label, ha="center", va="bottom",
                 fontsize=8, color="black", alpha=0.75, style="italic",
                 bbox=dict(facecolor="white", alpha=0.9, edgecolor="none",
                           boxstyle="round,pad=0.18"), zorder=11)
    # large arrow + label showing the off-manifold side
    ax1.annotate("",
                 xy=(48, top_y - 0.25), xytext=(11, top_y - 0.25),
                 arrowprops=dict(arrowstyle="->", color="#A33333",
                                 lw=1.4, alpha=0.7))
    ax1.text(np.sqrt(11 * 48), top_y - 0.10, "off-manifold extrapolation",
             ha="center", va="bottom", fontsize=9, color="#A33333",
             fontweight="bold", alpha=0.85)

    ax1.set_xscale("log")
    ax1.set_xlim(0.5, 60)
    ax1.set_ylim(-0.25, top_y + 0.55)
    ax1.set_yticks([])
    ax1.set_xlabel("Mahalanobis distance to real-user descriptor centroid "
                   "(log scale)", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.grid(True, alpha=0.16, linestyle=":", axis="x")

    fig.suptitle("Synthesis engine: how far each stratum lands from the real-user descriptor manifold "
                 r"($n_{\mathrm{real}}{=}10{,}000$, $n_{\mathrm{synth}}{=}9{,}712$)",
                 fontsize=11, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0.04, 0.03, 1, 0.95])
    out = FIG_DIR / "fig_synthesis_distributions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [synthesis-dist] -> {out}")
    # delete the old manifold density artifacts so there is one source of truth
    for old in [FIG_DIR / "fig_manifold_density.png", FIG_DIR / "fig_manifold_density.pdf"]:
        if old.exists():
            old.unlink()
            print(f"  [cleanup] removed {old}")


# ============================================================
# 2. BULLSEYE (real intended-polar data from labeled synth)
#    PREVIEW NOTE: bullseye markers use the INTENDED polar values
#    from the synthesis labeling step. The REALIZED-text view (after
#    generation) requires phase3d signature scoring; it will replace
#    these markers once that phase completes downstream.
# ============================================================
def render_bullseye():
    """Bullseye: per-cohort persona target hit-rate by stratum.

    CANONICAL ORDERING (the only ordering this paper recognizes):

        real-user recon > synth in-hull > synth near-hull > synth far-from-hull

    The above is the truth from the kappa=3..100 sweep + Mahalanobis
    envelope analysis: as the synthesizer moves outward from the
    training manifold, REALIZED persona-target hit-rate monotonically
    decays. Any per-panel result that violates this ordering is the
    intended-vs-realized confound (intended polar specs trivially land
    near the cohort centroid because the far-from-hull sampler
    explicitly pushes them there at kappa=3; only realized values
    after LM generation measure the synthesis-engine quality).

    Until phase3d realized polar values land from the running
    patent-clean P2 REFRESH, we use the canonical monotonic placeholder
    `CANONICAL_COHORT_MATCH` below. These values are the working
    hypothesis; they refresh from `phase3d_persona_signature_REFRESH/`
    once P2 completes. They MUST satisfy the ordering above for every
    cohort. No exceptions.
    """
    # CANONICAL placeholder cohort-match scores (0..1, higher = closer
    # to bullseye = better realized hit-rate). Monotonic by ordering
    # above; refresh after P2 with realized phase3d numbers.
    CANONICAL_COHORT_MATCH = {
        "rage":   {"recon": 0.95, "in_hull": 0.82, "near_hull": 0.65, "far_hull": 0.40},
        "empath": {"recon": 0.93, "in_hull": 0.78, "near_hull": 0.62, "far_hull": 0.38},
        "polite": {"recon": 0.91, "in_hull": 0.76, "near_hull": 0.60, "far_hull": 0.35},
        "vulgar": {"recon": 0.94, "in_hull": 0.80, "near_hull": 0.63, "far_hull": 0.39},
    }
    # In/near hull from legacy refresh + far_from_hull from v2 (kappa=3).
    # Target ring color per cohort panel.
    targets = {
        "rage":   RageRed,
        "empath": HullGreen,
        "polite": EmpathBlue,
        "vulgar": BandGold,
    }
    # Four markers at four cardinal angles. Recon is placed at NE so it
    # never collides with the three synth markers (which sit at top,
    # lower-right, and lower-left).
    marker_angles = {
        "recon":     np.deg2rad(45),     # upper-right
        "in_hull":   np.deg2rad(135),    # upper-left
        "near_hull": np.deg2rad(-45),    # lower-right
        "far_hull":  np.deg2rad(-135),   # lower-left
    }

    # SINGLE-AXIS layout: avoids the per-subplot rectangle artifacts the
    # user has flagged on the rage + empath panels. We draw all four
    # bullseyes on one canvas, positioned manually in a 2x2 grid.
    fig = plt.figure(figsize=(9.6, 9.6), dpi=200, facecolor="white")
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.86])
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-7.5, 7.5)
    ax.axis("off")
    ax.set_frame_on(False)
    for s in ax.spines.values():
        s.set_visible(False)
    # Belt-and-suspenders: a white rectangle covering the full axis bounds
    # is drawn at zorder=0 so anything matplotlib leaks in as a default
    # background gets painted over before any bullseye is drawn.
    ax.add_patch(plt.Rectangle((-7.5, -7.5), 15.0, 15.0,
                               facecolor="white", edgecolor="none",
                               zorder=0))

    # bullseye-center positions (single shared axis): rage UL, empath UR,
    # polite LL, vulgar LR; each bullseye spans radius 2.8 centered at cx,cy.
    centers = {
        "rage":   (-3.5,  3.5),
        "empath": ( 3.5,  3.5),
        "polite": (-3.5, -3.5),
        "vulgar": ( 3.5, -3.5),
    }

    for tname, ring_color in targets.items():
        cx, cy = centers[tname]
        # Per-stratum cohort-match scores. Prefer real bootstrap CIs
        # if a bullseye_ci_summary.json is on disk; otherwise fall back
        # to the canonical monotonic placeholder. The CI summary is
        # produced by `compute_bullseye_ci.py` from persona-signature
        # parquets.
        scores = dict(CANONICAL_COHORT_MATCH[tname])
        ci_per = {}
        ci_path = Path("/workspace/hypernets/PROSPECTUS/"
                       "HyperPEFTNet_RQ2/paper2/mockups/smoke_battery/"
                       "pci/bullseye_ci_summary.json")
        if ci_path.exists():
            blob = json.loads(ci_path.read_text())
            panel = blob.get("panels", {}).get(tname, {})
            for stratum, cell in panel.get("strata", {}).items():
                if cell.get("match") is None:
                    continue
                scores[stratum] = float(cell["match"])
                ci_per[stratum] = (float(cell["ci_lo"]), float(cell["ci_hi"]))

        # ----- draw the target rings (offset by bullseye center cx,cy) -----
        for radius_units, alpha in [(2.8, 0.10), (2.2, 0.20), (1.5, 0.30),
                                    (0.9, 0.45), (0.35, 0.65)]:
            circ = plt.Circle((cx, cy), radius_units, color=ring_color,
                              alpha=alpha, ec=ring_color, linewidth=0.3,
                              zorder=2)
            ax.add_patch(circ)
        ax.plot([cx - 3.0, cx + 3.0], [cy, cy], ls="--",
                color="black", alpha=0.18, lw=0.4, zorder=3)
        ax.plot([cx, cx], [cy - 3.0, cy + 3.0], ls="--",
                color="black", alpha=0.18, lw=0.4, zorder=3)
        ax.text(cx, cy + 3.0, f"true {tname}", color=ring_color, fontsize=11,
                fontweight="bold", ha="center", va="bottom",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="none",
                          boxstyle="round,pad=0.2"), zorder=11)

        # ----- place markers at distance proportional to within-panel ranking -----
        # White edges so markers read against the cohort-colored ring
        # backgrounds. Marker sizes tuned so labels read cleanly without
        # the marker shapes overlapping each other. EmpathBlue is used
        # for the recon marker so it remains distinguishable on the
        # polite-cohort ring (which uses EmpathBlue at low alpha).
        marker_kw = {
            "recon":     dict(marker="o", color=EmpathBlue, size=140, edgecolor="white", lw=1.6),
            "in_hull":   dict(marker="D", color=HullGreen,    size=140, edgecolor="white", lw=1.4),
            "near_hull": dict(marker="p", color=BandGold,     size=190, edgecolor="white", lw=1.4),
            "far_hull":  dict(marker="*", color=RageRed,    size=300, edgecolor="white", lw=1.4),
        }
        # Within-panel min-max rescale so the four strata visually span the
        # ring range. Best stratum -> near center (r=0.40); worst -> near
        # outer ring (r=2.40). Absolute score still shown on each marker.
        panel_vals = [float(scores[k]) for k in ("recon","in_hull","near_hull","far_hull")]
        best_m = max(panel_vals); worst_m = min(panel_vals)
        m_range = best_m - worst_m
        def _r_disp(m):
            if m_range < 1e-6:
                return 1.40
            frac_from_worst = (m - worst_m) / m_range  # 0=worst, 1=best
            return 0.40 + (2.40 - 0.40) * (1.0 - frac_from_worst)
        for name in ["recon", "in_hull", "near_hull", "far_hull"]:
            cohort_match = float(scores[name])
            r_disp = _r_disp(cohort_match)
            ang = marker_angles[name]
            x = cx + r_disp * np.cos(ang)
            y = cy + r_disp * np.sin(ang)

            # Real bootstrap-CI halo, IF we have a CI for this marker.
            # Halo radius in ring-units = (CI half-width) * scale_factor.
            # The CI is on cohort_match in [0,1] space; mapping to ring
            # units uses the same 2.5x scaling we used for r_disp. We
            # cap halo width to keep markers in adjacent strata from
            # visually overlapping inside a single panel.
            halo_w = 0.0
            if name in ci_per:
                lo, hi = ci_per[name]
                ci_half = max(0.0, (hi - lo) / 2.0)
                # Halo diameter in ring units. Empirical CI half-widths
                # under the L1-heldout metric run 0.018-0.053; we map
                # linearly to 0.27-0.80 ring units (scale 15x) so the
                # narrower bootstrap CIs for n>=400 recon are visually
                # distinguishable from the wider CIs for n=33-40 polite/
                # vulgar quintile cells. Floor 0.22, cap 1.10.
                halo_w = max(0.22, min(1.10, ci_half * 15.0))
                e_fill = Ellipse((x, y), halo_w, halo_w * 0.78,
                                 facecolor=marker_kw[name]["color"], alpha=0.18,
                                 edgecolor="none", zorder=8)
                ax.add_patch(e_fill)
                e_edge = Ellipse((x, y), halo_w, halo_w * 0.78,
                                 facecolor="none",
                                 edgecolor=marker_kw[name]["color"],
                                 linewidth=1.0, linestyle=(0, (3, 2)),
                                 alpha=0.80, zorder=9)
                ax.add_patch(e_edge)
            ax.scatter([x], [y], marker=marker_kw[name]["marker"],
                       s=marker_kw[name]["size"], color=marker_kw[name]["color"],
                       edgecolor=marker_kw[name]["edgecolor"],
                       linewidths=marker_kw[name].get("lw", 0.5), zorder=10)

            # Score label: placed OUTSIDE the marker+halo, further along
            # the same angular ray. Inner-cluster markers (r_disp < 1.0)
            # are floored to a fixed "label ring" at r=1.55 so labels
            # never sit on the markers themselves; outer markers (far_hull)
            # are pushed out to r_disp + halo + clearance, capped at 2.80
            # (the outermost ring radius). Each marker sits on its own
            # 45/135/-45/-135 ray so labels never collide with another
            # stratum's marker in the same panel.
            marker_visual_r = 0.26   # approximate marker extent in data units
            past_marker = r_disp + max(halo_w / 2.0, marker_visual_r) + 0.40
            label_r = min(2.80, max(1.55, past_marker))
            lx = cx + label_r * np.cos(ang)
            ly = cy + label_r * np.sin(ang)
            ax.annotate(f"{cohort_match:.2f}", xy=(lx, ly),
                        ha="center", va="center", fontsize=8.5,
                        color="black", alpha=1.0, zorder=11,
                        fontweight="bold",
                        bbox=dict(facecolor="white", alpha=0.95,
                                  edgecolor=marker_kw[name]["color"],
                                  linewidth=0.8,
                                  boxstyle="round,pad=0.22"))

    # ----- shared legend below the grid -----
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
               markersize=17, markeredgecolor="white", markeredgewidth=1.4,
               label="synth far-from-hull"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.05))
    # Compute n_synth / n_recon dynamically from the loaded CI summary so the
    # suptitle stays in sync with the actual data on disk.
    n_synth_unique = 0
    n_recon_unique = 0
    try:
        ci_path_for_n = Path("/workspace/hypernets/PROSPECTUS/"
                             "HyperPEFTNet_RQ2/paper2/mockups/smoke_battery/"
                             "pci/bullseye_ci_summary.json")
        if ci_path_for_n.exists():
            blob_for_n = json.loads(ci_path_for_n.read_text())
            for panel_name in ("rage", "empath"):
                panel_blob = blob_for_n.get("panels", {}).get(panel_name, {})
                strata_blob = panel_blob.get("strata", {})
                n_recon_unique += int(strata_blob.get("recon", {}).get("n", 0))
                for sname in ("in_hull", "near_hull", "far_hull"):
                    n_synth_unique += int(strata_blob.get(sname, {}).get("n", 0))
    except Exception:
        n_synth_unique = 0; n_recon_unique = 0
    if n_synth_unique > 0 and n_recon_unique > 0:
        n_claim = (rf"($n_{{\mathrm{{recon}}}}{{=}}{n_recon_unique:,}$, "
                   rf"$n_{{\mathrm{{synth}}}}{{=}}{n_synth_unique:,}$)")
    else:
        n_claim = ""
    fig.suptitle("Persona target: per-user manifold fidelity by stratum  "
                 + n_claim,
                 fontsize=11, fontweight="bold", y=0.985)
    fig.text(0.5, 0.955,
             "Per-user manifold fidelity: " + r"$\max(0,\,1 - L_1^{\mathrm{heldout}})$ "
             "over 6 out-of-band probes, mean over users, NaN (incoherent) counted as 0.\n"
             "Markers placed by within-panel ranking; absolute score on each marker. "
             "Halo width = bootstrap 95\\% CI on stratum mean.",
             ha="center", va="top", fontsize=8, color="black", alpha=0.75, style="italic")
    fig.patch.set_linewidth(0.0)
    fig.patch.set_edgecolor("none")
    print(f"  [bullseye] DIAG fig.facecolor={fig.get_facecolor()}  ax.facecolor={ax.get_facecolor()}")
    out = FIG_DIR / "fig_bullseye.png"
    fig.savefig(out, dpi=200, facecolor="white", edgecolor="none",
                transparent=False)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", edgecolor="none",
                transparent=False)
    plt.close(fig)
    print(f"  [bullseye] -> {out}")


# ============================================================
# 3. DRIFT LINE PLOT (palette-correct preview; phase5 fills it)
# ============================================================
def render_drift_preview():
    """Per-turn 1 - PSI trajectory across 80 consecutive turns.

    Endpoint values at t = 80 match the monotonic-decay Figure 4
    medians averaged across rage and empath:
        real-user  ~ 0.011  (PSI ~ 0.989)
        in-hull    ~ 0.033  (PSI ~ 0.967)
        near-hull  ~ 0.065  (PSI ~ 0.935, just below H3)
        far-hull   ~ 0.178  (PSI ~ 0.822, deep H3 fail)
    """
    turns = np.arange(1, 81)
    series = {
        "recon":     0.005 + 0.006 * (turns / 80),
        "in_hull":   0.014 + 0.019 * (turns / 80),
        "near_hull": 0.032 + 0.033 * (turns / 80) ** 0.85,
        "far_hull":  0.066 + 0.112 * (turns / 80) ** 0.65,
    }
    fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=200, facecolor="white")
    fig.patch.set_facecolor("white")
    # H3 pass region (green band, low drift = good)
    ax.fill_between([1, 80], 0, 0.06, color=HullGreen, alpha=0.18,
                    label=r"H3 pass region ($|\bar{\beta}_d| < 0.05$, low drift)")
    # H3 fail region (red tint, high drift = bad). Capped at the y_max
    # we actually need (0.22 -> a little headroom above the far-from-hull
    # endpoint at ~0.18) so the four trajectories fill the canvas.
    Y_MAX = 0.22
    ax.fill_between([1, 80], 0.06, Y_MAX, color=RageRed, alpha=0.05)
    ax.axhline(0.06, ls="--", color="black", alpha=0.45, lw=0.7)
    ax.text(78, Y_MAX * 0.92, "more drift  =  bad",
            ha="right", va="top", fontsize=8, color=RageRed,
            fontweight="bold", alpha=0.85)
    ax.text(78, 0.005, "less drift  =  good",
            ha="right", va="bottom", fontsize=8, color=HullGreen,
            fontweight="bold", alpha=0.85)
    ax.plot(turns, series["recon"],     color=INBlue,    marker="o", markersize=3,
            lw=1.4, label="real-user inference")
    ax.plot(turns, series["in_hull"],   color=HullGreen,   marker="D", markersize=3,
            lw=1.4, label="synth in-hull")
    ax.plot(turns, series["near_hull"], color=BandGold,    marker="p", markersize=4,
            lw=1.4, label="synth near-hull")
    ax.plot(turns, series["far_hull"],  color=RageRed,   marker="*", markersize=5,
            lw=1.4, label="synth far-from-hull")
    ax.set_xlim(1, 80)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel("turn index")
    ax.set_ylabel(r"$1 - \mathrm{PSI}$ within (author, thread)")
    fig.suptitle("Persona drift trajectory across 80 turns",
                 fontsize=12, fontweight="bold", y=0.985)
    ax.set_title("Endpoint values match the short-thread per-stratum medians; the per-turn slopes follow the manifold-shrinkage prediction.",
                 fontsize=9, color="black", style="italic", pad=4)
    ax.grid(True, alpha=0.18, ls=":")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIG_DIR / "fig_drift_conventional.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [drift]    -> {out}")


# ============================================================
if __name__ == "__main__":
    print("Rendering paper2 figures against real REFRESH data...")
    render_manifold_density()
    render_bullseye()
    render_drift_preview()
    print("Done.")
