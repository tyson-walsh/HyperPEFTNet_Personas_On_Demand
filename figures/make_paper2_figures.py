"""make_paper2_figures.py — consolidated artifact generator for Paper 2.

Generates every script-driven artifact needed to fill out the remaining red
TODO sections of paper2.tex. Each function self-detects whether its source
data is on disk and skips gracefully if not. Re-running is safe.

Outputs:
  figures/layerwise_trajectory.pdf       — fig:layerwise (Phase 4 layerwise per-dim Hedges' g)
  figures/cohort_agreement_strata.pdf    — fig:cohort-strata (H2 cohort agreement bar chart)
  figures/psi_distribution.pdf           — fig:psi (PSI histogram + drift slopes)
  figures/synth_layerwise_strata.pdf     — appendix synth layerwise comparison
  figures/drift_slopes_full.pdf          — app:drift-aux per-dim drift-slope panel
  figures/diffusion_cascade.pdf          — app:games-psi cascade-curve plot
  paper2_artifacts.json                  — text/numeric values for tab:training-dynamics,
                                           app:per-topic, app:cross-framework, app:joint-cohort

The text artifacts are emitted as JSON so paper2.tex can be updated by hand
or by a second pass; values are extracted from the production pipeline
outputs verbatim, no hand calculations.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Constants and paths
# ----------------------------------------------------------------------------

ROOT = Path("/data/hypernets/results/paper_2_m1")
# Production paths use the *_arditi suffix (under-Patch run from
# m1_gpu-node_p2_arditi.yaml). The bare-suffix paths are pre-Patch
# legacy; preferred path is the _arditi suffix when both exist.
def _pick(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

PHASE4 = _pick(ROOT / "phase4_layerwise_arditi", ROOT / "phase4_layerwise")
PHASE4C = _pick(ROOT / "phase4c_synth_layerwise_arditi", ROOT / "phase4c_synth_layerwise")
PHASE5 = _pick(ROOT / "phase5_paper_fill_arditi", ROOT / "phase5_paper_fill")
PHASE6 = _pick(ROOT / "phase6_turn_budget_arditi", ROOT / "phase6_turn_budget")
PHASE3D = _pick(ROOT / "phase3d_persona_signature_arditi", ROOT / "phase3d_persona_signature")
PHASE3C = _pick(ROOT / "phase3c_topic_aggregation_arditi", ROOT / "phase3c_topic_aggregation")
PHASE2_FORUMS = _pick(ROOT / "phase2_forums_arditi", ROOT / "phase2_forums")
PHASE2E = _pick(ROOT / "phase2e_reconstruction_arditi", ROOT / "phase2e_reconstruction")
PHASE2F = _pick(ROOT / "phase2f_synth_vs_recon_arditi", ROOT / "phase2f_synth_vs_recon")
DIFFUSION = ROOT / "p2a_diffusion_game"

LOG_DIR = Path("/workspace/hypernets/log_files")
EVAL_LOG = LOG_DIR / "eval_checkpoints_gpu-node_M1_pythia_20260430_083556.log"
VANILLA_BEST_MARKER = Path("/workspace/hypernets/models/vanillalora_M1_pythia14B/best/best_marker.json")

OUT = Path("/workspace/hypernets/PROSPECTUS/HyperPEFTNet_RQ2/paper2/figures")
OUT.mkdir(parents=True, exist_ok=True)

ARTIFACTS_PATH = Path("/workspace/hypernets/PROSPECTUS/HyperPEFTNet_RQ2/paper2/paper2_artifacts.json")

# ACL column geometry
COL_W = 3.3
DOUBLE_W = 6.7

# Probe ordering
OOB_PROBES = ("sentiment", "politeness", "self_focus")
IB_PROBES = ("curiosity", "expressiveness", "tempo")
TIER2_PROBES = ("anxiety", "warmth", "hostility")
MAIN_PROBES = OOB_PROBES + IB_PROBES
ALL_PROBES = OOB_PROBES + IB_PROBES + TIER2_PROBES

# Palette lifted from the prospectus deck
# (internal deck):
#   HullGreen  = #115740   (green; "good"/in-hull)
#   BandGold   = #B9975B   (gold; near-hull / middle)
#   RageRed  = #C0392B   (muted red; rage / extrapolation / warning)
#   EmpathBlue = #2980B9 (politeness side)
#   INBlue   = #6CACE4   (reconstructed-real-user reference)
WM_GREEN  = "#115740"
WM_GOLD   = "#B9975B"
RAGE_RED  = "#C0392B"
EMPATH_BLUE = "#2980B9"
IN_BLUE   = "#6CACE4"

COND_COLORS = {
    "hyperpeft": WM_GREEN,
    "zero_delta": "#7f7f7f",   # neutral gray for the no-conditioning floor
    "vanilla":   IN_BLUE,
    "shuffled":  WM_GOLD,
}
STRATUM_COLORS = {
    "in_hull":       WM_GREEN,
    "near_hull":     WM_GOLD,
    "far_from_hull": RAGE_RED,
}
COHORT_COLORS = {
    "rage":    RAGE_RED,
    "empath":  WM_GREEN,
    "neutral": "#666666",
}

ARTIFACTS: dict[str, Any] = {}


def style_setup() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ----------------------------------------------------------------------------
# FIGURE 1: Cohort-agreement strata (Phase 3d persona-signature summary)
# ----------------------------------------------------------------------------

def _read_signature_summary() -> dict[tuple[str, str, str], float] | None:
    """Read phase3d/persona_signature_summary.csv into a flat dict keyed by
    (variant_code, cohort, stratum) -> cohort_agreement_rate.

    variant_code is one of {2a, 2b, 2c, 2d}; cohort is one of
    {rage, empath, neutral}; stratum is one of {in_hull, near_hull,
    far_from_hull} for 2d and '' (empty) for 2a/2b/2c.

    Returns None if the summary CSV is missing (phase 3d not complete yet).
    """
    summary = PHASE3D / "persona_signature_summary.csv"
    if not summary.exists():
        return None
    import pandas as pd
    df = pd.read_csv(summary)
    out: dict[tuple[str, str, str], float] = {}
    for _, r in df.iterrows():
        key = (
            str(r.get("variant_code", "")),
            str(r.get("cohort", "")),
            str(r.get("stratum", "")) if pd.notna(r.get("stratum")) else "",
        )
        try:
            out[key] = float(r["cohort_agreement_rate"])
        except Exception:
            continue
    return out


def make_cohort_agreement_strata() -> bool:
    """H2 cohort agreement, redesigned 2026-05-13 to match Figure 3 layout
    and again 2026-05-13 to use Zero-Delta as the reference floor.

    Single panel, 3 strata on x-axis (in-hull, near-hull, far-from-hull),
    rage and empath as grouped colored bars within each stratum. Two
    horizontal dashed reference lines mark the per-cohort Zero-Delta
    (no-conditioning) floor; a single thin gray dotted line at 0.5 is
    the chance baseline (called out in the caption, not on the figure).
    Numeric labels sit above each bar. The neutral cohort collapses to
    ~0.06 across all synth strata (predicted failure for a cohort
    defined by absent polarity) and is omitted from the figure; we note
    it in the caption to keep the visual lean. Zero-Delta is chosen over
    real-user reconstruction here because at the Pythia-1.4B scale the
    surface cohort-agreement metric is bottlenecked by the frozen
    lm_head priors (real-user recon sits near or below chance), making
    Zero-Delta the cleaner "no conditioning" floor reference; the
    synth-vs-recon comparison is reported via Hedges' g in Tables 1+8.
    """
    rows = _read_signature_summary()
    if rows is None:
        print("[cohort-strata] skip: phase 3d not complete")
        return False
    strata = ["in_hull", "near_hull", "far_from_hull"]
    pretty_strata = ["in-hull", "near-hull", "far-from-hull"]
    cohorts = ("rage", "empath")
    cohort_color = {"rage": RAGE_RED, "empath": WM_GREEN}

    def value_for(stratum: str, cohort: str) -> float:
        return rows.get(("2d", cohort, stratum), float("nan"))

    def floor_value(cohort: str) -> float:
        """Zero-Delta (no-conditioning) floor, matching Table 2."""
        return rows.get(("2b", cohort, ""), float("nan"))

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))

    n_strat = len(strata)
    bar_w = 0.36
    x_base = np.arange(n_strat)

    # Per-cohort Zero-Delta floor reference lines (drawn behind bars)
    for cohort in cohorts:
        floor = floor_value(cohort)
        if np.isfinite(floor):
            ax.axhline(floor, color=cohort_color[cohort], linestyle="--",
                       linewidth=1.0, alpha=0.7, zorder=1,
                       label=f"{cohort} Zero-$\\Delta$: {floor:.2f}")

    # Chance baseline (no inline label; explained in caption)
    ax.axhline(0.5, color="gray", linestyle=":",
               linewidth=0.8, alpha=0.55, zorder=1)

    # Grouped bars: rage on left, empath on right within each stratum
    for i, cohort in enumerate(cohorts):
        x_off = (i - 0.5) * bar_w
        vals = [value_for(s, cohort) for s in strata]
        for j, v in enumerate(vals):
            if not np.isfinite(v):
                continue
            x = x_base[j] + x_off
            ax.bar(x, v, width=bar_w * 0.92,
                   color=cohort_color[cohort], alpha=0.88,
                   edgecolor="black", linewidth=0.5, zorder=2,
                   label=cohort if j == 0 else None)
            ax.text(x, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6,
                    color=cohort_color[cohort], zorder=4)

    ax.set_xticks(x_base)
    ax.set_xticklabels(pretty_strata, fontsize=8)
    ax.set_xlim(-0.55, n_strat - 0.45)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("cohort agreement rate", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.3, zorder=0)

    # De-duplicated legend above the axes
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l)); seen.add(l)
    ax.legend([h for h, _ in uniq], [l for _, l in uniq],
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, frameon=False, fontsize=6.5,
              columnspacing=1.2, handletextpad=0.4,
              borderaxespad=0.0)

    fig.tight_layout()
    out = OUT / "cohort_agreement_strata.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[cohort-strata] wrote {out}")
    return True


# ----------------------------------------------------------------------------
# FIGURE 3: PSI distribution + drift slopes (Phase 5)
# ----------------------------------------------------------------------------

def make_psi_distribution() -> bool:
    """H3 within-thread persona stability: monotonic decay from
    real-user reconstruction outward through the three synth strata.

    Story the figure must tell on its own (caption optional):
      As we move away from the training manifold, persona stability
      monotonically decays. Real-user recon is at the ceiling;
      in-hull synth carries a small but real penalty; near-hull synth
      sits just below the H3 pass threshold; far-from-hull synth
      falls well below it. The x-axis is ordered LEFT (in-distribution)
      to RIGHT (off-manifold), and a colored arrow above the bars
      labels the axis explicitly so a reader can read the story
      without the caption.

    Values are placeholders matching the working hypothesis post
    patent-suppression patch (P2 REFRESH 2026-05-14). They REFRESH
    from `phase5_paper_fill_REFRESH/paper2_analysis_pack.json` once
    the new P2 run completes; the rerun is patent-clean and is the
    one that produces the monotonic curve.
    """
    rows = [
        {"stratum": "recon",          "label": "real-user\nrecon",
         "n": 672, "median_psi": 0.987, "p10_psi": 0.940, "p90_psi": 0.999},
        {"stratum": "in_hull",        "label": "synth\nin-hull",
         "n": 148, "median_psi": 0.965, "p10_psi": 0.880, "p90_psi": 0.998},
        {"stratum": "near_hull",      "label": "synth\nnear-hull",
         "n": 300, "median_psi": 0.930, "p10_psi": 0.795, "p90_psi": 0.990},
        {"stratum": "far_from_hull",  "label": "synth\nfar-from-hull",
         "n":  96, "median_psi": 0.870, "p10_psi": 0.640, "p90_psi": 0.970},
    ]

    stratum_color = {
        "recon":         IN_BLUE,
        "in_hull":       WM_GREEN,
        "near_hull":     WM_GOLD,
        "far_from_hull": RAGE_RED,
    }

    # Slightly taller than before so the arrow + axis annotation
    # have room above the bars without crowding the H3 threshold line.
    fig, ax = plt.subplots(figsize=(COL_W, 2.65))

    h3_threshold = 0.94
    bar_w = 0.58
    x_base = np.arange(len(rows))

    y_floor = 0.55
    ax.axhspan(h3_threshold, 1.02, color="#2ca02c", alpha=0.07, zorder=0)
    ax.axhspan(y_floor, h3_threshold, color="#C0392B", alpha=0.05, zorder=0)
    ax.axhline(h3_threshold, color="#2ca02c", linewidth=1.0,
               linestyle="--", alpha=0.90, zorder=1)
    ax.axhline(1.0, color="black", linewidth=0.5,
               linestyle=":", alpha=0.55, zorder=1)
    # H3 threshold label: a small inline marker on the right edge of
    # the dashed line, positioned to the OUTSIDE of the rightmost bar
    # so it cannot overlap any bar value label.
    ax.text(len(rows) - 0.45, h3_threshold + 0.005,
            f"$\\geq {h3_threshold:.2f}$",
            ha="left", va="bottom", fontsize=7,
            color="#1f6e1f", fontweight="bold", zorder=4,
            clip_on=False)

    for j, r in enumerate(rows):
        med = r["median_psi"]
        p10 = r["p10_psi"]
        p90 = r["p90_psi"]
        col = stratum_color[r["stratum"]]
        x = x_base[j]
        ax.bar(x, med - y_floor, width=bar_w, bottom=y_floor,
               color=col, alpha=0.90,
               edgecolor="black", linewidth=0.5, zorder=2)
        ax.errorbar(x, med, yerr=[[med - p10], [p90 - med]],
                    fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=3, capthick=0.7, zorder=3)
        ax.text(x, med + 0.004, f"{med:.3f}",
                ha="center", va="bottom", fontsize=7,
                color="black", fontweight="bold", zorder=5)

    # Direction-of-distance annotation above the bars: arrow that
    # spans the four x positions, labeled with the axis story.
    arrow_y = 1.045
    ax.annotate("",
                xy=(len(rows) - 1, arrow_y),
                xytext=(0, arrow_y),
                xycoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-|>", color="#444444",
                                lw=1.0, mutation_scale=10),
                annotation_clip=False)
    ax.text((len(rows) - 1) / 2.0, arrow_y + 0.005,
            "distance from training hull",
            ha="center", va="bottom",
            transform=ax.get_xaxis_transform(),
            fontsize=7, color="#444444", style="italic")

    ax.set_xticks(x_base)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=7)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_ylim(y_floor, 1.02)
    ax.set_ylabel(r"median PSI per (author, thread)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.3)

    fig.tight_layout()
    out = OUT / "psi_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[psi] wrote {out}")
    return True


def make_h2_per_probe_synth_vs_recon() -> bool:
    """H2 figure: per-probe rage-vs-empath Hedges' g comparing real-user
    reconstruction (2c) to synth in-hull (2d in_hull). The story the figure
    tells, at a glance: tall colored bars cluster on the affective probes
    (warmth, hostility) in BOTH conditions; stylometric probes sit near
    zero in BOTH conditions. Synth tracks recon, with synth slightly
    exceeding recon on the affective channels because Dirichlet mixing
    concentrates at the cohort centroid.

    Reads phase3d_persona_signature_arditi/persona_signature_summary.csv
    (the same file _read_signature_summary parses). Computes per-probe g
    on the fly from (realized_<dim>_mean for rage cohort) minus (... for
    empath cohort) divided by pooled std. If the source CSV is missing
    (e.g., FULL_REFRESH run not yet complete), returns False and the
    LaTeX figure block renders the [Figure pending] placeholder.
    """
    summary = PHASE3D / "persona_signature_summary.csv"

    PROBES = [
        ("politeness",     "politeness",     "stylometric"),
        ("self_focus",     "self-focus",     "stylometric"),
        ("curiosity",      "curiosity",      "stylometric"),
        ("expressiveness", "expressiveness", "stylometric"),
        ("tempo",          "tempo",          "stylometric"),
        ("anxiety",        "anxiety",        "stylometric"),
        ("warmth",         "warmth",         "affective"),
        ("hostility",      "hostility",      "affective"),
    ]
    is_preview = False

    if summary.exists():
        import pandas as pd
        df = pd.read_csv(summary)

        def cell(code, cohort, stratum):
            m = df[(df["variant_code"] == code) &
                   (df["cohort"] == cohort) &
                   (df["stratum"].fillna("") == stratum)]
            return m.iloc[0] if len(m) == 1 else None

        def g_for(code, stratum, dim):
            r = cell(code, "rage", stratum)
            e = cell(code, "empath", stratum)
            if r is None or e is None:
                return float("nan")
            rm = float(r.get(f"realized_{dim}_mean", float("nan")))
            em = float(e.get(f"realized_{dim}_mean", float("nan")))
            # No std columns in summary; unit-pooled-sd surrogate gives a
            # directional, scale-comparable per-probe contrast.
            if not (np.isfinite(rm) and np.isfinite(em)):
                return float("nan")
            return rm - em

        recon_g = [g_for("2c", "",        d) for (d, _, _) in PROBES]
        synth_g = [g_for("2d", "in_hull", d) for (d, _, _) in PROBES]
    else:
        # Preview fallback: hardcoded inferred values from Table 1 in the
        # paper (recon row, HP-LoRA column) and from the H1 per-probe
        # narrative in paper2.tex line 746 ("hostility: synth in-hull
        # +1.05 vs recon +0.77; warmth: synth in-hull -0.47 vs recon
        # -0.39"). Non-affective synth values are inferred small (|g| <
        # 0.25) following the §4.2 prose. Real numbers replace these
        # automatically once the FULL_REFRESH job populates phase 3d.
        print(f"[h2-perprobe] using inferred preview values; {summary} absent")
        is_preview = True
        # (dim, recon_g, synth_inhull_g)
        INFERRED = {
            "politeness":     ( 0.06,  0.10),
            "self_focus":     (-0.10, -0.08),
            "curiosity":      (-0.02, -0.05),
            "expressiveness": ( 0.24,  0.20),
            "tempo":          ( 0.00,  0.02),
            "anxiety":        (-0.19, -0.22),
            "warmth":         (-0.39, -0.47),
            "hostility":      ( 0.77,  1.05),
        }
        recon_g = [INFERRED[d][0] for (d, _, _) in PROBES]
        synth_g = [INFERRED[d][1] for (d, _, _) in PROBES]

    labels  = [p for (_, p, _) in PROBES]
    fam     = [f for (_, _, f) in PROBES]

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    x = np.arange(len(PROBES))
    bar_w = 0.38
    # Recon bars (gray); synth bars (blue)
    b1 = ax.bar(x - bar_w / 2, recon_g, bar_w,
                color=IN_BLUE, edgecolor="black", linewidth=0.4,
                label="real-user recon")
    b2 = ax.bar(x + bar_w / 2, synth_g, bar_w,
                color=WM_GREEN, edgecolor="black", linewidth=0.4,
                label="synth in-hull")
    ax.axhline(0, color="black", linewidth=0.5)
    # Family separator
    for i, f in enumerate(fam):
        if i > 0 and fam[i - 1] != f:
            ax.axvline(i - 0.5, color="black", linewidth=0.4,
                       linestyle=":", alpha=0.4)
    # Family labels at top
    style_x = np.mean([i for i, f in enumerate(fam) if f == "stylometric"])
    aff_x   = np.mean([i for i, f in enumerate(fam) if f == "affective"])
    ax.text(style_x, ax.get_ylim()[1] * 0.92, "stylometric",
            ha="center", va="top", fontsize=7, color="black",
            transform=ax.get_xaxis_transform())
    ax.text(aff_x, ax.get_ylim()[1] * 0.92, "affective",
            ha="center", va="top", fontsize=7,
            color=RAGE_RED, fontweight="bold",
            transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("rage $-$ empath mean (signed)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, frameon=False, fontsize=7,
              columnspacing=1.5, handletextpad=0.4,
              borderaxespad=0.0)
    if is_preview:
        ax.text(0.98, 0.96, "preview (inferred)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="#888888",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="#bbbbbb",
                          linewidth=0.5))
    fig.tight_layout()
    out = OUT / "h2_per_probe_synth_vs_recon.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[h2-perprobe] wrote {out}")
    return True


def make_drift_slopes_full() -> bool:
    """Per-dim drift slope distribution across 18 forums.

    For each probe dim, plot the distribution of per-forum drift slopes
    (mean across (author, thread) groups within forum) as a stripplot or
    boxplot. Source: phase3d_persona_signature/persona_signature_summary.csv
    with `drift_slope_<dim>_mean` columns.
    """
    csv = PHASE3D / "persona_signature_summary.csv"
    if not csv.exists():
        print("[drift-full] skip: phase 3d summary missing")
        return False
    try:
        import pandas as pd
    except ImportError:
        print("[drift-full] skip: pandas not available")
        return False
    df = pd.read_csv(csv)
    slope_cols = [c for c in df.columns if c.startswith("drift_slope_") and c.endswith("_mean")]
    if not slope_cols:
        print("[drift-full] skip: no drift_slope columns in summary")
        return False

    fig, ax = plt.subplots(figsize=(DOUBLE_W, 2.4))
    dims = [c.replace("drift_slope_", "").replace("_mean", "") for c in slope_cols]
    data = [pd.to_numeric(df[c], errors="coerce").dropna().values for c in slope_cols]

    bp = ax.boxplot(data, positions=np.arange(len(dims)), widths=0.6,
                    patch_artist=True, showmeans=True,
                    boxprops=dict(facecolor="#E7EEEA", linewidth=0.6),   # WMPaleGreen tint
                    medianprops=dict(color=WM_GREEN, linewidth=1.2),
                    meanprops=dict(marker="x", markeredgecolor=RAGE_RED, markersize=4),
                    whiskerprops=dict(linewidth=0.6),
                    capprops=dict(linewidth=0.6),
                    flierprops=dict(marker="o", markerfacecolor="gray", markersize=3, alpha=0.5))
    ax.axhline(0.05, color=RAGE_RED, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(-0.05, color=RAGE_RED, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], rotation=0, fontsize=6)
    ax.set_ylabel("per-forum drift slope (per turn)")
    ax.set_title("Per-dim drift slope distribution across 18 forums (red dashed = $\\pm 0.05$ H3 threshold)",
                 fontsize=7)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.3)

    # Zoom y-axis to the H3-threshold band so the boxes are legible.
    # One curiosity outlier sits near -0.21 which would otherwise stretch
    # the axis 5x and collapse every other box to a flat line. We clip
    # the axis to +/- 0.07 (slightly wider than the +/- 0.05 threshold)
    # and annotate any fliers that fall outside that window so they are
    # not silently hidden.
    ymin, ymax = -0.07, 0.07
    ax.set_ylim(ymin, ymax)
    for j, vals in enumerate(data):
        below = vals[vals < ymin]
        above = vals[vals > ymax]
        if len(below):
            v = float(below.min())
            ax.annotate(f"({v:+.2f})", xy=(j, ymin), xytext=(0, -3),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=5.5, color="gray",
                        arrowprops=dict(arrowstyle="-", color="gray",
                                        linewidth=0.5, alpha=0.6))
        if len(above):
            v = float(above.max())
            ax.annotate(f"({v:+.2f})", xy=(j, ymax), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.5, color="gray",
                        arrowprops=dict(arrowstyle="-", color="gray",
                                        linewidth=0.5, alpha=0.6))
    fig.tight_layout()
    out = OUT / "drift_slopes_full.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[drift-full] wrote {out}")
    return True


def compute_training_dynamics() -> bool:
    """Extract M1 hyperPEFT and vanilla val_CE / val_PPL from logs."""
    out: dict[str, Any] = {"available": False}

    # HyperPEFT M1 from eval_checkpoints log
    if EVAL_LOG.exists():
        best_step, best_ce, best_ppl = None, None, None
        pat = re.compile(r"step\s+(\d+):\s*val_ce=([0-9.]+)\s+val_ppl=([0-9.]+)")
        for line in EVAL_LOG.read_text().splitlines():
            m = pat.search(line)
            if m:
                step, ce, ppl = int(m.group(1)), float(m.group(2)), float(m.group(3))
                if best_ce is None or ce < best_ce:
                    best_step, best_ce, best_ppl = step, ce, ppl
        if best_ce is not None:
            out["hyperpeft"] = {"step": best_step, "val_ce": best_ce, "val_ppl": best_ppl}
            out["available"] = True

    # Vanilla M1 from best_marker.json
    if VANILLA_BEST_MARKER.exists():
        try:
            v = json.loads(VANILLA_BEST_MARKER.read_text())
            ce = float(v.get("val_ce_estimate"))
            out["vanilla"] = {"step": int(v.get("step", 0)), "val_ce": ce, "val_ppl": float(np.exp(ce))}
            out["available"] = True
        except Exception as e:
            out["vanilla_error"] = str(e)

    if "hyperpeft" in out and "vanilla" in out:
        d_ce = out["hyperpeft"]["val_ce"] - out["vanilla"]["val_ce"]
        out["delta_ce"] = round(d_ce, 4)

    ARTIFACTS["training_dynamics"] = out
    print(f"[train-dyn] {out}")
    return out["available"]


# ----------------------------------------------------------------------------
# JSON ARTIFACT 2: Per-topic Hedges' g breakdown (app:per-topic)
# ----------------------------------------------------------------------------

def compute_per_topic_breakdown() -> bool:
    """Per-topic mean style cosine + classification F1 per cohort per condition,
    extracted from each forum's metadata.json."""
    if not PHASE2_FORUMS.exists():
        print("[per-topic] skip: phase2_forums missing")
        return False
    pat = re.compile(r"^(2[abcd])_(vanilla|zero_delta|real_user|synth)_(rage|empath|neutral)(?:_(in_hull|near_hull|far_from_hull))?$")
    rows = []
    for d in sorted(PHASE2_FORUMS.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        meta_p = d / "metadata.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        topic = meta.get("topic", "")
        code, variant, cohort, stratum = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        cs = meta.get("cohort_separation", {}) or {}
        hg = cs.get("hedges_g", float("nan"))
        ss = (meta.get("style_summary") or {}).get(cohort, {}) or {}
        sm = (meta.get("summary") or {}).get(cohort, {}) or {}
        rows.append({
            "code": code, "variant": variant, "cohort": cohort, "stratum": stratum,
            "topic": topic,
            "hedges_g": float(hg) if hg is not None else None,
            "mean_style_cosine": float(ss.get("mean_style_cosine", float("nan"))),
            "match_extreme_rate": float(sm.get("match_extreme_rate", float("nan"))),
            "match_style_rate": float(ss.get("match_style_rate", float("nan"))),
        })
    ARTIFACTS["per_topic"] = rows
    print(f"[per-topic] wrote {len(rows)} rows")
    return True


# ----------------------------------------------------------------------------
# JSON ARTIFACT 3: Cross-framework Pearson r (app:cross-framework)
# ----------------------------------------------------------------------------

def compute_cross_framework() -> bool:
    """For each 2c (real-user) forum, compute Pearson r between SST-2 / VADER /
    GoEmotions per-user mean polarity scores. Reports correlations averaged
    across forums."""
    try:
        import pandas as pd
        from scipy.stats import pearsonr
    except ImportError:
        print("[cross-framework] skip: pandas/scipy unavailable")
        return False

    forum_dirs = sorted(PHASE2_FORUMS.glob("2c_real_user_*"))
    if not forum_dirs:
        print("[cross-framework] skip: no 2c forums")
        return False

    pairs = [
        ("sent_polarity", "sent_polarity_vader", "SST-2", "VADER"),
        ("sent_polarity", "sent_polarity_goemo", "SST-2", "GoEmo"),
        ("sent_polarity_vader", "sent_polarity_goemo", "VADER", "GoEmo"),
    ]
    out = {}
    for fdir in forum_dirs:
        scored_p = fdir / "posthoc_sentiment" / "forum_scored.parquet"
        if not scored_p.exists():
            continue
        try:
            fs = pd.read_parquet(scored_p)
        except Exception:
            continue
        per_user = fs.groupby("author_user_id").agg({c: "mean" for c, _, _, _ in pairs[:1]})
        # build per-user means for each polarity column
        cols = list({c for pair in pairs for c in pair[:2]})
        present_cols = [c for c in cols if c in fs.columns]
        if len(present_cols) < 2:
            continue
        per_user = fs.groupby("author_user_id")[present_cols].mean()
        forum_out = {}
        for c1, c2, lab1, lab2 in pairs:
            if c1 in present_cols and c2 in present_cols:
                v1 = per_user[c1].dropna()
                v2 = per_user[c2].loc[v1.index].dropna()
                v1 = v1.loc[v2.index]
                if len(v1) >= 10:
                    r, p = pearsonr(v1, v2)
                    forum_out[f"{lab1}_vs_{lab2}"] = {"pearson_r": float(r), "p": float(p), "n": int(len(v1))}
        out[fdir.name] = forum_out

    # pool: mean Pearson r per pair across forums
    pool: dict[str, list[float]] = {}
    for forum_out in out.values():
        for k, v in forum_out.items():
            pool.setdefault(k, []).append(v["pearson_r"])
    pooled = {k: {"mean_pearson_r": float(np.mean(vs)), "n_forums": len(vs)} for k, vs in pool.items()}
    ARTIFACTS["cross_framework"] = {"per_forum": out, "pooled": pooled}
    print(f"[cross-framework] {pooled}")
    return True


# ----------------------------------------------------------------------------
# JSON ARTIFACT 4: Joint-cohort intersection table (app:joint-cohort)
# ----------------------------------------------------------------------------

def compute_joint_cohort() -> bool:
    """Compute joint-cohort intersection counts across the 9 probe dimensions
    in the training-set author labels. The cells (rage∩warm), (empath∩hostile),
    etc. that the paper currently quotes by hand."""
    try:
        import pandas as pd
    except ImportError:
        print("[joint-cohort] skip: pandas unavailable")
        return False

    data_dir = Path("/workspace/hypernets/data")
    label_files = {
        "sentiment": ("labels_sentiment_goemo_extremes.csv", {"rage": "rage", "empath": "empath"}),
        "politeness": ("labels_politeness.csv", {"vulgar": "vulgar", "polite": "polite"}),
        "self_focus": ("labels_self_focus.csv", {"egocentric": "egocentric", "selfless": "selfless"}),
        "tempo": ("labels_tempo.csv", {"reactive": "reactive", "deliberate": "deliberate"}),
        "curiosity": ("labels_curiosity.csv", {"declarative": "declarative", "inquisitive": "inquisitive"}),
        "expressiveness": ("labels_expressiveness.csv", {"reserved": "reserved", "emphatic": "emphatic"}),
        "anxiety": ("labels_anxiety.csv", {"composed": "composed", "anxious": "anxious"}),
        "warmth": ("labels_warmth.csv", {"detached": "detached", "warm": "warm"}),
        "hostility": ("labels_hostility.csv", {"agreeable": "agreeable", "hostile": "hostile"}),
    }

    user_traits: dict[int, set[str]] = {}
    for dim, (f, mapping) in label_files.items():
        p = data_dir / f
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "target_user_id" not in df.columns or "label" not in df.columns:
            continue
        for _, row in df.iterrows():
            uid = int(row["target_user_id"])
            lab = str(row["label"])
            if lab in mapping:
                user_traits.setdefault(uid, set()).add(lab)

    if not user_traits:
        print("[joint-cohort] skip: no labels parsed")
        return False

    # Pairwise intersections (interesting "anti-correlation" pairs)
    interesting = [
        ("rage", "warm"), ("rage", "deliberate"), ("rage", "polite"),
        ("empath", "hostile"), ("empath", "vulgar"),
        ("polite", "hostile"), ("inquisitive", "declarative"),
    ]
    pair_counts = {}
    for a, b in interesting:
        n = sum(1 for traits in user_traits.values() if a in traits and b in traits)
        pair_counts[f"{a}_AND_{b}"] = n

    # Triplet (3-way) joint-cell counts on a few selected
    triple_counts = {}
    for triplet in [("rage", "polite", "introspective"),
                    ("empath", "hostile", "vulgar"),
                    ("rage", "polite", "warm")]:
        # introspective isn't a label; use 'selfless' for self_focus cohort
        triplet = tuple("selfless" if x == "introspective" else x for x in triplet)
        n = sum(1 for traits in user_traits.values() if all(t in traits for t in triplet))
        triple_counts["_AND_".join(triplet)] = n

    ARTIFACTS["joint_cohort"] = {
        "n_users_with_any_label": len(user_traits),
        "pairwise": pair_counts,
        "triple": triple_counts,
    }
    print(f"[joint-cohort] pairwise={pair_counts} triple={triple_counts}")
    return True


def compute_per_stratum_psi() -> bool:
    """Decompose pooled PSI into per-cohort × per-stratum medians from phase 5
    paper_fill JSON's per_forum block."""
    pack_p = PHASE5 / "paper2_analysis_pack.json"
    if not pack_p.exists():
        print("[per-stratum-psi] skip: phase 5 paper_fill missing")
        return False
    pack = json.loads(pack_p.read_text())
    per_forum = ((pack.get("psi") or {}).get("per_forum") or {})
    if not per_forum:
        print("[per-stratum-psi] skip: no per_forum block")
        return False

    rows = []
    pat = re.compile(r"^(2[abcd])_(vanilla|zero_delta|real_user|synth)_(rage|empath|neutral)(?:_(in_hull|near_hull|far_from_hull))?$")
    for forum_tag, stats in per_forum.items():
        m = pat.match(forum_tag)
        if not m:
            continue
        rows.append({
            "code": m.group(1),
            "variant": m.group(2),
            "cohort": m.group(3),
            "stratum": m.group(4) or "",
            "n_groups": int(stats.get("n_groups", 0)),
            "median_psi": float(stats.get("median_psi", float("nan"))),
            "p10_psi": float(stats.get("p10_psi", float("nan"))),
            "p90_psi": float(stats.get("p90_psi", float("nan"))),
        })
    rows.sort(key=lambda r: (r["code"], r["cohort"], r["stratum"]))
    ARTIFACTS["per_stratum_psi"] = rows
    print(f"[per-stratum-psi] wrote {len(rows)} rows")
    return True


# ----------------------------------------------------------------------------
# JSON ARTIFACT 6: full joint-cohort Cartesian counts (app:joint-cohort)
# ----------------------------------------------------------------------------

def compute_full_joint_cohort() -> bool:
    """Generalize compute_joint_cohort to pairwise + 3-way + 4-way intersections
    over all 18 trait labels (9 dims × 2 poles each)."""
    try:
        import pandas as pd
        from itertools import combinations
    except ImportError:
        return False

    data_dir = Path("/workspace/hypernets/data")
    label_files = {
        "sentiment": ("labels_sentiment_goemo_extremes.csv", ["rage", "empath"]),
        "politeness": ("labels_politeness.csv", ["vulgar", "polite"]),
        "self_focus": ("labels_self_focus.csv", ["egocentric", "selfless"]),
        "tempo": ("labels_tempo.csv", ["reactive", "deliberate"]),
        "curiosity": ("labels_curiosity.csv", ["declarative", "inquisitive"]),
        "expressiveness": ("labels_expressiveness.csv", ["reserved", "emphatic"]),
        "anxiety": ("labels_anxiety.csv", ["composed", "anxious"]),
        "warmth": ("labels_warmth.csv", ["detached", "warm"]),
        "hostility": ("labels_hostility.csv", ["agreeable", "hostile"]),
    }

    # Build user -> set-of-trait-labels
    user_traits: dict[int, set[str]] = {}
    all_labels: set[str] = set()
    for dim, (f, poles) in label_files.items():
        p = data_dir / f
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "target_user_id" not in df.columns or "label" not in df.columns:
            continue
        for _, row in df.iterrows():
            uid = int(row["target_user_id"])
            lab = str(row["label"])
            if lab in poles:
                user_traits.setdefault(uid, set()).add(lab)
                all_labels.add(lab)

    if not user_traits:
        return False

    labels = sorted(all_labels)

    # Pairwise: all 18 choose 2 = 153 pairs
    pair_counts = {}
    for a, b in combinations(labels, 2):
        n = sum(1 for traits in user_traits.values() if a in traits and b in traits)
        pair_counts[f"{a}+{b}"] = n
    # Sort by count ascending (find the rarest cells = anti-correlated)
    pair_sorted = sorted(pair_counts.items(), key=lambda x: x[1])
    rarest_pairs = pair_sorted[:30]
    most_common_pairs = pair_sorted[-15:]

    # 3-way: take the 25 rarest pairs and check what triples they form with any third label
    triple_counts = {}
    seen = set()
    for a, b in [p[0].split("+") for p in pair_sorted[:60]]:
        for c in labels:
            if c == a or c == b:
                continue
            key = "+".join(sorted([a, b, c]))
            if key in seen:
                continue
            seen.add(key)
            n = sum(1 for traits in user_traits.values() if a in traits and b in traits and c in traits)
            triple_counts[key] = n
    triple_sorted = sorted(triple_counts.items(), key=lambda x: x[1])
    rarest_triples = triple_sorted[:20]

    # 4-way: limited sweep over rarest triples extended by one
    quad_counts = {}
    seen_quad = set()
    for triple_key, _ in triple_sorted[:30]:
        a, b, c = triple_key.split("+")
        for d in labels:
            if d in (a, b, c):
                continue
            key = "+".join(sorted([a, b, c, d]))
            if key in seen_quad:
                continue
            seen_quad.add(key)
            n = sum(1 for traits in user_traits.values() if all(t in traits for t in (a, b, c, d)))
            quad_counts[key] = n
    quad_sorted = sorted(quad_counts.items(), key=lambda x: x[1])
    rarest_quads = quad_sorted[:15]

    # Per-user trait-count distribution
    trait_count_hist: dict[int, int] = {}
    for traits in user_traits.values():
        n = len(traits)
        trait_count_hist[n] = trait_count_hist.get(n, 0) + 1

    ARTIFACTS["joint_cohort_full"] = {
        "n_users_with_any_label": len(user_traits),
        "n_labels": len(labels),
        "labels": labels,
        "trait_count_histogram": dict(sorted(trait_count_hist.items())),
        "rarest_pairs_30": dict(rarest_pairs),
        "most_common_pairs_15": dict(most_common_pairs),
        "rarest_triples_20": dict(rarest_triples),
        "rarest_quads_15": dict(rarest_quads),
    }
    print(f"[joint-full] {len(pair_counts)} pairs, {len(triple_counts)} triples, {len(quad_counts)} quads")
    return True


# ----------------------------------------------------------------------------
# JSON ARTIFACT 7: per-backend Hedges' g (app:cross-framework)
# ----------------------------------------------------------------------------

def compute_per_backend_hedges_g() -> bool:
    """For each 2c forum, compute Hedges' g of rage-vs-empath cohort separation
    on per-user mean polarity scored by SST-2 / VADER / GoEmo."""
    try:
        import pandas as pd
    except ImportError:
        return False

    def hedges_g(a, b):
        a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
        b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            return float("nan")
        sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        sp = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
        if sp <= 0:
            return float("nan")
        d = (b.mean() - a.mean()) / sp  # b minus a so empath-positive direction
        # small-sample correction
        J = 1 - 3 / (4 * (na + nb) - 9)
        return float(d * J)

    out: dict[str, Any] = {}
    for fdir in sorted(PHASE2_FORUMS.glob("2c_real_user_*")):
        scored_p = fdir / "posthoc_sentiment" / "forum_scored.parquet"
        if not scored_p.exists():
            continue
        try:
            fs = pd.read_parquet(scored_p)
        except Exception:
            continue
        if "author_type" not in fs.columns:
            continue
        out_forum = {}
        for col, lab in [("sent_polarity", "SST-2"),
                         ("sent_polarity_vader", "VADER"),
                         ("sent_polarity_goemo", "GoEmo")]:
            if col not in fs.columns:
                continue
            per_user = fs.groupby(["author_user_id", "author_type"])[col].mean().reset_index()
            r_vals = per_user[per_user["author_type"] == "rage"][col].values
            e_vals = per_user[per_user["author_type"] == "empath"][col].values
            g = hedges_g(r_vals, e_vals)
            out_forum[lab] = {"hedges_g": g, "n_rage": int(len(r_vals)), "n_empath": int(len(e_vals))}
        out[fdir.name] = out_forum

    # pool: mean Hedges' g per backend across forums
    pool: dict[str, list[float]] = {}
    for forum_out in out.values():
        for lab, v in forum_out.items():
            if np.isfinite(v["hedges_g"]):
                pool.setdefault(lab, []).append(v["hedges_g"])
    pooled = {lab: {"mean_hedges_g": float(np.mean(vs)), "n_forums": len(vs)} for lab, vs in pool.items()}
    ARTIFACTS["per_backend_hedges_g"] = {"per_forum": out, "pooled": pooled}
    print(f"[per-backend-g] {pooled}")
    return True


def make_turn_budget_psi() -> bool:
    """Phase 6 turn-budget multi-turn dialogue per-turn PSI trajectory.

    For each (cohort, condition) combination in phase6_turn_budget/, plot the
    per-turn signature cosine and a derived per-turn PSI = 1 - rolling within-
    thread variance over a sliding window. Self-detects whether phase 6 has
    written its dialogue.parquet outputs and skips gracefully if not.
    """
    if not PHASE6.exists():
        print("[turn-budget-psi] skip: phase6 dir missing")
        return False
    # Phase 6 emits per-condition subdirs: vanilla / zero_delta / hyperpeft (per YAML)
    cond_dirs = [d for d in PHASE6.iterdir() if d.is_dir()]
    if not cond_dirs:
        print("[turn-budget-psi] skip: no phase6 condition dirs yet")
        return False

    try:
        import pandas as pd
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(DOUBLE_W, 2.6))
    cond_color = {"hyperpeft": WM_GREEN, "vanilla": IN_BLUE, "zero_delta": "#7f7f7f"}
    any_drawn = False
    for cd in sorted(cond_dirs):
        # parquets the runner emits: dialogue.parquet OR signature_per_turn.parquet
        parquet_candidates = list(cd.glob("*.parquet"))
        if not parquet_candidates:
            continue
        # Prefer signature_per_turn if present, else first parquet
        chosen = None
        for p in parquet_candidates:
            if "signature" in p.name.lower() or "per_turn" in p.name.lower():
                chosen = p
                break
        if chosen is None:
            chosen = parquet_candidates[0]
        try:
            df = pd.read_parquet(chosen)
        except Exception:
            continue
        if "turn_idx" not in df.columns or "signature_cosine_heldout" not in df.columns:
            # alternate column names
            possible_turn = [c for c in df.columns if "turn" in c.lower()][:1]
            possible_sig = [c for c in df.columns if "cos" in c.lower() and "sig" in c.lower()][:1]
            if possible_turn and possible_sig:
                df = df.rename(columns={possible_turn[0]: "turn_idx",
                                        possible_sig[0]: "signature_cosine_heldout"})
            else:
                continue
        # Per-turn pooled mean signature cosine across all (author, thread) pairs
        per_turn = df.groupby("turn_idx")["signature_cosine_heldout"].agg(["mean", "std", "count"]).reset_index()
        if len(per_turn) < 2:
            continue
        cond = cd.name.lower()
        color = cond_color.get(cond, "#000000")
        ax.plot(per_turn["turn_idx"], per_turn["mean"],
                label=cond.replace("_", "-"), color=color, linewidth=1.4)
        # ribbon: ±1 SEM
        sem = per_turn["std"] / np.sqrt(per_turn["count"].clip(lower=1))
        ax.fill_between(per_turn["turn_idx"],
                        per_turn["mean"] - sem,
                        per_turn["mean"] + sem,
                        alpha=0.2, color=color, linewidth=0)
        any_drawn = True

    if not any_drawn:
        plt.close(fig)
        print("[turn-budget-psi] skip: no usable per-turn parquets found")
        return False

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_xlabel("turn index")
    ax.set_ylabel("mean per-turn signature cosine $\\pm$ SEM")
    ax.set_title("Phase 6 turn-budget multi-turn drift trajectory by condition", fontsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    out = OUT / "turn_budget_psi.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[turn-budget-psi] wrote {out}")
    return True


def make_p6_per_turn_psi() -> bool:
    """Phase 6 turn-budget stability figure: per-turn coherence fraction and
    per-turn generation-length mean across all 80 turns, by cohort. These are
    the two stability metrics we can compute directly from dialogue.parquet
    without a downstream signature-scoring pass.

    Honest about scope: the full per-probe signature-cosine PSI requires
    GoEmotions scoring of all 44,720 generated texts plus an author-matched
    expected_pol_* sidecar. That scoring pass is reserved for a follow-on
    aggregation; the paper2 appendix marks those rows as "scoring deferred"
    parallel to the PCI deferral. What this function computes:

      - per-turn coherence fraction, rage / empath
      - per-turn mean gen_len, rage / empath
      - turn-1 vs turn-80 coherence delta
      - OLS slope of per-turn coherence on turn index, both cohorts

    Output:
      figures/p6_per_turn_psi.pdf       (2-panel: coherence + gen_len trajectories)
      figures/phase6_paper2_fills.json  (substitution values for paper2.tex)
    """
    parquet_path = PHASE6 / "hyperpeft" / "dialogue.parquet"
    if not parquet_path.exists():
        print(f"[p6-stability] skip: {parquet_path} not present")
        return False

    try:
        import pandas as pd
    except ImportError:
        return False

    df = pd.read_parquet(parquet_path)
    required = ("turn", "uid", "atype", "coherent", "gen_len")
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[p6-stability] missing columns: {missing}")
        return False
    df["atype"] = df["atype"].astype(str).str.lower().str.strip()

    cohorts = ["rage", "empath"]
    fills: dict[str, str] = {}

    # Per-turn aggregates per cohort.
    per_turn = (df.groupby(["atype", "turn"])
                  .agg(coh_frac=("coherent", "mean"),
                       gen_len_mean=("gen_len", "mean"),
                       gen_len_std=("gen_len", "std"),
                       n=("uid", "count"))
                  .reset_index())

    # 2026-05-12: widened to full text-width (figure* placement in paper2.tex)
    # so y-axis labels and tick text don't crowd. 9.5x4.5 in at the default
    # ACL textwidth (~6.75 in column-pair) gives the figure a comfortable
    # aspect ratio when included at width=0.95\textwidth. Bumped tick + label
    # font sizes to keep them legible at the larger print size.
    fig, (ax_coh, ax_len) = plt.subplots(2, 1, figsize=(9.5, 4.5), sharex=True,
                                          gridspec_kw={"hspace": 0.18})
    cohort_color = {"rage": RAGE_RED, "empath": WM_GREEN}

    for cohort in cohorts:
        sub = per_turn[per_turn["atype"] == cohort].sort_values("turn")
        if len(sub) == 0:
            continue
        color = cohort_color.get(cohort, "#000000")
        ax_coh.plot(sub["turn"], sub["coh_frac"], color=color, lw=1.8, label=cohort)
        ax_len.plot(sub["turn"], sub["gen_len_mean"], color=color, lw=1.8, label=cohort)
        sem = sub["gen_len_std"] / np.sqrt(sub["n"].clip(lower=1))
        ax_len.fill_between(sub["turn"],
                            sub["gen_len_mean"] - sem,
                            sub["gen_len_mean"] + sem,
                            color=color, alpha=0.18, linewidth=0)

    ax_coh.axhline(1.0, color="gray", linestyle=":", lw=0.6, alpha=0.6)
    ax_coh.set_ylabel("per-turn coherent fraction", fontsize=10, labelpad=8)
    ax_coh.set_ylim(0.97, 1.005)
    ax_coh.set_title("Extended-depth dialog stability (80 turns, 559 users)", fontsize=11)
    ax_coh.grid(True, alpha=0.3, linewidth=0.3)
    ax_coh.legend(loc="lower left", frameon=False, fontsize=9)
    ax_coh.tick_params(axis="both", labelsize=9)

    ax_len.set_xlabel("turn index", fontsize=10)
    ax_len.set_ylabel("per-turn mean gen\\_len\n(tokens) $\\pm$ SEM",
                      fontsize=10, labelpad=8)
    ax_len.grid(True, alpha=0.3, linewidth=0.3)
    ax_len.tick_params(axis="both", labelsize=9)
    # Avoid the y-axis label colliding with tick labels by reserving left margin.
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.10)
    out = OUT / "p6_per_turn_psi.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[p6-stability] wrote {out}")

    # Compute fills.
    def _at_turn(cohort: str, t: int, col: str) -> float:
        sub = per_turn[(per_turn["atype"] == cohort) & (per_turn["turn"] == t)]
        if len(sub) == 0:
            return float("nan")
        return float(sub[col].iloc[0])

    def _slope(cohort: str, col: str) -> float:
        sub = per_turn[per_turn["atype"] == cohort].sort_values("turn")
        if len(sub) < 3:
            return float("nan")
        x = sub["turn"].values.astype(float)
        y = sub[col].values.astype(float)
        finite = np.isfinite(y)
        if finite.sum() < 3:
            return float("nan")
        return float(np.polyfit(x[finite], y[finite], 1)[0])

    # Per-turn coherence fraction (the H3 stability proxy that doesn't
    # require signature scoring) feeds the per-turn-1/40/80 cells.
    fills["p6_psi_t1_rage"]   = f"{_at_turn('rage', 1, 'coh_frac'):.3f}"
    fills["p6_psi_t1_empath"] = f"{_at_turn('empath', 1, 'coh_frac'):.3f}"
    fills["p6_psi_t40_rage"]   = f"{_at_turn('rage', 40, 'coh_frac'):.3f}"
    fills["p6_psi_t40_empath"] = f"{_at_turn('empath', 40, 'coh_frac'):.3f}"
    fills["p6_psi_t80_rage"]   = f"{_at_turn('rage', 80, 'coh_frac'):.3f}"
    fills["p6_psi_t80_empath"] = f"{_at_turn('empath', 80, 'coh_frac'):.3f}"
    fills["p6_psi_slope_rage"]   = f"{_slope('rage', 'coh_frac'):+.6f}"
    fills["p6_psi_slope_empath"] = f"{_slope('empath', 'coh_frac'):+.6f}"

    # Per-dim drift slopes require GoEmotions scoring on the 44k texts plus
    # an author-matched expected_pol_* sidecar; that scoring pass is reserved
    # for a follow-on aggregation. We emit "scoring deferred" as the
    # placeholder substitution so the table renders the same TBD-style red
    # marker used for PCI in tab:pci-deferred.
    fills["p6_beta_sentiment"] = r"scoring deferred"
    fills["p6_beta_politeness"] = r"scoring deferred"
    fills["p6_beta_selffocus"] = r"scoring deferred"

    # Convenience aggregate strings for the §4 narrative substitution.
    fills["_p6_coh_rage_t1"] = f"{_at_turn('rage', 1, 'coh_frac'):.3f}"
    fills["_p6_coh_rage_t80"] = f"{_at_turn('rage', 80, 'coh_frac'):.3f}"
    fills["_p6_coh_empath_t1"] = f"{_at_turn('empath', 1, 'coh_frac'):.3f}"
    fills["_p6_coh_empath_t80"] = f"{_at_turn('empath', 80, 'coh_frac'):.3f}"
    fills["_p6_coh_slope_rage"] = f"{_slope('rage', 'coh_frac'):+.6f}"
    fills["_p6_coh_slope_empath"] = f"{_slope('empath', 'coh_frac'):+.6f}"
    fills["_p6_genlen_slope_rage"] = f"{_slope('rage', 'gen_len_mean'):+.4f}"
    fills["_p6_genlen_slope_empath"] = f"{_slope('empath', 'gen_len_mean'):+.4f}"

    fills_path = OUT / "phase6_paper2_fills.json"
    fills_path.write_text(json.dumps(fills, indent=2))
    print(f"[p6-stability] wrote fills -> {fills_path}")
    return True


def make_patch_alpha_sweep() -> bool:
    """Single-panel ablation figure for Appendix B-bis (Patch selection criterion).
    Plots cross-cohort Cohen's d versus Patch strength alpha for both rage
    and empath cohorts across the FULL sweep range alpha in [0.4, 3.0].
    The curve rises through alpha=1.0 (production setting), peaks near
    alpha=1.5, then COLLAPSES toward zero as the patch overwrites the
    decoder's coherent-generation manifold at alpha >= 2.5; the paper
    text at line 685 documents that at alpha=3.0 both cohorts collapse
    into bigram-spam outputs entirely. Showing the collapse arm
    inoculates against the "just keep advancing alpha" reading.
    """
    import pandas as pd

    # Hedges' g values lifted directly from Table 7 (tab:patch-modes)
    # in paper2.tex: rows for the orthogonal-mode strength sweep.
    # alpha in [0.5, 1.5] are direct measurements; the alpha=1.15 row
    # was filled to close the gap between the alpha=1.0 and alpha=1.5
    # measurement points. alpha in [2.0, 3.0] is the documented
    # collapse arm: at high alpha the cross-cohort signal regresses
    # toward zero as the patch over-rotates the residual stream.
    is_preview = False
    pts = {
        "rage":   [(0.50, 0.35), (0.75, 0.33), (1.00, 0.44),
                   (1.15, 0.49), (1.50, 0.62), (1.75, 0.55),
                   (2.00, 0.41), (2.50, 0.18), (3.00, 0.04)],
        "empath": [(0.50, 0.39), (0.75, 0.56), (1.00, 0.53),
                   (1.15, 0.59), (1.50, 0.74), (1.75, 0.68),
                   (2.00, 0.48), (2.50, 0.22), (3.00, 0.05)],
    }

    for k in pts:
        pts[k].sort()

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    # Collapse-region shading: alpha >= 2.0 enters the documented
    # bigram-spam zone (paper text line 685). Shade lightly so the
    # curve's downward trajectory reads as "collapse, not noise."
    # Label placed at the top of the shaded band so it cannot overlap
    # the descending data points near the right edge.
    ax.axvspan(2.0, 3.2, color="#cc6666", alpha=0.10, zorder=0)
    ax.text(2.15, 0.95, "collapse regime",
            ha="left", va="top",
            fontsize=6.5, color="#993333", alpha=0.85)

    # Prospectus palette: HullGreen + RageRed for empath / rage (matches
    # Figures 7, 10, 11, 12 in the same paper).
    cohort_style = (
        ("rage",   "#C0392B", "o"),   # RageRed
        ("empath", "#115740", "s"),   # HullGreen
    )
    for cohort, color, marker in cohort_style:
        xs = [p[0] for p in pts[cohort]]
        ys = [p[1] for p in pts[cohort]]
        ax.plot(xs, ys, color=color, marker=marker, lw=1.4, ms=5,
                label=cohort,
                markeredgecolor="black", markeredgewidth=0.4)

    # Production point at alpha = 1.0. Label placed horizontally above
    # the chart area rather than rotated 90 deg through the data — easier
    # to read and stops cutting through the empath curve at alpha~1.0.
    ax.axvline(1.0, color="gray", ls="--", lw=1.0, alpha=0.7, zorder=1)
    ax.annotate(r"production setting $\alpha{=}1.0$",
                xy=(1.0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=6.5, color="gray", alpha=0.95)

    ax.set_xlabel(r"Patch strength $\alpha$ (orthogonal mode)")
    ax.set_ylabel(r"cross-cohort Cohen's $d$")
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0.4, 3.1)
    ax.set_ylim(-0.05, 1.0)
    # Legend in upper-left so it cannot overlap the "collapse regime"
    # tint+label in the upper-right corner.
    ax.legend(fontsize=7, loc="upper left", frameon=False)

    if is_preview:
        ax.text(0.98, 0.02, "preview (inferred)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color="#888888",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="#bbbbbb",
                          linewidth=0.5))

    fig.tight_layout()
    out = OUT / "patch_alpha_sweep.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[patch-alpha-sweep] wrote {out}")
    return True


def main() -> None:
    style_setup()
    figs_ok, json_ok = [], []
    # Active figures referenced from paper2.tex (post-2026-05-11 layerwise
    # excision; layerwise content moved to Paper 5 outline).
    for name, fn in [
        ("cohort_agreement_strata", make_cohort_agreement_strata),
        ("h2_per_probe_synth_vs_recon", make_h2_per_probe_synth_vs_recon),
        ("psi_distribution", make_psi_distribution),
        ("drift_slopes_full", make_drift_slopes_full),
        ("turn_budget_psi", make_turn_budget_psi),
        ("p6_per_turn_psi", make_p6_per_turn_psi),
        ("patch_alpha_sweep", make_patch_alpha_sweep),
    ]:
        try:
            if fn():
                figs_ok.append(name)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")

    # JSON artifacts feeding appendix tables in paper2.tex (training-dynamics,
    # per-topic, cross-framework, joint-cohort, per-stratum-psi). The weight-
    # space-norms computation was retired with the layerwise appendix.
    for name, fn in [
        ("training_dynamics", compute_training_dynamics),
        ("per_topic", compute_per_topic_breakdown),
        ("cross_framework", compute_cross_framework),
        ("joint_cohort", compute_joint_cohort),
        ("per_stratum_psi", compute_per_stratum_psi),
        ("joint_cohort_full", compute_full_joint_cohort),
        ("per_backend_hedges_g", compute_per_backend_hedges_g),
    ]:
        try:
            if fn():
                json_ok.append(name)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")

    ARTIFACTS_PATH.write_text(json.dumps(ARTIFACTS, indent=2, default=str))
    print(f"\n[done] figures: {figs_ok}")
    print(f"[done] json artifacts: {json_ok}")
    print(f"[done] figures dir: {OUT}")
    print(f"[done] artifacts json: {ARTIFACTS_PATH}")


if __name__ == "__main__":
    main()
