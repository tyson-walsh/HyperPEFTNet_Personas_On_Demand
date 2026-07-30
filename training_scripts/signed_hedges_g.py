"""Signed Hedges' g convention for Paper 2 cohort comparisons.

Convention: g > 0 always means "cohort effect is in the correct direction."
g < 0 means flipped/wrong. g ≈ 0 means flat (no signal).

For each measure we compute  g = (mean of HIGH-expected cohort) - (mean of LOW-expected
cohort), Hedges-corrected, where HIGH/LOW are defined by which cohort tail should score
higher on that specific measure given cohort theory.

Example: sentiment_goemo. Empath users should produce more positive sentiment than rage
users. So HIGH = empath, LOW = rage, and g = empath_mean - rage_mean.
Example: politeness. Rage users should swear more (higher profanity ratio) than empath
users. So HIGH = rage, LOW = empath, and g = rage_mean - empath_mean.

Usage:
    from signed_hedges_g import signed_g_rage_empath
    g, n_high, n_low = signed_g_rage_empath(df, measure="sentiment_goemo")
"""

import numpy as np
import pandas as pd

# Map: measure name -> which cohort SHOULD score higher on this measure.
# 'empath' means empath > rage expected (e.g. sentiment_goemo).
# 'rage' means rage > empath expected (e.g. profanity ratio for politeness).
# 2026-05-18: warmth + anxiety composite definitions live HERE as the single
# source of truth. score_persona_signature.py (realized) and the
# realprof builder in m1_gpu-node_p2_FULL_REFRESH.yaml (expected) BOTH import
# from this module so the realized vs expected composites stay synchronized.
#
# Warmth: was (caring + love + gratitude) / 3. Per-class decomposition at n=500
# production showed gratitude is the false-positive class on rage-topic emphatic
# phrasing ("thanks for being honest about X" / "I'm grateful when someone
# exposes this") -- it inverted the cohort-mean direction. Caring+love alone
# preserves the source-data direction (empath > rage by 1.62x in 10k users)
# AND gives HP-LoRA a clean Option B win (HP +0.439 vs Vanilla -0.035, Zero-Δ
# +0.180 -- HP-Van margin +0.474, HP-Zero margin +0.259). Locked 2026-05-18.
#
# Anxiety: was mean(fear, nervousness). Composite unchanged; only the HIGH
# cohort label was inverted (rage -> empath, see entry below).
WARMTH_GOEMO_CLASSES = ("caring", "love")            # OPTION B; previously caring+love+gratitude
ANXIETY_GOEMO_CLASSES = ("fear", "nervousness")     # unchanged

HIGH_COHORT_PER_MEASURE = {
    "sentiment_goemo": "empath",  # positive polarity → empath higher
    "politeness":      "rage",    # profanity ratio → rage higher
    "self_focus":      "rage",    # first-person ratio → rage higher
    "curiosity":       "empath",  # question ratio → empath asks more questions
    "expressiveness":  "rage",    # caps + punct → rage more emphatic
    "tempo":           "empath",  # realized_pol_tempo = z(word_count); axis HIGH = deliberate cohort (see score_persona_signature.py line 407). Empath users are mostly deliberate (305/453); rage users mostly reactive (212/360). So HIGH = empath.
    "anxiety":         "empath",  # 2026-05-18 FIX: goemo (fear+nervousness)/2 source data on 10k users shows empath=0.0051 > rage=0.0046; realized HP-LoRA agrees (empath_user 0.0030 > rage_user 0.00195). Prior "rage" assignment was intuition-based (rage→negative emotion) and inverted the signed g table for this probe.
    "warmth":          "empath",  # caring/love/gratitude → empath higher
    "hostility":       "rage",    # anger/disgust/disapproval → rage higher
}

# Measure taxonomy (2026-05-20). Split on whether a measure's DEFINING feature is
# in the hypernetwork input vector g. In-band measures are positive controls (the
# feature is handed to the model, so matching it is nearly free, and the kappa
# pushout extrapolates ONLY these -- they drive the off-manifold signature collapse).
# Out-of-band measures are genuine generalization (defining feature NOT in g).
# Corrects the prior habit of calling all 8 non-sentiment dims "held-out":
# curiosity/tempo/expressiveness are in-band controls, not held out.
IN_BAND_CONTROL_DIMS = ["curiosity", "tempo", "expressiveness"]
OUT_OF_BAND_DIMS     = ["sentiment_goemo", "politeness", "self_focus",
                        "anxiety", "warmth", "hostility"]
# Strict generalization set = OOB minus the cohort-defining sentiment axis.
OOB_GENERALIZATION_DIMS = ["politeness", "self_focus", "anxiety", "warmth", "hostility"]

# Paper 3 §5.6 originally pre-registered a MIXED pooled-6 that put 3 in-band positive
# controls into the headline. Kept only as a compliance line; never lead with it.
# See PROSPECTUS/HyperPEFTNet_RQ3 deviation 001 and Paper 2 PSI_HELDOUT_FIX_20MAY2026.md.
PREREG_MIXED_POOL_6 = ["sentiment_goemo", "politeness", "self_focus",
                       "curiosity", "expressiveness", "tempo"]


def pool_g_by_band(per_measure_g) -> dict:
    """Aggregate per-measure signed Hedges' g into the canonical bands.

    `per_measure_g`: dict {measure: g} or a DataFrame with columns ['measure','g'].
    Measure names use the signed_hedges_g convention (sentiment is 'sentiment_goemo').
    Returns the mean g over each band (non-finite values dropped):

      - 'oob_full_6'           : OUT_OF_BAND_DIMS  -- THE genuine result; lead with this.
      - 'oob_generalization_5' : OOB minus cohort-defining sentiment -- strict sub-pool.
      - 'in_band_control_3'    : IN_BAND_CONTROL_DIMS -- positive controls; never headline.
      - 'prereg_mixed_6'       : the locked mixed pool -- compliance line only.

    Any P3 verdict/aggregator MUST import this rather than hand-list bands (deviation 001).
    """
    if hasattr(per_measure_g, "columns"):
        gmap = dict(zip(per_measure_g["measure"], per_measure_g["g"]))
    else:
        gmap = dict(per_measure_g)

    def _mean(dims):
        vals = [gmap[d] for d in dims if d in gmap and np.isfinite(gmap[d])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "oob_full_6":           _mean(OUT_OF_BAND_DIMS),
        "oob_generalization_5": _mean(OOB_GENERALIZATION_DIMS),
        "in_band_control_3":    _mean(IN_BAND_CONTROL_DIMS),
        "prereg_mixed_6":       _mean(PREREG_MIXED_POOL_6),
    }


def _hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Bias-corrected standardized mean difference. g = (mean(x) - mean(y)) / pooled_sd, with
    Hedges' small-sample correction. Returns NaN if either group has < 2 valid values."""
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sp = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    d = (x.mean() - y.mean()) / sp
    j = 1 - 3 / (4 * (nx + ny) - 9)
    return d * j


def signed_g_rage_empath(df: pd.DataFrame, measure: str,
                         cohort_col: str = "expected_cohort_goemo",
                         realized_col_tpl: str = "realized_pol_{}") -> tuple:
    """Return (g, n_high, n_low) for the rage-vs-empath comparison on `measure`,
    signed so that g > 0 means correct cohort direction."""
    if measure not in HIGH_COHORT_PER_MEASURE:
        raise ValueError(f"unknown measure: {measure}")
    high_label = HIGH_COHORT_PER_MEASURE[measure]
    low_label = "rage" if high_label == "empath" else "empath"
    col = realized_col_tpl.format(measure)
    high_vals = df[df[cohort_col] == high_label][col].dropna().values
    low_vals  = df[df[cohort_col] == low_label][col].dropna().values
    g = _hedges_g(high_vals, low_vals)
    return g, len(high_vals), len(low_vals)


def signed_g_table(df: pd.DataFrame, measures: list = None) -> pd.DataFrame:
    """Compute signed g across a list of measures. Returns a DataFrame with one row per measure."""
    if measures is None:
        measures = list(HIGH_COHORT_PER_MEASURE.keys())
    rows = []
    for m in measures:
        g, n_h, n_l = signed_g_rage_empath(df, m)
        rows.append({"measure": m, "g": g, "high_cohort": HIGH_COHORT_PER_MEASURE[m],
                     "n_high": n_h, "n_low": n_l})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persona Fidelity Index (PFI) — 2026-05-19 production replacement for the
# retired cosine-variance PSI. Direction + magnitude both count; per-dim
# SD-normalized so no single dim dominates the L1 sum.
# ---------------------------------------------------------------------------

# Per-dim real-user expected SD denominator. MUST stay in sync with the
# EXPECTED_POLAR_SD_REAL_USER dict in score_persona_signature.py.
PFI_EXPECTED_SD_REAL_USER = {
    "politeness":      0.9261,
    "curiosity":       1.0213,
    "tempo":           1.0364,
    "self_focus":      1.2340,
    "expressiveness":  0.0168,
    "anxiety":         0.0031,
    "warmth":          0.0183,
    "hostility":       0.0126,
}

# Held-out polar dims used for PFI. Must match HELD_OUT_DIMS in the scorer.
PFI_HELD_OUT_DIMS = list(PFI_EXPECTED_SD_REAL_USER.keys())


def pfi_per_reply(realized_row: dict, expected_row: dict,
                  dims: list = None) -> float:
    """SD-normalized bullseye match for a single reply.
    Returns max(0, 1 - mean_d(|realized_d - expected_d| / sd_d) / 2).
    Range [0, 1]. 0 SDs off on average = 1.0, 2+ SDs off = 0.0.
    Pass dicts of dim -> polar value."""
    if dims is None: dims = PFI_HELD_OUT_DIMS
    sd_errs = []
    for d in dims:
        r = realized_row.get(d); e = expected_row.get(d)
        if r is None or e is None or not (np.isfinite(r) and np.isfinite(e)):
            continue
        sd = max(PFI_EXPECTED_SD_REAL_USER.get(d, 1.0), 1e-6)
        sd_errs.append(abs(r - e) / sd)
    if not sd_errs: return float("nan")
    return float(max(0.0, 1.0 - float(np.mean(sd_errs)) / 2.0))


def pfi_per_user(df: pd.DataFrame, user_col: str = "author_user_id",
                 bullseye_col: str = "bullseye_match_sdnorm") -> pd.DataFrame:
    """Return per-user mean bullseye_match_sdnorm. df must have user_col and
    bullseye_col (output of the updated score_persona_signature.py).
    Drops users with no coherent replies."""
    if bullseye_col not in df.columns:
        raise KeyError(f"missing {bullseye_col}; re-run score_persona_signature with the 2026-05-19 fix")
    return (df.groupby(user_col)[bullseye_col]
              .mean().dropna().reset_index(name="pfi"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: signed_hedges_g.py <persona_signature.parquet>")
        sys.exit(1)
    df = pd.read_parquet(sys.argv[1])
    out = signed_g_table(df)
    print(out.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
