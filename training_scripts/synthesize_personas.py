#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthesize_personas.py — Native synthetic-persona generator for
Pythia-1.4B.

Generates novel synthetic users by sampling new g-vectors from the manifold
of training users, with three sampling strata:

    in_hull        Dirichlet mixture over k nearest neighbors. The synthetic
                   point lies inside the convex hull of its anchors with high
                   probability (alpha=2.0 default), and the density gate
                   accepts only points whose k-NN density is above the p_dens
                   percentile of training-point densities.
    near_hull      Looser anchors (k larger), Dirichlet alpha=1.0; density
                   gate set lower (e.g. p_dens=0.50).
    far_from_hull  Random extrapolations along principal axes scaled by 1.5x
                   the per-component std; density gate disabled (these are
                   extrapolations by construction).

Optionally tests each candidate's offset norm under the trained hypernetwork
(`||H_phi(g_synth)||_2 <= tau_off`) so synthetic personas don't push the
LoRA-B injection past the trained clamp envelope.

Output
------
  synthetic_personas.parquet   one row per accepted candidate (target_user_id,
                                stratum, gstat_*..., source_anchors)
  synthesis_metadata.json      acceptance rates, stratum coverage, gates
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

LOG = logging.getLogger("synthesize_personas")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def _knn_density(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance to k nearest neighbors -> inverse density (lower = denser)."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=int(k) + 1).fit(X)
    dists, _ = nn.kneighbors(X)
    return dists[:, 1:].mean(axis=1)   # exclude self at column 0


def _sample_in_hull(X: np.ndarray, anchor_idxs: np.ndarray,
                     k: int, alpha: float, n_samples: int,
                     rng: np.random.Generator
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dirichlet-mixture sampling: each candidate is a convex combination of
    k randomly chosen training points from `anchor_idxs`. Returns the
    candidates, per-candidate anchor indices, and per-candidate mixing
    weights. Weights MUST be persisted for downstream label projection
    (label(synth) = sum_i w_i * label(anchor_i) for continuous dims).
    k is clamped to len(anchor_idxs) if the pool is smaller than requested."""
    k_eff = int(min(int(k), len(anchor_idxs)))
    if k_eff < 1:
        raise ValueError(
            f"_sample_in_hull: anchor pool is empty or k={k} invalid"
        )
    out = np.empty((int(n_samples), X.shape[1]), dtype=np.float32)
    anchor_log = np.empty((int(n_samples), k_eff), dtype=np.int64)
    weights_log = np.empty((int(n_samples), k_eff), dtype=np.float32)
    for i in range(int(n_samples)):
        chosen = rng.choice(anchor_idxs, size=k_eff, replace=False)
        weights = rng.dirichlet([float(alpha)] * k_eff)
        out[i] = (X[chosen] * weights[:, None]).sum(axis=0).astype(np.float32)
        anchor_log[i] = chosen
        weights_log[i] = weights.astype(np.float32)
    return out, anchor_log, weights_log


def _sample_far_extrapolation(X: np.ndarray, n_samples: int,
                               magnitude: float, rng: np.random.Generator
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample candidates along principal axes scaled by magnitude * std.
    Returns candidates and per-candidate PCA coefficients (log). The
    coefficients are the inputs to the reconstruction and are persisted
    so that downstream label projection can use the kNN-g-space
    fallback rather than anchor-weighted averaging (no anchors exist for
    far_from_hull)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    u, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    pcs = vt[: min(8, vt.shape[0])]   # top 8 directions
    out = np.empty((int(n_samples), X.shape[1]), dtype=np.float32)
    coefs_log = np.empty((int(n_samples), pcs.shape[0]), dtype=np.float32)
    for i in range(int(n_samples)):
        coefs = rng.normal(0.0, float(magnitude), size=pcs.shape[0])
        delta = (coefs[:, None] * pcs * sd).sum(axis=0)
        out[i] = (mu + delta).astype(np.float32)
        coefs_log[i] = coefs.astype(np.float32)
    return out, coefs_log


def _density_gate(candidates: np.ndarray, X_train: np.ndarray,
                   k: int, percentile: float) -> np.ndarray:
    """Accept candidates whose distance to k-NN among training is below the
    `percentile` of training-point inter-distances (denser-or-similar)."""
    from sklearn.neighbors import NearestNeighbors
    train_density = _knn_density(X_train, k=int(k))
    cutoff = float(np.percentile(train_density, float(percentile) * 100.0))
    nn = NearestNeighbors(n_neighbors=int(k)).fit(X_train)
    dists, _ = nn.kneighbors(candidates)
    cand_density = dists.mean(axis=1)
    return cand_density <= cutoff


def _offset_norm_gate(candidates: np.ndarray, hyper_dir: Optional[str],
                       base_model: str, tau_off: float,
                       X_train: Optional[np.ndarray] = None,
                       adaptive_mult: float = 2.0,
                       feature_names_json: str = "") -> np.ndarray:
    """Accept candidates whose hypernet-emitted offset norm <= tau_off.

    Optional: skipped if hyper_dir is empty / None.

    If `tau_off <= 0`, the threshold is calibrated adaptively as
    `adaptive_mult * median(||H(X_train)||_2)` so that roughly half of
    real training anchors would pass and `adaptive_mult` controls how
    far outside the training-delta distribution a synth persona may sit.
    This is robust across checkpoints; a fixed literal (e.g., 10.0) does
    not generalize because delta-vector dimensionality and per-role
    scales vary with lora_rank, emit_both, and target_modules.
    """
    if not hyper_dir:
        return np.ones(len(candidates), dtype=bool)
    # Lazy import to keep this script light when hypernet isn't loaded.
    # The hypernetwork inference engine is not bundled in the public release
    # (the offset-norm gate needs the trained checkpoint, which is not
    # redistributed), so fall back to accepting all candidates if it is
    # unavailable rather than failing the run.
    try:
        from legacy_engine import HyperPEFTEngine
    except Exception:
        LOG.warning("offset-norm gate skipped: hypernetwork inference engine "
                    "is not available in this release; accepting all "
                    "candidate profiles without the offset-norm gate.")
        return np.ones(len(candidates), dtype=bool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eng = HyperPEFTEngine(
        base_ckpt=base_model, hyper_ckpt=hyper_dir,
        feature_names_json=feature_names_json,
        lora_rank=24, lora_alpha=48.0,
        clamp=0.020, delta_scale=1.0,
        online=True, qlora=False, device=device,
    )

    def _norms(batch_np: np.ndarray) -> np.ndarray:
        out = np.zeros(len(batch_np), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(batch_np), 32):
                chunk = batch_np[i:i + 32]
                g_t = torch.tensor(chunk, dtype=torch.float32, device=device)
                delta = eng.hyper(g_t)
                n = torch.linalg.norm(delta.view(delta.shape[0], -1),
                                      dim=1).cpu().numpy()
                out[i:i + 32] = n
        return out

    thr = float(tau_off)
    if thr <= 0.0:
        if X_train is None or len(X_train) == 0:
            LOG.warning("tau_off<=0 requested but X_train unavailable; "
                        "falling back to hyper_dir=None (no gate)")
            return np.ones(len(candidates), dtype=bool)
        sample_idx = np.random.default_rng(142).choice(
            len(X_train), size=min(256, len(X_train)), replace=False,
        )
        train_norms = _norms(X_train[sample_idx])
        med = float(np.median(train_norms))
        p95 = float(np.percentile(train_norms, 95))
        thr = adaptive_mult * med
        LOG.info("[tau_off=auto] train-norm median=%.4f p95=%.4f "
                 "-> threshold=%.4f (mult=%.2f x median)",
                 med, p95, thr, adaptive_mult)

    cand_norms = _norms(candidates)
    accept = cand_norms <= thr
    LOG.info("[offset_gate] cand_norm min=%.4f med=%.4f max=%.4f "
             "threshold=%.4f accepted=%d/%d",
             float(cand_norms.min()) if len(cand_norms) else 0.0,
             float(np.median(cand_norms)) if len(cand_norms) else 0.0,
             float(cand_norms.max()) if len(cand_norms) else 0.0,
             thr, int(accept.sum()), len(cand_norms))
    return accept


def synthesize(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(int(args.seed))
    author_df = pd.read_parquet(args.author_parquet)
    if "target_user_id" not in author_df.columns:
        raise KeyError("author_parquet must include 'target_user_id'")

    feature_names: List[str]
    if args.feature_names_json and Path(args.feature_names_json).exists():
        raw = json.loads(Path(args.feature_names_json).read_text())
        # Checkpoints write {"feature_names": [...]}; older call sites passed a bare list.
        if isinstance(raw, dict):
            feature_names = list(raw.get("feature_names") or raw.get("features") or [])
            if not feature_names:
                raise ValueError(
                    f"{args.feature_names_json}: dict has no 'feature_names' or 'features' key; got keys={list(raw.keys())}"
                )
        elif isinstance(raw, list):
            feature_names = [str(x) for x in raw]
        else:
            raise ValueError(
                f"{args.feature_names_json}: expected list or dict, got {type(raw).__name__}"
            )
    else:
        feature_names = [c for c in author_df.columns
                         if c.startswith("gstat_")][: int(args.K)]
    K = min(int(args.K), len(feature_names))
    LOG.info("Using K=%d features: %s%s",
             K, feature_names[:5], " ..." if K > 5 else "")

    # Held-out probe columns read directly by `label_synthetic_personas.py`.
    # Any column not already in the K-feature set must be projected onto each
    # synth row (anchor-weighted mix for in_hull/near_hull, kNN mean for
    # far_from_hull). The hypernetwork's conditioning vector is unchanged;
    # these are auxiliary columns for the downstream labeler only.
    PROBE_PROJECT_COLS = [
        "gstat_profanity_ratio",    # politeness
        "gstat_question_ratio",     # curiosity
        "gstat_reply_delay_mean",   # tempo
        "gstat_firstperson_ratio",  # self_focus
        "gstat_caps_ratio",         # expressiveness (caps leg)
        "gstat_punct_ratio",        # expressiveness (punct leg)
    ]
    missing_probe_cols = [c for c in PROBE_PROJECT_COLS
                          if c not in author_df.columns]
    if missing_probe_cols:
        raise KeyError(
            f"author_parquet missing probe columns {missing_probe_cols}; "
            "cannot project held-out probes onto synth rows."
        )
    held_out_probe_cols = [c for c in PROBE_PROJECT_COLS
                           if c not in feature_names[:K]]
    LOG.info("Held-out probe columns to project onto synth: %s",
             held_out_probe_cols)

    # Build X (train pool) -- restrict to labeled cohort users if labels given
    keep_uids: Optional[set] = None
    if args.labels_csv and Path(args.labels_csv).exists():
        ldf = pd.read_csv(args.labels_csv)
        keep_uids = set(int(u) for u in ldf["target_user_id"])
        LOG.info("Restricting to %d labeled users", len(keep_uids))

    rows = []
    for _, r in author_df.iterrows():
        uid = int(r["target_user_id"])
        if keep_uids is not None and uid not in keep_uids:
            continue
        vec = np.zeros(K, dtype=np.float32)
        for i, fname in enumerate(feature_names[:K]):
            v = r.get(fname, 0.0)
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    fv = 0.0
            except Exception:
                fv = 0.0
            vec[i] = fv
        rows.append((uid, vec))
    if not rows:
        raise RuntimeError("No usable training rows; check labels_csv.")

    uids_train = np.array([r[0] for r in rows], dtype=np.int64)
    X_train = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
    LOG.info("Loaded %d training users (K=%d)", len(uids_train), K)

    # Held-out probe values aligned to uids_train ordering. Anchors index into
    # this array the same way they index into X_train, so mixing weights can
    # be reused verbatim for the projection.
    if held_out_probe_cols:
        probe_lookup = author_df.set_index("target_user_id")[held_out_probe_cols]
        X_probe_train = probe_lookup.loc[uids_train].to_numpy(dtype=np.float32)
    else:
        X_probe_train = np.zeros((len(uids_train), 0), dtype=np.float32)

    # ---------- Targeted-synthesis plumbing ----------
    # Approach A (`--target_method filter`): intersect the anchor pool with
    # one or more data/labels_*.csv constraints before Dirichlet mixing so the
    # convex combination inherits the targeted cell by construction. Approach
    # B (steer) and C (hybrid) remain reserved. See Appendix K.3b / K.4 of
    # zzzzz_COMPLETE_TECHNICAL_REFERENCE.md.
    target_method = str(getattr(args, "target_method", "none") or "none").lower()
    if target_method not in {"none", "filter", "steer", "hybrid"}:
        raise ValueError(
            f"--target_method must be one of none/filter/steer/hybrid; got {target_method!r}"
        )
    target_spec_default: Dict[str, Any] = {"method": target_method, "labels": []}
    filter_eligible_idx: Optional[np.ndarray] = None
    if target_method == "filter":
        if not args.target_label:
            raise ValueError(
                "--target_method filter requires at least one --target_label DIM:VALUE"
            )
        constraints: List[Tuple[str, str]] = []
        for raw in args.target_label:
            parts = str(raw).split(":")
            if len(parts) < 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    f"--target_label expects DIM:VALUE[:ALPHA]; got {raw!r}"
                )
            constraints.append((parts[0], parts[1]))
        labels_dir = (Path(args.labels_csv).parent
                      if args.labels_csv else Path("data"))
        eligible_uids: Optional[set] = None
        per_dim_counts: Dict[str, int] = {}
        for dim, val in constraints:
            p = labels_dir / f"labels_{dim}.csv"
            if not p.exists():
                raise FileNotFoundError(
                    f"labels CSV missing for target dim {dim!r}: {p}"
                )
            df = pd.read_csv(p)
            if ("target_user_id" not in df.columns
                    or "label" not in df.columns):
                raise ValueError(
                    f"{p} missing required columns (target_user_id, label)"
                )
            sub = set(int(u) for u in
                      df.loc[df["label"] == val, "target_user_id"].tolist())
            if not sub:
                uniq = sorted(df["label"].astype(str).unique().tolist())
                raise ValueError(
                    f"--target_label {dim}:{val} selects 0 real users in {p}; "
                    f"valid labels: {uniq}"
                )
            per_dim_counts[f"{dim}:{val}"] = len(sub)
            eligible_uids = sub if eligible_uids is None else (eligible_uids & sub)
        if not eligible_uids:
            raise ValueError(
                f"Filter intersection is EMPTY for constraints {constraints}. "
                "This joint cell is cross-cohort-impossible on the real manifold "
                "(see zzzzz_COMPLETE_TECHNICAL_REFERENCE.md Appendix K.3b). "
                "Filter-mixing cannot synthesize it; per-dim counts: "
                f"{per_dim_counts}."
            )
        uid_to_idx = {int(u): i for i, u in enumerate(uids_train.tolist())}
        filter_eligible_idx = np.array(
            sorted(uid_to_idx[u] for u in eligible_uids if u in uid_to_idx),
            dtype=np.int64,
        )
        k_floor = max(int(args.k_in_hull), int(args.k_near_hull))
        if filter_eligible_idx.size < k_floor:
            raise ValueError(
                f"Filter intersection has {filter_eligible_idx.size} users "
                f"inside the training pool, fewer than max(k_in_hull, "
                f"k_near_hull) = {k_floor}. Broaden the filter or reduce k."
            )
        LOG.info(
            "[target] filter constraints=%s -> %d eligible (of %d training); "
            "per-dim intersections: %s",
            constraints, int(filter_eligible_idx.size),
            len(uids_train), per_dim_counts,
        )
        target_spec_default = {
            "method": "filter",
            "labels": [{"dim": d, "value": v} for d, v in constraints],
            "n_eligible": int(filter_eligible_idx.size),
            "per_dim_counts": per_dim_counts,
        }
    elif target_method in ("steer", "hybrid"):
        raise NotImplementedError(
            f"--target_method {target_method!r} is reserved. Approach A "
            "(filter) ships in this release; B/C remain pending the Paper 2 "
            "Table 2 follow-on decision. See K.4."
        )
    LOG.info("[target] method=%s", target_method)

    # ---------- Sample three strata ----------
    accepted_rows: List[Dict[str, Any]] = []
    stratum_stats: Dict[str, Dict[str, int]] = {}

    def _record(stratum: str, candidates: np.ndarray,
                 anchors: Optional[np.ndarray],
                 weights: Optional[np.ndarray],
                 pca_coefs: Optional[np.ndarray],
                 spec: Optional[Dict[str, Any]] = None):
        # Density + offset-norm gates
        if stratum == "far_from_hull":
            dens_keep = np.ones(len(candidates), dtype=bool)
        else:
            dens_pct = float(args.in_hull_density_pct) if stratum == "in_hull" \
                else float(args.near_hull_density_pct)
            dens_keep = _density_gate(candidates, X_train,
                                      k=int(args.k_density),
                                      percentile=dens_pct)
        off_keep = _offset_norm_gate(
            candidates, args.hyper_dir, args.base_model,
            float(args.tau_off), X_train=X_train,
            adaptive_mult=float(args.tau_off_adaptive_mult),
            feature_names_json=args.feature_names_json,
        )
        keep = dens_keep & off_keep
        kept_n = int(np.sum(keep))
        stratum_stats[stratum] = {
            "n_proposed": int(len(candidates)),
            "n_density_pass": int(np.sum(dens_keep)),
            "n_offset_pass": int(np.sum(off_keep)),
            "n_accepted": kept_n,
        }
        # Project held-out probe columns onto every candidate (pre-filter so
        # the projection is consistent with the K-feature Dirichlet mix).
        if held_out_probe_cols and X_probe_train.shape[1] > 0:
            if (stratum in ("in_hull", "near_hull")
                    and anchors is not None and weights is not None):
                y_probe = np.einsum(
                    "nk,nkc->nc",
                    weights.astype(np.float32),
                    X_probe_train[anchors],
                ).astype(np.float32)
            else:
                # far_from_hull: kNN mean over training set in K-space.
                d2 = (
                    (candidates ** 2).sum(axis=1, keepdims=True)
                    + (X_train ** 2).sum(axis=1)[None, :]
                    - 2.0 * candidates @ X_train.T
                )
                k_nn = int(min(10, X_train.shape[0]))
                idx = np.argpartition(d2, kth=k_nn - 1, axis=1)[:, :k_nn]
                y_probe = X_probe_train[idx].mean(axis=1).astype(np.float32)
        else:
            y_probe = np.zeros((len(candidates), 0), dtype=np.float32)
        # Build rows
        synth_id_base = 10_000_000 + sum(s["n_accepted"]
                                          for s in stratum_stats.values())
        for i, ok in enumerate(keep):
            if not ok:
                continue
            row: Dict[str, Any] = {
                "target_user_id": int(synth_id_base + i),
                "stratum": stratum,
                "source_anchors": ([] if anchors is None
                                    else anchors[i].tolist()),
                # Resolve anchor indices -> real user IDs so downstream label
                # projection can look up per-anchor label values without
                # reconstructing the synthesize-time X_train ordering.
                "source_anchor_uids": ([] if anchors is None
                                        else uids_train[anchors[i]].tolist()),
                # Persisted mixing weights (in_hull / near_hull) or PCA
                # coefficients (far_from_hull). JSON-encoded to keep the
                # parquet schema stable across variable-k runs.
                "mixing_weights_json": (json.dumps(weights[i].tolist())
                                         if weights is not None else ""),
                "pca_coefs_json": (json.dumps(pca_coefs[i].tolist())
                                    if pca_coefs is not None else ""),
                "target_spec_json": json.dumps(spec or target_spec_default),
            }
            for j, fname in enumerate(feature_names[:K]):
                row[fname] = float(candidates[i, j])
            for c_idx, col_name in enumerate(held_out_probe_cols):
                row[col_name] = float(y_probe[i, c_idx])
            accepted_rows.append(row)
        LOG.info("[%s] proposed=%d density=%d offset=%d -> accepted=%d",
                 stratum, len(candidates),
                 int(np.sum(dens_keep)), int(np.sum(off_keep)), kept_n)

    anchor_pool = (filter_eligible_idx
                   if filter_eligible_idx is not None
                   else np.arange(X_train.shape[0]))
    # in_hull
    cands, anchors, weights = _sample_in_hull(
        X_train, anchor_pool,
        k=int(args.k_in_hull), alpha=float(args.alpha_in_hull),
        n_samples=int(args.n_per_stratum), rng=rng,
    )
    _record("in_hull", cands, anchors, weights, None)
    # near_hull
    cands, anchors, weights = _sample_in_hull(
        X_train, anchor_pool,
        k=int(args.k_near_hull), alpha=float(args.alpha_near_hull),
        n_samples=int(args.n_per_stratum), rng=rng,
    )
    _record("near_hull", cands, anchors, weights, None)
    # far_from_hull. Under filter mode, PCA extrapolation past a filtered
    # subset does not inherit filter semantics (extrapolated candidates can
    # land anywhere in g-space and their label comes from full-pool kNN
    # regression), so the stratum is skipped. Default (none) behavior is
    # unchanged.
    if target_method == "filter":
        LOG.info(
            "[far_from_hull] skipped under target_method=filter "
            "(extrapolation past a filtered anchor subset is not semantically targeted)"
        )
    else:
        cands_far, pca_coefs = _sample_far_extrapolation(
            X_train, n_samples=int(args.n_per_stratum),
            magnitude=float(args.far_magnitude), rng=rng,
        )
        _record("far_from_hull", cands_far, None, None, pca_coefs)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(accepted_rows)
    out_df.to_parquet(out_dir / "synthetic_personas.parquet", index=False)

    meta = {
        "schema": "synth_v1",
        "n_train_users": int(len(uids_train)), "K": K,
        "feature_names": feature_names[:K],
        "held_out_probe_cols": held_out_probe_cols,
        "stratum_stats": stratum_stats,
        "n_accepted_total": int(len(accepted_rows)),
        "args": vars(args),
    }
    (out_dir / "synthesis_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    LOG.info("Wrote %d accepted personas to %s", len(accepted_rows), out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize personas")
    p.add_argument("--author_parquet", type=str, required=True)
    p.add_argument("--labels_csv", type=str, default="",
                   help="Restrict training pool to labeled users (recommended).")
    p.add_argument("--feature_names_json", type=str, default="",
                   help="Optional explicit feature manifest (else uses gstat_*).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--n_per_stratum", type=int, default=3500,
                   help="Raised from 500 to 3500 "
                        "(post-gate yield ~3333/stratum x 3 strata ~= 10k synths, "
                        "matching the real-user cohort size). See Appendix K.5 of "
                        "zzzzz_COMPLETE_TECHNICAL_REFERENCE.md for the power analysis.")

    # Stratum knobs
    p.add_argument("--k_in_hull", type=int, default=5)
    p.add_argument("--alpha_in_hull", type=float, default=2.0,
                   help="Higher alpha -> more uniform mixture -> deeper inside hull.")
    p.add_argument("--in_hull_density_pct", type=float, default=0.95,
                   help="Density gate percentile for in-hull (0-1).")
    p.add_argument("--k_near_hull", type=int, default=12)
    p.add_argument("--alpha_near_hull", type=float, default=1.0)
    p.add_argument("--near_hull_density_pct", type=float, default=0.50)
    p.add_argument("--far_magnitude", type=float, default=1.5)
    p.add_argument("--k_density", type=int, default=10)

    # Offset-norm gate
    p.add_argument("--hyper_dir", type=str, default="",
                   help="If set, candidates are gated by ||H(g)||_2 <= tau_off.")
    p.add_argument("--base_model", type=str, default="EleutherAI/pythia-1.4b")
    p.add_argument("--tau_off", type=float, default=0.0,
                   help="Offset-norm cap. If <=0 (default), auto-calibrate "
                        "as tau_off_adaptive_mult * median(||H(X_train)||).")
    p.add_argument("--tau_off_adaptive_mult", type=float, default=2.0,
                   help="Multiplier on median training delta norm when "
                        "--tau_off <= 0 (auto mode). Larger = looser gate.")

    # Targeted-synthesis surface (reserved; default `none` = three-stratum behavior).
    # Implementations for filter / steer / hybrid are deferred to the Paper 2
    # Table 2 follow-on experiment; the current release raises NotImplementedError
    # for any non-`none` value. See Appendix K.4 for the intended semantics.
    p.add_argument("--target_method", type=str, default="none",
                   choices=["none", "filter", "steer", "hybrid"],
                   help="Targeted-synthesis regime. `none` = three-stratum default "
                        "(default three-stratum behavior). `filter` / `steer` / "
                        "`hybrid` reserved for Paper 2 Table 2 follow-on.")
    p.add_argument("--target_label", action="append", default=[],
                   metavar="DIM:BIN[:ALPHA]",
                   help="Repeatable target-label spec. Example: "
                        "`--target_label sentiment_goemo:rage "
                        "--target_label politeness:high:1.5`. Ignored when "
                        "--target_method=none.")
    p.add_argument("--target_manifest", type=str, default="",
                   help="Optional YAML manifest for batched (method x target) "
                        "cross-products. Ignored when --target_method=none.")

    p.add_argument("--seed", type=int, default=142)
    return p.parse_args()


def main() -> None:
    synthesize(parse_args())


if __name__ == "__main__":
    main()
