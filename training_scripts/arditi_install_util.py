#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
arditi_install_util.py — Standalone Arditi residual-stream patch installer
==========================================================================

This is the standalone equivalent of ``_install_arditi_patch`` in
``build_hyperlora_forum.py``. It installs per-layer forward hooks on a
transformer block stack to project (or "ablate") a learned direction out of
the residual stream, optionally per-cohort or per-user.

Engineering contract
--------------------
* Operates directly on an HF causal-LM ``model`` (not on a ForumBuilder).
* Stores all per-user lookup tensors AS ATTRIBUTES ON ``model``, so the
  per-batch UID assignment pattern from build_hyperlora_forum.py
  (``model._arditi_batch_uids_tensor = ...``) still works unchanged:

      model._arditi_batch_user_ids
      model._arditi_batch_uids_tensor
      model._arditi_label_signs           [n_users_arr, 9]
      model._arditi_cohort_idx            [n_users_arr]
      model._arditi_polar_dim_names       List[str]
      model._arditi_orth_perp_by_layer    Dict[int (lk=ell+1), Tensor [4,H]]
      model._arditi_cohort_dirs_by_layer  Dict[int, Tensor [4,H]]
      model._arditi_orth_multi_by_layer   Dict[int, Tensor [n_users_arr, H]]
      model._arditi_originals             List[(layer, orig_forward)]
      model._arditi_meta                  Dict[str, Any]

* All ``self.engine.model`` references in the original method are mapped to
  ``model`` directly. ``self.log`` becomes the passed-in ``log`` parameter.
  ``self.engine.device`` becomes the passed-in ``device`` parameter (or
  inferred from model).

Modes
-----
``orthogonal`` is the production default for HyperPEFT-LoRA at alpha=1.0,
layers=15-23.

Other modes mirror the original method exactly:
  single, signed_sentiment, signed_multi, per_cohort_dir, orthogonal,
  orthogonal_multi.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch


# ---------------------------------------------------------------------
# Constants — duplicated from build_hyperlora_forum.py to minimize churn
# in the 5700-line legacy file. Keep these in sync if labels change.
# ---------------------------------------------------------------------

_ARDITI_POLAR_PAIRS = {
    "labels_sentiment_goemo.csv": ("rage", "empath"),
    "labels_politeness.csv":      ("vulgar", "polite"),
    "labels_self_focus.csv":      ("egocentric", "selfless"),
    "labels_curiosity.csv":       ("declarative", "inquisitive"),
    "labels_expressiveness.csv":  ("reserved", "emphatic"),
    "labels_tempo.csv":           ("reactive", "deliberate"),
    "labels_anxiety.csv":         ("anxious", "composed"),
    "labels_warmth.csv":          ("detached", "warm"),
    "labels_hostility.csv":       ("hostile", "agreeable"),
}

_ARDITI_CSV_TO_DIM = {
    "labels_sentiment_goemo.csv": "sentiment",
    "labels_politeness.csv":      "politeness",
    "labels_self_focus.csv":      "self_focus",
    "labels_curiosity.csv":       "curiosity",
    "labels_expressiveness.csv":  "expressiveness",
    "labels_tempo.csv":           "tempo",
    "labels_anxiety.csv":         "anxiety",
    "labels_warmth.csv":          "warmth",
    "labels_hostility.csv":       "hostility",
}


def arditi_load_user_labels(label_csv_dir: Path,
                            label_files: List[str]) -> Dict[int, Dict[str, str]]:
    """Build user_id -> {dim_name: label_string_lower} from per-dim CSVs.

    Missing or malformed CSVs are silently skipped (returns empty dict for
    that dim). A completely empty result is allowed; callers can detect it
    by ``len(result) == 0``.
    """
    out: Dict[int, Dict[str, str]] = {}
    for fname in label_files:
        path = Path(label_csv_dir) / fname
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "target_user_id" not in df.columns or "label" not in df.columns:
            continue
        dim_name = _ARDITI_CSV_TO_DIM.get(
            fname, fname.replace("labels_", "").replace(".csv", ""))
        for _, row in df.iterrows():
            try:
                uid = int(row["target_user_id"])
            except Exception:
                continue
            lab = str(row["label"]).strip().lower()
            out.setdefault(uid, {})[dim_name] = lab
    return out


def install_arditi_on_model(
    model,
    *,
    directions_path: str,
    alpha: float,
    layer_spec: str,
    mode: str = "orthogonal",
    label_csv_dir: str = "/tmp",
    label_files: Optional[List[str]] = None,
    log: Optional[logging.Logger] = None,
    profiles_uids: Optional[Iterable[int]] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Install Arditi residual-stream patch on ``model``.

    Parameters
    ----------
    model : nn.Module
        An HF causal-LM (or PEFT-wrapped causal-LM). The transformer block
        stack is auto-detected via the same family walk as the original
        ``_install_arditi_patch``.
    directions_path : str
        Path to the safetensors file produced by
        ``extract_arditi_directions.py --sampling multi_axis``.
    alpha : float
        Scalar gain on the projection subtraction. Production: 1.0.
    layer_spec : str
        Layer spec like ``"15-23"`` or ``"all"`` or ``"0,4,8"``.
    mode : str
        One of: single, signed_sentiment, signed_multi, per_cohort_dir,
        orthogonal, orthogonal_multi. Default: ``orthogonal``.
    label_csv_dir : str
        Directory containing the 9 per-dim label CSVs.
    label_files : Optional[List[str]]
        Subset of label CSV filenames to load. Defaults to all 9.
    log : Optional[logging.Logger]
        Logger to write status messages to. Defaults to a module logger.
    profiles_uids : Optional[Iterable[int]]
        User IDs that this model will be asked to generate for. Used to
        size the per-user lookup tensors. If neither this NOR
        ``label_csv_dir`` produces any user IDs, the patch installs in
        a degraded no-op mode (with a warning) instead of crashing.
    device : Optional[torch.device]
        Device for the lookup tables. Defaults to the model's first
        parameter's device.
    """
    if log is None:
        log = logging.getLogger("arditi_install_util")
        if not log.handlers:
            log.setLevel(logging.INFO)

    from safetensors.torch import load_file as _load_safetensors  # type: ignore

    p = Path(directions_path)
    if not p.exists():
        raise FileNotFoundError(f"Arditi directions file not found: {directions_path}")
    raw = _load_safetensors(str(p))
    if not raw:
        raise RuntimeError(f"Arditi directions file is empty: {directions_path}")

    # ------------------------------------------------------------------
    # Locate the transformer block stack. Same walk-down as the original.
    # ------------------------------------------------------------------
    obj = model
    # 2026-05-16: added `backbone` to the walk-down so PEFTHypernetModel
    # (which wraps the PEFT-wrapped backbone as self.backbone) is unwrapped.
    # Order matters: `backbone` first so we descend through it before the
    # generic `base_model`/`model` PEFT wrappers underneath.
    for attr in ("backbone", "base_model", "model"):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
    if hasattr(obj, "gpt_neox"):
        layers = obj.gpt_neox.layers
        family = "gpt_neox"
    elif hasattr(obj, "transformer") and hasattr(obj.transformer, "h"):
        layers = obj.transformer.h
        family = "gpt2"
    elif hasattr(obj, "embed_in") and hasattr(obj, "layers"):
        layers = obj.layers
        family = "gpt_neox"
    elif hasattr(obj, "layers"):
        layers = obj.layers
        family = "llama"
    else:
        raise RuntimeError("Could not locate transformer block stack on model.")
    n_layers = len(layers)

    # Parse layer spec
    spec = (layer_spec or "").strip().lower()
    if spec in ("all", "*"):
        target_layers = list(range(n_layers))
    elif "-" in spec:
        a, b = spec.split("-", 1)
        target_layers = list(range(int(a), int(b) + 1))
    elif "," in spec:
        target_layers = [int(s.strip()) for s in spec.split(",") if s.strip()]
    else:
        target_layers = [int(spec)] if spec else list(range(n_layers))
    target_layers = [ell for ell in target_layers if 0 <= ell < n_layers]
    if not target_layers:
        raise RuntimeError(f"layer_spec {layer_spec!r} resolved to empty target list")

    # ------------------------------------------------------------------
    # Resolve device. Prefer caller-supplied; else use model's first param.
    # ------------------------------------------------------------------
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Group raw keys by family.
    # ------------------------------------------------------------------
    grouped: Dict[str, Dict[int, torch.Tensor]] = {}
    for key, t in raw.items():
        if key.startswith("layer_") and "/" not in key:
            family_name = "main"
            try:
                ell = int(key.replace("layer_", ""))
            except ValueError:
                continue
        else:
            parts = key.split("/")
            try:
                ell = int(parts[-1].replace("layer_", ""))
            except ValueError:
                continue
            family_name = "/".join(parts[:-1])
        v = t.to(device=device, dtype=torch.float32).contiguous()
        n = float(torch.linalg.norm(v).item())
        if n >= 1e-8:
            v = v / n
        grouped.setdefault(family_name, {})[ell] = v

    if not grouped:
        raise RuntimeError("No directions decoded from safetensors; check key format.")
    log.info("[arditi] direction families found: %s",
             ", ".join(sorted(grouped.keys())))

    # ------------------------------------------------------------------
    # Validate mode requirements
    # ------------------------------------------------------------------
    mode = (mode or "single").lower()
    if mode in ("single", "signed_sentiment", "orthogonal") and "main" not in grouped:
        raise RuntimeError(f"mode={mode} requires 'main/' direction family in {p}")
    if mode == "signed_multi":
        missing = [d for d in _ARDITI_CSV_TO_DIM.values() if f"dim/{d}" not in grouped]
        if missing:
            log.warning("[arditi] signed_multi: missing dim families %s; "
                        "those axes will be skipped per-user", missing)
    if mode == "per_cohort_dir":
        if not any(k.startswith("cohort/") for k in grouped):
            raise RuntimeError("mode=per_cohort_dir requires 'cohort/' direction family")
    if mode == "orthogonal":
        if not any(k.startswith("persona/") for k in grouped):
            raise RuntimeError("mode=orthogonal requires 'persona/' direction family")

    # ------------------------------------------------------------------
    # Load per-user labels (only modes that need them).
    # ------------------------------------------------------------------
    user_labels: Dict[int, Dict[str, str]] = {}
    if mode in ("signed_sentiment", "signed_multi", "per_cohort_dir",
                "orthogonal", "orthogonal_multi"):
        labf = label_files or list(_ARDITI_CSV_TO_DIM.keys())
        user_labels = arditi_load_user_labels(Path(label_csv_dir), labf)
        log.info("[arditi] loaded labels for %d users across %d dims",
                 len(user_labels), len(labf))

    # Per-batch slots consumed by the patched forwards
    model._arditi_batch_user_ids = None
    model._arditi_batch_uids_tensor = None

    polar_dim_names = list(_ARDITI_CSV_TO_DIM.values())

    # ------------------------------------------------------------------
    # Size lookup tensors. max(uid) over labels + supplied profile pool.
    # ------------------------------------------------------------------
    n_polar = len(polar_dim_names)
    COHORT_KEYS = ["rage", "empath", "calm"]

    max_uid = -1
    if user_labels:
        max_uid = max(max_uid, max(int(k) for k in user_labels.keys()))
    if profiles_uids is not None:
        try:
            uids_list = [int(u) for u in profiles_uids]
            if uids_list:
                max_uid = max(max_uid, max(uids_list))
        except Exception:
            pass
    n_users_arr = max(max_uid + 1, 1)

    if n_users_arr <= 1 and mode in ("signed_sentiment", "signed_multi",
                                     "per_cohort_dir", "orthogonal",
                                     "orthogonal_multi"):
        log.warning(
            "[arditi] n_users_arr=%d (no labels and no profile pool supplied); "
            "per-user lookup will be empty and the patch will be a no-op for "
            "every batch. Hooks are installed for compatibility but will fall "
            "through to the unhooked path.",
            n_users_arr,
        )

    # Per-user polar signs [n_users_arr, n_polar]
    label_signs = torch.zeros((n_users_arr, n_polar), dtype=torch.float32, device=device)
    for d, fname in enumerate(_ARDITI_POLAR_PAIRS.keys()):
        lo_label, hi_label = _ARDITI_POLAR_PAIRS[fname]
        dim_name = _ARDITI_CSV_TO_DIM[fname]
        for uid_k, labs in user_labels.items():
            v = labs.get(dim_name, "")
            if v == lo_label:
                label_signs[int(uid_k), d] = +1.0
            elif v == hi_label:
                label_signs[int(uid_k), d] = -1.0
    model._arditi_label_signs = label_signs

    # Per-user cohort idx
    cohort_idx_arr = torch.full((n_users_arr,), 3, dtype=torch.long, device=device)
    for uid_k, labs in user_labels.items():
        sent = labs.get("sentiment", "")
        if sent in COHORT_KEYS:
            cohort_idx_arr[int(uid_k)] = COHORT_KEYS.index(sent)
    model._arditi_cohort_idx = cohort_idx_arr
    model._arditi_polar_dim_names = polar_dim_names

    # Per-layer stacks
    model._arditi_cohort_dirs_by_layer = {}     # type: Dict[int, torch.Tensor]
    model._arditi_orth_perp_by_layer = {}       # type: Dict[int, torch.Tensor]
    model._arditi_orth_multi_by_layer = {}      # type: Dict[int, torch.Tensor]

    # Infer hidden dim
    H_dim = None
    for fam_name, by_ell in grouped.items():
        if by_ell:
            any_t = next(iter(by_ell.values()))
            H_dim = int(any_t.shape[-1])
            break
    if H_dim is None:
        raise RuntimeError("could not infer hidden dim from directions file")

    if mode == "per_cohort_dir":
        for ell in target_layers:
            lk = ell + 1
            stack = torch.zeros((4, H_dim), dtype=torch.float32, device=device)
            for c, cohort_name in enumerate(COHORT_KEYS):
                d = grouped.get(f"cohort/{cohort_name}", {}).get(lk)
                if d is not None:
                    stack[c] = d
            model._arditi_cohort_dirs_by_layer[lk] = stack

    if mode == "orthogonal":
        for ell in target_layers:
            lk = ell + 1
            r_main = grouped.get("main", {}).get(lk)
            if r_main is None:
                continue
            stack = torch.zeros((4, H_dim), dtype=torch.float32, device=device)
            stack[3] = r_main
            for c, cohort_name in enumerate(COHORT_KEYS):
                pvec = grouped.get(f"persona/{cohort_name}", {}).get(lk)
                if pvec is None:
                    stack[c] = r_main
                    continue
                dot = (r_main * pvec).sum()
                r_perp = r_main - dot * pvec
                nrm = torch.linalg.norm(r_perp).clamp_min(1e-8)
                stack[c] = r_perp / nrm
            model._arditi_orth_perp_by_layer[lk] = stack

    if mode == "orthogonal_multi":
        active_mask = (label_signs.abs() > 0.5)
        for ell in target_layers:
            lk = ell + 1
            r_main = grouped.get("main", {}).get(lk)
            if r_main is None:
                continue
            polar_stack = torch.zeros((n_polar, H_dim), dtype=torch.float32, device=device)
            polar_present = torch.zeros((n_polar,), dtype=torch.bool, device=device)
            for d, dim_name in enumerate(polar_dim_names):
                v = grouped.get(f"dim/{dim_name}", {}).get(lk)
                if v is not None:
                    polar_stack[d] = v
                    polar_present[d] = True
            per_user_perp = r_main.unsqueeze(0).expand(n_users_arr, -1).contiguous()
            active_layer = active_mask & polar_present.unsqueeze(0)
            for d in range(n_polar):
                if not bool(polar_present[d].item()):
                    continue
                u = polar_stack[d]
                un = torch.linalg.norm(u).clamp_min(1e-8)
                u = u / un
                dot = (per_user_perp * u.unsqueeze(0)).sum(dim=-1, keepdim=True)
                gate = active_layer[:, d].to(per_user_perp.dtype).unsqueeze(-1)
                per_user_perp = per_user_perp - gate * dot * u.unsqueeze(0)
            norms = torch.linalg.norm(per_user_perp, dim=-1, keepdim=True).clamp_min(1e-8)
            per_user_perp = (per_user_perp / norms).contiguous()
            bad = ~torch.isfinite(per_user_perp).all(dim=-1)
            n_bad = int(bad.sum().item())
            if n_bad > 0:
                log.warning(
                    "[arditi] orthogonal_multi layer=%d: %d/%d user rows had NaN/Inf "
                    "after Gram-Schmidt; zeroing", lk, n_bad, n_users_arr,
                )
                per_user_perp[bad] = 0.0
            row_norms = torch.linalg.norm(per_user_perp, dim=-1)
            blown = row_norms > 10.0
            n_blown = int(blown.sum().item())
            if n_blown > 0:
                log.warning(
                    "[arditi] orthogonal_multi layer=%d: %d user rows had post-norm > 10; zeroing",
                    lk, n_blown,
                )
                per_user_perp[blown] = 0.0
            model._arditi_orth_multi_by_layer[lk] = per_user_perp

    log.info(
        "[arditi] precomputed | n_users=%d | n_polar=%d | tables=%s",
        n_users_arr, n_polar,
        {
            "cohort_dirs_layers": len(model._arditi_cohort_dirs_by_layer),
            "orth_perp_layers": len(model._arditi_orth_perp_by_layer),
            "orth_multi_layers": len(model._arditi_orth_multi_by_layer),
        },
    )

    def _per_layer_main(ell):
        return grouped.get("main", {}).get(ell + 1)

    def _per_layer_dim(ell, dim_name):
        return grouped.get(f"dim/{dim_name}", {}).get(ell + 1)

    def _per_layer_cohort(ell, cohort):
        return grouped.get(f"cohort/{cohort}", {}).get(ell + 1)

    # ------------------------------------------------------------------
    # Build per-mode patched forward closures
    # ------------------------------------------------------------------
    def _make_patched(orig, ell):
        def patched(*args, **kwargs):
            outputs = orig(*args, **kwargs)
            was_tuple = isinstance(outputs, tuple)
            hs = outputs[0] if was_tuple else outputs
            if not torch.is_tensor(hs) or hs.ndim < 2:
                return outputs
            B = int(hs.shape[0])
            batch_uids = getattr(model, "_arditi_batch_user_ids", None)

            if mode == "single":
                r = _per_layer_main(ell)
                if r is None:
                    return outputs
                r = r.to(dtype=hs.dtype, device=hs.device)
                proj = (hs * r).sum(dim=-1, keepdim=True)
                hs_new = hs - (float(alpha) * proj) * r

            elif mode == "signed_sentiment":
                r = _per_layer_main(ell)
                if r is None:
                    return outputs
                r = r.to(dtype=hs.dtype, device=hs.device)
                signs = torch.zeros((B,), dtype=hs.dtype, device=hs.device)
                if batch_uids is not None and len(batch_uids) == B:
                    for i, uid in enumerate(batch_uids):
                        sent = user_labels.get(int(uid), {}).get("sentiment", "")
                        if sent == "rage":
                            signs[i] = +1.0
                        elif sent == "empath":
                            signs[i] = -1.0
                proj = (hs * r).sum(dim=-1, keepdim=True)
                a_per_row = (signs * float(alpha)).view(B, 1, 1)
                hs_new = hs - a_per_row * proj * r

            elif mode == "signed_multi":
                hs_new = hs
                for fname, (lo_label, hi_label) in _ARDITI_POLAR_PAIRS.items():
                    dim_name = _ARDITI_CSV_TO_DIM[fname]
                    r = _per_layer_dim(ell, dim_name)
                    if r is None:
                        continue
                    r = r.to(dtype=hs.dtype, device=hs.device)
                    signs = torch.zeros((B,), dtype=hs.dtype, device=hs.device)
                    if batch_uids is not None and len(batch_uids) == B:
                        for i, uid in enumerate(batch_uids):
                            lab = user_labels.get(int(uid), {}).get(dim_name, "")
                            if lab == lo_label:
                                signs[i] = +1.0
                            elif lab == hi_label:
                                signs[i] = -1.0
                    proj = (hs_new * r).sum(dim=-1, keepdim=True)
                    a_per_row = (signs * float(alpha)).view(B, 1, 1)
                    hs_new = hs_new - a_per_row * proj * r

            elif mode == "per_cohort_dir":
                if batch_uids is None or len(batch_uids) != B:
                    return outputs
                r_rows = []
                for uid in batch_uids:
                    sent = user_labels.get(int(uid), {}).get("sentiment", "")
                    cohort_key = sent if sent in ("rage", "empath", "calm") else None
                    rv = _per_layer_cohort(ell, cohort_key) if cohort_key else None
                    if rv is None:
                        rv = torch.zeros(hs.shape[-1], dtype=hs.dtype, device=hs.device)
                    else:
                        rv = rv.to(dtype=hs.dtype, device=hs.device)
                    r_rows.append(rv)
                R = torch.stack(r_rows, dim=0)
                proj = (hs * R.unsqueeze(1)).sum(dim=-1, keepdim=True)
                hs_new = hs - (float(alpha) * proj) * R.unsqueeze(1)

            elif mode == "orthogonal":
                batch_uids_t = getattr(model, "_arditi_batch_uids_tensor", None)
                if batch_uids_t is None or int(batch_uids_t.numel()) != B:
                    return outputs
                stack = model._arditi_orth_perp_by_layer.get(ell + 1)
                if stack is None:
                    return outputs
                n_table = int(model._arditi_cohort_idx.numel())
                in_range = (batch_uids_t >= 0) & (batch_uids_t < n_table)
                safe_uids = torch.where(in_range, batch_uids_t,
                                        torch.zeros_like(batch_uids_t))
                cohort_idx_b = torch.where(
                    in_range,
                    model._arditi_cohort_idx[safe_uids],
                    torch.full_like(batch_uids_t, 3),
                )
                orig_dtype = hs.dtype
                hs_f = hs.to(torch.float32)
                R = stack[cohort_idx_b].to(torch.float32)
                if not torch.isfinite(R).all():
                    return outputs
                proj = (hs_f * R.unsqueeze(1)).sum(dim=-1, keepdim=True)
                if not torch.isfinite(proj).all():
                    return outputs
                hs_new = (hs_f - (float(alpha) * proj) * R.unsqueeze(1)).to(orig_dtype)

            elif mode == "orthogonal_multi":
                batch_uids_t = getattr(model, "_arditi_batch_uids_tensor", None)
                if batch_uids_t is None or int(batch_uids_t.numel()) != B:
                    return outputs
                perp_users = model._arditi_orth_multi_by_layer.get(ell + 1)
                if perp_users is None:
                    return outputs
                n_rows = int(perp_users.shape[0])
                if int(batch_uids_t.max().item()) >= n_rows or int(batch_uids_t.min().item()) < 0:
                    return outputs
                R = perp_users[batch_uids_t].to(dtype=hs.dtype)
                if not torch.isfinite(R).all():
                    return outputs
                proj = (hs * R.unsqueeze(1)).sum(dim=-1, keepdim=True)
                if not torch.isfinite(proj).all():
                    return outputs
                hs_new = hs - (float(alpha) * proj) * R.unsqueeze(1)

            else:
                r = _per_layer_main(ell)
                if r is None:
                    return outputs
                r = r.to(dtype=hs.dtype, device=hs.device)
                proj = (hs * r).sum(dim=-1, keepdim=True)
                hs_new = hs - (float(alpha) * proj) * r

            if was_tuple:
                return (hs_new,) + outputs[1:]
            return hs_new
        return patched

    # ------------------------------------------------------------------
    # Install patched forwards
    # ------------------------------------------------------------------
    model._arditi_originals = []  # type: List[Tuple[Any, Any]]
    installed_layers: List[int] = []
    for ell in target_layers:
        if mode in ("single", "signed_sentiment", "orthogonal") and _per_layer_main(ell) is None:
            log.warning("[arditi] no main/<%d> direction; skipping layer %d", ell + 1, ell)
            continue
        layer = layers[ell]
        orig = layer.forward
        layer.forward = _make_patched(orig, ell)
        model._arditi_originals.append((layer, orig))
        installed_layers.append(ell)

    if not installed_layers:
        raise RuntimeError("No layers patched; check direction file keys vs layer_spec.")

    model._arditi_meta = {
        "directions_path": str(p),
        "mode": mode,
        "alpha": float(alpha),
        "layer_spec": str(layer_spec),
        "applied_layers": installed_layers,
        "n_layers_total": int(n_layers),
        "model_family": family,
        "n_user_labels_loaded": len(user_labels),
        "direction_families_present": sorted(grouped.keys()),
    }
    log.info(
        "[arditi] installed | mode=%s | family=%s | alpha=%.3f | layers=%s | direction families=%s",
        mode, family, float(alpha),
        ",".join(str(ell) for ell in installed_layers),
        ",".join(sorted(grouped.keys())),
    )
