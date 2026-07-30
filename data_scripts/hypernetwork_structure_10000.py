#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hypernetwork_structure_10000.py
===============================

Hypernetwork for HyperPEFT-LoRA (PEFT–LoRA B-surface offsets)
-------------------------------------------------------

This module defines a hypernetwork wrapper that conditions a frozen PEFT–LoRA
surface on an author/user global feature vector **g** (global-static features).

Given:
  • a PEFT-wrapped CausalLM backbone with LoRA modules (A, B), and
  • a per-sample global feature vector g ∈ ℝ^G,

the hypernetwork H maps g → δθ, where δθ is the flattened concatenation of
additive offsets δB for every LoRA B matrix across the backbone:

    δθ = concat_j vec(δB_j)     with   δB_j ∈ ℝ^{out_j×r}

Stateless injection is performed per-forward via module hooks:

    y = y + scaling · (A(dropout(x)) @ δBᵀ)

Emission topologies:
  1) Single-head (flat): one head for all roles
  2) Role-split multi-head: separate heads for {qkv, o_proj, mlp, other}

Optional dictionary-coded emission (either topology):
  For each role r, learn a dictionary D_r (atoms) and predict coefficients α_r(g):

    α_r = W_r h      and     δθ_r = α_r @ D_rᵀ

Optional layer-context gating:
  Predict per-placeholder gates β_j(g) ∈ [-1, 1] to scale δB_j at runtime.

Pipeline integration notes
--------------------------
• The dataset wrapper `hypernetwork_dataset_10000.py` constructs per-user
  `global_features` from `gstat_*` columns and can optionally return
  `global_mask` indicating whether the global vector is non-zero.
• This wrapper accepts `global_mask` to force δ=0 on masked rows (prevents
  silent-join failures from injecting arbitrary deltas).

Smoke test:
  Run this file directly to instantiate either:
    (A) a tiny HF+PEFT LoRA model (if `peft` is installed), or
    (B) a lightweight built-in dummy LoRA backbone (no external deps),
  attach the hypernetwork, and execute a forward pass to validate placeholder
  discovery and delta injection.

    python hypernetwork_structure_10000.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Optional environment knobs
# --------------------------------------------------------------------------- #

ENV_HEAD_MODE = "HN_HEAD_MODE"                    # "single" | "multi"
ENV_DELTA_GAIN = "HN_DELTA_GAIN"                  # global gain applied after role scaling
ENV_DISABLE_GATES = "HN_DISABLE_GATES"            # truthy => bypass layer-context gates
ENV_GATES_SCALE = "HN_GATES_SCALE"                # [0,1] gate amplitude multiplier
ENV_SANITY_DELTA_NOISE = "HN_SANITY_DELTA_NOISE"  # add noise to delta for debugging
ENV_DICT_OUT_CHUNK = "HN_DICT_OUT_CHUNK"          # output chunk size for dict-coded head
ENV_DICT_INIT_STD = "HN_DICT_INIT_STD"            # std for dictionary atoms init
ENV_ALPHA_INIT_STD = "HN_ALPHA_INIT_STD"          # std for alpha-head init (dict-coded)
ENV_ZERO_INIT_OUT = "HN_ZERO_INIT_HYPER_OUT"      # truthy => initialize emission head to produce ~0 delta


# --------------------------------------------------------------------------- #
# Small utils
# --------------------------------------------------------------------------- #

def _is_truthy(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _role_of_leaf(leaf: str) -> str:
    """Map module leaf name to canonical role buckets.

    Recognized layouts:
      - q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
      - GPT-NeoX (Pythia): query_key_value, dense, dense_h_to_4h, dense_4h_to_h
    """
    # QKV (q_proj/k_proj/v_proj)
    if leaf in {"q_proj", "k_proj", "v_proj"}:
        return "qkv"
    # GPT-NeoX fused QKV (Pythia)
    if leaf in {"query_key_value"}:
        return "qkv"
    # attention output (o_proj / dense)
    if leaf in {"o_proj", "out_proj", "dense"}:
        return "o_proj"
    # gated MLP (gate/up/down)
    if leaf in {"gate_proj", "up_proj", "down_proj"}:
        return "mlp"
    # GPT-NeoX two-layer MLP
    if leaf in {"dense_h_to_4h", "dense_4h_to_h"}:
        return "mlp"
    return "other"


def _resolve_decoder_backbone(model: nn.Module) -> nn.Module:
    """
    For a PEFT-wrapped CausalLM, return the underlying decoder/base model
    (e.g., a decoder-only HF model). For models where a decoder is not separable, return
    the best candidate that supports forward(input_ids, attention_mask, ...).
    """
    def follow(obj: Any, path: Sequence[str]) -> Optional[Any]:
        cur = obj
        for name in path:
            if not hasattr(cur, name):
                return None
            cur = getattr(cur, name)
        return cur

    # Preferred paths for PeftModelForCausalLM layouts
    preferred_paths = [
        ("base_model", "model", "model"),          # PeftModel -> LoraModel -> CausalLM -> Decoder
        ("module", "base_model", "model", "model"),
        ("model", "model"),                        # CausalLM -> Decoder
    ]
    for path in preferred_paths:
        obj = follow(model, path)
        if obj is not None and hasattr(obj, "embed_tokens") and callable(getattr(obj, "forward", None)):
            return obj

    # Secondary candidates: accept anything callable
    secondary_paths = [
        ("base_model", "model"),
        ("module", "base_model", "model"),
        ("model",),
        ("module", "model"),
        ("transformer",),
        ("base_model",),
        ("module",),
    ]
    for path in secondary_paths:
        obj = follow(model, path)
        if obj is not None and callable(getattr(obj, "forward", None)):
            return obj

    # Global scan fallback
    for mod in model.modules():
        if hasattr(mod, "embed_tokens") and callable(getattr(mod, "forward", None)):
            return mod
    for mod in model.modules():
        if callable(getattr(mod, "forward", None)):
            return mod

    raise RuntimeError("Could not locate a decoder/backbone module to call forward().")


class LockedDropout1D(nn.Module):
    """
    Dropout applied to feature vectors with one independent mask per sample.
    """
    def __init__(self, p: float):
        super().__init__()
        self.p = float(max(0.0, min(1.0, p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        if x.dim() != 2:
            return F.dropout(x, p=self.p, training=True)
        B, D = x.shape
        mask = x.new_empty(B, D).bernoulli_(1.0 - self.p).div_(1.0 - self.p)
        return x * mask


# --------------------------------------------------------------------------- #
# Output heads
# --------------------------------------------------------------------------- #

class HyperOutHead(nn.Module):
    """
    Single-head emission: h -> δθ (dense or factorized).

    If hyper_out_rank > 0:
        δθ = (W_rank h) @ T^T
    else:
        δθ = W_out h
    """
    def __init__(
        self,
        total_out_dim: int,
        hidden_dim: int,
        *,
        hyper_out_rank: int = 0,
        bias: bool = False,
        zero_init: bool = False,
    ):
        super().__init__()
        self.total_out_dim = int(total_out_dim)
        self.hidden_dim = int(hidden_dim)
        self.rank = int(max(0, hyper_out_rank))
        self.zero_init = bool(zero_init)

        if self.rank > 0:
            self.proj1 = nn.Linear(self.hidden_dim, self.rank, bias=False)
            self.table = nn.Parameter(torch.empty(self.total_out_dim, self.rank))

            # Safe "zero delta" init: zero proj1 so z=0, keep table non-zero so proj1 can learn immediately.
            if self.zero_init:
                nn.init.zeros_(self.proj1.weight)
                nn.init.normal_(self.table, mean=0.0, std=0.01)
            else:
                nn.init.kaiming_uniform_(self.proj1.weight, a=math.sqrt(5))
                nn.init.normal_(self.table, mean=0.0, std=0.01)
        else:
            self.dense = nn.Linear(self.hidden_dim, self.total_out_dim, bias=bias)
            if self.zero_init:
                nn.init.zeros_(self.dense.weight)
                if bias and self.dense.bias is not None:
                    nn.init.zeros_(self.dense.bias)
            else:
                nn.init.kaiming_uniform_(self.dense.weight, a=math.sqrt(5))
                if bias and self.dense.bias is not None:
                    nn.init.zeros_(self.dense.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "dense"):
            return self.dense(h)
        z = self.proj1(h)           # [B, r]
        return z @ self.table.t()   # [B, P]


@dataclass
class RoleSpec:
    """
    Contiguous segment of δθ for a role.
    """
    name: str
    size: int
    rank: int = 0
    dict_k: int = 0


class RoleSplitOutHead(nn.Module):
    """
    Per-role emission heads concatenated into δθ.
    """
    class _DenseHead(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, *, zero_init: bool = False):
            super().__init__()
            self.proj = nn.Linear(int(in_dim), int(out_dim), bias=False)
            if zero_init:
                nn.init.zeros_(self.proj.weight)
            else:
                nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            return self.proj(h)

    class _FactorizedHead(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, rank: int, *, zero_init: bool = False):
            super().__init__()
            self.proj1 = nn.Linear(int(in_dim), int(rank), bias=False)
            self.table = nn.Parameter(torch.empty(int(out_dim), int(rank)))

            if zero_init:
                nn.init.zeros_(self.proj1.weight)
                nn.init.normal_(self.table, mean=0.0, std=0.01)
            else:
                nn.init.kaiming_uniform_(self.proj1.weight, a=math.sqrt(5))
                nn.init.normal_(self.table, mean=0.0, std=0.01)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            z = self.proj1(h)
            return z @ self.table.t()

    def __init__(self, hidden_dim: int, roles: List[RoleSpec], *, zero_init: bool = False):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.roles = list(roles)

        self.heads = nn.ModuleDict()
        for spec in self.roles:
            out_dim = int(spec.size)
            if out_dim <= 0:
                continue
            if spec.rank and int(spec.rank) > 0:
                self.heads[spec.name] = self._FactorizedHead(self.hidden_dim, out_dim, int(spec.rank), zero_init=zero_init)
            else:
                self.heads[spec.name] = self._DenseHead(self.hidden_dim, out_dim, zero_init=zero_init)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        parts = [self.heads[spec.name](h) for spec in self.roles if spec.name in self.heads]
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


class DictCodedOutHead(nn.Module):
    """
    Dictionary-coded emission per role:

        alpha_r = W_r h                 [B, K]
        delta_r = alpha_r @ D_r^T       where D_r is [size, K] -> [B, size]

    Regularization:
      - alpha_l1: L1 penalty on alpha
      - dict_ortho: orthogonality penalty on normalized dictionary columns

    Chunking:
      Use env HN_DICT_OUT_CHUNK or set_out_chunk() to compute alpha @ D^T in slices
      along the output dimension, reducing GEMM workspace pressure for large 'size'.
    """
    def __init__(
        self,
        hidden_dim: int,
        roles: List[RoleSpec],
        *,
        alpha_l1: float = 0.0,
        dict_ortho: float = 0.0,
        out_chunk: int = 0,
        zero_init: bool = False,
        alpha_init_std: Optional[float] = None,
        dict_init_std: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.roles = list(roles)
        self.alpha_l1 = float(alpha_l1)
        self.dict_ortho = float(dict_ortho)

        # Init stds from env unless explicit
        if alpha_init_std is None:
            alpha_init_std = _env_float(ENV_ALPHA_INIT_STD, 0.0)  # 0.0 => kaiming
        if dict_init_std is None:
            dict_init_std = _env_float(ENV_DICT_INIT_STD, 0.02)

        self._alpha_init_std = float(alpha_init_std)
        self._dict_init_std = float(dict_init_std)

        env_chunk = _env_int(ENV_DICT_OUT_CHUNK, 0)
        self._out_chunk = int(max(0, out_chunk or env_chunk))

        self.alpha_heads = nn.ModuleDict()
        self.dict_tables = nn.ParameterDict()

        for spec in self.roles:
            K = int(max(1, spec.dict_k))
            head = nn.Linear(self.hidden_dim, K, bias=False)

            # Safe "zero delta" init: zero alpha head, keep D non-zero so alpha head can learn.
            if zero_init:
                nn.init.zeros_(head.weight)
            else:
                if self._alpha_init_std and self._alpha_init_std > 0.0:
                    nn.init.normal_(head.weight, mean=0.0, std=self._alpha_init_std)
                else:
                    nn.init.kaiming_uniform_(head.weight, a=math.sqrt(5))

            self.alpha_heads[spec.name] = head

            D = nn.Parameter(torch.empty(int(spec.size), K))
            nn.init.normal_(D, mean=0.0, std=self._dict_init_std)
            self.dict_tables[spec.name] = D

        self._last_aux: Dict[str, torch.Tensor] = {}

    def set_out_chunk(self, n: int) -> None:
        self._out_chunk = int(max(0, n))

    def aux_losses(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_aux) if self._last_aux else {}

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        aux_terms: List[torch.Tensor] = []

        for spec in self.roles:
            alpha = self.alpha_heads[spec.name](h)  # [B, K]
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            D = self.dict_tables[spec.name]         # [size, K]

            # alpha @ D^T
            if self._out_chunk and self._out_chunk > 0:
                Dt = D.t()  # [K, size]
                chunk_parts: List[torch.Tensor] = []
                size = int(D.shape[0])
                for start in range(0, size, self._out_chunk):
                    end = min(start + self._out_chunk, size)
                    chunk_parts.append(alpha @ Dt[:, start:end])
                part = torch.cat(chunk_parts, dim=1)
            else:
                part = alpha @ D.t()

            part = torch.nan_to_num(part, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(part)

            if self.alpha_l1 > 0.0:
                aux_terms.append(alpha.abs().mean() * self.alpha_l1)

            if self.dict_ortho > 0.0:
                D_norm = F.normalize(D, p=2, dim=0)
                gram = D_norm.t() @ D_norm
                I = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
                aux_terms.append((gram - I).pow(2).mean() * self.dict_ortho)

        if aux_terms:
            aux = torch.stack(aux_terms).sum()
            aux = torch.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0)
            self._last_aux = {"dict_aux_loss": aux}
        else:
            self._last_aux = {}

        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


# --------------------------------------------------------------------------- #
# Hypernetwork trunk
# --------------------------------------------------------------------------- #

class FlatHypernetwork(nn.Module):
    """
    Global-only hypernetwork:
        g -> h -> δθ
    """
    def __init__(
        self,
        global_input_dim: int,
        hidden_dim: int,
        peft_param_count: int,
        *,
        activation: str = "relu",
        dropout_p_global: float = 0.0,
        clamp_range: Optional[float] = None,
        head_mode: str = "single",
        role_specs: Optional[List[RoleSpec]] = None,
        dict_mode: bool = False,
        dict_k_global: int = 0,
        hyper_out_rank: int = 0,
        alpha_l1: float = 0.0,
        dict_ortho: float = 0.0,
        zero_init_out: Optional[bool] = None,
        alpha_init_std: Optional[float] = None,
        dict_init_std: Optional[float] = None,
    ):
        super().__init__()
        self.global_input_dim = int(global_input_dim)
        self.hidden_dim = int(hidden_dim)
        self._peft_param_count = int(peft_param_count)

        self.clamp_range = None if (clamp_range is None or clamp_range <= 0.0) else float(clamp_range)
        self.head_mode = str(head_mode).lower()
        self.dict_mode = bool(dict_mode)

        if zero_init_out is None:
            zero_init_out = _is_truthy(os.environ.get(ENV_ZERO_INIT_OUT, "0"))
        self._zero_init_out = bool(zero_init_out)

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "leaky_relu":
            act = nn.LeakyReLU(0.01, inplace=False)
        else:
            raise ValueError("activation must be one of {'relu','silu','gelu','leaky_relu'}")

        self.drop_g = LockedDropout1D(dropout_p_global)
        self.net = nn.Sequential(
            nn.Linear(self.global_input_dim, self.hidden_dim),
            act,
        )
        # Trunk init: always non-degenerate (do NOT zero-init, to avoid dead activations).
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        role_specs = list(role_specs or [])
        if self.head_mode not in {"single", "multi"}:
            raise ValueError("head_mode must be 'single' or 'multi'.")

        if self.head_mode == "single":
            if self.dict_mode:
                K = int(max(1, dict_k_global))
                roles = [RoleSpec(name="all", size=self._peft_param_count, dict_k=K)]
                self.out_head = DictCodedOutHead(
                    hidden_dim=self.hidden_dim,
                    roles=roles,
                    alpha_l1=float(alpha_l1),
                    dict_ortho=float(dict_ortho),
                    zero_init=self._zero_init_out,
                    alpha_init_std=alpha_init_std,
                    dict_init_std=dict_init_std,
                )
            else:
                self.out_head = HyperOutHead(
                    total_out_dim=self._peft_param_count,
                    hidden_dim=self.hidden_dim,
                    hyper_out_rank=int(max(0, hyper_out_rank)),
                    bias=False,
                    zero_init=self._zero_init_out,
                )
        else:
            if not role_specs:
                raise ValueError("head_mode='multi' requires non-empty role_specs.")
            P_roles = int(sum(int(rs.size) for rs in role_specs))
            if P_roles != self._peft_param_count:
                raise ValueError("Sum of role sizes does not equal peft_param_count.")
            if self.dict_mode:
                if any(int(rs.dict_k) <= 0 for rs in role_specs):
                    raise ValueError("dict_mode=True requires dict_k>0 for each RoleSpec.")
                self.out_head = DictCodedOutHead(
                    hidden_dim=self.hidden_dim,
                    roles=role_specs,
                    alpha_l1=float(alpha_l1),
                    dict_ortho=float(dict_ortho),
                    zero_init=self._zero_init_out,
                    alpha_init_std=alpha_init_std,
                    dict_init_std=dict_init_std,
                )
            else:
                self.out_head = RoleSplitOutHead(hidden_dim=self.hidden_dim, roles=role_specs, zero_init=self._zero_init_out)

        self._out_chunk_size = 0

        # MC-dropout (Gal & Ghahramani 2016): inference-time dropout on hypernet
        # hidden activations. Default 0.0 = no-op. Used by RQ3 lever sweep
        # (mc_dropout_rate). Set via set_mc_dropout_rate(); applied in forward()
        # with training=True so it remains active in eval mode.
        self._mc_dropout_rate: float = 0.0

    @property
    def peft_param_count(self) -> int:
        return int(self._peft_param_count)

    def set_mc_dropout_rate(self, rate: float) -> None:
        """Set MC-dropout rate for inference-time uncertainty injection.
        When > 0, F.dropout is applied to the hypernet hidden activations even
        in eval mode (training=True). Default 0.0 is a no-op."""
        self._mc_dropout_rate = float(max(0.0, min(1.0, rate)))

    def set_output_chunk_size(self, n: int) -> None:
        self._out_chunk_size = int(max(0, n))
        if hasattr(self.out_head, "set_out_chunk"):
            try:
                self.out_head.set_out_chunk(self._out_chunk_size)  # type: ignore[attr-defined]
            except Exception:
                pass

    def aux_losses(self) -> Dict[str, torch.Tensor]:
        if hasattr(self.out_head, "aux_losses"):
            try:
                return self.out_head.aux_losses()  # type: ignore[attr-defined]
            except Exception:
                return {}
        return {}

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        if global_features.dim() != 2 or global_features.size(1) != self.global_input_dim:
            raise ValueError(
                "Expected global_features shape (B,{}) got {}".format(self.global_input_dim, tuple(global_features.shape))
            )

        # Defensive: sanitize global features to prevent NaN poisoning.
        g_in = torch.nan_to_num(global_features, nan=0.0, posinf=0.0, neginf=0.0)

        g = self.drop_g(g_in)
        h = self.net(g)
        # MC-dropout (inference-time, training=True keeps it active in eval).
        # No-op when _mc_dropout_rate == 0.0 (the default).
        if self._mc_dropout_rate > 0.0:
            h = torch.nn.functional.dropout(h, p=float(self._mc_dropout_rate),
                                            training=True)
        delta = self.out_head(h)

        sigma = _env_float(ENV_SANITY_DELTA_NOISE, 0.0)
        if sigma > 0.0:
            delta = delta + torch.randn_like(delta) * float(sigma)

        if self.clamp_range is not None:
            delta = torch.clamp(delta, -self.clamp_range, self.clamp_range)
        return delta


# --------------------------------------------------------------------------- #
# PEFT surface discovery
# --------------------------------------------------------------------------- #

def _iter_lora_modules(peft_model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """Yield (name, module) for modules that own LoRA A/B objects."""
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            yield name, mod


def _build_placeholder_plan(peft_model: nn.Module) -> Tuple[List[str], List[int], List[str], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Discover LoRA placeholders and build a stable plan.

    Returns:
      names:       qualified module names (sorted)
      slice_sizes: numel(out_features * r) for each B
      roles:       role bucket per placeholder
      shapesB:     (out_features, r) per placeholder
      shapesA:     (in_features, r) per placeholder
    """
    def _pick_adapter_entry(obj: Any, adapter: str) -> Optional[nn.Module]:
        if obj is None:
            return None
        if isinstance(obj, nn.ModuleDict):
            if adapter in obj:
                return obj[adapter]
            for _, v in obj.items():
                return v
            return None
        if isinstance(obj, dict):
            if adapter in obj:
                return obj[adapter]
            for _, v in obj.items():
                return v
            return None
        if isinstance(obj, nn.Module):
            return obj
        return None

    active = getattr(peft_model, "active_adapter", None) or "default"
    if isinstance(active, (list, tuple)):
        active = active[0] if active else "default"
    active = str(active)

    found: List[Tuple[str, int, str, Tuple[int, int]]] = []
    for full_name, mod in _iter_lora_modules(peft_model):
        A_lin = _pick_adapter_entry(getattr(mod, "lora_A", None), active)
        B_lin = _pick_adapter_entry(getattr(mod, "lora_B", None), active)
        if A_lin is None or B_lin is None:
            continue

        A_w = getattr(A_lin, "weight", None)
        B_w = getattr(B_lin, "weight", None)
        if not isinstance(A_w, torch.Tensor) or not isinstance(B_w, torch.Tensor):
            continue

        # A: [r, in], B: [out, r]
        try:
            rA = int(A_w.shape[0])
            out = int(B_w.shape[0])
            rB = int(B_w.shape[1])
        except Exception:
            continue
        if rA <= 0 or out <= 0 or rA != rB:
            continue

        in_feat = int(A_w.shape[1])
        leaf = full_name.rsplit(".", 1)[-1]
        role = _role_of_leaf(leaf)
        found.append((full_name, out * rA, role, (out, rA), (in_feat, rA)))

    found.sort(key=lambda x: x[0])
    names = [x[0] for x in found]
    slice_sizes = [int(x[1]) for x in found]
    roles = [x[2] for x in found]
    shapesB = [x[3] for x in found]
    shapesA = [x[4] for x in found]
    return names, slice_sizes, roles, shapesB, shapesA


def _make_role_specs_from_placeholders(
    roles_order: List[str],
    ph_roles: List[str],
    ph_sizes: List[int],
    *,
    per_role_rank: Optional[Dict[str, int]] = None,
    per_role_dictk: Optional[Dict[str, int]] = None,
) -> List[RoleSpec]:
    """
    Aggregate placeholder sizes by role, producing RoleSpec in roles_order.

    per_role_rank:  {"qkv": 32, "o_proj": 16, ...} (optional)
    per_role_dictk: {"qkv": 64, "o_proj": 64, ...} (optional)
    """
    agg: Dict[str, int] = {r: 0 for r in roles_order}
    for r, sz in zip(ph_roles, ph_sizes):
        agg[r] = agg.get(r, 0) + int(sz)

    out: List[RoleSpec] = []
    for r in roles_order:
        size = int(agg.get(r, 0))
        if size <= 0:
            continue
        rank = int(per_role_rank.get(r, 0)) if per_role_rank else 0
        dict_k = int(per_role_dictk.get(r, 0)) if per_role_dictk else 0
        out.append(RoleSpec(name=r, size=size, rank=rank, dict_k=dict_k))
    return out


# --------------------------------------------------------------------------- #
# PEFT wrapper with stateless delta injection via hooks
# --------------------------------------------------------------------------- #

class PEFTHypernetModel(nn.Module):
    """
    Wrapper around a PEFT-LoRA model that injects per-forward δB predicted by a hypernetwork.

    Public attributes expected by training:
      - backbone: base PEFT model
      - hypernet: FlatHypernetwork
      - _last_delta: last flattened delta vector (post scaling/gating/clamp/mask)
      - _hook_touches: how many module hooks contributed in the last forward (debug)
    """
    def __init__(
        self,
        base_peft_model: nn.Module,
        hypernet: FlatHypernetwork,
        *,
        clamp_range: float = 0.02,
        global_columns: Optional[List[str]] = None,
        group_scales: Optional[Dict[str, float]] = None,
        use_layer_context: bool = False,
        ctx_in_dim: Optional[int] = None,
        ctx_embed_dim: int = 32,
        ctx_init_scale: float = 0.05,
        emit_both: bool = False,
    ):
        super().__init__()
        self.backbone = base_peft_model
        self.hypernet = hypernet

        self.global_columns = list(global_columns or [])
        self.group_scales = dict(group_scales or {})
        self.clamp_range = float(max(0.0, clamp_range))
        self._emit_both = bool(emit_both)

        # Placeholder plan (stable)
        names, slice_sizes, roles, shapesB, shapesA = _build_placeholder_plan(self.backbone)
        if not names:
            raise RuntimeError("No LoRA placeholders found on the PEFT model.")

        # Ordered map name -> module, aligned with plan
        named_lookup = dict(self.backbone.named_modules())
        ph_map: "OrderedDict[str, nn.Module]" = OrderedDict()
        kept_sizes: List[int] = []
        kept_roles: List[str] = []
        kept_shapes_B: List[Tuple[int, int]] = []
        kept_shapes_A: List[Tuple[int, int]] = []

        for nm, sz, role, shpB, shpA in zip(names, slice_sizes, roles, shapesB, shapesA):
            mod = named_lookup.get(nm, None)
            if mod is None:
                raise RuntimeError("Placeholder '{}' from plan is missing in named_modules().".format(nm))
            if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
                raise RuntimeError("Module '{}' does not expose lora_A/lora_B as expected.".format(nm))
            ph_map[nm] = mod
            kept_sizes.append(int(sz))
            kept_roles.append(str(role))
            kept_shapes_B.append(tuple(shpB))
            kept_shapes_A.append(tuple(shpA))

        self._placeholders = ph_map
        self._slice_sizes = kept_sizes          # B sizes: [out_i * r_i, ...]
        self._placeholder_roles = kept_roles
        self._placeholder_shapes_B = kept_shapes_B
        self._placeholder_shapes_A = kept_shapes_A
        self._slice_sizes_A = [int(inf * r) for (inf, r) in kept_shapes_A]

        # Surface size sanity: hypernet output must match discovered LoRA surface
        P_B = int(sum(self._slice_sizes))
        P_A = int(sum(self._slice_sizes_A)) if self._emit_both else 0
        P_surface = P_B + P_A
        if int(self.hypernet.peft_param_count) != P_surface:
            raise ValueError(
                f"Hypernet peft_param_count ({int(self.hypernet.peft_param_count)}) != "
                f"discovered LoRA surface ({P_surface}) [emit_both={self._emit_both}, P_B={P_B}, P_A={P_A}]."
            )
        if len(self._slice_sizes) != len(self._placeholders):
            raise RuntimeError(
                f"Placeholder count mismatch: slice_sizes={len(self._slice_sizes)} vs placeholders={len(self._placeholders)}."
            )

        # Hooks
        self._hook_handles: List[Any] = []
        self._delta_A_for_forward: Optional[List[torch.Tensor]] = None
        for idx, (_name, mod) in enumerate(self._placeholders.items()):
            self._hook_handles.append(mod.register_forward_hook(self._make_injection_hook(idx)))

        # Optional layer-context gating
        self._use_ctx = bool(use_layer_context)
        self._ctx_embed_dim = int(max(1, ctx_embed_dim))
        if self._use_ctx:
            if ctx_in_dim is None or int(ctx_in_dim) <= 0:
                raise ValueError("use_layer_context=True requires ctx_in_dim > 0")
            self.ctx_proj = nn.Linear(int(ctx_in_dim), self._ctx_embed_dim, bias=False)
            nn.init.normal_(self.ctx_proj.weight, mean=0.0, std=float(ctx_init_scale))
            self.layer_emb = nn.Parameter(torch.empty(len(self._placeholders), self._ctx_embed_dim))
            nn.init.normal_(self.layer_emb, mean=0.0, std=float(ctx_init_scale))
        else:
            self.ctx_proj = None
            self.layer_emb = None

        # Per-forward caches (support grad checkpointing recomputation)
        self._delta_for_forward: Optional[List[torch.Tensor]] = None
        self._g_for_forward: Optional[torch.Tensor] = None
        self._row_mask_for_forward: Optional[torch.Tensor] = None  # float mask [B,1] or None
        self._force_zero_flag: bool = False

        self._last_delta: Optional[torch.Tensor] = None
        self._hook_touches: int = 0

        # Elastic-LoRA rank mask (RQ3 lever sweep). When set to int < trained
        # LoRA rank, the trailing rank columns of A and rank rows of B in the
        # emitted delta tensors are zeroed. None (default) = no mask.
        # Inference-time only; does not affect _last_delta L2 contributions.
        self._rank_mask: Optional[int] = None

        # Paper 5 H4a/H4b causal LoRA decomposition (2026-05-11). When set,
        # zero out the per-site delta for sites whose block index is in
        # _block_mask OR whose role is in _role_mask. Both inference-time
        # only; do not affect training. None (default) = no mask.
        #
        # _site_to_block: list[int]   site_idx -> decoder block number
        # _site_to_role:  list[str]   site_idx -> role name ("qkv","o_proj",
        #                              "mlp_up","mlp_down" or similar)
        # Parsed from self._placeholders keys at init time below.
        self._block_mask: Optional[set] = None
        self._role_mask: Optional[set] = None
        self._site_to_block: List[int] = []
        self._site_to_role: List[str] = []
        for name in self._placeholders.keys():
            # Parse "<...>.layers.<idx>.<...>.<role>.lora_B.default" pattern
            block_idx = -1
            role_name = "unknown"
            try:
                parts = name.split(".")
                for k, tok in enumerate(parts):
                    if tok == "layers" and k + 1 < len(parts):
                        block_idx = int(parts[k + 1])
                        break
                # Role: walk parts and pick the last submodule keyword we
                # recognize before the lora_B suffix.
                lower = name.lower()
                if "query_key_value" in lower or "q_proj" in lower or "k_proj" in lower or "v_proj" in lower:
                    role_name = "qkv"
                elif "o_proj" in lower or ("attention" in lower and ".dense" in lower):
                    role_name = "o_proj"
                elif "dense_h_to_4h" in lower or "gate_proj" in lower or "up_proj" in lower:
                    role_name = "mlp_up"
                elif "dense_4h_to_h" in lower or "down_proj" in lower:
                    role_name = "mlp_down"
                elif "mlp" in lower:
                    role_name = "mlp"
            except Exception:
                pass
            self._site_to_block.append(block_idx)
            self._site_to_role.append(role_name)

    # -------------------- lifecycle --------------------

    def set_rank_mask(self, rank: Optional[int] = None) -> None:
        """Set elastic-LoRA rank mask for inference. When `rank` is an int
        less than the trained LoRA rank, the trailing rank columns of A and
        the trailing rank rows of B in the emitted per-site delta tensors are
        zeroed. Pass None to disable. Mirrors the Pythia-160M ModelLoRAQKV
        semantics."""
        self._rank_mask = int(rank) if rank is not None else None

    def set_block_mask(self, block_indices: Optional[List[int]]) -> None:
        """Paper 5 H4a: zero the per-site delta for any site whose decoder
        block index is in ``block_indices``. Pass None or [] to disable.
        Inference-time only; does not affect training. Per-site mapping is
        built from self._placeholders at __init__ time.

        Example:
            model.set_block_mask([0, 1, 2])  # ablate deltas at blocks 0-2
        """
        if block_indices is None:
            self._block_mask = None
        else:
            self._block_mask = {int(b) for b in block_indices}

    def set_role_mask(self, roles: Optional[List[str]]) -> None:
        """Paper 5 H4b: zero the per-site delta for any site whose role is
        in ``roles``. Accepted values: "qkv", "o_proj", "mlp", "mlp_up",
        "mlp_down". Pass None or [] to disable.

        Example:
            model.set_role_mask(["qkv"])  # ablate δ at all QKV sites
            model.set_role_mask(["o_proj"])  # ablate δ at all O-projection sites
        """
        if roles is None:
            self._role_mask = None
        else:
            self._role_mask = {str(r).strip().lower() for r in roles if r}

    def remove_hooks(self) -> None:
        """Remove injection hooks (useful for cleanup in interactive sessions)."""
        for h in getattr(self, "_hook_handles", []) or []:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []

    # -------------------- injection hook --------------------

    def _make_injection_hook(self, idx: int):
        """
        Add y_extra = scaling * (A(dropout(x)) @ δBᵀ) to the module output.

        δB is predicted by the hypernetwork per-forward and cached in self._delta_for_forward.
        For gradient checkpoint recomputation, δB can be rebuilt lazily from saved g (+ row mask).
        """
        def _pick_adapter_parts(mod: nn.Module) -> Tuple[Optional[nn.Module], Optional[nn.Module], float]:
            active = getattr(mod, "active_adapter", None)
            if active is None:
                active = getattr(self.backbone, "active_adapter", None)
            if isinstance(active, (list, tuple)):
                active = active[0] if active else "default"
            if not active:
                active = "default"
            active = str(active)

            def _resolve_entry(raw: Any) -> Optional[nn.Module]:
                if raw is None:
                    return None
                if isinstance(raw, nn.ModuleDict):
                    if active in raw:
                        return raw[active]
                    for _, v in raw.items():
                        return v
                    return None
                if isinstance(raw, dict):
                    if active in raw:
                        return raw[active]
                    for _, v in raw.items():
                        return v
                    return None
                if isinstance(raw, nn.Module):
                    return raw
                return None

            A = _resolve_entry(getattr(mod, "lora_A", None))
            B = _resolve_entry(getattr(mod, "lora_B", None))
            D = _resolve_entry(getattr(mod, "lora_dropout", None))

            scl = getattr(mod, "scaling", 1.0)
            if isinstance(scl, dict):
                try:
                    scaling = float(scl[active]) if active in scl else float(next(iter(scl.values())))
                except Exception:
                    scaling = 1.0
            else:
                try:
                    scaling = float(scl)
                except Exception:
                    scaling = 1.0

            return A, B, D, scaling

        def _coerce_output_tensor(out: Any) -> Tuple[Optional[torch.Tensor], Optional[Tuple[Any, ...]]]:
            """
            Some modules might return tuples. Return (tensor, wrapper_tuple_or_None).
            If wrapper_tuple_or_None is not None, caller should reconstruct the tuple.
            """
            if isinstance(out, torch.Tensor):
                return out, None
            if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                return out[0], out
            return None, None

        def _diag(reason: str) -> None:
            # One-shot per-forward diagnostic logger. Records the FIRST
            # bailout reason seen during a forward and decrements the
            # counter so later hooks don't spam the log.
            try:
                if getattr(self, "_hook_diag_remaining", 0) > 0:
                    self._hook_diag_last_reason = reason
                    self._hook_diag_remaining -= 1
                    try:
                        import logging as _lg
                        _lg.getLogger(__name__).warning(
                            "[inject-hook bailout] idx=%d reason=%s "
                            "delta_for_forward_is_list=%s len=%s this_none=%s "
                            "g_is_None=%s force_zero=%s",
                            int(idx), reason,
                            isinstance(self._delta_for_forward, list),
                            (len(self._delta_for_forward)
                             if isinstance(self._delta_for_forward, list) else "n/a"),
                            (self._delta_for_forward[idx] is None
                             if (isinstance(self._delta_for_forward, list)
                                 and idx < len(self._delta_for_forward)) else "n/a"),
                            (self._g_for_forward is None),
                            bool(getattr(self, "_force_zero_flag", False)),
                        )
                    except Exception:
                        pass
            except Exception:
                pass

        def _hook(mod: nn.Module, inputs: Tuple[Any, ...], output: Any):
            out_tensor, out_tuple = _coerce_output_tensor(output)
            if out_tensor is None:
                _diag("out_tensor_is_None")
                return output
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                _diag("inputs_empty_or_not_tensor")
                return output

            # Get delta for this hook: use pre-computed cache (initial forward)
            # or recompute from features (backward recomputation).
            delta_B = None
            delta_A = None
            if (self._delta_for_forward is not None
                    and idx < len(self._delta_for_forward)
                    and self._delta_for_forward[idx] is not None):
                delta_B = self._delta_for_forward[idx]
                if (self._delta_A_for_forward is not None
                        and idx < len(self._delta_A_for_forward)):
                    delta_A = self._delta_A_for_forward[idx]
            else:
                # Fallback: recompute deltas from cached features.
                # Each hook independently recomputes — no cross-segment
                # caching — so each checkpoint segment gets its own
                # computation graph for correct gradient flow.
                g = self._g_for_forward
                if g is None:
                    _diag("fallback_g_is_None")
                    return output
                try:
                    delta_B_list, delta_A_list, _ = self._emit_delta_parts(
                        g,
                        force_zero=self._force_zero_flag,
                        row_mask=self._row_mask_for_forward,
                    )
                    delta_B = delta_B_list[idx]
                    if delta_A_list is not None:
                        delta_A = delta_A_list[idx]
                except Exception as _e:
                    _diag(f"fallback_emit_exception:{type(_e).__name__}")
                    # torch.utils.checkpoint signals "stop recomputing" via the
                    # private _StopRecomputationError. Swallowing it leaves the
                    # checkpoint state machine's target_frame.early_stop flag
                    # latched on, which trips an AssertionError when the next
                    # segment starts recomputation. Let it propagate so PyTorch
                    # can complete its early-stop sequence cleanly.
                    if type(_e).__name__ == "_StopRecomputationError":
                        raise
                    return output

            if delta_B is None:
                _diag("delta_B_is_None_after_resolve")
                return output
            delta_B = torch.nan_to_num(delta_B, nan=0.0, posinf=0.0, neginf=0.0)

            # Paper 5 H4a/H4b causal masking: if this site's block index is
            # in _block_mask OR this site's role is in _role_mask, zero the
            # delta at this site only (others active). delta_A is also zeroed
            # so the emit_both x@δA^T@B^T path collapses for masked sites.
            _masked_block = (self._block_mask is not None
                             and idx < len(self._site_to_block)
                             and self._site_to_block[idx] in self._block_mask)
            _masked_role = (self._role_mask is not None
                            and idx < len(self._site_to_role)
                            and self._site_to_role[idx] in self._role_mask)
            if _masked_block or _masked_role:
                delta_B = torch.zeros_like(delta_B)
                if delta_A is not None:
                    delta_A = torch.zeros_like(delta_A)

            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.dim() < 2:
                _diag("input_x_not_tensor_or_dim_lt_2")
                return output

            A, B_lin, D, scaling = _pick_adapter_parts(mod)
            if A is None:
                _diag(f"A_is_None_active='{getattr(mod, 'active_adapter', None)}'_"
                      f"has_lora_A={hasattr(mod, 'lora_A')}_"
                      f"lora_A_type={type(getattr(mod, 'lora_A', None)).__name__}")
                return output

            # align batch if needed
            if x.shape[0] != delta_B.shape[0]:
                if delta_B.shape[0] == 1:
                    delta_B = delta_B.expand(x.shape[0], -1, -1)
                else:
                    delta_B = delta_B[:1].expand(x.shape[0], -1, -1)

            try:
                # Compute a = A(dropout(x)) without tracking grads through backbone/LoRA params.
                with torch.no_grad():
                    a = x
                    if D is not None:
                        a = D(a)
                    # PEFT LoRA adapters default to FP32 while the backbone is
                    # loaded FP16/BF16 at inference. F.linear(a, A.weight) then
                    # raises "mat1/mat2 dtype mismatch" and the hook silently
                    # bails via the outer try/except, zeroing the delta path.
                    # Cast a to A's weight dtype before the projection.
                    try:
                        A_w_dtype = next(A.parameters()).dtype
                        if a.dtype != A_w_dtype:
                            a = a.to(A_w_dtype)
                    except StopIteration:
                        pass
                    a = A(a)
                a = a.detach()

                # Match dtypes for the contraction.
                if a.dtype != delta_B.dtype:
                    a = a.to(delta_B.dtype)

                # B-path: a @ δB^T → [B, ..., out]
                y_B = torch.einsum("b...r,bor->b...o", a, delta_B)

                # A-path (emit_both): x @ δA^T @ B^T → [B, ..., out]
                # Computed in FP32 via torch.bmm/matmul to avoid the BF16 SM90
                # TMA/WGMMA kernel path that SIGILL'd on the GPU node across every
                # image+driver combo (see Dockerfile.gpu-node_fail.log §23).
                # Einsum "b...i,bir->b...r" dispatched to a Hopper batched GEMM
                # whose opcode was not executable on the GPU node's PyTorch 2.7
                # build under either 570.124.06 or 595.58.03. Bmm+FP32 routes
                # through plain cuBLAS and is the known-good path.
                y_A = None
                if delta_A is not None and B_lin is not None:
                    try:
                        x_det = x.detach()
                        if delta_A.shape[0] != x_det.shape[0]:
                            if delta_A.shape[0] == 1:
                                delta_A = delta_A.expand(x_det.shape[0], -1, -1)
                            else:
                                delta_A = delta_A[:1].expand(x_det.shape[0], -1, -1)
                        # Flatten seq/... into a single middle dim for bmm:
                        #   x_det: [B, *S, I] -> x2d: [B, N, I] where N = prod(S)
                        B_sz = x_det.shape[0]
                        I_sz = x_det.shape[-1]
                        mid_shape = x_det.shape[1:-1]           # the ... part
                        x2d = x_det.reshape(B_sz, -1, I_sz)
                        # FP32 compute for dA path (avoids Hopper BF16 batched
                        # GEMM path that was the SIGILL source on SM90).
                        x32 = x2d.to(torch.float32)
                        dA32 = delta_A.to(torch.float32)
                        # y_proj2d: [B, N, r] = bmm([B,N,I], [B,I,r])
                        y_proj2d = torch.bmm(x32, dA32)
                        B_w32 = B_lin.weight.detach().to(torch.float32)
                        # y_A2d: [B, N, out] = y_proj2d @ B_w32^T
                        y_A2d = torch.matmul(y_proj2d, B_w32.t())
                        # Restore middle shape and cast back to y_B's dtype.
                        y_A = y_A2d.reshape(B_sz, *mid_shape, -1).to(y_B.dtype)
                    except Exception:
                        y_A = None

                y_extra = y_B if y_A is None else (y_B + y_A)
                y_extra = torch.nan_to_num(y_extra, nan=0.0, posinf=0.0, neginf=0.0)
                if y_extra.dtype != out_tensor.dtype:
                    y_extra = y_extra.to(dtype=out_tensor.dtype)

                self._hook_touches += 1
                new_out = out_tensor + (scaling * y_extra)

                if out_tuple is not None:
                    # Reconstruct tuple with updated first element.
                    return (new_out,) + tuple(out_tuple[1:])
                return new_out
            except Exception as _e:
                _diag(f"main_try_exception:{type(_e).__name__}:{str(_e)[:120]}")
                # See note above: do not swallow checkpoint's early-stop signal.
                if type(_e).__name__ == "_StopRecomputationError":
                    raise
                return output

        return _hook

    # -------------------- helper API --------------------

    def set_output_chunk_size(self, n: int) -> None:
        try:
            self.hypernet.set_output_chunk_size(int(n))
        except Exception:
            pass

    def set_runtime_clamp(self, v: float) -> None:
        """
        Update the clamp used at injection time. Hypernet emission clamp is left disabled
        to avoid double-clamping; only runtime injection clamp is applied.
        """
        self.clamp_range = float(max(0.0, v))
        try:
            self.hypernet.clamp_range = None
        except Exception:
            pass

    def aux_losses(self) -> Dict[str, torch.Tensor]:
        return self.hypernet.aux_losses()

    def context_parameters(self) -> List[torch.nn.Parameter]:
        """
        Return layer-context trainables if enabled (useful for optimizer param groups).
        """
        if not self._use_ctx:
            return []
        params: List[torch.nn.Parameter] = []
        if self.ctx_proj is not None:
            params.extend(list(self.ctx_proj.parameters()))
        if self.layer_emb is not None:
            params.append(self.layer_emb)
        return params

    def detach_placeholder_state(self) -> Dict[str, torch.Tensor]:
        """
        Export LoRA A/B tensors from the PEFT surface for inspection/debugging.
        """
        state: Dict[str, torch.Tensor] = {}

        def _active_adapter_of(mod: nn.Module) -> str:
            a = getattr(mod, "active_adapter", None)
            if isinstance(a, (list, tuple)):
                a = a[0] if a else None
            if a:
                return str(a)
            a = getattr(self.backbone, "active_adapter", None)
            if isinstance(a, (list, tuple)):
                a = a[0] if a else None
            return str(a) if a else "default"

        def _pick_linear(obj: Any, key: str) -> Optional[nn.Module]:
            if obj is None:
                return None
            if isinstance(obj, nn.ModuleDict):
                if key in obj:
                    return obj[key]
                for _, v in obj.items():
                    return v
                return None
            if isinstance(obj, dict):
                if key in obj:
                    return obj[key]
                for _, v in obj.items():
                    return v
                return None
            if isinstance(obj, nn.Module):
                return obj
            return None

        for name, m in self._placeholders.items():
            try:
                active = _active_adapter_of(m)
                A_lin = _pick_linear(getattr(m, "lora_A", None), active)
                B_lin = _pick_linear(getattr(m, "lora_B", None), active)
                A_w = getattr(A_lin, "weight", None) if A_lin is not None else None
                B_w = getattr(B_lin, "weight", None) if B_lin is not None else None
                if isinstance(A_w, torch.Tensor):
                    state[name + ".lora_A.weight"] = A_w.detach().cpu().clone()
                if isinstance(B_w, torch.Tensor):
                    state[name + ".lora_B.weight"] = B_w.detach().cpu().clone()
            except Exception:
                continue
        return state

    def _normalize_row_mask(
        self,
        global_mask: Optional[torch.Tensor],
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Normalize global_mask to a float tensor of shape [B,1] with values in {0,1}.
        Returns None if mask is not usable.
        """
        if global_mask is None or not isinstance(global_mask, torch.Tensor):
            return None
        try:
            m = global_mask
            if m.dim() == 2 and m.size(1) == 1:
                m = m.view(-1)
            elif m.dim() == 0:
                m = m.view(1)
            m = m.to(device=device)
            # Non-zero => keep
            m = (m != 0).to(dtype=dtype).view(-1, 1)
            if m.shape[0] == 1 and batch_size > 1:
                m = m.expand(batch_size, 1)
            if m.shape[0] != batch_size:
                return None
            return m
        except Exception:
            return None

    def _compute_gates(self, g: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._use_ctx:
            return None
        if _is_truthy(os.environ.get(ENV_DISABLE_GATES, "0")):
            return None
        assert self.ctx_proj is not None and self.layer_emb is not None
        ctx = self.ctx_proj(g)                          # [B, E]
        raw = ctx @ self.layer_emb.t()                  # [B, J]
        raw = raw / max(1.0, math.sqrt(float(self._ctx_embed_dim)))
        scale = _env_float(ENV_GATES_SCALE, 1.0)
        scale = float(min(1.0, max(0.0, scale)))
        return torch.tanh(raw) * scale                  # [B, J]

    def _apply_group_scales(self, parts: List[torch.Tensor]) -> List[torch.Tensor]:
        if not parts:
            return parts
        gain = float(_env_float(ENV_DELTA_GAIN, 1.0))

        if not self.group_scales:
            return [p * gain for p in parts] if gain != 1.0 else parts

        out: List[torch.Tensor] = []
        for i, p in enumerate(parts):
            role = self._placeholder_roles[i]
            s = float(self.group_scales.get(role, 1.0))
            out.append(p * (s * gain))
        return out

    def _emit_delta_parts(
        self,
        g: torch.Tensor,
        *,
        force_zero: bool,
        row_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], torch.Tensor]:
        """
        Emit per-placeholder δB (and optionally δA) tensors and the flattened δ vector.

        Returns:
          reshaped_B: list of [B, out_i, r] tensors (one per LoRA site)
          reshaped_A: list of [B, in_i, r] tensors when emit_both=True, else None
          delta_flat:  [B, P] flattened delta (B-only or A+B concatenated)

        row_mask:
          Optional float mask [B,1] with {0,1}. Applied after scaling/gating/clamp,
          zeroing deltas for masked rows. This is crucial when `global_mask` is
          used together with gradient checkpointing (hook recomputation).
        """
        if g.dim() != 2 or g.size(1) != self.hypernet.global_input_dim:
            raise ValueError("global_features must be shape (B, {}).".format(self.hypernet.global_input_dim))

        Bsz = int(g.shape[0])
        P_B = int(sum(self._slice_sizes))
        P_A = int(sum(self._slice_sizes_A)) if self._emit_both else 0
        P = P_B + P_A

        if force_zero:
            delta = g.new_zeros(Bsz, P)
        else:
            delta = self.hypernet(g)  # [B, P]

        # ---- Split and process B deltas (first P_B elements) ----
        delta_B_raw = delta[:, :P_B]
        parts_B: List[torch.Tensor] = []
        cur = 0
        for sz in self._slice_sizes:
            parts_B.append(delta_B_raw[:, cur:cur + sz])
            cur += sz

        parts_B = self._apply_group_scales(parts_B)

        gates = self._compute_gates(g)
        if gates is not None:
            parts_B = [p * gates[:, i:i+1] for i, p in enumerate(parts_B)]

        parts_B = [torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0) for p in parts_B]

        cr = float(max(0.0, self.clamp_range))
        if cr > 0.0:
            for i in range(len(parts_B)):
                parts_B[i] = torch.clamp(parts_B[i], -cr, cr)

        # ---- Split and process A deltas (next P_A elements) ----
        parts_A: Optional[List[torch.Tensor]] = None
        if self._emit_both:
            delta_A_raw = delta[:, P_B:]
            parts_A = []
            cur = 0
            for sz in self._slice_sizes_A:
                parts_A.append(delta_A_raw[:, cur:cur + sz])
                cur += sz

            parts_A = self._apply_group_scales(parts_A)
            if gates is not None:
                parts_A = [p * gates[:, i:i+1] for i, p in enumerate(parts_A)]
            parts_A = [torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0) for p in parts_A]
            if cr > 0.0:
                for i in range(len(parts_A)):
                    parts_A[i] = torch.clamp(parts_A[i], -cr, cr)

        # ---- Row mask (global_mask) ----
        if (row_mask is not None) and isinstance(row_mask, torch.Tensor):
            try:
                rm = row_mask
                if rm.dim() == 2 and rm.size(1) == 1:
                    rm2 = rm
                elif rm.dim() == 1:
                    rm2 = rm.view(-1, 1)
                else:
                    rm2 = None
                if rm2 is not None:
                    if rm2.shape[0] == 1 and Bsz > 1:
                        rm2 = rm2.expand(Bsz, 1)
                    if rm2.shape[0] == Bsz:
                        rm2 = rm2.to(device=parts_B[0].device, dtype=parts_B[0].dtype)
                        parts_B = [p * rm2 for p in parts_B]
                        if parts_A is not None:
                            parts_A = [p * rm2 for p in parts_A]
            except Exception:
                pass

        # ---- Flatten for auxiliary losses ----
        # Chunked cat: reduces fan-in per kernel launch. 112-way cats on SM90
        # via the CUDA compat shim have produced SIGILL at r=32 emit_both;
        # chunks of 8 change the kernel tile and avoid that specific crash.
        def _chunked_cat(tensors, dim=1, chunk=8):
            if len(tensors) <= chunk:
                return torch.cat(tensors, dim=dim) if len(tensors) > 1 else tensors[0]
            groups = [torch.cat(tensors[i:i + chunk], dim=dim)
                      for i in range(0, len(tensors), chunk)]
            return torch.cat(groups, dim=dim) if len(groups) > 1 else groups[0]

        delta_flat_B = _chunked_cat(parts_B, dim=1)
        if parts_A is not None:
            delta_flat_A = _chunked_cat(parts_A, dim=1)
            delta_flat = torch.cat([delta_flat_B, delta_flat_A], dim=1)
        else:
            delta_flat = delta_flat_B

        # ---- Reshape B to [B, out, r] ----
        reshaped_B: List[torch.Tensor] = []
        for p, (out_f, r) in zip(parts_B, self._placeholder_shapes_B):
            reshaped_B.append(p.view(Bsz, int(out_f), int(r)))

        # ---- Reshape A to [B, in, r] (if emit_both) ----
        reshaped_A: Optional[List[torch.Tensor]] = None
        if parts_A is not None:
            reshaped_A = []
            for p, (in_f, r) in zip(parts_A, self._placeholder_shapes_A):
                reshaped_A.append(p.view(Bsz, int(in_f), int(r)))

        # ---- Elastic-LoRA rank mask (RQ3 lever) ----
        # Zero the trailing (r - mask) rank columns/rows of each per-site
        # delta tensor when an inference-time mask is set. Default no-op.
        rm = self._rank_mask
        if rm is not None and rm > 0:
            for i in range(len(reshaped_B)):
                _, out_f, r = reshaped_B[i].shape
                if int(rm) < int(r):
                    reshaped_B[i] = reshaped_B[i].clone()
                    reshaped_B[i][:, :, int(rm):] = 0.0
            if reshaped_A is not None:
                for i in range(len(reshaped_A)):
                    _, in_f, r = reshaped_A[i].shape
                    if int(rm) < int(r):
                        reshaped_A[i] = reshaped_A[i].clone()
                        reshaped_A[i][:, :, int(rm):] = 0.0

        return reshaped_B, reshaped_A, delta_flat

    def _is_grad_ckpt_enabled(self) -> bool:
        def _flag_on(obj: Any) -> bool:
            try:
                v = getattr(obj, "gradient_checkpointing", None)
                if isinstance(v, bool) and v:
                    return True
            except Exception:
                pass
            try:
                v2 = getattr(obj, "is_gradient_checkpointing", None)
                if isinstance(v2, bool) and v2:
                    return True
            except Exception:
                pass
            try:
                cfg = getattr(obj, "config", None)
                if cfg is not None and bool(getattr(cfg, "gradient_checkpointing", False)):
                    return True
            except Exception:
                pass
            return False

        try:
            dec = _resolve_decoder_backbone(self.backbone)
        except Exception:
            dec = self.backbone

        if _flag_on(self.backbone) or _flag_on(dec):
            return True

        try:
            for m in dec.modules():
                if _flag_on(m):
                    return True
        except Exception:
            pass
        return False

    # -------------------- forward --------------------

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        global_features: torch.Tensor,
        global_mask: Optional[torch.Tensor] = None,
        return_hidden_only: bool = True,
        deltas_applied: bool = True,
        force_zero_delta: bool = False,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        **_unused: Any,
    ):
        """
        Forward through the decoder/backbone with per-forward delta injection.

        If global_mask is provided (shape [B] or [B,1] or [1]), rows with mask==0
        are forced to use δ=0 (useful for missing-author fallbacks). The mask is
        also saved and reapplied if hooks need to recompute δ during grad-ckpt.
        """
        self._hook_touches = 0
        # One-shot bailout diagnostic: the injection hook logs its first
        # early-return reason (A is None, delta_for_forward[idx] is None,
        # exception in emit, etc.) when this counter is > 0, then decrements.
        # Set to 1 per forward so we get exactly one diag line per forward.
        self._hook_diag_remaining = 1
        self._hook_diag_last_reason = None

        force_zero = bool(force_zero_delta or (not deltas_applied))

        # Save for checkpoint recomputation
        self._g_for_forward = global_features
        self._force_zero_flag = force_zero

        row_mask = self._normalize_row_mask(
            global_mask,
            batch_size=int(global_features.shape[0]),
            device=global_features.device,
            dtype=torch.float32,
        )
        self._row_mask_for_forward = row_mask

        reshaped_B, reshaped_A, delta_vec = self._emit_delta_parts(global_features, force_zero=force_zero, row_mask=row_mask)

        self._delta_for_forward = reshaped_B
        self._delta_A_for_forward = reshaped_A
        self._last_delta = delta_vec

        decoder = _resolve_decoder_backbone(self.backbone)
        need_hidden = bool(return_hidden_only or output_hidden_states)

        outs = decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_hidden_states=need_hidden,
            return_dict=True,
        )

        # Always clear pre-computed delta cache. During backward
        # recomputation, hooks independently recompute deltas from
        # _g_for_forward, creating separate computation graphs per
        # checkpoint segment (avoids "backward through graph second time").
        self._delta_for_forward = None
        self._delta_A_for_forward = None
        if not self._is_grad_ckpt_enabled():
            # Without grad ckpt, hooks are never re-invoked — clear everything
            self._g_for_forward = None
            self._row_mask_for_forward = None
            self._force_zero_flag = False

        if return_hidden_only:
            if hasattr(outs, "last_hidden_state") and outs.last_hidden_state is not None:
                return outs.last_hidden_state
            if hasattr(outs, "hidden_states") and outs.hidden_states is not None:
                return outs.hidden_states[-1]
            if isinstance(outs, dict):
                if "last_hidden_state" in outs and outs["last_hidden_state"] is not None:
                    return outs["last_hidden_state"]
                if "hidden_states" in outs and outs["hidden_states"] is not None:
                    return outs["hidden_states"][-1]
            if isinstance(outs, tuple) and len(outs) > 0 and isinstance(outs[0], torch.Tensor) and outs[0].dim() == 3:
                return outs[0]
            raise RuntimeError("Decoder did not return hidden states.")
        return outs


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

# Import-time sanity guard: catch accidental dedents of critical methods
try:
    _ = getattr(PEFTHypernetModel, "_make_injection_hook")
    _ = getattr(PEFTHypernetModel, "forward")
except Exception as _e:
    raise RuntimeError(
        "hypernetwork_structure_10000.py: PEFTHypernetModel critical methods are missing. "
        "Check indentation of _make_injection_hook and forward."
    ) from _e


def make_from_dims(
    *,
    peft_model: nn.Module,
    global_dim: int,
    instance_dim: int = 0,                    # unused (kept for signature compatibility)
    hidden_dim: int = 512,
    mode: str = "flat",                       # unused (compat)
    clamp_range: Optional[float] = None,      # emission clamp (rare); injection clamp via inject_clamp
    activation: str = "relu",
    dropout_p_global: float = 0.0,
    dropout_p_instance: float = 0.0,          # unused
    inject_clamp: float = 0.02,
    global_columns: Optional[List[str]] = None,
    instance_columns: Optional[List[str]] = None,  # unused
    group_scales: Optional[Dict[str, float]] = None,
    enforce_gstat_only_in_flat: bool = True,  # unused (compat)
    use_layer_context: bool = False,
    ctx_in_dim: Optional[int] = None,
    ctx_embed_dim: int = 32,
    ctx_init_scale: float = 0.05,
    hyper_out_rank: int = 0,
    head_mode: str = "single",
    role_specs: Optional[List[RoleSpec]] = None,
    dict_mode: bool = False,
    dict_k_global: int = 0,
    dict_k_per_role: Optional[Dict[str, int]] = None,
    alpha_l1: float = 0.0,
    dict_ortho: float = 0.0,
    zero_init_out: Optional[bool] = None,
    alpha_init_std: Optional[float] = None,
    dict_init_std: Optional[float] = None,
    emit_both: bool = False,
) -> PEFTHypernetModel:
    """
    Construct a FlatHypernetwork and wrap it with PEFTHypernetModel.

    Notes:
      - If head_mode='multi' and role_specs is None, role sizes are derived by
        aggregating discovered LoRA placeholders into roles {qkv,o_proj,mlp,other}.
      - If dict_mode=True and head_mode='multi', dict_k_per_role can specify
        per-role dictionary sizes: {"qkv":64, "o_proj":64, "mlp":64, "other":0}.
      - If emit_both=True, the hypernetwork emits deltas for both LoRA A and B
        matrices, roughly doubling the output surface.
    """
    names, slice_sizes, roles, _shapesB, _shapesA = _build_placeholder_plan(peft_model)
    if not names:
        raise RuntimeError("No LoRA placeholders found to define the hypernet output surface.")
    peft_param_count = int(sum(slice_sizes))
    if emit_both:
        peft_param_count += int(sum(in_f * r for (in_f, r) in _shapesA))

    hm = str(head_mode or "").strip().lower()
    if not hm:
        hm = str(os.environ.get(ENV_HEAD_MODE, "single") or "single").strip().lower()
    if hm not in {"single", "multi"}:
        hm = "single"

    # When emit_both, role sizes must include both A and B surfaces
    ph_sizes_for_roles = slice_sizes
    if emit_both:
        ph_sizes_for_roles = [int(sz_b + in_f * r) for sz_b, (in_f, r) in zip(slice_sizes, _shapesA)]

    role_specs_final = list(role_specs or [])
    if hm == "multi" and not role_specs_final:
        roles_order = ["qkv", "o_proj", "mlp", "other"]
        _auto_rank = {r: int(hyper_out_rank) for r in roles_order} if int(hyper_out_rank) > 0 else None
        role_specs_final = _make_role_specs_from_placeholders(
            roles_order=roles_order,
            ph_roles=roles,
            ph_sizes=ph_sizes_for_roles,
            per_role_rank=_auto_rank,
            per_role_dictk=dict_k_per_role,
        )

    hyper = FlatHypernetwork(
        global_input_dim=int(global_dim),
        hidden_dim=int(hidden_dim),
        peft_param_count=int(peft_param_count),
        activation=str(activation),
        dropout_p_global=float(dropout_p_global),
        clamp_range=clamp_range,
        head_mode=hm,
        role_specs=role_specs_final if role_specs_final else None,
        dict_mode=bool(dict_mode),
        dict_k_global=int(dict_k_global),
        hyper_out_rank=int(hyper_out_rank),
        alpha_l1=float(alpha_l1),
        dict_ortho=float(dict_ortho),
        zero_init_out=zero_init_out,
        alpha_init_std=alpha_init_std,
        dict_init_std=dict_init_std,
    )

    if use_layer_context and (ctx_in_dim is None or int(ctx_in_dim) <= 0):
        ctx_in_dim = int(global_dim)

    return PEFTHypernetModel(
        base_peft_model=peft_model,
        hypernet=hyper,
        clamp_range=float(max(0.0, inject_clamp)),
        global_columns=global_columns or [],
        group_scales=group_scales or {},
        use_layer_context=bool(use_layer_context),
        ctx_in_dim=int(ctx_in_dim) if ctx_in_dim is not None else None,
        ctx_embed_dim=int(ctx_embed_dim),
        ctx_init_scale=float(ctx_init_scale),
        emit_both=bool(emit_both),
    )


# --------------------------------------------------------------------------- #
# Smoke test (CLI)
# --------------------------------------------------------------------------- #

def _parse_kv_csv(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    raw = (raw or "").strip()
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def _parse_kv_int_csv(raw: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    raw = (raw or "").strip()
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            out[k] = int(v)
        except Exception:
            continue
    return out


def _load_feature_manifest(path: str) -> Tuple[int, List[str]]:
    with open(path, "r") as f:
        obj = json.load(f)
    g_dim = int(obj.get("g_dim", obj.get("global_dim", 0)) or 0)
    cols = obj.get("global_columns", []) or []
    if not isinstance(cols, list):
        cols = []
    cols = [str(c) for c in cols]
    return g_dim, cols


# -------------------- dummy backbone for smoke test -------------------- #

class _DummyLoraLinear(nn.Module):
    """
    Minimal LoRA-wrapped Linear that mimics the PEFT attributes used by this script:
      - lora_A / lora_B (ModuleDict keyed by adapter name)
      - lora_dropout (ModuleDict)
      - scaling (dict)
      - active_adapter (str)
    """
    def __init__(self, in_features: int, out_features: int, r: int, *, dropout_p: float = 0.0, adapter: str = "default"):
        super().__init__()
        self.base = nn.Linear(int(in_features), int(out_features), bias=False)

        self.lora_A = nn.ModuleDict({adapter: nn.Linear(int(in_features), int(r), bias=False)})
        self.lora_B = nn.ModuleDict({adapter: nn.Linear(int(r), int(out_features), bias=False)})
        self.lora_dropout = nn.ModuleDict({adapter: nn.Dropout(float(dropout_p))})

        # match PEFT: scaling may be per-adapter
        self.scaling = {adapter: 1.0}
        self.active_adapter = adapter

        # init
        nn.init.kaiming_uniform_(self.base.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A[adapter].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[adapter].weight)  # typical LoRA init: B=0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter = self.active_adapter
        A = self.lora_A[adapter]
        B = self.lora_B[adapter]
        D = self.lora_dropout[adapter]
        scl = float(self.scaling.get(adapter, 1.0))
        return self.base(x) + scl * B(A(D(x)))


class _DummyCausalLM(nn.Module):
    """
    Tiny "CausalLM-like" module sufficient for smoke testing:
      - exposes embed_tokens
      - returns last_hidden_state in a dict-like object
      - contains LoRA placeholder leaves (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj)
    """
    def __init__(self, *, vocab_size: int = 256, hidden_size: int = 64, r: int = 8, dropout_p: float = 0.05):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=int(vocab_size))
        self.embed_tokens = nn.Embedding(int(vocab_size), int(hidden_size))

        # attention-ish
        self.q_proj = _DummyLoraLinear(hidden_size, hidden_size, r, dropout_p=dropout_p)
        self.k_proj = _DummyLoraLinear(hidden_size, hidden_size, r, dropout_p=dropout_p)
        self.v_proj = _DummyLoraLinear(hidden_size, hidden_size, r, dropout_p=dropout_p)
        self.o_proj = _DummyLoraLinear(hidden_size, hidden_size, r, dropout_p=dropout_p)

        # mlp-ish
        self.gate_proj = _DummyLoraLinear(hidden_size, hidden_size, r, dropout_p=dropout_p)
        self.up_proj = _DummyLoraLinear(hidden_size, hidden_size, r, dropout_p=dropout_p)
        self.down_proj = _DummyLoraLinear(hidden_size, hidden_size, r, dropout_p=dropout_p)

        self.lm_head = nn.Linear(int(hidden_size), int(vocab_size), bias=False)

        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.kaiming_uniform_(self.lm_head.weight, a=math.sqrt(5))

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: Any,
    ):
        x = self.embed_tokens(input_ids)  # [B,L,H]

        # simple "attention" + "mlp" chain to exercise all placeholders
        h = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        h = self.o_proj(h)
        h = self.down_proj(F.relu(self.up_proj(h)) + self.gate_proj(h))

        if return_dict:
            hs = (h,) if output_hidden_states else None
            return SimpleNamespace(last_hidden_state=h, hidden_states=hs)
        return (h,)


# -------------------- smoke runners -------------------- #

def _run_smoke_dummy(args: argparse.Namespace, *, global_dim: int, cols: List[str]) -> None:
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[str(args.dtype)]

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    backbone = _DummyCausalLM(vocab_size=256, hidden_size=64, r=int(args.lora_r), dropout_p=float(args.lora_dropout)).to(device=device, dtype=dtype)
    backbone.train()

    dict_k_per_role = _parse_kv_int_csv(args.dict_k_per_role)
    group_scales = _parse_kv_csv(args.group_scales)

    hm = str(args.head_mode).strip().lower()
    if not hm:
        hm = str(os.environ.get(ENV_HEAD_MODE, "single") or "single").strip().lower()

    hyper_model = make_from_dims(
        peft_model=backbone,
        global_dim=int(global_dim),
        hidden_dim=int(args.hidden_dim),
        inject_clamp=float(args.inject_clamp),
        head_mode=hm,
        dict_mode=bool(args.dict_mode),
        dict_k_global=int(args.dict_k_global),
        dict_k_per_role=dict_k_per_role if dict_k_per_role else None,
        hyper_out_rank=int(args.hyper_out_rank),
        use_layer_context=bool(args.use_layer_context),
        ctx_embed_dim=int(args.ctx_embed_dim),
        group_scales=group_scales if group_scales else None,
        global_columns=cols if cols else None,
        zero_init_out=args.zero_init_out,
    ).to(device)

    B = int(args.batch)
    L = int(args.seq_len)

    vocab = int(getattr(backbone.config, "vocab_size", 256))
    input_ids = torch.randint(low=0, high=max(1, vocab), size=(B, L), device=device)
    attention_mask = torch.ones((B, L), dtype=torch.long, device=device)

    g = torch.randn((B, int(global_dim)), device=device, dtype=torch.float32)
    # Optionally simulate missing global vectors
    if bool(args.simulate_mask):
        global_mask = torch.ones((B, 1), dtype=torch.int8, device=device)
        global_mask[0] = 0
    else:
        global_mask = None

    hidden = hyper_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_features=g,
        global_mask=global_mask,
        return_hidden_only=True,
        deltas_applied=True,
        force_zero_delta=False,
    )

    # logits sanity
    try:
        out_emb = hyper_model.backbone.get_output_embeddings()
        logits = out_emb(hidden)
        print("[smoke dummy] logits.shape =", tuple(logits.shape))
    except Exception:
        pass

    hidden0 = hyper_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_features=g,
        global_mask=global_mask,
        return_hidden_only=True,
        deltas_applied=False,
        force_zero_delta=True,
    )

    diff = (hidden - hidden0).float().norm().item()
    delta_norm = float(hyper_model._last_delta.float().norm().item()) if hyper_model._last_delta is not None else 0.0

    names, slice_sizes, roles, _shapesB, _shapesA = _build_placeholder_plan(hyper_model.backbone)
    role_counts = dict(Counter(roles))

    print("\nHyperPEFT-LoRA hypernetwork smoke test (DUMMY backbone)")
    print("--------------------------------------------------")
    print("device/dtype        :", str(device), str(dtype))
    print("batch/seq_len       :", B, L)
    print("global_dim          :", int(global_dim))
    print("hidden_dim          :", int(args.hidden_dim))
    print("head_mode           :", hm)
    print("dict_mode           :", bool(args.dict_mode))
    print("zero_init_out       :", bool(args.zero_init_out) if args.zero_init_out is not None else _is_truthy(os.environ.get(ENV_ZERO_INIT_OUT, "0")))
    print("placeholders        :", len(names))
    print("roles               :", role_counts)
    print("peft_param_count    :", sum(slice_sizes))
    print("HN_DELTA_GAIN       :", os.environ.get(ENV_DELTA_GAIN, "1.0"))
    print("inject_clamp        :", float(hyper_model.clamp_range))
    print("delta_norm          :", delta_norm)
    print("hidden.shape        :", tuple(hidden.shape))
    print("hidden_diff_norm    :", diff)
    print("hook_touches        :", int(hyper_model._hook_touches))
    if args.simulate_mask:
        print("simulate_mask       :", True, "(row 0 global_mask=0 => δ=0 for row 0)")

    aux = hyper_model.aux_losses()
    if aux:
        for k, v in aux.items():
            try:
                print("aux_loss {} : {}".format(k, float(v.detach().cpu().item())))
            except Exception:
                pass


def _run_smoke_peft(args: argparse.Namespace, *, global_dim: int, cols: List[str]) -> None:
    # Lazy imports for smoke test
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[str(args.dtype)]

    # Import peft early to fail fast (or allow auto fallback)
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

    tok_id = args.tokenizer_id or args.base_model_id
    tok = AutoTokenizer.from_pretrained(tok_id, token=args.hf_token, local_files_only=bool(args.local_files_only))
    if tok.pad_token_id is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        token=args.hf_token,
        local_files_only=bool(args.local_files_only),
        torch_dtype=dtype,
    )

    targets = [t.strip() for t in str(args.target_modules).split(",") if t.strip()]
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(base, lora_cfg)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    peft_model.to(device)
    peft_model.train()

    dict_k_per_role = _parse_kv_int_csv(args.dict_k_per_role)
    group_scales = _parse_kv_csv(args.group_scales)

    hm = str(args.head_mode).strip().lower()
    if not hm:
        hm = str(os.environ.get(ENV_HEAD_MODE, "single") or "single").strip().lower()

    hyper_model = make_from_dims(
        peft_model=peft_model,
        global_dim=int(global_dim),
        hidden_dim=int(args.hidden_dim),
        inject_clamp=float(args.inject_clamp),
        head_mode=hm,
        dict_mode=bool(args.dict_mode),
        dict_k_global=int(args.dict_k_global),
        dict_k_per_role=dict_k_per_role if dict_k_per_role else None,
        hyper_out_rank=int(args.hyper_out_rank),
        use_layer_context=bool(args.use_layer_context),
        ctx_embed_dim=int(args.ctx_embed_dim),
        group_scales=group_scales if group_scales else None,
        global_columns=cols if cols else None,
        zero_init_out=args.zero_init_out,
    ).to(device)

    # Random inputs
    B = int(args.batch)
    L = int(args.seq_len)
    vocab = int(getattr(peft_model.config, "vocab_size", 32000))
    input_ids = torch.randint(low=0, high=max(1, vocab), size=(B, L), device=device)
    attention_mask = torch.ones((B, L), dtype=torch.long, device=device)
    g = torch.randn((B, int(global_dim)), device=device, dtype=torch.float32)

    hidden = hyper_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_features=g,
        return_hidden_only=True,
        deltas_applied=True,
        force_zero_delta=False,
    )

    # Compute logits with LM head (optional sanity check)
    try:
        out_emb = hyper_model.backbone.get_output_embeddings()
        logits = out_emb(hidden)
        print("[smoke peft] logits.shape =", tuple(logits.shape))
    except Exception:
        pass

    hidden0 = hyper_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_features=g,
        return_hidden_only=True,
        deltas_applied=False,
        force_zero_delta=True,
    )

    diff = (hidden - hidden0).float().norm().item()
    delta_norm = float(hyper_model._last_delta.float().norm().item()) if hyper_model._last_delta is not None else 0.0

    names, slice_sizes, roles, _shapesB, _shapesA = _build_placeholder_plan(hyper_model.backbone)
    role_counts = dict(Counter(roles))

    print("\nHyperPEFT-LoRA hypernetwork smoke test (PEFT backbone)")
    print("-----------------------------------------------")
    print("base_model_id       :", args.base_model_id)
    print("device/dtype        :", str(device), str(dtype))
    print("batch/seq_len       :", B, L)
    print("global_dim          :", int(global_dim))
    print("hidden_dim          :", int(args.hidden_dim))
    print("head_mode           :", hm)
    print("dict_mode           :", bool(args.dict_mode))
    print("zero_init_out       :", bool(args.zero_init_out) if args.zero_init_out is not None else _is_truthy(os.environ.get(ENV_ZERO_INIT_OUT, "0")))
    print("placeholders        :", len(names))
    print("roles               :", role_counts)
    print("peft_param_count    :", sum(slice_sizes))
    print("HN_DELTA_GAIN       :", os.environ.get(ENV_DELTA_GAIN, "1.0"))
    print("inject_clamp        :", float(hyper_model.clamp_range))
    print("delta_norm          :", delta_norm)
    print("hidden.shape        :", tuple(hidden.shape))
    print("hidden_diff_norm    :", diff)
    print("hook_touches        :", int(hyper_model._hook_touches))

    aux = hyper_model.aux_losses()
    if aux:
        for k, v in aux.items():
            try:
                print("aux_loss {} : {}".format(k, float(v.detach().cpu().item())))
            except Exception:
                pass


def _smoke_main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Smoke test for hypernetwork_structure_10000.py")

    ap.add_argument(
        "--smoke_mode",
        type=str,
        default="auto",
        choices=["auto", "peft", "dummy"],
        help="auto: prefer PEFT if available; peft: require peft+transformers; dummy: no external deps.",
    )
    ap.add_argument(
        "--simulate_mask",
        action="store_true",
        help="Dummy-mode only: set global_mask[0]=0 to validate per-row masking path.",
    )

    ap.add_argument("--base_model_id", type=str, default="hf-internal-testing/tiny-random-GPTNeoXForCausalLM")
    ap.add_argument("--tokenizer_id", type=str, default=None)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=32)

    ap.add_argument("--global_dim", type=int, default=136)
    ap.add_argument("--feature_manifest", type=str, default="")

    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--head_mode", type=str, default="")          # default from env if empty
    ap.add_argument("--dict_mode", action="store_true")
    ap.add_argument("--dict_k_global", type=int, default=64)
    ap.add_argument("--dict_k_per_role", type=str, default="qkv=64,o_proj=64,mlp=64,other=0")
    ap.add_argument("--hyper_out_rank", type=int, default=0)

    ap.add_argument("--use_layer_context", action="store_true")
    ap.add_argument("--ctx_embed_dim", type=int, default=32)

    ap.add_argument("--group_scales", type=str, default="")        # e.g. "qkv=1.0,o_proj=1.0,mlp=1.0"
    ap.add_argument("--inject_clamp", type=float, default=0.02)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    ap.add_argument(
        "--zero_init_out",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override HN_ZERO_INIT_HYPER_OUT for the smoke test (default: env-driven).",
    )

    ap.add_argument("--seed", type=int, default=142)
    ap.add_argument("-f", "--f", help=argparse.SUPPRESS)
    args, _unknown = ap.parse_known_args(argv)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    cols: List[str] = []
    global_dim = int(args.global_dim)
    if args.feature_manifest:
        try:
            g_dim, cols = _load_feature_manifest(args.feature_manifest)
            if g_dim > 0:
                global_dim = int(g_dim)
        except Exception:
            cols = []

    mode = str(args.smoke_mode).strip().lower()

    # Prefer PEFT only if available; otherwise fall back to dummy.
    if mode in {"auto", "peft"}:
        try:
            import peft  # noqa: F401
            have_peft = True
        except Exception as e:
            have_peft = False
            if mode == "peft":
                print("[smoke] Requested peft mode but peft is not installed:", e)
                print("[smoke] Install peft (and transformers) or rerun with --smoke_mode dummy.")
                return
            print("[smoke] peft not installed; running dummy smoke test instead.")
            mode = "dummy"

    if mode == "dummy":
        _run_smoke_dummy(args, global_dim=global_dim, cols=cols)
        return

    # mode == "peft"
    try:
        _run_smoke_peft(args, global_dim=global_dim, cols=cols)
    except Exception as e:
        # Avoid raising SystemExit in notebooks/VS Code; print a clean error instead.
        print("[smoke] PEFT smoke test failed:", repr(e))
        if os.environ.get("HN_SMOKE_RAISE", "").strip():
            raise


if __name__ == "__main__":
    _smoke_main()