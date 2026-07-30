#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_hyperlora.py  —  Training Pipeline for HyperPEFT-LoRA over PEFT–LoRA
=====================================================================

This script trains a **flat, global-static hypernetwork** that predicts
per-sample additive offsets δθ over a frozen PEFT–LoRA surface of a
pretrained causal LM (Pythia-1.4B). The backbone stays frozen (BF16).
Only the hypernetwork (and optional layer-context gates)
is updated; the backbone and base LoRA weights are not optimized.

Mathematical Context
--------------------
Let x ∈ ℕ^L be the input token sequence (context ⧺ reply) and y the
next-token labels (−100 where masked). Let g ∈ ℝ^G denote the **global,
persona-level feature vector** for the current user. The backbone has LoRA
modules with matrices (A_j ∈ ℝ^{r_j×d_in}, B_j ∈ ℝ^{d_out×r_j}), scaling s_j.

The hypernetwork H maps features to a flattened offset δθ ∈ ℝ^P, sliced to
match LoRA B_j blocks:
    δθ = H(g),      δB_j ∈ ℝ^{d_out×r_j}  (concatenate over j to form δθ).

During the forward pass, each LoRA module contributes:
    y_j = y_j + s_j · ( A_j(x) (B_j + δB_j)ᵀ )

We optimize a blended objective with teacher-forced CE, a **context-only
boundary** probe to guard against leakage, and a small quadratic penalty
on δθ:
    L = L_TF + λ_ctx · L_CTX + λ_δ · ||δθ||²

• Teacher-forced CE (reply tokens only):
    L_TF = − Σ_t log p(y_t | x_{≤t−1}, g; θ̄, B+δB)

• Context-only boundary CE (first reply token predicted from context only):
    L_CTX = − log p(y_{t₀} | x_{<t₀}, g'; θ̄, B)    with g' ∈ {0, mean(g), g}

A clamp schedule controls δθ magnitude at injection, c_t ∈ [c_min, c_max]:
    δθ̃ = clip(δθ, −c_t, +c_t)

Where it Fits in the Ablation Study
-----------------------------------
This trainer exercises the **Global-Static-Only** setting:
    • Features: global author vectors g only (no per-instance dynamics).
    • Injection: additive δB over LoRA B per sample, stateless each forward.
    • Heads: the emission architecture (single vs multi head, dictionary-coded)
      is configured in the hypernetwork module and toggled via CLI flags here.
    • Probes: teacher-forced CE/PPL and context-only boundary CE/PPL track
      fluency and leakage, respectively.

Implementation Outline
----------------------
1) **Backbone & PEFT attach**
   • Load the tokenizer/model (with `--online` + `--hf_token` when pulling
     a gated model). Optionally enable QLoRA (nf4, double-quant).
   • Attach LoRA to attention/MLP targets and freeze base weights.

2) **Dataset & features**
   • Load Parquet splits and an author table with gstat_* columns; filter
     and blocklist leaky globals. Build a dataset that yields tokenized
     batches plus `global_features ∈ ℝ^G`.

3) **Hypernetwork wrapper**
   • Construct the hypernetwork (single/multi head, optional dictionary-coded
     emission, optional layer-context gates). Role/group scales normalize
     qkv/o_proj/mlp magnitudes. Injection uses hooks over LoRA B_j.

4) **Training loop**
   • Pack inputs as `[context | <|reply|> | trimmed target]`.
   • Forward (hidden-only) → compute teacher-forced CE via lm_head on the
     selected reply tokens. Compute context-only boundary CE by masking future
     tokens and disabling δθ as configured. Add L2(δθ) with warm multiplier.
   • Apply clamp schedule for δθ, AMP/grad-clip, AdamW, and optional EMA.
   • Log rich telemetry: clamp, ‖δθ‖₂, var of top-k dims, δθ·prev, cos(prev),
     CE/PPL (TF, CTX), hypernet/context param counts, and per-role scales.

5) **Evaluation & uncertainty (optional)**
   • Periodically evaluate on validation with TF CE/PPL and CTX CE/PPL.
   • Optional MC-Dropout / noise-conditioning over g to log predictive entropy
     and mutual information.

6) **Interpretability exports**
   • Gradient-based feature importance, permutation importance.
   • Integrated Gradients snapshots.
   • Per-module δθ L2 CSV (using the hypernet directly).

7) **Checkpointing & artifacts**
   • Save hypernetwork weights, frozen PEFT placeholder snapshot (for reference),
     tokenizer, a feature manifest, and a training summary (metrics + hparams).
   • Keep a “best by metric” subdirectory based on validation CE or CTX CE.
   • Optional dynamic-INT8 export of the hypernetwork for CPU inference.

8) **Distributed & stability**
   • DDP with `broadcast_buffers=False` to avoid forward-time desyncs.
   • A preflight synthetic forward verifies NCCL/BF16/P2P settings early.
"""

from __future__ import annotations

import os
import sys

# ---- RMM unified memory (high-memory GPU: addresses 480 GB via NVLink-C2C) -----------
# Must run BEFORE `import torch` — torch lazily inits CUDA on import in some
# environments, and the allocator can only be swapped before first CUDA context.
if os.environ.get("HN_USE_UNIFIED_MEMORY", "0") == "1":
    try:
        try:
            os.chdir("/tmp")
        except OSError:
            pass
        import rmm
        from rmm.allocators.torch import rmm_torch_allocator
        import torch
        rmm.reinitialize(pool_allocator=True, managed_memory=True, logging=False)
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"[rmm] Unified memory allocator active (RMM {rmm.__version__})", flush=True)
    except Exception as _rmm_err:
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"[rmm] Failed to init: {_rmm_err}; falling back to default allocator", flush=True)

import gc
import json
import time
import math
import random
import logging
import argparse
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import contextlib
from contextlib import nullcontext
import numpy as np
logger = logging.getLogger(__name__)

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Enable TF32 tensor-core precision for fp32 matmuls (Ampere+/Hopper).
# Default PyTorch precision is "highest"; "medium" uses TF32 (19-bit mantissa)
# which is 3-8× faster on standard GPU/high-memory GPU with negligible accuracy difference.
torch.set_float32_matmul_precision("medium")

# AMP
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# v18 FP16-autocast override (set HN_FORCE_FP16_AUTOCAST=1 to force FP16 even
# when the hardware supports BF16). Diagnostic lever for high-memory GPU/SM90 SIGILLs
# where the BF16 TMA path hits an illegal opcode against host driver 570.124.
_HN_FORCE_FP16 = os.environ.get("HN_FORCE_FP16_AUTOCAST", "0").lower() in ("1", "true", "yes")

# bitsandbytes (optional for QLoRA). bnb prints its own stderr traceback at
# import time when the CUDA binary for the current toolkit version is missing
# (e.g. NGC 26.03 ships no libbitsandbytes_cuda132.so). That's harmless for
# this codepath -- BF16 emit_both does not use bnb -- but the traceback
# pollutes training logs. Redirect stderr during the import so only HAVE_BNB
# records the outcome.
import io as _io
import contextlib as _contextlib
_bnb_stderr = _io.StringIO()
try:
    with _contextlib.redirect_stderr(_bnb_stderr):
        import bitsandbytes as bnb  # noqa: F401
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False
del _bnb_stderr, _io, _contextlib

# safetensors (optional)
try:
    from safetensors.torch import save_file as save_safetensors
    from safetensors.torch import load_file as load_safetensors
    HAVE_SFT = True
except Exception:
    HAVE_SFT = False
    save_safetensors = None  # type: ignore
    load_safetensors = None  # type: ignore

# Dynamic quantization (optional compression export)
try:
    from torch.ao.quantization import quantize_dynamic as _quantize_dynamic
except Exception:
    try:
        from torch.quantization import quantize_dynamic as _quantize_dynamic  # type: ignore
    except Exception:
        _quantize_dynamic = None  # type: ignore

# Spectral norm parametrize (robust import)
try:
    from torch.nn.utils.parametrizations import spectral_norm as _sn_wrap
except Exception:
    from torch.nn.utils import spectral_norm as _sn_wrap  # type: ignore

# Hugging Face
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed as hf_set_seed,
    utils as hf_utils,
)

try:
    from transformers import BitsAndBytesConfig
    HAVE_BNB_CFG = True
except Exception:
    BitsAndBytesConfig = None  # type: ignore
    HAVE_BNB_CFG = False

# PEFT
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ---------- Local modules (dataset + hypernetwork wrapper) ----------
_CANDIDATE_SCRIPT_DIRS = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parents[1] / "data_scripts",
    Path("/workspace/hypernets/data_scripts"),
]
for _p in _CANDIDATE_SCRIPT_DIRS:
    if _p.exists():
        sys.path.append(str(_p))

from hypernetwork_dataset_10000 import (  # type: ignore  # noqa: E402
    HypernetGlobalOnlyDataset10000,
    LEAKY_GLOBAL_SENTIMENT,
    LEAKY_GLOBAL_BEAST_SENT,
)

from hypernetwork_structure_10000 import (  # type: ignore  # noqa: E402
    PEFTHypernetModel,
    make_from_dims,
    RoleSpec,
)

# ------------------------- Constants -------------------------
# Leave HF token blank here (user will paste / pass via CLI). Script also supports --hf_token_file and env.
HF_TOKEN: str = ""

REPLY_SEP_TOKEN = "<|reply|>"
REPLY_END_TOKEN = "<|eoreply|>"
CONTEXT_SEP_TOKEN = "<|context|>"

REPLY_SEP_ID: Optional[int] = None
REPLY_END_ID: Optional[int] = None
BOS_ID: Optional[int] = None
EOS_ID: Optional[int] = None

REPLY_BUDGET = 256
MIN_CLAMP = 0.02

_PACK_SANITY_PRINTED = False

ROLE_QKV = {"q_proj", "k_proj", "v_proj", "query_key_value"}
ROLE_O = {"o_proj", "out_proj", "dense"}
ROLE_MLP = {"gate_proj", "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h"}
ALL_ROLES_DEFAULT = ["qkv", "o_proj", "mlp", "other"]

K6_FEATURES_DEFAULT = [
    "gstat_user_ttr",
    "gstat_user_post_rate",
    "gstat_user_subreddit_entropy",
    "gstat_user_sr_max_share",
    "gstat_punct_ratio",
    "gstat_question_ratio",
]


# ------------------------- IO helpers -------------------------
def atomic_write_text(path: Union[str, Path], text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, p)


def read_resume_pointer(ptr_path: Union[str, Path]) -> Tuple[Optional[Path], int]:
    """
    resume.state format:
      CHECKPOINT=/path/to/hypernetwork_last.safetensors
      COMPLETED_STEPS=1234
    """
    try:
        p = Path(ptr_path)
        if not p.exists():
            return None, 0
        ckpt: Optional[Path] = None
        steps = 0
        with open(p, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("CHECKPOINT="):
                    ckpt = Path(line.split("=", 1)[1])
                elif line.startswith("COMPLETED_STEPS="):
                    try:
                        steps = int(line.split("=", 1)[1])
                    except Exception:
                        steps = 0
        if ckpt is not None and ckpt.exists():
            return ckpt, max(0, steps)
    except Exception:
        pass
    return None, 0


def write_resume_pointer(ptr_path: Union[str, Path], checkpoint: Union[str, Path], completed_steps: int) -> None:
    text = f"CHECKPOINT={Path(checkpoint).as_posix()}\nCOMPLETED_STEPS={int(completed_steps)}\n"
    atomic_write_text(ptr_path, text)


def load_state_dict_any(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        if HAVE_SFT and p.suffix == ".safetensors" and load_safetensors is not None:
            return load_safetensors(str(p))
        return torch.load(str(p), map_location="cpu")
    except Exception:
        return None


def save_state_dict_any(path: Union[str, Path], state: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")

    cpu_state: Dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            cpu_state[k] = v.detach().cpu()
        else:
            cpu_state[k] = v

    try:
        if HAVE_SFT and p.suffix == ".safetensors" and save_safetensors is not None:
            save_safetensors(cpu_state, str(tmp))
        else:
            torch.save(cpu_state, str(tmp))
        os.replace(tmp, p)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

def safe_read_parquet(path: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    p = str(path)
    cols = list(columns) if columns is not None else None
    try:
        return pd.read_parquet(p, columns=cols, engine="pyarrow", memory_map=True, dtype_backend="pyarrow")
    except TypeError:
        return pd.read_parquet(p, columns=cols, engine="pyarrow", memory_map=True)
    except Exception:
        if cols is not None:
            return pd.read_parquet(p, columns=cols)
        return pd.read_parquet(p)

# ------------------------- Dist helpers -------------------------
def dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if dist_is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist_is_initialized() else 1


def is_main() -> bool:
    return rank() == 0


def ddp_allreduce_sums(device: torch.device, *vals: float) -> Tuple[float, ...]:
    if not dist_is_initialized():
        return vals
    t = torch.tensor(vals, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return tuple(t.tolist())


class GracefulShutdown(BaseException):
    """
    Raised from SIGTERM/SIGINT handlers so torchrun/elastic sees a clean exit
    (instead of propagating DataLoader worker termination as a job failure).
    """
    pass


_SHUTDOWN_REQUESTED: bool = False
_SHUTDOWN_SIGNAL: Optional[int] = None


def mark_shutdown(signum: int) -> None:
    global _SHUTDOWN_REQUESTED, _SHUTDOWN_SIGNAL
    _SHUTDOWN_REQUESTED = True
    _SHUTDOWN_SIGNAL = int(signum)


def shutdown_requested() -> bool:
    return bool(_SHUTDOWN_REQUESTED)


def shutdown_signal() -> Optional[int]:
    return _SHUTDOWN_SIGNAL


# ------------------------- Misc helpers -------------------------
def ppl(ce: float) -> float:
    return math.exp(ce) if ce < 50 else float("inf")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    hf_set_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True


def maybe_enable_online(online: bool) -> None:
    if online:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"


def assert_hf_auth(repo_id: str, token: Optional[str], online: bool) -> None:
    if not online:
        return
    try:
        from huggingface_hub import list_repo_files
    except Exception:
        # If hub isn't available, best-effort only.
        return

    if token is None or str(token).strip() == "":
        raise RuntimeError(
            "HF token is empty/undefined but --online was requested and the repo may be gated.\n"
            f"repo_id={repo_id}\n"
            "Pass --hf_token, set HF_TOKEN/HUGGINGFACEHUB_API_TOKEN, or use --hf_token_file."
        )
    try:
        _ = list_repo_files(repo_id, token=token)
    except Exception as e:
        raise RuntimeError(
            f"Hugging Face authentication failed for '{repo_id}'. "
            f"Ensure the token has access and the license is accepted. Original error: {e}"
        ) from e


def resolve_hf_token(args: argparse.Namespace) -> str:
    """
    Priority:
      1) --hf_token (non-empty)
      2) --hf_token_file (first non-empty line)
      3) env HF_TOKEN / HUGGINGFACEHUB_API_TOKEN / HUGGINGFACE_HUB_TOKEN
      4) module constant HF_TOKEN (blank placeholder by default)
    """
    t = ""
    if getattr(args, "hf_token", None):
        t = str(args.hf_token).strip()
        if t:
            return t

    tok_file = str(getattr(args, "hf_token_file", "") or "").strip()
    if tok_file:
        try:
            with open(tok_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        return s
        except Exception:
            pass

    for k in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        ev = os.environ.get(k, "").strip()
        if ev:
            return ev

    return str(HF_TOKEN or "").strip()


# ------------------------- Token helpers -------------------------
def add_special_tokens(tok: AutoTokenizer, extra_special_tokens: Optional[List[str]]) -> Tuple[AutoTokenizer, int]:
    required = [REPLY_SEP_TOKEN, REPLY_END_TOKEN, CONTEXT_SEP_TOKEN]
    extra = [t for t in (extra_special_tokens or []) if isinstance(t, str) and t.strip()]

    seen = set()
    tokens: List[str] = []
    for t in required + extra:
        if not t or t in seen:
            continue
        tokens.append(t)
        seen.add(t)

    if not tokens:
        return tok, 0

    cur = set(getattr(tok, "additional_special_tokens", []) or [])
    need = [t for t in tokens if t not in cur]
    if not need:
        return tok, 0

    added = tok.add_special_tokens({"additional_special_tokens": need})
    return tok, int(added or 0)


def set_special_token_ids(tok: AutoTokenizer) -> None:
    global REPLY_SEP_ID, REPLY_END_ID, BOS_ID, EOS_ID

    BOS_ID = tok.bos_token_id
    EOS_ID = tok.eos_token_id

    try:
        REPLY_SEP_ID = tok.convert_tokens_to_ids(REPLY_SEP_TOKEN)
    except Exception:
        REPLY_SEP_ID = None

    try:
        REPLY_END_ID = tok.convert_tokens_to_ids(REPLY_END_TOKEN)
    except Exception:
        REPLY_END_ID = None

    if REPLY_SEP_ID is None or REPLY_END_ID is None:
        raise RuntimeError(
            "Tokenizer is missing required reply boundary special tokens.\n"
            f"reply_sep_id={REPLY_SEP_ID} reply_end_id={REPLY_END_ID}\n"
            f"unk_token_id={tok.unk_token_id} tokenizer={tok.__class__.__name__}\n"
            f"additional_special_tokens={getattr(tok, 'additional_special_tokens', None)}"
        )

    if tok.unk_token_id is not None:
        if REPLY_SEP_ID == tok.unk_token_id or REPLY_END_ID == tok.unk_token_id:
            raise RuntimeError("Reply boundary special tokens were not added correctly (mapped to unk_token_id).")


@torch.no_grad()
def init_token_rows_from_id(backbone: nn.Module, new_token_ids: List[int], src_id: int) -> None:
    if not new_token_ids:
        return

    try:
        in_emb = backbone.get_input_embeddings()
        out_emb = backbone.get_output_embeddings()
    except Exception:
        return

    W_in = getattr(in_emb, "weight", None)
    if W_in is None:
        return

    W_out = getattr(out_emb, "weight", None) if out_emb is not None else None
    b_out = getattr(out_emb, "bias", None) if out_emb is not None else None

    src_id = int(src_id)
    with torch.no_grad():
        for tid in new_token_ids:
            tid = int(tid)
            if tid < 0 or tid >= int(W_in.size(0)):
                continue
            if 0 <= src_id < int(W_in.size(0)):
                W_in[tid].copy_(W_in[src_id])
            if W_out is not None and tid < int(W_out.size(0)) and 0 <= src_id < int(W_out.size(0)):
                W_out[tid].copy_(W_out[src_id])
            if b_out is not None and tid < int(b_out.size(0)) and 0 <= src_id < int(b_out.size(0)):
                b_out[tid].copy_(b_out[src_id])


def enable_trainable_special_token_rows(
    peft_or_base_model: nn.Module,
    new_token_ids: List[int],
) -> List[nn.Parameter]:
    """
    Enable training ONLY for newly-added token rows via gradient masking.
    """
    if not new_token_ids:
        return []
    ids = sorted({int(x) for x in new_token_ids if int(x) >= 0})
    if not ids:
        return []

    try:
        in_emb = peft_or_base_model.get_input_embeddings()
    except Exception:
        in_emb = None

    try:
        out_emb = peft_or_base_model.get_output_embeddings()
    except Exception:
        out_emb = None

    params: List[nn.Parameter] = []

    def install_row_mask(param: nn.Parameter) -> None:
        def _hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None or grad.ndim < 2:
                return grad
            V = int(grad.size(0))
            mask = torch.zeros(V, dtype=torch.bool, device=grad.device)
            for tid in ids:
                if 0 <= tid < V:
                    mask[tid] = True
            grad[~mask] = 0.0
            return grad

        try:
            param.register_hook(_hook)
        except Exception:
            pass

    if in_emb is not None and hasattr(in_emb, "weight") and isinstance(in_emb.weight, nn.Parameter):
        in_emb.weight.requires_grad_(True)
        install_row_mask(in_emb.weight)
        params.append(in_emb.weight)

    if out_emb is not None and hasattr(out_emb, "weight") and isinstance(out_emb.weight, nn.Parameter):
        out_emb.weight.requires_grad_(True)
        install_row_mask(out_emb.weight)
        params.append(out_emb.weight)

    if out_emb is not None and hasattr(out_emb, "bias") and isinstance(getattr(out_emb, "bias"), nn.Parameter):
        out_emb.bias.requires_grad_(True)
        params.append(out_emb.bias)

    return params


# ------------------------- LoRA / role meta helpers -------------------------
def role_of_leaf(leaf: str) -> str:
    if leaf in ROLE_QKV:
        return "qkv"
    if leaf in ROLE_O:
        return "o_proj"
    if leaf in ROLE_MLP:
        return "mlp"
    return "other"


def extract_lora_meta(peft_model: nn.Module) -> List[Dict[str, Any]]:
    meta: List[Dict[str, Any]] = []
    active_adapter: Optional[str] = getattr(peft_model, "active_adapter", None)

    for full_name, mod in peft_model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue

        A_lin = None
        B_lin = None

        # Some PEFT versions store lora_A/lora_B as ModuleDict(adapter_name -> Linear)
        try:
            A_dict = getattr(mod, "lora_A", None)
            B_dict = getattr(mod, "lora_B", None)

            if isinstance(A_dict, nn.ModuleDict) and isinstance(B_dict, nn.ModuleDict):
                keys_A = list(A_dict.keys())
                keys_B = list(B_dict.keys())
                common = [k for k in keys_A if k in keys_B]

                adapter = active_adapter if (active_adapter in common) else (common[0] if common else None)
                if adapter is None:
                    continue
                A_lin = A_dict[adapter]
                B_lin = B_dict[adapter]
            elif hasattr(mod.lora_A, "weight") and hasattr(mod.lora_B, "weight"):
                A_lin = mod.lora_A
                B_lin = mod.lora_B
        except Exception:
            continue

        if A_lin is None or B_lin is None:
            continue

        A_w = getattr(A_lin, "weight", None)
        B_w = getattr(B_lin, "weight", None)
        if A_w is None or B_w is None:
            continue

        try:
            r, fan_in = tuple(int(x) for x in A_w.shape)       # [r, in_features]
            fan_out, rB = tuple(int(x) for x in B_w.shape)      # [out_features, r]
            if r != rB:
                continue

            leaf = full_name.rsplit(".", 1)[-1]
            role = role_of_leaf(leaf)

            meta.append(
                {
                    "name": full_name,
                    "leaf": leaf,
                    "role": role,
                    "A_shape": (int(r), int(fan_in)),
                    "B_shape": (int(fan_out), int(r)),
                    "fan_in": int(fan_in),
                    "fan_out": int(fan_out),
                    "B_numel": int(fan_out * r),
                }
            )
        except Exception:
            continue

    return meta


def compute_group_scales(meta: List[Dict[str, Any]], mode: str = "fan_in") -> Dict[str, float]:
    if mode == "none" or not meta:
        return {}
    buckets: Dict[str, List[int]] = {}
    for z in meta:
        f_in, f_out = z["fan_in"], z["fan_out"]
        key = z["role"]
        buckets.setdefault(key, [])
        if mode == "fan_in":
            buckets[key].append(int(f_in))
        else:
            buckets[key].append(int(0.5 * (f_in + f_out)))
    scales: Dict[str, float] = {}
    for role, fans in buckets.items():
        if not fans:
            continue
        m = sum(fans) / max(1, len(fans))
        scales[role] = float(1.0 / math.sqrt(max(1.0, float(m))))
    return scales


def zero_linear_(lin: nn.Linear) -> None:
    nn.init.zeros_(lin.weight)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


def zero_init_last_linear(module: nn.Module) -> int:
    last: Optional[nn.Linear] = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        zero_linear_(last)
        return 1
    return 0


def apply_spectral_norm(module: nn.Module, scope: str = "last") -> int:
    """
    Apply spectral norm to Linear layers.
    Skips dictionary alpha_heads to avoid issues with zero-init.
    """
    if scope == "none":
        return 0

    skip = set()
    try:
        out_head = getattr(module, "out_head", None)
        alpha_heads = getattr(out_head, "alpha_heads", None)
        if isinstance(alpha_heads, nn.ModuleDict):
            for _, lin in alpha_heads.items():
                if isinstance(lin, nn.Linear):
                    skip.add(lin)
    except Exception:
        pass

    wraps = 0
    if scope == "last":
        last: Optional[nn.Linear] = None
        for m in module.modules():
            if isinstance(m, nn.Linear) and m not in skip:
                last = m
        if last is not None:
            _sn_wrap(last)
            wraps += 1
    else:
        for m in module.modules():
            if isinstance(m, nn.Linear) and m not in skip:
                _sn_wrap(m)
                wraps += 1
    return wraps


class EMA:
    def __init__(self, params: List[nn.Parameter], decay: float):
        self.decay = float(decay)
        self.shadow: Dict[int, torch.Tensor] = {}
        for p in params:
            if isinstance(p, nn.Parameter) and p.requires_grad:
                self.shadow[id(p)] = p.detach().clone()

    @torch.no_grad()
    def update(self, params: List[nn.Parameter]) -> None:
        if self.decay <= 0.0:
            return
        for p in params:
            if not (isinstance(p, nn.Parameter) and p.requires_grad):
                continue
            k = id(p)
            if k not in self.shadow:
                self.shadow[k] = p.detach().clone()
            self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, params: List[nn.Parameter]) -> None:
        for p in params:
            if isinstance(p, nn.Parameter) and p.requires_grad:
                setattr(p, "_ema_backup", p.detach().clone())
                k = id(p)
                if k in self.shadow:
                    p.data.copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, params: List[nn.Parameter]) -> None:
        for p in params:
            if hasattr(p, "_ema_backup"):
                p.data.copy_(getattr(p, "_ema_backup"))
                delattr(p, "_ema_backup")


def install_grad_sanitizers_on(module: nn.Module, also_params: Optional[List[nn.Parameter]] = None) -> None:
    def sanitize(grad: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        for p in module.parameters():
            if isinstance(p, nn.Parameter) and p.requires_grad:
                p.register_hook(sanitize)
        if also_params:
            for p in also_params:
                if isinstance(p, nn.Parameter) and p.requires_grad:
                    p.register_hook(sanitize)
        logging.info("Installed non-finite gradient sanitizers on %s.", module.__class__.__name__)
    except Exception as e:
        logging.warning("Gradient sanitizer installation failed on %s: %s", module.__class__.__name__, e)


def is_cuda_oom_error(err: BaseException) -> bool:
    msg = str(err).lower()
    signatures = [
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cuda error: memory allocation",
        "cudnn_status_alloc_failed",
        "hip error out of memory",
    ]
    return any(s in msg for s in signatures)


@torch.no_grad()
def hypernet_grad_stats(hnet: nn.Module) -> Dict[str, float]:
    alpha_sq = 0.0
    dict_sq = 0.0
    trunk_sq = 0.0
    head_sq = 0.0
    total_sq = 0.0

    for n, p in hnet.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        n2 = float(g.norm(p=2).item()) ** 2
        total_sq += n2

        if "out_head.alpha_heads" in n:
            alpha_sq += n2
        elif "out_head.dict_tables" in n:
            dict_sq += n2
        elif "out_head.heads" in n:
            head_sq += n2
        else:
            trunk_sq += n2

    return {
        "total_grad_norm": math.sqrt(total_sq) if total_sq > 0 else 0.0,
        "trunk_grad_norm": math.sqrt(trunk_sq) if trunk_sq > 0 else 0.0,
        "alpha_head_grad_norm": math.sqrt(alpha_sq) if alpha_sq > 0 else 0.0,
        "dict_table_grad_norm": math.sqrt(dict_sq) if dict_sq > 0 else 0.0,
        "role_head_grad_norm": math.sqrt(head_sq) if head_sq > 0 else 0.0,
    }

# ─────────────────────── Behavioral Probe Head ───────────────────────
class BehavioralProbeHead(nn.Module):
    """Linear probe on reply-region hidden states → user behavioral targets.

    Gradient flows back through the backbone and delta injection, giving
    the hypernetwork direct supervision to produce persona-aware deltas.
    """

    def __init__(self, hidden_dim: int, n_targets: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, n_targets, bias=True)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        """pooled_hidden: [B, H] → predictions: [B, n_targets]"""
        return self.proj(self.drop(pooled_hidden))


def build_hypernet_optimizer(
    hnet: nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    alpha_mult: float = 1.0,
    dict_mult: float = 1.0,
    ctx_params: Optional[List[nn.Parameter]] = None,
    special_token_params: Optional[List[nn.Parameter]] = None,
    alpha_weight_decay: Optional[float] = None,
    dict_weight_decay: Optional[float] = None,
    special_lr_mult: float = 0.2,
    special_weight_decay: float = 0.0,
):
    """
    AdamW param groups:
      • trunk (everything except α and D): base_lr, weight_decay
      • α-heads (codes):                   base_lr * alpha_mult, alpha_weight_decay
      • D tables (atoms):                  base_lr * dict_mult,  dict_weight_decay
      • context gate params (if any):      base_lr, weight_decay
      • special token rows:                base_lr*special_lr_mult, special_weight_decay
    """
    if alpha_weight_decay is None:
        alpha_weight_decay = weight_decay
    if dict_weight_decay is None:
        dict_weight_decay = weight_decay

    try:
        seen = set()
        cast_n = 0

        def maybe_cast(p: nn.Parameter) -> None:
            nonlocal cast_n
            if not isinstance(p, nn.Parameter):
                return
            if not p.requires_grad:
                return
            pid = id(p)
            if pid in seen:
                return
            seen.add(pid)
            if p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
                cast_n += 1

        for _, p in hnet.named_parameters():
            maybe_cast(p)
        if ctx_params:
            for p in ctx_params:
                maybe_cast(p)
        if special_token_params:
            for p in special_token_params:
                maybe_cast(p)

        if cast_n > 0:
            logging.info(f"Casted {cast_n} trainable params to fp32 for AMP scaler compatibility.")
    except Exception:
        pass

    alpha_params: List[nn.Parameter] = []
    dict_params: List[nn.Parameter] = []
    trunk_params: List[nn.Parameter] = []

    for name, p in hnet.named_parameters():
        if not p.requires_grad:
            continue
        if "out_head.alpha_heads" in name:
            alpha_params.append(p)
        elif "out_head.dict_tables" in name:
            dict_params.append(p)
        else:
            trunk_params.append(p)

    groups: List[Dict[str, Any]] = []
    if trunk_params:
        groups.append({"params": trunk_params, "lr": float(base_lr), "weight_decay": float(weight_decay)})
    if alpha_params:
        groups.append(
            {
                "params": alpha_params,
                "lr": float(base_lr) * float(alpha_mult),
                "weight_decay": float(alpha_weight_decay),
            }
        )
    if dict_params:
        groups.append(
            {
                "params": dict_params,
                "lr": float(base_lr) * float(dict_mult),
                "weight_decay": float(dict_weight_decay),
            }
        )
    if ctx_params:
        groups.append({"params": ctx_params, "lr": float(base_lr), "weight_decay": float(weight_decay)})
    if special_token_params:
        groups.append(
            {
                "params": special_token_params,
                "lr": float(base_lr) * float(special_lr_mult),
                "weight_decay": float(special_weight_decay),
            }
        )

    try:
        opt = torch.optim.AdamW(groups, fused=True)
    except TypeError:
        try:
            opt = torch.optim.AdamW(groups, foreach=True)
        except TypeError:
            opt = torch.optim.AdamW(groups)
    return opt

def zero_grad_dict_tables_if_frozen(hnet: nn.Module, cur_step: int, freeze_steps: int) -> None:
    if freeze_steps <= 0 or cur_step >= freeze_steps:
        return
    out_head = getattr(hnet, "out_head", None)
    dict_tables = getattr(out_head, "dict_tables", None)
    if isinstance(dict_tables, nn.ParameterDict):
        for _, p in dict_tables.items():
            if isinstance(p, nn.Parameter) and p.grad is not None:
                p.grad.zero_()


# ------------------------- Backbone helpers -------------------------
def get_lm_head(model: nn.Module) -> nn.Module:
    try:
        head = model.get_output_embeddings()
        if head is not None:
            return head
    except Exception:
        pass
    for obj in (model, getattr(model, "base_model", None), getattr(model, "module", None)):
        if obj is not None and hasattr(obj, "lm_head"):
            return getattr(obj, "lm_head")
    for name, mod in model.named_modules():
        if name.endswith("lm_head"):
            return mod
    raise RuntimeError("Could not resolve lm_head / output embeddings.")


def resolve_backbone(model: nn.Module) -> nn.Module:
    def follow(obj: Any, path: Tuple[str, ...]) -> Optional[Any]:
        cur = obj
        for name in path:
            if not hasattr(cur, name):
                return None
            cur = getattr(cur, name)
        return cur

    candidates = [
        ("base_model", "model"),
        ("model",),
        ("module", "base_model", "model"),
        ("module", "model"),
        ("transformer",),
        ("base_model",),
        ("module",),
    ]
    for path in candidates:
        obj = follow(model, path)
        if obj is not None and callable(getattr(obj, "forward", None)):
            return obj
    # Cross-architecture heuristic: match the decoder body by the input-embedding
    # attribute name. Pythia (GPT-NeoX) uses `embed_in`; some HF models use `embed_tokens`.
    for mod in model.modules():
        if hasattr(mod, "embed_tokens") and callable(getattr(mod, "forward", None)):
            return mod
    for mod in model.modules():
        if hasattr(mod, "embed_in") and callable(getattr(mod, "forward", None)):
            return mod
    raise RuntimeError("Could not locate decoder backbone (looked for embed_tokens and embed_in [GPT-NeoX/Pythia]).")


# ------------------------- Input packing & CE -------------------------
def make_concat_inputs(batch: Dict[str, torch.Tensor], pad_id: int, L: int) -> None:
    """
    Produces in-place:
      batch["input_ids"], batch["attention_mask"], batch["labels"]

    Packed:
      [context_tokens | <|reply|> | reply_tokens (trimmed) | <|eoreply|>]

    Labels:
      -100 for context + <|reply|>
      reply tokens (and <|eoreply|>) supervised
    """
    global _PACK_SANITY_PRINTED

    if REPLY_SEP_ID is None or REPLY_END_ID is None:
        raise RuntimeError("Reply boundary token ids are unset. Call set_special_token_ids(tok) after tokenizer load.")

    if "context_input_ids" in batch:
        ctx_ids = batch["context_input_ids"]
        ctx_attn = batch["context_attention_mask"]
        tgt_ids = batch["target_input_ids"]
        tgt_attn = batch["target_attention_mask"]
    else:
        if ("input_ids" not in batch) or ("attention_mask" not in batch) or ("labels" not in batch):
            raise KeyError(
                "Batch is missing required keys for packing. "
                "Expected either: "
                "{context_input_ids,context_attention_mask,target_input_ids,target_attention_mask} "
                "or: {input_ids,attention_mask,labels}."
            )

        # The dataset provides a full packed sequence (prompt+reply) in input_ids and a label mask:
        #   labels == -100 for prompt (context)
        #   labels != -100 for reply (targets)
        # We must split to avoid leaking reply tokens into the context and to avoid extracting pad/eos as targets.
        full_ids = batch["input_ids"]
        full_attn = batch["attention_mask"]
        full_labels = batch["labels"]
        if full_labels.dtype != torch.long:
            full_labels = full_labels.to(dtype=torch.long)

        B = int(full_ids.size(0))
        T = int(full_ids.size(1))
        dev = full_ids.device

        ctx_ids = torch.full((B, T), int(pad_id), dtype=torch.long, device=dev)
        ctx_attn = torch.zeros((B, T), dtype=torch.long, device=dev)

        tgt_ids = torch.full((B, T), int(pad_id), dtype=torch.long, device=dev)
        tgt_attn = torch.zeros((B, T), dtype=torch.long, device=dev)

        for i in range(B):
            n = int(full_attn[i].sum().item())
            if n <= 0:
                continue

            ids_i = full_ids[i, :n]
            labs_i = full_labels[i, :n]

            # reply positions are exactly where labels != -100
            tgt_pos = (labs_i != -100).nonzero(as_tuple=False).flatten()

            if tgt_pos.numel() == 0:
                # No supervised reply tokens; treat everything as context
                c_tokens = ids_i
                t_tokens = ids_i.new_empty((0,), dtype=torch.long)
            else:
                first = int(tgt_pos[0].item())
                c_tokens = ids_i[:first]
                t_tokens = ids_i.index_select(0, tgt_pos)

            c_len = int(c_tokens.numel())
            t_len = int(t_tokens.numel())

            if c_len > 0:
                ctx_ids[i, :c_len] = c_tokens
                ctx_attn[i, :c_len] = 1

            if t_len > 0:
                tgt_ids[i, :t_len] = t_tokens
                tgt_attn[i, :t_len] = 1

    device = ctx_ids.device
    B = int(ctx_ids.size(0))
    L = int(L)

    input_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, L), dtype=torch.long, device=device)
    labels = torch.full((B, L), -100, dtype=torch.long, device=device)

    ctx_lens = ctx_attn.sum(dim=1).to(dtype=torch.long)
    tgt_lens = tgt_attn.sum(dim=1).to(dtype=torch.long)

    stripped_ctx_eos = 0
    stripped_tgt_bos = 0
    stripped_tgt_eos = 0

    for i in range(B):
        c_len = int(ctx_lens[i].item())
        t_len = int(tgt_lens[i].item())

        c = ctx_ids[i, :c_len].tolist()
        if EOS_ID is not None:
            while c and c[-1] == EOS_ID:
                c.pop()
                stripped_ctx_eos += 1

        t = tgt_ids[i, :t_len].tolist()
        if BOS_ID is not None:
            while t and t[0] == BOS_ID:
                t = t[1:]
                stripped_tgt_bos += 1
        if EOS_ID is not None:
            while t and t[-1] == EOS_ID:
                t.pop()
                stripped_tgt_eos += 1

        max_t = max(0, min(int(REPLY_BUDGET), L - (len(c) + 2)))
        if len(t) > max_t:
            t = t[:max_t]

        packed = c + [int(REPLY_SEP_ID)] + t + [int(REPLY_END_ID)]
        n = min(L, len(packed))

        if n > 0:
            input_ids[i, :n] = torch.as_tensor(packed[:n], dtype=torch.long, device=device)
            attention_mask[i, :n] = 1

        start = len(c) + 1
        if start < L:
            tgt_end = min(L, start + len(t))
            if tgt_end > start:
                labels[i, start:tgt_end] = input_ids[i, start:tgt_end]
            re_pos = start + len(t)
            if re_pos < L:
                labels[i, re_pos] = int(REPLY_END_ID)

    if (not _PACK_SANITY_PRINTED) and (stripped_ctx_eos or stripped_tgt_bos or stripped_tgt_eos):
        _PACK_SANITY_PRINTED = True
        logging.info(
            "[pack] stripped totals so far: ctx_eos=%d tgt_bos=%d tgt_eos=%d",
            stripped_ctx_eos,
            stripped_tgt_bos,
            stripped_tgt_eos,
        )

    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["labels"] = labels


def reply_ce_sums_from_hidden(
    hidden: torch.Tensor,  # [B, L-1, H]
    labels: torch.Tensor,  # [B, L]
    lm_head: nn.Module,
    chunk_tokens: int = 256,
    pos_weight_boost: float = 0.0,
) -> Dict[str, Any]:
    """
    Token-sum NLL over reply region plus start/mid/end thirds.

    If *pos_weight_boost* > 0, each token's loss is weighted by
    ``w = 1 + boost * (1 - frac)`` where *frac* ∈ [0, 1] is the token's
    relative position inside its reply.  Early tokens (style markers) get
    higher weight; the last token gets weight ≈ 1.0.  The returned "sum"
    is then a *weighted* sum and "count" remains the raw token count so
    the caller's ``sum / count`` equals a weighted-mean CE.
    """
    tgt = labels[:, 1:]
    mask = tgt != -100
    n_tokens = int(mask.sum().item())

    z = hidden.new_tensor(0.0, dtype=torch.float32)
    if n_tokens == 0:
        return {
            "sum": z,
            "count": 0,
            "start_sum": z,
            "start_count": 0,
            "mid_sum": z,
            "mid_count": 0,
            "end_sum": z,
            "end_count": 0,
        }

    idx = mask.nonzero(as_tuple=False)
    hidden_sel = hidden[idx[:, 0], idx[:, 1], :]
    targets = tgt[idx[:, 0], idx[:, 1]]

    row_counts = mask.sum(dim=1).to(dtype=torch.long)
    row_first = mask.float().argmax(dim=1).to(dtype=torch.long)
    rel = (idx[:, 1] - row_first[idx[:, 0]]).to(dtype=torch.long)
    denom = row_counts[idx[:, 0]].clamp(min=1)

    frac = (rel.to(dtype=torch.float32) + 0.5) / denom.to(dtype=torch.float32)
    seg = torch.where(
        denom == 1,
        torch.zeros_like(rel),
        torch.where(
            frac < (1.0 / 3.0),
            torch.zeros_like(rel),
            torch.where(
                frac < (2.0 / 3.0),
                torch.ones_like(rel),
                torch.full_like(rel, 2),
            ),
        ),
    )

    total_sum = hidden_sel.new_zeros((), dtype=torch.float32)
    total_sum_raw = hidden_sel.new_zeros((), dtype=torch.float32)  # unweighted
    start_sum = hidden_sel.new_zeros((), dtype=torch.float32)
    mid_sum = hidden_sel.new_zeros((), dtype=torch.float32)
    end_sum = hidden_sel.new_zeros((), dtype=torch.float32)

    start_count_t = hidden_sel.new_zeros((), dtype=torch.long)
    mid_count_t = hidden_sel.new_zeros((), dtype=torch.long)
    end_count_t = hidden_sel.new_zeros((), dtype=torch.long)

    # Pre-compute position weights if boost is active (B8).
    if pos_weight_boost > 0.0:
        pos_w = (1.0 + pos_weight_boost * (1.0 - frac)).to(dtype=torch.float32)
    else:
        pos_w = None

    N = hidden_sel.size(0)
    # Cross-architecture dtype guard: when this function is called outside
    # autocast (e.g., eval-time first-token accuracy on Pythia), hidden_sel
    # may not match lm_head.weight dtype. Resolve once here and cast each
    # chunk consistently so F.linear sees matching dtypes regardless of how
    # the upstream caller staged autocast.
    try:
        _lmh_dtype = next(lm_head.parameters()).dtype
    except Exception:
        _lmh_dtype = hidden_sel.dtype
    for s in range(0, N, chunk_tokens):
        e = min(s + chunk_tokens, N)
        logits = lm_head(hidden_sel[s:e].to(dtype=_lmh_dtype)).float()   # FP32 for stable log_softmax
        loss_vec = F.cross_entropy(logits, targets[s:e], reduction="none")

        total_sum_raw = total_sum_raw + loss_vec.sum()  # always unweighted

        if pos_w is not None:
            loss_vec = loss_vec * pos_w[s:e]

        total_sum = total_sum + loss_vec.sum()

        seg_chunk = seg[s:e]

        m0 = seg_chunk == 0
        if m0.any():
            start_sum = start_sum + loss_vec[m0].sum()
            start_count_t = start_count_t + m0.sum()

        m1 = seg_chunk == 1
        if m1.any():
            mid_sum = mid_sum + loss_vec[m1].sum()
            mid_count_t = mid_count_t + m1.sum()

        m2 = seg_chunk == 2
        if m2.any():
            end_sum = end_sum + loss_vec[m2].sum()
            end_count_t = end_count_t + m2.sum()

    return {
        "sum": total_sum,
        "sum_raw": total_sum_raw,  # always unweighted (== sum when pos_weight_boost=0)
        "count": int(N),
        "start_sum": start_sum,
        "start_count": int(start_count_t.item()),
        "mid_sum": mid_sum,
        "mid_count": int(mid_count_t.item()),
        "end_sum": end_sum,
        "end_count": int(end_count_t.item()),
    }


def reply_ce_loss_from_hidden(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    chunk_tokens: int = 256,
    pos_weight_boost: float = 0.0,
) -> torch.Tensor:
    sums = reply_ce_sums_from_hidden(hidden, labels, lm_head, chunk_tokens=chunk_tokens,
                                     pos_weight_boost=pos_weight_boost)
    N = int(sums["count"])
    if N <= 0:
        return hidden.new_tensor(0.0, dtype=torch.float32)
    return sums["sum"] / max(1, N)

class FeatureDensityGate:
    def __init__(
        self,
        train_g: torch.Tensor,
        *,
        k: int = 16,
        percentile: float = 99.5,
        probe_size: int = 2048,
        multiplier: float = 1.0,
        eps: float = 1e-6,
    ):
        assert train_g.dim() == 2
        self.k = int(k)
        self.percentile = float(percentile)
        self.probe_size = int(probe_size)
        self.multiplier = float(multiplier)
        self.eps = float(eps)

        self.device = train_g.device
        self.mean = train_g.mean(dim=0)
        self.std = train_g.std(dim=0).clamp(min=self.eps)
        self.ref_g = train_g.detach()
        self.ref_z = (train_g - self.mean) / self.std

        n = int(self.ref_z.shape[0])
        if n <= 1 or self.k <= 0:
            self.threshold = float("inf")
            return

        k_eff = min(self.k, max(1, n - 1))
        m = min(self.probe_size, n)
        idx = torch.randperm(n, device=self.device)[:m]
        probe = self.ref_z[idx]

        dist = torch.cdist(probe, self.ref_z)
        dist[torch.arange(m, device=self.device), idx] = float("inf")
        kth = dist.kthvalue(k_eff, dim=1).values

        q = max(0.0, min(1.0, self.percentile / 100.0))
        thr = torch.quantile(kth, q).item()
        self.threshold = float(thr) * self.multiplier

    def score(self, g: torch.Tensor) -> torch.Tensor:
        if self.k <= 0 or not math.isfinite(float(self.threshold)):
            return torch.zeros((g.shape[0],), device=g.device, dtype=torch.float32)
        z = (g - self.mean.to(g.device)) / self.std.to(g.device)
        dist = torch.cdist(z, self.ref_z.to(g.device))
        k_eff = min(self.k, dist.shape[1])
        return dist.kthvalue(k_eff, dim=1).values


@contextlib.contextmanager
def _hypernet_dropout_ctx(model: torch.nn.Module, keep_dropout: bool):
    m = model.module if hasattr(model, "module") else model
    old_m_train = m.training
    old_h_train = getattr(m, "hypernet", None).training if hasattr(m, "hypernet") else old_m_train
    old_b_train = getattr(m, "backbone", None).training if hasattr(m, "backbone") else old_m_train
    try:
        if keep_dropout:
            m.train(True)
            if hasattr(m, "hypernet"):
                m.hypernet.train(True)
            if hasattr(m, "backbone"):
                m.backbone.train(False)
        else:
            m.train(False)
            if hasattr(m, "hypernet"):
                m.hypernet.train(False)
            if hasattr(m, "backbone"):
                m.backbone.train(False)
        yield
    finally:
        m.train(old_m_train)
        if hasattr(m, "hypernet"):
            m.hypernet.train(old_h_train)
        if hasattr(m, "backbone"):
            m.backbone.train(old_b_train)


def mc_delta_mean_variance(
    model: torch.nn.Module,
    g: torch.Tensor,
    *,
    n_samples: int,
    sample_dims: int = 4096,
    keep_dropout: bool = True,
) -> torch.Tensor:
    n_samples = int(n_samples)
    if n_samples <= 1:
        return torch.zeros((g.shape[0],), device=g.device, dtype=torch.float32)

    m = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        with _hypernet_dropout_ctx(model, keep_dropout=keep_dropout):
            _, _, d0 = m._emit_delta_parts(g, force_zero=False, row_mask=None)
            d0 = d0.detach().float()
            P = int(d0.shape[1])
            k = min(int(sample_dims), P)
            idx = torch.randint(0, P, (k,), device=g.device)

            samples = [d0[:, idx]]
            for _ in range(n_samples - 1):
                _, _, d = m._emit_delta_parts(g, force_zero=False, row_mask=None)
                samples.append(d.detach().float()[:, idx])

            stack = torch.stack(samples, dim=0)
            var = stack.var(dim=0, unbiased=False)
            return var.mean(dim=1)
        
def boundary_ce_from_hidden(
    hidden: torch.Tensor,       # [B, L-1, H] from input_ids[:, :-1]
    input_ids: torch.Tensor,    # [B, L]
    labels: torch.Tensor,       # [B, L]
    lm_head: nn.Module,
) -> Tuple[torch.Tensor, int]:
    """
    CE on first reply token predicted from the last context hidden state.

    In the packed format [context | REPLY_SEP | reply | REPLY_END], the first
    supervised position (labels != -100) is the first reply *content* token,
    which follows the unsupervised REPLY_SEP.  We take the hidden state at
    the REPLY_SEP position (fi - 1) and compute CE against the first reply
    token (input_ids[fi]).

    Expected behaviour of the absolute CE values:
    - CTXzero (force_zero_delta) will be HIGH (15-25+ nats) because the
      frozen base model has never been pre-trained on the REPLY_SEP special
      token and cannot accurately predict the first reply word from it.
    - CTXzero will DRIFT upward over training because the trainable special-
      token embeddings change, shifting downstream hidden states while the
      frozen backbone weights cannot adapt.
    - The informative metric is the DELTA (CTXzero - CTXapply), not the
      absolute value.  A positive delta would indicate the hypernetwork
      deltas are leaking user information into context representations.

    Returns (ce_sum, kept_rows).
    """
    reply_mask = labels != -100
    has_reply = reply_mask.any(dim=1)
    first_idx = reply_mask.float().argmax(dim=1)
    keep = has_reply & (first_idx > 0)

    kept = int(keep.sum().item())
    if kept <= 0:
        return hidden.new_tensor(0.0, dtype=torch.float32), 0

    rows = torch.arange(input_ids.size(0), device=input_ids.device)[keep]
    fi = first_idx[keep].long()
    last_ctx = (fi - 1).long()

    h = hidden[rows, last_ctx, :]
    try:
        h = h.to(dtype=next(lm_head.parameters()).dtype)
    except Exception:
        pass
    logits = lm_head(h).float()   # FP32 for numerical stability in log_softmax
    targets = input_ids[rows, fi]
    ce_sum = F.cross_entropy(logits, targets, reduction="sum")
    return ce_sum, kept


# ------------------------- Feature augmentation -------------------------
def augment_global_features(
    g: torch.Tensor,
    noise_sigma: float = 0.0,
    dropout_p: float = 0.0,
    mixup_p: float = 0.0,
    mixup_alpha: float = 0.0,
    clamp_abs: float = 0.0,
    return_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    info: Dict[str, float] = {
        "noise_sigma": float(noise_sigma or 0.0),
        "dropout_p": float(dropout_p or 0.0),
        "mixup_p": float(mixup_p or 0.0),
        "mixup_alpha": float(mixup_alpha or 0.0),
        "clamp_abs": float(clamp_abs or 0.0),
        "did_mixup": 0.0,
        "mixup_lam": 0.0,
        "drop_frac": 0.0,
    }

    if clamp_abs and clamp_abs > 0:
        g = g.clamp(-clamp_abs, clamp_abs)

    if noise_sigma and noise_sigma > 0:
        g = g + torch.randn_like(g) * float(noise_sigma)

    if dropout_p and dropout_p > 0:
        mask = (torch.rand_like(g) > float(dropout_p)).float()
        info["drop_frac"] = float((1.0 - mask.mean()).item())
        g = g * mask

    if mixup_p and mixup_p > 0 and mixup_alpha and mixup_alpha > 0:
        if torch.rand((), device=g.device).item() < float(mixup_p):
            perm = torch.randperm(g.size(0), device=g.device)
            a = float(mixup_alpha)
            lam = float(torch.distributions.Beta(a, a).sample().item())
            info["did_mixup"] = 1.0
            info["mixup_lam"] = float(lam)
            g = lam * g + (1.0 - lam) * g[perm]

    if return_info:
        return g, info
    return g


# ------------------------- LoRA / QLoRA loader -------------------------
def load_backbone_with_peft_lora(
    base_model_id: str,
    *,
    qlora: bool,
    online: bool,
    target_modules: List[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    token: Optional[str] = None,
    extra_special_tokens: Optional[List[str]] = None,
    tokenizer_max_len: int = 2048,
    torch_dtype: Optional[torch.dtype] = None,
    use_fast_tokenizer: bool = True,
    device_map: Optional[Union[str, Dict[str, int]]] = None,
    use_grad_ckpt: bool = False,
) -> Tuple[AutoTokenizer, nn.Module, int, List[int]]:
    logging.info("Loading base model: %s (qlora=%s, online=%s)", base_model_id, str(qlora), str(online))
    have_bnb = bool(HAVE_BNB and HAVE_BNB_CFG and qlora)

    if torch_dtype is None:
        # Default to bf16 on bf16-capable GPUs (Hopper / high-memory GPU / A100 / standard GPU /
        # Ampere+ via emulation). Autocast in this codebase is bf16, so
        # loading the base model in fp16 (the previous default) caused an
        # implicit cast on every matmul between fp16 weights and bf16
        # activations, plus surfaced as `mat1 != mat2 dtype` crashes on
        # eval paths that ran outside autocast (M1 Pythia 2026-04-26
        # eval_split crash at first-token-accuracy block). bf16 keeps
        # weights and activations on the same path.
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    eff_token = (token or "").strip() or None

    if not online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

    tok = AutoTokenizer.from_pretrained(
        base_model_id,
        use_fast=use_fast_tokenizer,
        token=eff_token,
        trust_remote_code=True,
        model_max_length=int(tokenizer_max_len),
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tok, added = add_special_tokens(tok, extra_special_tokens)
    if added:
        logging.info("Added %d special tokens to tokenizer; resizing embeddings later.", added)

    if have_bnb:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_cfg,
            device_map=device_map or "auto",
            token=eff_token,
            trust_remote_code=True,
        )
    else:
        # Try FlashAttention2 first (drops attention activation memory ~10x
        # on Pythia/GPT-NeoX which uses full multi-head attention without
        # GQA). Falls back to "sdpa" then to default if FA2 isn't available
        # in this image. Pythia needs this; GQA models would not, because
        # GQA already drops attention memory. Pythia at MICROBATCH=16 saturated
        # HBM without it (2026-04-26 audit).
        _model_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map=device_map or ("auto" if torch.cuda.is_available() else None),
            token=eff_token,
            trust_remote_code=True,
        )
        base_model = None
        for _attn_impl in ("flash_attention_2", "sdpa", None):
            try:
                _kw = dict(_model_kwargs)
                if _attn_impl is not None:
                    _kw["attn_implementation"] = _attn_impl
                base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **_kw)
                logging.info("[attn] base model loaded with attn_implementation=%s",
                             str(_attn_impl) if _attn_impl else "(default)")
                break
            except Exception as _ae:
                logging.warning("[attn] from_pretrained with attn_implementation=%s failed: %s; trying next fallback",
                                str(_attn_impl) if _attn_impl else "(default)", _ae)
                base_model = None
        if base_model is None:
            # Should not reach here; the no-attn-impl path matches the original
            # call exactly. But guard against an empty base_model just in case.
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch_dtype,
                device_map=device_map or ("auto" if torch.cuda.is_available() else None),
                token=eff_token,
                trust_remote_code=True,
            )

    old_vocab: Optional[int] = None
    try:
        old_vocab = int(base_model.get_input_embeddings().weight.size(0))
    except Exception:
        old_vocab = None

    new_token_ids: List[int] = []
    if added:
        try:
            base_model.resize_token_embeddings(len(tok))
        except Exception as e:
            logging.warning("resize_token_embeddings failed after adding special tokens: %s", e)

    set_special_token_ids(tok)

    if added:
        try:
            if old_vocab is not None and old_vocab < len(tok):
                new_token_ids = list(range(int(old_vocab), int(len(tok))))
            else:
                new_token_ids = []
            init_token_rows_from_id(base_model, new_token_ids=new_token_ids, src_id=int(tok.eos_token_id or 0))
        except Exception as e:
            logging.warning("Special-token row init failed: %s", e)

    if use_grad_ckpt:
        try:
            base_model.config.use_cache = False
        except Exception:
            pass
        # The non-reentrant checkpoint variant (use_reentrant=False) does a
        # strict tensor-metadata check between forward and recomputation.
        # Our inject-hook's fallback path (recompute deltas from features when
        # the per-segment cache is cleared) produces extra hypernet-internal
        # tensors during recomputation that weren't in the initial forward's
        # save list, so the strict check trips. The reentrant variant skips
        # that check, treating the segment as an opaque function. Toggle via
        # env var so we can pick: HN_GRAD_CKPT_REENTRANT=1 -> reentrant=True.
        _ckpt_reentrant = bool(int(os.environ.get("HN_GRAD_CKPT_REENTRANT", "0") or "0"))
        try:
            if hasattr(base_model, "gradient_checkpointing_enable"):
                try:
                    base_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": _ckpt_reentrant}
                    )
                except TypeError:
                    base_model.gradient_checkpointing_enable()
                logging.info(
                    "[grad_ckpt] enabled with use_reentrant=%s", _ckpt_reentrant
                )
        except Exception as e:
            logging.warning("gradient_checkpointing_enable failed: %s", e)

    if have_bnb:
        try:
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=bool(use_grad_ckpt))
        except TypeError:
            try:
                base_model = prepare_model_for_kbit_training(base_model)
            except Exception as e:
                logging.warning("prepare_model_for_kbit_training failed: %s", e)
        except Exception as e:
            logging.warning("prepare_model_for_kbit_training failed: %s", e)

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(base_model, peft_cfg)

    pad_id = int(tok.pad_token_id)
    return tok, peft_model, pad_id, new_token_ids


def safe_lora_init(peft_model: nn.Module) -> None:
    """
    Zero LoRA B so ΔW≈0 at t=0 (A stays as initialized).
    """
    for n, p in peft_model.named_parameters():
        if "lora_B" in n:
            with torch.no_grad():
                p.zero_()


def ensure_lora_basis_initialized(
    peft_model: nn.Module,
    *,
    init_std: float = 5e-4,
    layerwise_scale: str = "fan_in",
) -> None:
    """
    Ensure frozen LoRA A factors are non-zero.
    """
    def fan_in_std(w: torch.Tensor) -> float:
        fan_in = w.shape[1] if w.ndim == 2 else max(1, w.numel())
        return (1.0 / float(fan_in)) ** 0.5

    with torch.no_grad():
        for m in peft_model.modules():
            if not (hasattr(m, "lora_A") and hasattr(m, "lora_B")):
                continue
            try:
                items_A = m.lora_A.items()
                items_B = m.lora_B.items()
            except Exception:
                continue
            adapters = set(k for k, _ in items_A) & set(k for k, _ in items_B)
            if not adapters:
                continue
            for adapter in adapters:
                A = m.lora_A[adapter]
                if not hasattr(A, "weight"):
                    continue
                w = A.weight
                if w is None or w.numel() == 0:
                    continue
                if float(w.detach().abs().mean().item()) < 1e-8:
                    std = float(init_std)
                    if layerwise_scale == "fan_in" and w.ndim == 2 and w.shape[1] > 0:
                        std = max(1e-8, float(init_std) * fan_in_std(w))
                    torch.nn.init.normal_(w, mean=0.0, std=std)


# ------------------------- Schedulers -------------------------
def build_scheduler(
    opt: torch.optim.Optimizer,
    steps: int,
    warm_frac: float = 0.05,
    min_lr: float = 3.0e-4,
):
    warm = max(1, int(steps * warm_frac))
    rest = max(1, steps - warm)
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, warm),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rest, eta_min=float(min_lr)),
        ],
        [warm],
    )


def compute_clamp(abs_step: int, *, unclamp_steps: int, unclamp_duration: int, min_clamp: float, max_clamp: float) -> float:
    if abs_step < unclamp_steps:
        return float(min_clamp)
    if unclamp_duration <= 0:
        return float(max_clamp)
    frac = (abs_step - unclamp_steps) / float(max(1, unclamp_duration))
    frac = float(min(1.0, max(0.0, frac)))
    return float(min_clamp + 0.5 * (1.0 - math.cos(math.pi * frac)) * (max_clamp - min_clamp))


def boundary_weight_at_step(abs_step: int, *, base: float, wmax: float, warmup: int) -> float:
    if warmup <= 0:
        return float(wmax)
    alpha = min(max(abs_step, 0) / float(max(1, warmup)), 1.0)
    return float(base + (wmax - base) * alpha)


# ------------------------- Eval -------------------------
@torch.no_grad()
def eval_split(
    model: Union[PEFTHypernetModel, torch.nn.parallel.DistributedDataParallel],
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    seq_len: int,
    use_amp: bool,
    progress_pct: Optional[float] = None,
    max_batches: int = 0,
    log_every: int = 200,
    mc_samples: int = 0,
    mc_keep_dropout: bool = False,
    noise_sigma: float = 0.0,
    eval_microbatch_size: int = 0,
) -> Tuple[float, float, Optional[Dict[str, float]]]:
    """
    Returns:
        (teacher-forced token CE, boundary CE with δ disabled (ctx_zero), metrics dict)
    Also includes ctx_apply_ce and ctx_delta in metrics dict.
    """
    model.eval()
    tot_tf = 0.0
    tok_tf = 0.0

    tot_tf_start = 0.0
    tok_tf_start = 0.0
    tot_tf_mid = 0.0
    tok_tf_mid = 0.0
    tot_tf_end = 0.0
    tok_tf_end = 0.0

    tot_ctx_zero = 0.0
    samp_ctx_zero = 0.0
    tot_ctx_apply = 0.0
    samp_ctx_apply = 0.0

    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if (bf16_ok and not _HN_FORCE_FP16) else torch.float16
    chunk_tokens = int(os.getenv("CE_CHUNK_TOKENS", "256"))

    try_disable_delta = os.getenv("CTX_BOUNDARY_DISABLE_DELTA", "1").lower() in ("1", "true", "yes", "y")

    def backbone_of(m: nn.Module) -> nn.Module:
        return m.module.backbone if hasattr(m, "module") else m.backbone  # type: ignore

    lm_head = get_lm_head(backbone_of(model))

    per_rank_batch_cap = 0
    per_rank_sample_cap = 0
    if max_batches and max_batches > 0:
        per_rank_batch_cap = int(math.ceil(max_batches / max(1, world_size())))
    elif progress_pct is not None:
        per_rank_sample_cap = int(len(loader.dataset) * float(progress_pct) / max(1, world_size()))

    mc_entropy_sum = 0.0
    mc_mi_sum = 0.0
    mc_count = 0
    mc_max_tok = int(os.getenv("MC_MAX_TOK", "4096"))

    # B5: first-token accuracy tracking
    first_tok_correct = 0.0
    first_tok_total = 0.0

    seen_samples = 0
    for i, batch in enumerate(loader):
        if per_rank_batch_cap and i >= per_rank_batch_cap:
            break
        if per_rank_sample_cap and seen_samples >= per_rank_sample_cap:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        make_concat_inputs(batch, pad_id, seq_len)

        nt = int((batch["labels"] != -100).sum().item())
        if nt == 0:
            continue

        B_eval = batch["input_ids"].size(0)
        eff_eval_mb = int(eval_microbatch_size) if eval_microbatch_size > 0 else B_eval

        for mb_s in range(0, B_eval, eff_eval_mb):
            mb_e = min(B_eval, mb_s + eff_eval_mb)
            inp = batch["input_ids"][mb_s:mb_e, :-1]
            attn = batch["attention_mask"][mb_s:mb_e, :-1]
            gfeat = batch["global_features"][mb_s:mb_e]
            labels_mb = batch["labels"][mb_s:mb_e]
            input_ids_mb = batch["input_ids"][mb_s:mb_e]

            mb_nt = int((labels_mb != -100).sum().item())
            if mb_nt == 0:
                continue

            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                hidden_apply = model(
                    input_ids=inp,
                    attention_mask=attn,
                    labels=None,
                    global_features=gfeat,
                    return_hidden_only=True,
                    use_cache=False,
                    output_hidden_states=False,
                )
                tf_sums = reply_ce_sums_from_hidden(hidden_apply, labels_mb, lm_head, chunk_tokens=chunk_tokens)

            tot_tf += float(tf_sums["sum"].item())
            tok_tf += float(tf_sums["count"])

            tot_tf_start += float(tf_sums["start_sum"].item())
            tok_tf_start += float(tf_sums["start_count"])
            tot_tf_mid += float(tf_sums["mid_sum"].item())
            tok_tf_mid += float(tf_sums["mid_count"])
            tot_tf_end += float(tf_sums["end_sum"].item())
            tok_tf_end += float(tf_sums["end_count"])

            # B5: first-token accuracy — argmax at first reply position
            reply_mask = labels_mb != -100
            has_reply = reply_mask.any(dim=1)
            if has_reply.any():
                fi_idx = reply_mask.float().argmax(dim=1)
                keep_ft = has_reply & (fi_idx > 0)
                if keep_ft.any():
                    mb_size = mb_e - mb_s
                    rows_ft = torch.arange(mb_size, device=device)[keep_ft]
                    pos_ft = (fi_idx[keep_ft] - 1).long()  # hidden at position before first reply token
                    h_ft = hidden_apply[rows_ft, pos_ft, :]
                    # Pythia-1.4B-specific dtype guard: when this call site
                    # runs OUTSIDE autocast (the parent with-block ended at
                    # line 1908), h_ft retains the dtype of the autocast'd
                    # forward (bf16) but lm_head.weight may be fp32 on
                    # Pythia (HF defaults differ). Force h_ft
                    # to lm_head dtype so the F.linear matmul sees matched
                    # dtypes regardless of how the model was loaded.
                    try:
                        _lmh_dtype_ft = next(lm_head.parameters()).dtype
                        h_ft = h_ft.to(dtype=_lmh_dtype_ft)
                    except Exception:
                        pass
                    logits_ft = lm_head(h_ft).float()   # FP32 for accurate argmax
                    pred_ft = logits_ft.argmax(dim=-1)
                    target_ft = input_ids_mb[rows_ft, fi_idx[keep_ft].long()]
                    first_tok_correct += float((pred_ft == target_ft).sum().item())
                    first_tok_total += float(keep_ft.sum().item())

            # ctx_apply from hidden_apply (δ applied)
            ce_apply_sum, kept_apply = boundary_ce_from_hidden(hidden_apply, input_ids_mb, labels_mb, lm_head)
            tot_ctx_apply += float(ce_apply_sum.item())
            samp_ctx_apply += float(kept_apply)

            # ctx_zero: run a no-grad forward with δ disabled (force_zero_delta)
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                hidden_zero = model(
                    input_ids=inp,
                    attention_mask=attn,
                    labels=None,
                    global_features=gfeat,
                    return_hidden_only=True,
                    force_zero_delta=bool(try_disable_delta),
                    use_cache=False,
                    output_hidden_states=False,
                )
            ce_zero_sum, kept_zero = boundary_ce_from_hidden(hidden_zero, input_ids_mb, labels_mb, lm_head)
            tot_ctx_zero += float(ce_zero_sum.item())
            samp_ctx_zero += float(kept_zero)

            # MC uncertainty (optional)
            if mc_samples and mc_samples > 0:
                prev_training = model.training
                if mc_keep_dropout:
                    model.train()
                else:
                    model.eval()

                tgt = labels_mb[:, 1:]
                mask = tgt != -100
                if mask.sum() > 0:
                    idx = mask.nonzero(as_tuple=False)
                    if idx.size(0) > mc_max_tok:
                        idx = idx[:mc_max_tok]
                    logp_stack = None
                    for _m in range(int(mc_samples)):
                        g_mc = gfeat
                        if noise_sigma and noise_sigma > 0.0:
                            g_mc = g_mc + torch.randn_like(g_mc) * float(noise_sigma)
                        hidden_mc = model(
                            input_ids=inp,
                            attention_mask=attn,
                            labels=None,
                            global_features=g_mc,
                            return_hidden_only=True,
                            use_cache=False,
                            output_hidden_states=False,
                        )
                        hidden_sel = hidden_mc[idx[:, 0], idx[:, 1], :]
                        try:
                            hidden_sel = hidden_sel.to(dtype=next(lm_head.parameters()).dtype)
                        except Exception:
                            pass
                        logits_mc = lm_head(hidden_sel).float()
                        logp_mc = F.log_softmax(logits_mc, dim=-1)
                        logp_stack = logp_mc.unsqueeze(0) if logp_stack is None else torch.cat([logp_stack, logp_mc.unsqueeze(0)], dim=0)

                    if logp_stack is not None:
                        logp_mean = torch.logsumexp(logp_stack, dim=0) - math.log(float(mc_samples))
                        p_mean = logp_mean.exp()
                        H_pred = (-(p_mean * logp_mean).sum(dim=-1)).mean().item()
                        H_inner = (-(logp_stack.exp() * logp_stack).sum(dim=-1)).mean().item()
                        mi = H_pred - H_inner

                        mc_entropy_sum += H_pred
                        mc_mi_sum += mi
                        mc_count += 1

                if mc_keep_dropout:
                    model.train(prev_training)

        seen_samples += int(batch["input_ids"].size(0))
        if log_every and is_main() and ((i + 1) % log_every == 0):
            logging.info("[eval] processed %d batches on rank %d", i + 1, rank())

    (
        tot_tf,
        tok_tf,
        tot_tf_start,
        tok_tf_start,
        tot_tf_mid,
        tok_tf_mid,
        tot_tf_end,
        tok_tf_end,
        tot_ctx_zero,
        samp_ctx_zero,
        tot_ctx_apply,
        samp_ctx_apply,
        first_tok_correct,
        first_tok_total,
    ) = ddp_allreduce_sums(
        device,
        tot_tf,
        tok_tf,
        tot_tf_start,
        tok_tf_start,
        tot_tf_mid,
        tok_tf_mid,
        tot_tf_end,
        tok_tf_end,
        tot_ctx_zero,
        samp_ctx_zero,
        tot_ctx_apply,
        samp_ctx_apply,
        first_tok_correct,
        first_tok_total,
    )

    ce_tf_avg = tot_tf / max(1.0, tok_tf)
    ce_tf_start_avg = (tot_tf_start / tok_tf_start) if tok_tf_start > 0 else float("nan")
    ce_tf_mid_avg = (tot_tf_mid / tok_tf_mid) if tok_tf_mid > 0 else float("nan")
    ce_tf_end_avg = (tot_tf_end / tok_tf_end) if tok_tf_end > 0 else float("nan")

    ce_ctx_zero_avg = (tot_ctx_zero / samp_ctx_zero) if samp_ctx_zero > 0 else float("nan")
    ce_ctx_apply_avg = (tot_ctx_apply / samp_ctx_apply) if samp_ctx_apply > 0 else float("nan")

    mc_metrics: Optional[Dict[str, float]] = None
    if mc_samples and mc_count > 0:
        e_sum, mi_sum, c = ddp_allreduce_sums(device, mc_entropy_sum, mc_mi_sum, float(mc_count))
        if c > 0:
            mc_metrics = {
                "predictive_entropy": float(e_sum / c),
                "mutual_info": float(mi_sum / c),
            }

    if mc_metrics is None:
        mc_metrics = {}
    mc_metrics["ctx_apply_ce"] = float(ce_ctx_apply_avg)
    mc_metrics["ctx_zero_ce"] = float(ce_ctx_zero_avg)
    if math.isfinite(ce_ctx_zero_avg) and math.isfinite(ce_ctx_apply_avg):
        mc_metrics["ctx_delta"] = float(ce_ctx_zero_avg - ce_ctx_apply_avg)

    mc_metrics["reply_start_ce"] = float(ce_tf_start_avg)
    mc_metrics["reply_mid_ce"] = float(ce_tf_mid_avg)
    mc_metrics["first_tok_acc"] = float(first_tok_correct / max(1.0, first_tok_total)) if first_tok_total > 0 else float("nan")
    mc_metrics["reply_end_ce"] = float(ce_tf_end_avg)

    return float(ce_tf_avg), float(ce_ctx_zero_avg), mc_metrics


# --------------------- FRG (Free-Running Generation) sidecar --------------------
@torch.no_grad()
def frg_delta_profile(
    model: Union[PEFTHypernetModel, torch.nn.parallel.DistributedDataParallel],
    ds: "HypernetGlobalOnlyDataset10000",
    cohort_ids: Dict[str, List[int]],
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """
    Delta-profile diagnostic: compute delta stats per cohort (empath vs rage).
    Returns dict with frg_* metrics.
    """
    mlocal = model.module if hasattr(model, "module") else model
    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if (bf16_ok and not _HN_FORCE_FP16) else torch.float16

    # Build uid→row index for fast lookup
    uid_to_rows: Dict[int, List[int]] = {}
    for idx in range(len(ds)):
        uid = int(ds.df.iloc[idx].get("target_user_id", -1))
        uid_to_rows.setdefault(uid, []).append(idx)

    results: Dict[str, float] = {}
    cohort_deltas: Dict[str, List[torch.Tensor]] = {}

    for cohort_name, uids in cohort_ids.items():
        deltas = []
        for uid in uids:
            rows = uid_to_rows.get(uid, [])
            if not rows:
                continue
            row_idx = rows[0]  # one sample per user
            sample = ds[row_idx]
            gfeat = sample["global_features"].unsqueeze(0).to(device)

            with autocast(device_type=device.type, enabled=bool(use_amp), dtype=amp_dtype):
                # Emit delta only (we don't need the full forward)
                try:
                    _, _, delta = mlocal._emit_delta_parts(gfeat, force_zero=False, row_mask=None)
                    deltas.append(delta.detach().float().squeeze(0))
                except Exception:
                    pass

        if deltas:
            stacked = torch.stack(deltas, dim=0)  # [N, K]
            rms = stacked.pow(2).mean(dim=1).sqrt()  # [N]
            results[f"frg_{cohort_name}_delta_rms_mean"] = float(rms.mean().item())
            results[f"frg_{cohort_name}_delta_rms_std"] = float(rms.std().item())
            results[f"frg_{cohort_name}_n"] = float(len(deltas))
            cohort_deltas[cohort_name] = deltas

    # Inter-cohort cosine: mean(empath_delta) vs mean(rage_delta)
    if "empath" in cohort_deltas and "rage" in cohort_deltas:
        c_e = torch.stack(cohort_deltas["empath"], dim=0).mean(dim=0)
        c_r = torch.stack(cohort_deltas["rage"], dim=0).mean(dim=0)
        cos = float(F.cosine_similarity(c_e.unsqueeze(0), c_r.unsqueeze(0)).item())
        results["frg_inter_cohort_cos"] = cos
        # If cos ≈ 1.0, the model isn't differentiating cohorts
        results["frg_delta_separation"] = 1.0 - cos

        # Live cohort separation Hedges' g on the delta-vector L2 norm.
        # The delta-RMS is a single-scalar summary of the per-user offset;
        # cohort separation on this scalar is a fast surrogate for the
        # paper's hidden-state probe Hedges' g and catches a
        # surface-tie regime mid-training rather than at end-of-eval.
        # The full hidden-state probe (mean-pooled penultimate residual,
        # cosine-silhouette / k-NN / logistic / AUC) is too heavy for
        # per-FRG-step logging; this delta-norm Hedges' g is the
        # one-line live indicator.
        try:
            rms_e = torch.stack(cohort_deltas["empath"], dim=0).pow(2).mean(dim=1).sqrt()
            rms_r = torch.stack(cohort_deltas["rage"], dim=0).pow(2).mean(dim=1).sqrt()
            n_e, n_r = rms_e.numel(), rms_r.numel()
            if n_e >= 2 and n_r >= 2:
                m_e = float(rms_e.mean().item())
                m_r = float(rms_r.mean().item())
                v_e = float(rms_e.var(unbiased=True).item())
                v_r = float(rms_r.var(unbiased=True).item())
                pooled_sd = math.sqrt(((n_e - 1) * v_e + (n_r - 1) * v_r) / max(1, n_e + n_r - 2))
                if pooled_sd > 1e-12:
                    cohen_d = (m_e - m_r) / pooled_sd
                    df = n_e + n_r - 2
                    j = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
                    results["frg_delta_rms_hedges_g"] = float(j * cohen_d)
                else:
                    results["frg_delta_rms_hedges_g"] = float("nan")
                # AUC (Mann-Whitney U) on delta-rms scalar
                try:
                    s_all = torch.cat([rms_e, rms_r], dim=0).cpu().numpy()
                    y_all = ([1] * n_e) + ([0] * n_r)
                    order = np.argsort(s_all, kind="mergesort")
                    ranks = np.empty(len(s_all), dtype=np.float64)
                    ranks[order] = np.arange(1, len(s_all) + 1, dtype=np.float64)
                    sum_r_pos = float(ranks[: n_e].sum())
                    U = sum_r_pos - n_e * (n_e + 1) / 2.0
                    results["frg_delta_rms_auc"] = float(U / (n_e * n_r))
                except Exception:
                    results["frg_delta_rms_auc"] = float("nan")
        except Exception:
            pass

        # Cohort-mean delta L2 norms (raw magnitudes, easy to inspect at a glance)
        try:
            results["frg_empath_delta_mean_l2"] = float(c_e.pow(2).sum().sqrt().item())
            results["frg_rage_delta_mean_l2"] = float(c_r.pow(2).sum().sqrt().item())
        except Exception:
            pass

    return results


def load_or_create_frg_cohort(
    cohort_file: str,
    ds: "HypernetGlobalOnlyDataset10000",
    n_per_cohort: int,
    val_uids: set,
    seed: int = 142,
    author_df: Optional[Any] = None,
    labels_file: str = "",
) -> Dict[str, List[int]]:
    """
    Load frg_cohort_ids.json or create it.

    Cohort selection priority:
      1. Existing frg_cohort_ids.json (cross-run consistency)
      2. labels_file (labels_sentiment.csv) — canonical quintile labels;
         selects only users labeled "rage" or "empath" (bottom/top 20%)
      3. Fallback: sort by gstat_user_sent_mean, take bottom/top 20%
         (quintile-aligned, NOT thirds)
    """
    if cohort_file and os.path.exists(cohort_file):
        with open(cohort_file, "r") as f:
            cohort = json.load(f)
        logging.info("[FRG] Loaded cohort from %s: %s",
                     cohort_file, {k: len(v) for k, v in cohort.items()})
        return cohort

    # Collect train user IDs
    all_uids = set()
    for idx in range(len(ds)):
        uid = int(ds.df.iloc[idx].get("target_user_id", -1))
        if uid not in val_uids:
            all_uids.add(uid)

    # --- Priority 1: canonical labels from labels_sentiment.csv ---
    rage_pool: List[int] = []
    empath_pool: List[int] = []

    if labels_file and os.path.exists(labels_file):
        labels_df = pd.read_csv(labels_file)
        _uid_col = "target_user_id"
        rage_pool = [int(u) for u in labels_df.loc[labels_df["label"] == "rage", _uid_col]
                     if int(u) in all_uids]
        empath_pool = [int(u) for u in labels_df.loc[labels_df["label"] == "empath", _uid_col]
                       if int(u) in all_uids]
        logging.info("[FRG] Loaded canonical labels from %s: %d rage, %d empath (in training set)",
                     labels_file, len(rage_pool), len(empath_pool))

    # --- Priority 2: fallback to quintile-aligned sort (bottom/top 20%) ---
    if not rage_pool or not empath_pool:
        uid_sent: Dict[int, float] = {}
        if author_df is not None and "gstat_user_sent_mean" in author_df.columns:
            _uid_col = "target_user_id" if "target_user_id" in author_df.columns else author_df.columns[0]
            for _, row in author_df.iterrows():
                uid = int(row.get(_uid_col, -1))
                if uid not in all_uids:
                    continue
                sent = float(row.get("gstat_user_sent_mean", float("nan")))
                if math.isfinite(sent):
                    uid_sent[uid] = sent

        if not uid_sent:
            logging.warning("[FRG] No sentiment data for cohort selection; using random split.")
            rng = random.Random(seed)
            uids_list = sorted(all_uids)
            rng.shuffle(uids_list)
            half = min(n_per_cohort, len(uids_list) // 2)
            rage_pool = uids_list[:half]
            empath_pool = uids_list[half:half * 2]
        else:
            sorted_by_sent = sorted(uid_sent.items(), key=lambda x: x[1])
            # Bottom/top 20% = quintile-aligned (matches labels_sentiment.csv bins)
            n_quintile = len(sorted_by_sent) // 5
            rage_pool = [uid for uid, _ in sorted_by_sent[:n_quintile]]
            empath_pool = [uid for uid, _ in sorted_by_sent[-n_quintile:]]
            logging.info("[FRG] Quintile fallback: %d rage (bottom 20%%), %d empath (top 20%%)",
                         len(rage_pool), len(empath_pool))

    rng = random.Random(seed)
    rng.shuffle(rage_pool)
    rng.shuffle(empath_pool)
    cohort = {
        "rage": rage_pool[:n_per_cohort],
        "empath": empath_pool[:n_per_cohort],
    }

    if cohort_file:
        try:
            with open(cohort_file, "w") as f:
                json.dump(cohort, f, indent=2)
            logging.info("[FRG] Saved cohort to %s", cohort_file)
        except Exception as e:
            logging.warning("[FRG] Could not save cohort file: %s", e)

    logging.info("[FRG] Created cohort: %s", {k: len(v) for k, v in cohort.items()})
    return cohort


def _frg_is_coherent(text: str, min_alpha_frac: float = 0.40,
                     min_words: int = 3, min_ascii_frac: float = 0.70) -> bool:
    """Light coherence check for FRG-generated text (mirrors build_hyperlora_forum)."""
    if not text or not text.strip():
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    chars = text.strip()
    if not chars:
        return False
    alpha_frac = sum(1 for c in chars if c.isalpha()) / len(chars)
    if alpha_frac < min_alpha_frac:
        return False
    ascii_frac = sum(1 for c in chars if ord(c) < 128) / len(chars)
    return ascii_frac >= min_ascii_frac


@torch.no_grad()
def frg_generate_and_score(
    model: Union[PEFTHypernetModel, "torch.nn.parallel.DistributedDataParallel"],
    ds: "HypernetGlobalOnlyDataset10000",
    cohort_ids: Dict[str, List[int]],
    device: torch.device,
    use_amp: bool,
    tokenizer: Any,
    max_new_tokens: int = 96,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_users_per_cohort: int = 25,
) -> Dict[str, float]:
    """
    FRG generation diagnostic: generate text per cohort user and report
    coherence fraction and mean generation length.
    """
    if tokenizer is None:
        return {}

    mlocal = model.module if hasattr(model, "module") else model
    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if (bf16_ok and not _HN_FORCE_FP16) else torch.float16

    # Build uid→row index lookup
    uid_to_rows: Dict[int, List[int]] = {}
    for idx in range(len(ds)):
        uid = int(ds.df.iloc[idx].get("target_user_id", -1))
        uid_to_rows.setdefault(uid, []).append(idx)

    pad_id = int(getattr(tokenizer, "pad_token_id", None) or
                 getattr(tokenizer, "eos_token_id", 0))
    eos_id = getattr(tokenizer, "eos_token_id", None)

    results: Dict[str, float] = {}

    for cohort_name, uids in cohort_ids.items():
        gen_lens: List[int] = []
        coherent_count = 0
        total_count = 0

        for uid in uids[:max_users_per_cohort]:
            rows = uid_to_rows.get(uid, [])
            if not rows:
                continue

            sample = ds[rows[0]]
            gfeat = sample["global_features"].unsqueeze(0).to(device)
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attn_mask = sample["attention_mask"].unsqueeze(0).to(device)

            try:
                with autocast(device_type=device.type, enabled=bool(use_amp), dtype=amp_dtype):
                    # Populate delta cache for injection hooks
                    mlocal._g_for_forward = gfeat
                    parts, _, _ = mlocal._emit_delta_parts(gfeat, force_zero=False)
                    mlocal._delta_for_forward = parts

                    # Generate with KV cache
                    mlocal.backbone.config.use_cache = True
                    gen_kwargs = dict(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        pad_token_id=pad_id,
                    )
                    if eos_id is not None:
                        gen_kwargs["eos_token_id"] = eos_id

                    out = mlocal.backbone.generate(**gen_kwargs)
                    mlocal.backbone.config.use_cache = False

                # Clean up state
                mlocal._g_for_forward = None
                mlocal._delta_for_forward = None

                # Decode only the generated tokens (strip prompt)
                prompt_len = input_ids.shape[1]
                gen_tokens = out[0, prompt_len:]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                n_gen = int(gen_tokens.shape[0])

                total_count += 1
                gen_lens.append(n_gen)
                if _frg_is_coherent(gen_text):
                    coherent_count += 1

            except Exception:
                # Clean up state on failure
                mlocal._g_for_forward = None
                mlocal._delta_for_forward = None
                mlocal.backbone.config.use_cache = False

        if total_count > 0:
            results[f"frg_{cohort_name}_gen_coherent_frac"] = float(coherent_count / total_count)
            results[f"frg_{cohort_name}_gen_mean_len"] = float(sum(gen_lens) / len(gen_lens)) if gen_lens else 0.0
            results[f"frg_{cohort_name}_gen_n"] = float(total_count)

    return results


def compute_composite(
    val_ce: float,
    delta_separation: float,
    persona_fidelity: float,
    w_sep: float = 0.5,
    w_pf: float = 0.3,
) -> float:
    """Composite training quality score (lower = better, same direction as val_ce).

    composite = val_ce - w_sep * delta_separation - w_pf * persona_fidelity

    Higher delta_separation and persona_fidelity *reduce* the composite,
    rewarding persona-differentiated models.
    """
    return val_ce - w_sep * delta_separation - w_pf * persona_fidelity


@torch.no_grad()
def frg_persona_fidelity(
    model: Union[PEFTHypernetModel, "torch.nn.parallel.DistributedDataParallel"],
    ds: "HypernetGlobalOnlyDataset10000",
    cohort_ids: Dict[str, List[int]],
    device: torch.device,
    use_amp: bool,
    tokenizer: Any,
    sent_model_path: str = "vader",  # kept for call-site compat, unused (VADER is sole backend)
    max_new_tokens: int = 48,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_users_per_cohort: int = 20,
) -> Dict[str, float]:
    """FRG persona fidelity: generate text per cohort user, score with VADER,
    and check whether the generated polarity matches the user's persona label.

    Rage users should produce negative-polarity text (compound < 0).
    Empath users should produce positive-polarity text (compound > 0).

    Returns dict with:
      frg_persona_fidelity       - mean(rage_preservation, empath_preservation)
      frg_rage_preservation      - fraction of rage users with polarity < 0
      frg_empath_preservation    - fraction of empath users with polarity > 0
      frg_rage_mean_polarity     - mean polarity of rage cohort
      frg_empath_mean_polarity   - mean polarity of empath cohort
    """
    if tokenizer is None:
        logging.warning("[FRG-PF] Skipping persona fidelity: tokenizer=%s", tokenizer is not None)
        return {}

    # Use VADER for sentiment scoring (CPU-only, no GPU memory overhead)
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError:
        logging.warning("[FRG-PF] Cannot import VADER (nltk not installed)")
        return {}
    try:
        _vader_sia = SentimentIntensityAnalyzer()
    except LookupError:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        _vader_sia = SentimentIntensityAnalyzer()

    def _vader_polarity_batch(texts):
        """Score texts with VADER, return compound scores in [-1, +1]."""
        return [_vader_sia.polarity_scores(t or "")["compound"] for t in texts]

    mlocal = model.module if hasattr(model, "module") else model
    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if (bf16_ok and not _HN_FORCE_FP16) else torch.float16

    # Build uid→row index lookup
    uid_to_rows: Dict[int, List[int]] = {}
    for idx in range(len(ds)):
        uid = int(ds.df.iloc[idx].get("target_user_id", -1))
        uid_to_rows.setdefault(uid, []).append(idx)

    pad_id = int(getattr(tokenizer, "pad_token_id", None) or
                 getattr(tokenizer, "eos_token_id", 0))
    eos_id = getattr(tokenizer, "eos_token_id", None)

    # Generate text per cohort
    cohort_texts: Dict[str, List[str]] = {}  # cohort → list of generated texts
    for cohort_name, uids in cohort_ids.items():
        texts: List[str] = []
        for uid in uids[:max_users_per_cohort]:
            rows = uid_to_rows.get(uid, [])
            if not rows:
                continue

            sample = ds[rows[0]]
            gfeat = sample["global_features"].unsqueeze(0).to(device)
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attn_mask = sample["attention_mask"].unsqueeze(0).to(device)

            try:
                with autocast(device_type=device.type, enabled=bool(use_amp), dtype=amp_dtype):
                    mlocal._g_for_forward = gfeat
                    parts, _, _ = mlocal._emit_delta_parts(gfeat, force_zero=False)
                    mlocal._delta_for_forward = parts

                    mlocal.backbone.config.use_cache = True
                    gen_kwargs = dict(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        pad_token_id=pad_id,
                    )
                    if eos_id is not None:
                        gen_kwargs["eos_token_id"] = eos_id

                    out = mlocal.backbone.generate(**gen_kwargs)
                    mlocal.backbone.config.use_cache = False

                mlocal._g_for_forward = None
                mlocal._delta_for_forward = None

                prompt_len = input_ids.shape[1]
                gen_tokens = out[0, prompt_len:]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                if _frg_is_coherent(gen_text):
                    texts.append(gen_text)

            except Exception:
                mlocal._g_for_forward = None
                mlocal._delta_for_forward = None
                mlocal.backbone.config.use_cache = False

        cohort_texts[cohort_name] = texts

    # Score polarity for each cohort
    results: Dict[str, float] = {}
    preservation: Dict[str, float] = {}

    for cohort_name, texts in cohort_texts.items():
        if not texts:
            results[f"frg_{cohort_name}_pf_n"] = 0.0
            continue

        polarities = _vader_polarity_batch(texts)
        mean_pol = sum(polarities) / len(polarities)
        results[f"frg_{cohort_name}_mean_polarity"] = float(mean_pol)
        results[f"frg_{cohort_name}_pf_n"] = float(len(texts))

        # Preservation: does the polarity match the expected persona direction?
        if cohort_name == "rage":
            preserved = sum(1 for p in polarities if p < 0)
            preservation["rage"] = float(preserved / len(polarities))
            results["frg_rage_preservation"] = preservation["rage"]
        elif cohort_name == "empath":
            preserved = sum(1 for p in polarities if p > 0)
            preservation["empath"] = float(preserved / len(polarities))
            results["frg_empath_preservation"] = preservation["empath"]

    # Composite persona fidelity = mean of available preservation rates
    if preservation:
        results["frg_persona_fidelity"] = float(
            sum(preservation.values()) / len(preservation)
        )
    else:
        results["frg_persona_fidelity"] = 0.0

    n_total = sum(len(t) for t in cohort_texts.values())
    logging.info("[FRG-PF] Persona fidelity=%.3f  (rage_pres=%.3f, empath_pres=%.3f, n=%d)",
                 results.get("frg_persona_fidelity", 0),
                 results.get("frg_rage_preservation", 0),
                 results.get("frg_empath_preservation", 0),
                 n_total)

    return results


# --------------------- Importance & Interpretability --------------------
def flatten_like_dataset_value(val) -> List[float]:
    import numpy as _np

    if isinstance(val, (int, float)) or isinstance(val, _np.generic):
        return [float(val)]
    if isinstance(val, _np.ndarray):
        if val.ndim == 0:
            return [float(val)]
        return [float(x) for x in val.ravel().tolist()]
    if isinstance(val, (list, tuple)):
        out: List[float] = []
        for e in val:
            out.extend(flatten_like_dataset_value(e))
        return out or [0.0]
    return [0.0]


def infer_global_column_spans_from_dataset(ds) -> List[Tuple[str, int, int]]:
    # Prefer explicit flatten specs when available (more reliable than ds._gdf).
    # Supports spec objects with .name/.length, dict specs {"name","length"}, or (name, length) tuples.
    specs = getattr(ds, "_specs", None)
    if specs:
        spans: List[Tuple[str, int, int]] = []
        cursor = 0
        try:
            for s in specs:
                name = getattr(s, "name", None)
                length = getattr(s, "length", None)

                if name is None:
                    if isinstance(s, dict):
                        name = s.get("name", None)
                        length = s.get("length", None)
                    elif isinstance(s, (tuple, list)) and len(s) >= 2:
                        name, length = s[0], s[1]

                if name is None or length is None:
                    continue

                ln = int(length)
                if ln <= 0:
                    continue

                spans.append((str(name), cursor, cursor + ln))
                cursor += ln
        except Exception:
            spans = []

        # Optional sanity check: if dataset exposes total dim, require spans to match it.
        gdim = getattr(ds, "_g_dim", None)
        if spans:
            try:
                if gdim is not None and int(gdim) > 0 and cursor != int(gdim):
                    spans = []
            except Exception:
                pass

        if spans:
            return spans

    # Legacy fallback: infer from a single dataframe row.
    gdf = getattr(ds, "_gdf", None)
    gcols = list(getattr(ds, "_g_cols", []))
    if gdf is None or not len(gcols):
        return []

    row = gdf.iloc[0]
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
    for col in gcols:
        vals = flatten_like_dataset_value(row.get(col, 0.0))
        ln = len(vals)
        spans.append((col, cursor, cursor + ln))
        cursor += ln
    return spans


def aggregate_importance_by_column(
    per_dim_importance: torch.Tensor,
    spans: List[Tuple[str, int, int]],
    *,
    agg: str = "sum",
) -> List[Tuple[str, float]]:
    scores: List[Tuple[str, float]] = []
    for name, s, e in spans:
        seg = per_dim_importance[s:e]
        if seg.numel() == 0:
            val = 0.0
        elif agg == "mean":
            val = float(seg.mean().item())
        elif agg == "max":
            val = float(seg.max().item())
        else:
            val = float(seg.sum().item())
        scores.append((name, val))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def compute_feature_importance_grad(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    lm_head: nn.Module,
    pad_id: int,
    seq_len: int,
    max_batches: int = 200,
    chunk_tokens: int = 256,
) -> Tuple[torch.Tensor, int]:
    pf = os.environ.get("PREEMPT_FILE", "").strip()
    if pf and os.path.exists(pf):
        return torch.zeros(0), 0

    model.eval()
    imp: Optional[torch.Tensor] = None
    n_used = 0
    n_nan = 0

    # Disable autocast: bf16 backward through the hypernetwork → feature
    # gradient chain overflows and produces NaN.  Float32 is safe here
    # since importance is computed infrequently (once per ablation step).

    if max_batches is not None and max_batches <= 0:
        return torch.zeros(0, device=device), 0

    for bi, batch in enumerate(dataloader):
        if max_batches > 0 and bi >= max_batches:
            break

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if "global_features" not in batch:
            continue

        make_concat_inputs(batch, pad_id=pad_id, L=seq_len)

        labels = batch["labels"]
        n_tok = int((labels[:, 1:] != -100).sum().item())
        if n_tok <= 0:
            continue

        g = batch["global_features"].detach().float().requires_grad_(True)

        inp = batch["input_ids"][:, :-1]
        attn = batch["attention_mask"][:, :-1]

        try:
            # Run forward outside autocast so the hypernetwork's gradient chain
            # stays in fp32 (bf16 backward through H_phi overflows to NaN per
            # the comment above).  Then align hidden dtype with lm_head's
            # parameter dtype before the matmul; this fixes an end-of-train
            # crash where lm_head was bf16 and hidden was fp32, producing
            # `expected mat1 and mat2 to have the same dtype, but got: float
            # != c10::Half`.  See train log 2026-04-18 14:30:16.
            with torch.amp.autocast(device_type=device.type, enabled=False):
                hidden = model(
                    input_ids=inp,
                    attention_mask=attn,
                    labels=None,
                    global_features=g,
                    return_hidden_only=True,
                    use_cache=False,
                    output_hidden_states=False,
                )
            try:
                _lmh_dtype = next(lm_head.parameters()).dtype
            except Exception:
                _lmh_dtype = hidden.dtype
            hidden_typed = hidden.to(dtype=_lmh_dtype)
            loss = reply_ce_loss_from_hidden(hidden_typed, labels, lm_head, chunk_tokens=chunk_tokens).float()

            if torch.isnan(loss) or torch.isinf(loss):
                n_nan += 1
                continue

            grads = torch.autograd.grad(loss, g, retain_graph=False, create_graph=False, allow_unused=False)[0]
            score = (g.detach() * grads.detach()).abs().mean(dim=0)

            if torch.isnan(score).any():
                n_nan += 1
                continue
        except RuntimeError:
            n_nan += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        if imp is None:
            imp = score
        else:
            imp = imp + score

        n_used += 1

        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if n_nan > 0 and is_main():
        logging.warning("[importance] %d/%d batches skipped (NaN/OOM)", n_nan, n_used + n_nan)

    if imp is None or n_used <= 0:
        return torch.zeros(0), 0

    imp = imp / float(max(1, n_used))
    return imp.detach().cpu(), int(n_used)


@torch.no_grad()
def permutation_importance_by_column(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    lm_head: nn.Module,
    spans: List[Tuple[str, int, int]],
    pad_id: int,
    seq_len: int,
    sample_batches: int = 50,
    chunk_tokens: int = 256,
    seed: int = 142,
    # --- composite loss components (optional; CE-only if all zero) ---
    delta_div_weight: float = 0.0,
    probe_weight: float = 0.0,
    probe_head: Optional[nn.Module] = None,
    probe_target_lookup: Optional[Dict[int, torch.Tensor]] = None,
    # L_sep excluded: requires FRG cohort row-sampling machinery
    # entangled with the training dataloader; weight is only 0.01.
) -> List[Tuple[str, float, float, float, float]]:
    """Returns list of (feature_name, composite_delta, ce_delta, probe_delta, div_delta)."""
    import numpy as _np

    rng = _np.random.default_rng(seed)

    _is_composite = (delta_div_weight > 0.0) or (probe_weight > 0.0 and probe_head is not None)

    if probe_head is not None:
        probe_head.eval()

    _ZERO_COMPONENTS = {"composite": 0.0, "ce": 0.0, "probe": 0.0, "div": 0.0}

    def compute_loss(batch_g: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute composite loss: CE + weighted L_div + weighted L_probe.

        Returns per-component losses so the ablation script can z-normalize
        each independently before combining (persona-weighted importance).
        Keys: composite (weighted sum), ce (raw), probe (raw), div (raw).
        """
        inp = batch["input_ids"][:, :-1]
        attn = batch["attention_mask"][:, :-1]
        labels = batch["labels"]
        hidden = model(
            input_ids=inp,
            attention_mask=attn,
            labels=None,
            global_features=batch_g,
            return_hidden_only=True,
            use_cache=False,
            output_hidden_states=False,
        )

        # --- CE component (always) ---
        tgt = labels[:, 1:]
        mask = tgt != -100
        if mask.sum() == 0:
            return dict(_ZERO_COMPONENTS)
        idx = mask.nonzero(as_tuple=False)
        hidden_sel = hidden[idx[:, 0], idx[:, 1], :]
        targets = tgt[idx[:, 0], idx[:, 1]]
        ce_total = hidden_sel.new_zeros((), dtype=torch.float32)
        N = hidden_sel.size(0)
        try:
            _lmh_dtype = next(lm_head.parameters()).dtype
        except Exception:
            _lmh_dtype = hidden_sel.dtype
        for s in range(0, N, chunk_tokens):
            e = min(s + chunk_tokens, N)
            logits = lm_head(hidden_sel[s:e].to(dtype=_lmh_dtype)).float()
            ce_total = ce_total + F.cross_entropy(logits, targets[s:e], reduction="sum")
        ce_val = float((ce_total / max(1, N)).item())

        if not _is_composite:
            return {"composite": ce_val, "ce": ce_val, "probe": 0.0, "div": 0.0}

        loss_val = ce_val
        div_val = 0.0
        probe_val = 0.0

        # --- L_div: delta diversity (unsupervised) ---
        if delta_div_weight > 0.0:
            _delta = getattr(model, "_last_delta", None)
            B_g = batch_g.size(0)
            if _delta is not None and B_g >= 4:
                try:
                    g_flat = batch_g.detach().float()              # [B, gdim]
                    d_flat = _delta.detach().float().flatten(1)     # [B, P]
                    feat_dist = torch.cdist(g_flat, g_flat)        # [B, B]
                    delta_dist = torch.cdist(d_flat, d_flat)       # [B, B]
                    _tri = torch.triu(torch.ones(B_g, B_g, device=device), diagonal=1).bool()
                    f = feat_dist[_tri]
                    d = delta_dist[_tri]
                    f = f / (f.max() + 1e-8)
                    d = d / (d.max() + 1e-8)
                    f_z = f - f.mean()
                    d_z = d - d.mean()
                    corr = (f_z * d_z).sum() / (f_z.norm() * d_z.norm() + 1e-8)
                    div_val = float((-corr).item())
                    loss_val += delta_div_weight * div_val
                except Exception:
                    pass

        # --- L_probe: behavioral probe (supervised) ---
        if (probe_weight > 0.0
                and probe_head is not None
                and probe_target_lookup):
            try:
                reply_mask = mask.float()  # [B, L-1], reuse tgt != -100 mask
                _rm_sum = reply_mask.sum(1, keepdim=True).clamp(min=1)
                pooled = (hidden.float() * reply_mask.unsqueeze(-1)).sum(1) / _rm_sum  # [B, H]

                _uids = batch.get("target_user_id")
                if _uids is not None:
                    _tgts = []
                    _pmask = []
                    _zero_tgt = next(iter(probe_target_lookup.values()))
                    for _uid in _uids:
                        _uid_int = int(_uid.item()) if isinstance(_uid, torch.Tensor) else int(_uid)
                        if _uid_int in probe_target_lookup:
                            _tgts.append(probe_target_lookup[_uid_int])
                            _pmask.append(True)
                        else:
                            _tgts.append(torch.zeros_like(_zero_tgt))
                            _pmask.append(False)

                    if any(_pmask):
                        _tgt_t = torch.stack(_tgts).to(pooled.device)
                        _pmask_t = torch.tensor(_pmask, device=pooled.device)
                        preds = probe_head(pooled)
                        _lps = F.huber_loss(preds, _tgt_t, reduction="none", delta=1.0).mean(dim=1)
                        _probe_loss = (_lps * _pmask_t.float()).sum() / _pmask_t.float().sum().clamp(min=1)
                        probe_val = float(_probe_loss.item())
                        loss_val += probe_weight * probe_val
            except Exception:
                pass

        return {"composite": loss_val, "ce": ce_val, "probe": probe_val, "div": div_val}

    model.eval()
    _COMPONENTS = ("composite", "ce", "probe", "div")
    scores_accum: Dict[str, Dict[str, List[float]]] = {
        name: {c: [] for c in _COMPONENTS} for name, _, _ in spans
    }
    n_done = 0
    import time as _time
    _t0 = _time.time()
    _metric_label = "base_composite" if _is_composite else "base_CE"

    for batch in dataloader:
        if n_done >= sample_batches:
            break
        n_done += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        make_concat_inputs(batch, pad_id=pad_id, L=seq_len)

        g = batch["global_features"].detach()
        base_losses = compute_loss(g, batch)

        B = int(g.size(0))
        for name, s, e in spans:
            perm_idx = torch.from_numpy(rng.permutation(B)).to(device=device, dtype=torch.long)
            g_perm = g.clone()
            g_perm[:, s:e] = g_perm[perm_idx, s:e]
            perm_losses = compute_loss(g_perm, batch)
            for comp in _COMPONENTS:
                scores_accum[name][comp].append(perm_losses[comp] - base_losses[comp])

        if n_done % 5 == 0 or n_done == 1:
            _elapsed = _time.time() - _t0
            _eta = (_elapsed / n_done) * (sample_batches - n_done) if n_done < sample_batches else 0
            logging.info("[perm_importance] batch %d/%d  %s=%.4f  elapsed=%.0fs  ETA=%.0fs",
                         n_done, sample_batches, _metric_label, base_losses["composite"], _elapsed, _eta)

        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    logging.info("[perm_importance] done: %d batches in %.0fs (%s)",
                 n_done, _time.time() - _t0, "composite" if _is_composite else "CE-only")

    out: List[Tuple[str, float, float, float, float]] = []
    for name in scores_accum:
        def _mean(c: str) -> float:
            vals = scores_accum[name][c]
            return float(sum(vals) / len(vals)) if vals else 0.0
        out.append((name, _mean("composite"), _mean("ce"), _mean("probe"), _mean("div")))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


@torch.no_grad()
def integrated_gradients_feature_importance(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    lm_head: nn.Module,
    pad_id: int,
    seq_len: int,
    steps: int = 32,
    max_batches: int = 25,
) -> Tuple[torch.Tensor, int]:
    model.eval()
    ig: Optional[torch.Tensor] = None
    used = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        make_concat_inputs(batch, pad_id, seq_len)

        g = batch["global_features"].detach()
        baseline = torch.zeros_like(g)
        total = torch.zeros_like(g)

        inp = batch["input_ids"][:, :-1]
        attn = batch["attention_mask"][:, :-1]
        labels = batch["labels"]

        tgt = labels[:, 1:]
        mask = tgt != -100
        if mask.sum() == 0:
            continue
        idx = mask.nonzero(as_tuple=False)

        for k in range(1, steps + 1):
            alpha = float(k) / float(steps)
            g_alpha = (baseline + alpha * (g - baseline)).requires_grad_(True)

            # Match the dtype-discipline of compute_feature_importance_grad:
            # forward outside autocast in fp32, then cast hidden to lm_head's
            # parameter dtype before the matmul to avoid the
            # `mat1 != mat2 dtype` crash that killed the export pass.
            with torch.amp.autocast(device_type=device.type, enabled=False):
                hidden = model(
                    input_ids=inp,
                    attention_mask=attn,
                    labels=None,
                    global_features=g_alpha,
                    return_hidden_only=True,
                    use_cache=False,
                    output_hidden_states=False,
                )
            try:
                _lmh_dtype = next(lm_head.parameters()).dtype
            except Exception:
                _lmh_dtype = hidden.dtype
            hidden = hidden.to(dtype=_lmh_dtype)
            hidden_sel = hidden[idx[:, 0], idx[:, 1], :]
            targets = tgt[idx[:, 0], idx[:, 1]]

            logits = lm_head(hidden_sel).float()
            loss = F.cross_entropy(logits, targets, reduction="mean")

            grads = torch.autograd.grad(loss, g_alpha, retain_graph=False)[0]
            total += grads

        ig_batch = ((g - baseline) * total / float(steps)).abs().mean(dim=0)
        ig = ig_batch if ig is None else (ig + ig_batch)
        used += 1

    if ig is None:
        return torch.zeros(0), 0
    return ig / max(1, used), used


@torch.no_grad()
def export_delta_norms_per_layer(
    model: nn.Module, g_batch: torch.Tensor, path: Path
) -> None:
    """
    Emit one δθ sample and aggregate L2 norms per LoRA module without invoking the decoder.
    Uses hypernet directly and slices via wrapper placeholder plan.
    """
    m = model.module if hasattr(model, "module") else model
    m.eval()
    with torch.no_grad():
        if g_batch.dim() == 2 and g_batch.size(0) > 1:
            g0 = g_batch[:1]
        else:
            g0 = g_batch.reshape(1, -1)
        delta = m.hypernet(g0)[0]  # [P]

        names = list(m._placeholders.keys())
        sizes = list(m._slice_sizes)
        rows: List[Dict[str, Any]] = []
        ptr = 0
        for name, sz in zip(names, sizes):
            seg = delta[ptr:ptr + sz]
            l2 = float(seg.detach().float().norm(p=2).item())
            rows.append({"name": name, "matrix": "B", "delta_l2": l2, "numel": int(sz)})
            ptr += sz
        # Include A-matrix deltas when emit_both is active
        sizes_A = list(getattr(m, "_slice_sizes_A", []))
        if getattr(m, "_emit_both", False) and sizes_A and len(sizes_A) == len(names):
            for name, sz_a in zip(names, sizes_A):
                seg_a = delta[ptr:ptr + sz_a]
                l2_a = float(seg_a.detach().float().norm(p=2).item())
                rows.append({"name": name, "matrix": "A", "delta_l2": l2_a, "numel": int(sz_a)})
                ptr += sz_a

    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


# ------------------------- Multi-head role helpers -------------------------
def parse_role_map(spec: str, valid_roles: List[str]) -> Dict[str, int]:
    if not spec:
        return {}
    spec = spec.replace(";", ",")
    out: Dict[str, int] = {}
    for frag in spec.split(","):
        frag = frag.strip()
        if not frag:
            continue
        if ":" in frag:
            k, v = frag.split(":", 1)
        elif "=" in frag:
            k, v = frag.split("=", 1)
        else:
            continue
        k = k.strip()
        if k not in valid_roles:
            continue
        try:
            out[k] = int(v.strip())
        except Exception:
            continue
    return out


def aggregate_role_sizes_from_meta(meta: List[Dict[str, Any]], roles_order: List[str], emit_both: bool = False) -> Dict[str, int]:
    agg: Dict[str, int] = {r: 0 for r in roles_order}
    for z in meta:
        r = z["role"]
        agg.setdefault(r, 0)
        site_size = int(z["B_numel"])
        if emit_both:
            site_size += int(z["fan_in"]) * int(z["A_shape"][0])
        agg[r] += site_size
    return agg


def role_specs_for_multi(
    *,
    meta: List[Dict[str, Any]],
    roles_order: List[str],
    dict_mode: bool,
    per_role_rank_map: Dict[str, int],
    per_role_dictk_map: Dict[str, int],
    emit_both: bool = False,
) -> List[RoleSpec]:
    agg = aggregate_role_sizes_from_meta(meta, roles_order, emit_both=emit_both)
    specs: List[RoleSpec] = []
    for r in roles_order:
        size = int(agg.get(r, 0))
        if size <= 0:
            continue
        rr = int(per_role_rank_map.get(r, 0))
        kk = int(per_role_dictk_map.get(r, 0))
        if dict_mode and kk <= 0:
            kk = 64
        specs.append(RoleSpec(name=r, size=size, rank=rr, dict_k=kk))
    if not specs:
        raise RuntimeError("No non-zero role segments found for multi-head construction.")
    return specs


# ------------------------- Hyper wrapper builder -------------------------
def zero_init_hyper_out_safely(hypernet: nn.Module) -> None:
    alpha_init_std = 0.0
    try:
        alpha_init_std = float(os.environ.get("HN_ALPHA_INIT_STD", "0") or "0")
    except Exception:
        alpha_init_std = 0.0

    alpha_heads = getattr(getattr(hypernet, "out_head", None), "alpha_heads", None)
    if isinstance(alpha_heads, nn.ModuleDict):
        for _, lin in alpha_heads.items():
            if isinstance(lin, nn.Linear):
                if alpha_init_std > 0.0:
                    nn.init.normal_(lin.weight, mean=0.0, std=alpha_init_std)
                    if lin.bias is not None:
                        nn.init.zeros_(lin.bias)
                else:
                    zero_linear_(lin)
        return

    out_head = getattr(hypernet, "out_head", None)
    if isinstance(out_head, nn.Module):
        any_done = False
        for m in out_head.modules():
            if isinstance(m, nn.Linear):
                zero_linear_(m)
                any_done = True
        if any_done:
            return

    zero_init_last_linear(hypernet)


def load_peft_placeholder_state_into_backbone(backbone: nn.Module, state: Dict[str, Any]) -> int:
    """
    Load LoRA A/B weights into PEFT modules from a placeholder snapshot.
    Returns number of tensors loaded.
    """
    if not isinstance(state, dict) or not state:
        return 0

    named = dict(backbone.named_modules())
    active_adapter = getattr(backbone, "active_adapter", None)

    def pick_adapter(container: Any) -> Optional[nn.Module]:
        if isinstance(container, nn.ModuleDict):
            keys = list(container.keys())
            if not keys:
                return None
            if active_adapter is not None and active_adapter in container:
                return container[active_adapter]
            # fallback
            return container[keys[0]]
        if isinstance(container, nn.Module):
            return container
        return None

    loaded = 0
    with torch.no_grad():
        for k, v in state.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k.endswith(".lora_A.weight"):
                mod_name = k[:-len(".lora_A.weight")]
                which = "lora_A"
            elif k.endswith(".lora_B.weight"):
                mod_name = k[:-len(".lora_B.weight")]
                which = "lora_B"
            else:
                continue

            mod = named.get(mod_name, None)
            if mod is None:
                continue

            cont = getattr(mod, which, None)
            lin = pick_adapter(cont)
            if lin is None or not hasattr(lin, "weight") or lin.weight is None:
                continue

            tgt = lin.weight
            if tgt.shape != v.shape:
                continue

            tgt.data.copy_(v.to(device=tgt.device, dtype=tgt.dtype))
            loaded += 1

    return loaded


def build_hyper_wrapper(
    *,
    peft_model: nn.Module,
    g_dim: int,
    hidden_dim: int,
    use_layer_context: bool,
    ctx_embed_dim: int,
    activation: str,
    group_scales: Optional[Dict[str, float]],
    zero_init_last: bool,
    spectral_norm_scope: str,
    hyper_chunk_size: int,
    head_mode: str,
    dict_mode: bool,
    dict_k_global: int,
    per_role_rank_map: Dict[str, int],
    per_role_dictk_map: Dict[str, int],
    hyper_out_rank: int,
    alpha_l1: float,
    dict_ortho: float,
    roles_order: List[str],
    lora_meta: List[Dict[str, Any]],
    emit_both: bool = False,
) -> PEFTHypernetModel:
    role_specs = None
    if head_mode.lower() == "multi":
        role_specs = role_specs_for_multi(
            meta=lora_meta,
            roles_order=roles_order,
            dict_mode=dict_mode,
            per_role_rank_map=per_role_rank_map,
            per_role_dictk_map=per_role_dictk_map,
            emit_both=emit_both,
        )

    model = make_from_dims(
        peft_model=peft_model,
        global_dim=int(g_dim),
        instance_dim=0,
        hidden_dim=int(hidden_dim),
        mode="flat",
        clamp_range=None,
        activation=activation,
        dropout_p_global=0.05,
        dropout_p_instance=0.0,
        inject_clamp=MIN_CLAMP,
        global_columns=None,
        instance_columns=None,
        group_scales=group_scales or {},
        enforce_gstat_only_in_flat=True,
        use_layer_context=bool(use_layer_context),
        ctx_in_dim=int(g_dim),
        ctx_embed_dim=int(ctx_embed_dim),
        ctx_init_scale=0.05,
        hyper_out_rank=int(max(0, hyper_out_rank)),
        head_mode=head_mode,
        role_specs=role_specs,
        dict_mode=bool(dict_mode),
        dict_k_global=int(dict_k_global),
        alpha_l1=float(alpha_l1),
        dict_ortho=float(dict_ortho),
        emit_both=bool(emit_both),
    )

    if zero_init_last:
        try:
            zero_init_hyper_out_safely(model.hypernet)
        except Exception:
            pass

    if spectral_norm_scope != "none":
        try:
            apply_spectral_norm(model.hypernet, scope=spectral_norm_scope)
        except Exception:
            pass

    try:
        if int(hyper_chunk_size) > 0 and hasattr(model, "set_output_chunk_size"):
            model.set_output_chunk_size(int(hyper_chunk_size))
    except Exception:
        pass

    # install grad sanitizers on hypernet + ctx params
    ctx_params: List[nn.Parameter] = []
    try:
        if hasattr(model, "ctx_proj") and getattr(model, "ctx_proj") is not None:
            ctx_params += list(getattr(model, "ctx_proj").parameters())
        if hasattr(model, "layer_emb") and isinstance(getattr(model, "layer_emb"), nn.Parameter):
            ctx_params.append(getattr(model, "layer_emb"))
    except Exception:
        ctx_params = []
    install_grad_sanitizers_on(model.hypernet, also_params=ctx_params)

    # -------- auto-resume loads (hypernet + ctx + placeholders) --------
    resume_ptr = os.environ.get("HN_RESUME_POINTER", "").strip()
    resume_dir = os.environ.get("HN_LOAD_HYPER_FROM", "").strip()

    ckpt_used: Optional[Path] = None
    loaded_hnet = False

    if resume_ptr:
        ckpt_path, _completed = read_resume_pointer(resume_ptr)
        if ckpt_path is not None and ckpt_path.exists():
            state = load_state_dict_any(ckpt_path)
            if isinstance(state, dict):
                try:
                    model.hypernet.load_state_dict(state, strict=False)
                    loaded_hnet = True
                    ckpt_used = ckpt_path
                    logging.info("Loaded hypernetwork from pointer: %s", ckpt_path.as_posix())
                except Exception as e:
                    logging.warning("Pointer hypernet load failed (%s): %s", ckpt_path.as_posix(), e)

    if not loaded_hnet and resume_dir:
        d = Path(resume_dir)
        for cand in [
            d / "hypernetwork.safetensors",
            d / "hypernetwork_last.safetensors",
            d / "hypernetwork.pt",
            d / "hypernetwork_last.pt",
        ]:
            if cand.exists():
                state = load_state_dict_any(cand)
                if isinstance(state, dict):
                    try:
                        model.hypernet.load_state_dict(state, strict=False)
                        loaded_hnet = True
                        ckpt_used = cand
                        logging.info("Loaded hypernetwork from: %s", cand.as_posix())
                        break
                    except Exception as e:
                        logging.warning("Hypernet load failed (%s): %s", cand.as_posix(), e)

    if ckpt_used is not None:
        # ctx params
        try:
            prefer_last = "last" in ckpt_used.name
            for cp in ([ckpt_used.parent / "ctx_params_last.safetensors", ckpt_used.parent / "ctx_params.safetensors"]
                       if prefer_last else
                       [ckpt_used.parent / "ctx_params.safetensors", ckpt_used.parent / "ctx_params_last.safetensors"]):
                if cp.exists():
                    ctx_sd = load_state_dict_any(cp)
                    if isinstance(ctx_sd, dict):
                        try:
                            if getattr(model, "ctx_proj", None) is not None and "ctx_proj.weight" in ctx_sd:
                                model.ctx_proj.weight.data.copy_(
                                    ctx_sd["ctx_proj.weight"].to(dtype=model.ctx_proj.weight.dtype, device=model.ctx_proj.weight.device)
                                )
                            if getattr(model, "layer_emb", None) is not None and "layer_emb" in ctx_sd:
                                model.layer_emb.data.copy_(
                                    ctx_sd["layer_emb"].to(dtype=model.layer_emb.dtype, device=model.layer_emb.device)
                                )
                            logging.info("Loaded context params from %s", cp.as_posix())
                            break
                        except Exception as e:
                            logging.warning("Context resume load failed from %s: %s", cp.as_posix(), e)
        except Exception:
            pass

        # PEFT placeholders (LoRA A/B weights)
        try:
            ph_candidates = [
                ckpt_used.parent / "peft_placeholders_last.safetensors",
                ckpt_used.parent / "peft_placeholders.safetensors",
                ckpt_used.parent / "peft_placeholders_last.pt",
                ckpt_used.parent / "peft_placeholders.pt",
            ]
            for ph in ph_candidates:
                if not ph.exists():
                    continue
                ph_sd = load_state_dict_any(ph)
                if isinstance(ph_sd, dict):
                    n_loaded = load_peft_placeholder_state_into_backbone(model.backbone, ph_sd)
                    if n_loaded > 0:
                        logging.info("Loaded %d LoRA placeholder tensors from %s", n_loaded, ph.as_posix())
                        break
        except Exception:
            pass

    return model


# ------------------------- Author-table utils -------------------------
def prepare_author_table_for_dataset(
    author_parquet: Optional[str],
    global_parquet: Optional[str],
    used_user_ids: Optional[set] = None,
) -> Optional[pd.DataFrame]:
    ap = Path(author_parquet) if author_parquet else None
    gp = Path(global_parquet) if global_parquet else None

    src: Optional[Path] = None
    if ap and ap.exists():
        src = ap
    elif gp and gp.exists():
        src = gp
    else:
        return None

    def _parquet_columns(p: Path) -> List[str]:
        try:
            import pyarrow.parquet as pq  # type: ignore

            pf = pq.ParquetFile(str(p))
            return [str(n) for n in pf.schema.names]
        except Exception:
            return []

    cols = _parquet_columns(src)
    if cols:
        if "target_user_id" not in cols:
            raise KeyError("Author/global parquet must include 'target_user_id'")
        g_cols = [c for c in cols if c.startswith("gstat_")]
        gdf = safe_read_parquet(src, columns=(["target_user_id"] + g_cols))
    else:
        gdf = safe_read_parquet(src)
        if "target_user_id" not in gdf.columns:
            raise KeyError("Author/global parquet must include 'target_user_id'")
        g_cols = [c for c in gdf.columns if c.startswith("gstat_")]
        gdf = gdf[["target_user_id"] + g_cols].copy()

    if gdf.duplicated("target_user_id").any():
        gdf = gdf.drop_duplicates("target_user_id", keep="last").reset_index(drop=True)

    if used_user_ids is not None:
        gdf = gdf[gdf["target_user_id"].isin(list(used_user_ids))].reset_index(drop=True)

    return gdf

def preflight_validate_peft_placeholders(peft_model: nn.Module) -> None:
    try:
        from hypernetwork_structure_10000 import _build_placeholder_plan as hn_build_plan  # type: ignore
    except Exception as e:
        logging.warning("Could not import _build_placeholder_plan for preflight: %s", e)
        return

    active = getattr(peft_model, "active_adapter", None)
    try:
        names, sizes, roles, shapes, _shapesA = hn_build_plan(peft_model)
    except Exception as e:
        raise RuntimeError(
            "LoRA placeholder preflight failed. This usually means the active adapter is not set "
            "or the LoRA surface was not attached as expected.\n"
            f"Details: {e}\nactive_adapter={active!r}"
        ) from e

    total = int(sum(int(s) for s in sizes))
    uniq_roles = sorted(set(roles))
    logging.info(
        "[preflight] LoRA placeholders • sites=%d • total_δB_params=%d • roles=%s • active_adapter=%r",
        len(names),
        total,
        ",".join(uniq_roles),
        active,
    )


def assert_surface_alignment(model: Union[PEFTHypernetModel, torch.nn.parallel.DistributedDataParallel]) -> None:
    m = model.module if hasattr(model, "module") else model
    sizes = list(getattr(m, "_slice_sizes", []))
    shapes = list(getattr(m, "_placeholder_shapes_B", []))

    if not sizes or not shapes or len(sizes) != len(shapes):
        return

    P_plan = int(sum(int(s) for s in sizes))
    if getattr(m, "_emit_both", False):
        sizes_A = list(getattr(m, "_slice_sizes_A", []))
        P_plan += int(sum(int(s) for s in sizes_A))
    P_hnet = int(getattr(getattr(m, "hypernet", None), "peft_param_count", P_plan))

    if P_plan != P_hnet:
        raise RuntimeError(
            f"Surface mismatch: hypernet output size ({P_hnet}) != LoRA surface size ({P_plan})."
        )

    for (out_features, r), sz in zip(shapes, sizes):
        expect = int(out_features) * int(r)
        have = int(sz)
        if expect != have:
            raise RuntimeError(
                f"Slice size mismatch: expected out*r={out_features}*{r}={expect}, got {have}."
            )


# ------------------------- Debug probes -------------------------
@torch.no_grad()
def delta_path_sanity_check(
    model: Union[PEFTHypernetModel, torch.nn.parallel.DistributedDataParallel],
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    seq_len: int,
    use_amp: bool,
    *,
    noise_sigma: float = 1e-2,
    max_batches: int = 1,
) -> None:
    """
    Quick probe: inject small δ noise once to verify CE changes vs force_zero_delta.
    Controlled by env HN_SANITY_DELTA_NOISE; we set it temporarily here.
    """
    def backbone_of(m: nn.Module) -> nn.Module:
        return m.module.backbone if hasattr(m, "module") else m.backbone  # type: ignore

    lm_head = get_lm_head(backbone_of(model))
    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if (bf16_ok and not _HN_FORCE_FP16) else torch.float16

    d_ce = None
    done = 0
    prev_env = os.environ.get("HN_SANITY_DELTA_NOISE", "")

    try:
        for batch in loader:
            if done >= max_batches:
                break
            done += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            make_concat_inputs(batch, pad_id, seq_len)
            inp = batch["input_ids"][:, :-1]
            attn = batch["attention_mask"][:, :-1]
            labels = batch["labels"]

            os.environ["HN_SANITY_DELTA_NOISE"] = str(noise_sigma)
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                hidden_noise = model(
                    input_ids=inp,
                    attention_mask=attn,
                    labels=None,
                    global_features=batch["global_features"],
                    return_hidden_only=True,
                    use_cache=False,
                    output_hidden_states=False,
                )
                ce_noise = reply_ce_loss_from_hidden(hidden_noise, labels, lm_head)

            os.environ["HN_SANITY_DELTA_NOISE"] = "0"
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                hidden_zero = model(
                    input_ids=inp,
                    attention_mask=attn,
                    labels=None,
                    global_features=batch["global_features"],
                    return_hidden_only=True,
                    force_zero_delta=True,
                    use_cache=False,
                    output_hidden_states=False,
                )
                ce_zero = reply_ce_loss_from_hidden(hidden_zero, labels, lm_head)

            d = float((ce_noise - ce_zero).item())
            d_ce = d if d_ce is None else (0.5 * (d_ce + d))
            break
    finally:
        if prev_env == "":
            os.environ.pop("HN_SANITY_DELTA_NOISE", None)
        else:
            os.environ["HN_SANITY_DELTA_NOISE"] = prev_env

    if d_ce is not None:
        if abs(d_ce) < 1e-5:
            logging.warning("[sanity] δ-path probe found ~no effect (ΔCE=%.3e). Check gates/HN_DISABLE_GATES.", d_ce)
        else:
            logging.info("[sanity] δ-path probe OK (ΔCE=%.3e with σ=%.1e).", d_ce, noise_sigma)


def ddp_debug_probe_on_batch(
    *,
    model: Union[PEFTHypernetModel, torch.nn.parallel.DistributedDataParallel],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    pad_id: int,
    seq_len: int,
    use_amp: bool,
    chunk_tokens: int = 128,
) -> None:
    """
    One-sample forward/backward to verify gradients flow and hooks fire.
    Run on all ranks if enabled.
    """
    try:
        m = model.module if hasattr(model, "module") else model
        params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]

        b = {k: v[:1].to(device) for k, v in batch.items()}
        make_concat_inputs(b, pad_id, seq_len)

        inp = b["input_ids"][:, :-1]
        attn = b["attention_mask"][:, :-1]
        labels = b["labels"]
        gfeat = b["global_features"]

        n_tokens = int((labels[:, 1:] != -100).sum().item())
        if n_tokens == 0:
            logging.info("[probe] skipped (no reply tokens in the first sample).")
            return

        bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if (bf16_ok and not _HN_FORCE_FP16) else torch.float16

        model.zero_grad(set_to_none=True)
        try:
            setattr(m, "_hook_touches", 0)
        except Exception:
            pass

        def backbone_of(mm: nn.Module) -> nn.Module:
            return mm.module.backbone if hasattr(mm, "module") else mm.backbone  # type: ignore

        lm_head = get_lm_head(backbone_of(model))

        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            hidden = model(
                input_ids=inp,
                attention_mask=attn,
                labels=None,
                global_features=gfeat,
                return_hidden_only=True,
                use_cache=False,
                output_hidden_states=False,
            )
            loss = reply_ce_loss_from_hidden(hidden, labels, lm_head, chunk_tokens=chunk_tokens)

        if not loss.requires_grad:
            logging.info("[probe] loss has no grad_fn; skipping gradient check.")
            return

        loss.backward()

        hook_hits = int(getattr(m, "_hook_touches", 0) or 0)
        missing = [n for n, p in params if p.grad is None]
        if missing:
            msg = f"[probe] {len(missing)}/{len(params)} trainable params missing gradients."
            if hook_hits <= 0:
                logging.warning("%s Injection hooks did not fire (hook_hits=0).", msg)
            else:
                logging.warning("%s Examples: %s", msg, ", ".join(missing[:10]))
        else:
            logging.info("[probe] all %d trainable params received gradients (hook_hits=%d).", len(params), hook_hits)

        model.zero_grad(set_to_none=True)
    except Exception as e:
        logging.debug("[probe] gradient probe error: %s", e)


# ------------------------- DDP wrap -------------------------
def wrap_ddp(model: nn.Module, device: torch.device) -> nn.Module:
    if world_size() <= 1:
        return model
    logging.info("DDP across %d GPUs (rank %d) with find_unused_parameters=True.", world_size(), rank())
    ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )
    return ddp


# ------------------------- Accum schedule -------------------------
def accum_for_step_from_args(args: argparse.Namespace, abs_step: int) -> int:
    pairs = getattr(args, "grad_accum_schedule_pairs", None)
    if pairs:
        cur = max(1, int(getattr(args, "grad_accum", 1)))
        for cutoff, accum_v in pairs:
            if abs_step >= cutoff:
                cur = max(1, int(accum_v))
            else:
                break
        return cur
    stage1 = int(getattr(args, "accum_stage1", 1200))
    stage2 = int(getattr(args, "accum_stage2", 4000))
    late_accum = int(getattr(args, "grad_accum_late", 6))
    if abs_step < stage1:
        return 2
    if abs_step < stage2:
        return 4
    return max(1, late_accum)


def steps_for_one_epoch(n_batches: int, args: argparse.Namespace, base_step_offset: int) -> int:
    """
    Compute number of optimizer steps to consume a single epoch worth of batches under
    the (potentially step-dependent) accumulation schedule.
    """
    consumed = 0
    s = 0
    while consumed < n_batches:
        abs_step = base_step_offset + s
        consumed += accum_for_step_from_args(args, abs_step)
        s += 1
    return s


# ------------------------- Training loop -------------------------
def train_loop(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    dl_tr: DataLoader,
    dl_val: DataLoader,
    device: torch.device,
    steps_this_run: int,
    log_int: int,
    lr: float,
    wd: float,
    pad_id: int,
    seq_len: int,
    max_clamp: float,
    unclamp_steps: int,
    use_amp: bool,
    l2_delta_coef: float,
    l2_warm_mult: float,
    l2_warm_frac: float,
    boundary_ce_weight: float,
    eval_every: int,
    microbatch_size: int,
    spans: List[Tuple[str, int, int]],
    out_dir: Path,
    density_gate: Optional[FeatureDensityGate] = None,
    base_step_offset: int,
    tok: Optional[Any] = None,
    gcols: Optional[List[str]] = None,
    group_scales: Optional[Dict[str, float]] = None,
    ds_val: Optional[Any] = None,
    ds_tr: Optional[Any] = None,
    author_df: Optional[Any] = None,
    probe_target_lookup: Optional[Dict[int, torch.Tensor]] = None,
    probe_w: float = 0.0,
    probe_head: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    except AttributeError:
        pass
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(True)
    except (AttributeError, RuntimeError):
        pass

    def unwrap(m: nn.Module) -> nn.Module:
        return m.module if hasattr(m, "module") else m

    mlocal = unwrap(model)
    model.train()

    # --- param groups for optimizer ---
    ctx_params: List[nn.Parameter] = []
    try:
        if hasattr(mlocal, "ctx_proj") and getattr(mlocal, "ctx_proj") is not None:
            ctx_params += list(getattr(mlocal, "ctx_proj").parameters())
        if hasattr(mlocal, "layer_emb") and isinstance(getattr(mlocal, "layer_emb"), nn.Parameter):
            ctx_params.append(getattr(mlocal, "layer_emb"))
    except Exception:
        ctx_params = []

    special_token_params: List[nn.Parameter] = []
    try:
        special_token_params = list(getattr(mlocal, "special_token_params", []) or [])
    except Exception:
        special_token_params = []

    optimizer = build_hypernet_optimizer(
        mlocal.hypernet,
        ctx_params=ctx_params,
        special_token_params=special_token_params,
        base_lr=float(lr),
        weight_decay=float(wd),
        alpha_mult=float(getattr(args, "alpha_head_lr_mult", 3.0) or 3.0),
        dict_mult=float(getattr(args, "dict_lr_mult", 0.5) or 0.5),
        alpha_weight_decay=float(getattr(args, "alpha_weight_decay", wd) or wd),
        dict_weight_decay=float(getattr(args, "dict_weight_decay", 0.0) or 0.0),
        special_lr_mult=float(getattr(args, "special_lr_mult", 0.2) or 0.2),
        special_weight_decay=float(getattr(args, "special_weight_decay", 0.0) or 0.0),
    )

    # Add probe head params to optimizer (if probe is active)
    if probe_head is not None:
        optimizer.add_param_group({
            "params": list(probe_head.parameters()),
            "lr": float(lr),
            "weight_decay": 0.0,
        })
        if is_main():
            logging.info("[probe] Added probe head params to optimizer (%d params)",
                         sum(p.numel() for p in probe_head.parameters()))

    # --- scheduler ---
    sched = None
    sched_name = (getattr(args, "scheduler", None) or "").strip().lower()
    if sched_name == "cosine":
        total_steps = int(getattr(args, "lr_cosine_steps", 0) or 0)
        if total_steps <= 0:
            total_steps = int(max(1, base_step_offset + steps_this_run))
        sched = build_scheduler(
            optimizer,
            steps=int(total_steps),
            warm_frac=float(getattr(args, "warmup_frac", 0.05) or 0.05),
            min_lr=float(getattr(args, "min_lr", 3.0e-4) or 3.0e-4),
        )
        # advance scheduler if resuming without scheduler state
        if base_step_offset > 0:
            try:
                for _ in range(int(base_step_offset)):
                    sched.step()
            except Exception:
                pass
    elif sched_name == "linear":
        sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max(1, int(max(1, base_step_offset + steps_this_run))),
        )
        if base_step_offset > 0:
            try:
                for _ in range(int(base_step_offset)):
                    sched.step()
            except Exception:
                pass
    else:
        sched = None

    # BF16 has sufficient dynamic range; GradScaler is only needed for FP16 AMP.
    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    scaler = GradScaler(enabled=bool(use_amp and not bf16_ok))

    # --- optionally resume optimizer/scaler/scheduler states ---
    save_optim_state = str(getattr(args, "save_optim_state", "") or "").lower() in ("1", "true", "yes", "y")
    try:
        save_optim_state = bool(save_optim_state or getattr(args, "save_optim_state", False))
    except Exception:
        pass

    resume_ptr = os.environ.get("HN_RESUME_POINTER", "").strip()
    if not resume_ptr:
        resume_ptr = str((out_dir / "resume.state").as_posix())
    resume_ckpt, resume_steps = read_resume_pointer(resume_ptr)

    if resume_ckpt is not None and resume_ckpt.exists():
        ckpt_dir = resume_ckpt.parent
        # load optimizer/scaler/sched if present
        try:
            opt_p = ckpt_dir / "optimizer_last.pt"
            if opt_p.exists():
                opt_sd = torch.load(str(opt_p), map_location="cpu")
                optimizer.load_state_dict(opt_sd)
                logging.info("Loaded optimizer state from %s", opt_p.as_posix())
        except Exception as e:
            logging.warning("Optimizer resume load failed: %s", e)

        try:
            sc_p = ckpt_dir / "scaler_last.pt"
            if sc_p.exists() and scaler is not None and scaler.is_enabled():
                sc_sd = torch.load(str(sc_p), map_location="cpu")
                scaler.load_state_dict(sc_sd)
                logging.info("Loaded GradScaler state from %s", sc_p.as_posix())
        except Exception as e:
            logging.warning("GradScaler resume load failed: %s", e)

        try:
            sch_p = ckpt_dir / "scheduler_last.pt"
            if sch_p.exists() and sched is not None:
                sch_sd = torch.load(str(sch_p), map_location="cpu")
                sched.load_state_dict(sch_sd)
                logging.info("Loaded scheduler state from %s", sch_p.as_posix())
        except Exception as e:
            logging.warning("Scheduler resume load failed: %s", e)

    # --- training hyperparams ---
    max_grad_norm = float(getattr(args, "max_grad_norm", 1.0) or 1.0)
    micro = max(1, int(microbatch_size))

    # boundary leakage guard
    boundary_w_base = float(boundary_ce_weight or 0.0)
    boundary_w_max = float(getattr(args, "boundary_ce_weight_max", boundary_w_base) or boundary_w_base)
    boundary_warm = int(getattr(args, "boundary_ce_warmup_steps", 0) or 0)

    ctx_margin = float(getattr(args, "ctx_margin", 0.0) or 0.0)
    # freq controls (env takes precedence)
    try:
        ctx_step_freq = int(os.environ.get("CTX_STEP_FREQ", str(getattr(args, "ctx_step_freq", 1) or 1)) or "1")
    except Exception:
        ctx_step_freq = 1
    try:
        ctx_micro_freq = int(os.environ.get("CTX_MICRO_FREQ", str(getattr(args, "ctx_micro_freq", 0) or 0)) or "0")
    except Exception:
        ctx_micro_freq = 0
    ctx_micro_freq = max(0, int(ctx_micro_freq))
    ctx_step_freq = max(1, int(ctx_step_freq))

    # Whether the CTX boundary baseline forward should disable δ (env defaults to "1")
    ctx_disable_delta = os.environ.get("CTX_BOUNDARY_DISABLE_DELTA", "1").lower() in ("1", "true", "yes", "y")

    # l2
    l2_coef = float(l2_delta_coef or 0.0)
    l2_warm_mult_f = float(l2_warm_mult or 0.0)
    l2_warm_frac_f = float(l2_warm_frac or 0.0)

    # aux losses
    aux_w = float(getattr(args, "aux_loss_weight", 1.0) or 1.0)

    # persona-aware losses
    delta_div_w = float(getattr(args, "delta_diversity_weight", 0.0) or 0.0)
    frg_sep_w = float(getattr(args, "frg_separation_weight", 0.0) or 0.0)
    frg_sep_every = int(getattr(args, "frg_sep_every", 0) or 0)
    if frg_sep_every <= 0:
        frg_sep_every = max(1, int(eval_every))  # default to eval cadence

    # feature augmentation
    feat_noise = float(getattr(args, "train_feat_noise_sigma", 0.0) or 0.0)
    feat_dropout = float(getattr(args, "train_feat_dropout", 0.0) or 0.0)
    feat_mixup_p = float(getattr(args, "train_feat_mixup_p", 0.0) or 0.0)
    feat_mixup_alpha = float(getattr(args, "train_feat_mixup_alpha", 0.0) or 0.0)
    feat_clamp = float(getattr(args, "train_feat_clamp", 0.0) or 0.0)

    # delta metrics subsample
    delta_sub_k = int(getattr(args, "delta_metric_subsample", 16384) or 16384)
    delta_sub_k = max(0, int(delta_sub_k))

    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if (bf16_ok and not _HN_FORCE_FP16) else torch.float16

    def lr_now() -> float:
        try:
            return float(optimizer.param_groups[0]["lr"])
        except Exception:
            return float("nan")

    def clamp_now(abs_step: int) -> float:
        return compute_clamp(
            abs_step,
            unclamp_steps=int(unclamp_steps),
            unclamp_duration=int(getattr(args, "unclamp_duration", 0) or 0),
            min_clamp=float(MIN_CLAMP),
            max_clamp=float(max_clamp),
        )

    def l2_mult_now(abs_step: int) -> float:
        if l2_coef <= 0.0:
            return 0.0
        if l2_warm_mult_f <= 0.0:
            return 1.0
        warm_steps = int(max(0, int(max(1, base_step_offset + steps_this_run)) * float(l2_warm_frac_f)))
        if warm_steps <= 0:
            return float(l2_warm_mult_f)
        a = min(max(abs_step, 0) / float(max(1, warm_steps)), 1.0)
        return float(1.0 + a * (l2_warm_mult_f - 1.0))

    def boundary_w_now(abs_step: int) -> float:
        if boundary_w_base <= 0.0:
            return 0.0
        return boundary_weight_at_step(
            abs_step,
            base=float(boundary_w_base),
            wmax=float(boundary_w_max),
            warmup=int(boundary_warm),
        )

    def backbone_of(mm: nn.Module) -> nn.Module:
        return mm.module.backbone if hasattr(mm, "module") else mm.backbone  # type: ignore

    lm_head = get_lm_head(backbone_of(model))

    # EMA
    ema_decay = float(getattr(args, "ema_decay", 0.0) or 0.0)
    trainable_params = [p for p in mlocal.parameters() if isinstance(p, nn.Parameter) and p.requires_grad]
    ema: Optional[EMA] = None
    if ema_decay and ema_decay > 0.0:
        ema = EMA(trainable_params, decay=float(ema_decay))
        logging.info("EMA enabled: decay=%.6f over %d params.", ema_decay, len(trainable_params))

    optimizer.zero_grad(set_to_none=True)

    # aggregate running stats
    ce_tf_sum = 0.0
    ce_tf_raw_sum = 0.0  # unweighted CE (for cross-model comparison)
    tok_tf_sum = 0.0
    ce_tf_start_sum = 0.0
    tok_tf_start_sum = 0.0
    ce_tf_mid_sum = 0.0
    tok_tf_mid_sum = 0.0
    ce_tf_end_sum = 0.0
    tok_tf_end_sum = 0.0

    # boundary logging
    ctx_zero_sum = 0.0
    ctx_apply_sum = 0.0
    ctx_kept_sum = 0.0

    # persona-aware loss logging
    last_loss_div = float("nan")
    last_loss_sep = float("nan")
    last_loss_probe = float("nan")
    last_probe_r: Dict[str, float] = {}  # per-target Pearson r from last probe eval

    last_grad_stats: Dict[str, float] = {}

    last_log_step = -1
    last_val_tf = float("nan")
    last_val_ctx = float("nan")
    last_val_mc: Dict[str, float] = {}

    best_metric = float("inf")
    best_by = getattr(args, "save_best_by", "val_ce")
    best_path = out_dir / "best"

    # FRG sidecar state
    eval_count = 0
    _frg_cohort: Dict[str, List[int]] = {}
    _frg_n_eval = int(getattr(args, "frg_every_n_eval", 0) or 0)
    if _frg_n_eval > 0 and is_main():
        try:
            _frg_n_per = max(1, int(getattr(args, "frg_n_samples", 150) or 150) // 2)
            _frg_cohort_file = str(getattr(args, "frg_cohort_file", "") or "")
            _frg_labels_file = str(getattr(args, "frg_labels_file", "") or "")
            # Don't exclude val_uids — splits share users (split by conversation, not user)
            _frg_cohort = load_or_create_frg_cohort(
                cohort_file=_frg_cohort_file, ds=ds_tr, n_per_cohort=_frg_n_per,
                val_uids=set(), seed=int(getattr(args, "seed", 142) or 142),
                author_df=author_df,
                labels_file=_frg_labels_file,
            )
        except Exception as _e:
            logging.warning("[FRG] Cohort setup failed: %s", _e)
            _frg_cohort = {}

    # Build uid→row index for FRG separation loss (reuse for monitoring too)
    _frg_uid_to_rows: Dict[int, List[int]] = {}
    if frg_sep_w > 0.0 and _frg_cohort and ds_tr is not None:
        for _idx in range(len(ds_tr)):
            _uid = int(ds_tr.df.iloc[_idx].get("target_user_id", -1))
            _frg_uid_to_rows.setdefault(_uid, []).append(_idx)
        logging.info("[FRG-sep] Built uid_to_rows index: %d users, %d rows", len(_frg_uid_to_rows), len(ds_tr))

    # last checkpoint cadence
    try:
        ckpt_every = int(os.environ.get("CKPT_EVERY_STEPS", "0") or "0")
    except Exception:
        ckpt_every = 0

    preempt_file = os.environ.get("PREEMPT_FILE", "").strip()

    def should_preempt() -> bool:
        if not preempt_file:
            return False
        try:
            return os.path.exists(preempt_file)
        except Exception:
            return False

    # delta monitoring
    P = int(getattr(getattr(mlocal, "hypernet", None), "peft_param_count", 0) or 0)
    delta_sub_idx: Optional[torch.Tensor] = None
    prev_delta_mean_sub: Optional[torch.Tensor] = None
    if P > 0 and delta_sub_k > 0:
        k = min(delta_sub_k, P)
        # sample indices on CPU
        try:
            perm = torch.randperm(P)[:k]
            delta_sub_idx = perm.to(device=device, dtype=torch.long)
        except Exception:
            delta_sub_idx = None

    # B6: build per-role index boundaries for enhanced delta monitoring
    role_boundaries: Dict[str, List[Tuple[int, int]]] = {}
    try:
        _sizes = getattr(mlocal, "_slice_sizes", None)
        _roles = getattr(mlocal, "_placeholder_roles", None)
        if _sizes is not None and _roles is not None and len(_sizes) == len(_roles):
            cur_b6 = 0
            for sz, role in zip(_sizes, _roles):
                role_boundaries.setdefault(role, []).append((cur_b6, cur_b6 + sz))
                cur_b6 += sz
            # Include A-matrix slices when emit_both is active
            _sizes_A = getattr(mlocal, "_slice_sizes_A", [])
            if getattr(mlocal, "_emit_both", False) and _sizes_A and len(_sizes_A) == len(_roles):
                for sz_a, role in zip(_sizes_A, _roles):
                    role_boundaries[role].append((cur_b6, cur_b6 + sz_a))
                    cur_b6 += sz_a
    except Exception:
        role_boundaries = {}

    # Save/Resume pointer path
    if not resume_ptr:
        resume_ptr = str((out_dir / "resume.state").as_posix())

    def save_last_checkpoint(abs_step: int) -> None:
        if not is_main():
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        save_model = unwrap(model)

        save_state_dict_any(out_dir / "hypernetwork_last.safetensors", save_model.hypernet.state_dict())
        save_state_dict_any(out_dir / "peft_placeholders_last.safetensors", save_model.detach_placeholder_state())

        if getattr(save_model, "ctx_proj", None) is not None and getattr(save_model, "layer_emb", None) is not None:
            ctx_sd = {
                "ctx_proj.weight": save_model.ctx_proj.weight.detach().cpu(),
                "layer_emb": save_model.layer_emb.detach().cpu(),
            }
            save_state_dict_any(out_dir / "ctx_params_last.safetensors", ctx_sd)

        if probe_head is not None:
            save_state_dict_any(out_dir / "probe_head_last.safetensors", probe_head.state_dict())

        if save_optim_state:
            try:
                torch.save(optimizer.state_dict(), str(out_dir / "optimizer_last.pt"))
                if scaler is not None and scaler.is_enabled():
                    torch.save(scaler.state_dict(), str(out_dir / "scaler_last.pt"))
                if sched is not None:
                    torch.save(sched.state_dict(), str(out_dir / "scheduler_last.pt"))
            except Exception as e:
                logging.warning("Optimizer/scaler/scheduler checkpoint save failed: %s", e)

        try:
            write_resume_pointer(resume_ptr, out_dir / "hypernetwork_last.safetensors", completed_steps=int(abs_step))
        except Exception:
            pass

    # training loop
    step_in_run = 0
    grad_step = 0
    ctx_micro_counter = 0

    # epoch control for DDP sampler
    epoch = 0
    sampler = getattr(dl_tr, "sampler", None)

    while step_in_run < int(steps_this_run):
        if isinstance(sampler, DistributedSampler):
            try:
                sampler.set_epoch(epoch)
            except Exception:
                pass
        epoch += 1

        progressed = False

        for batch in dl_tr:
            if step_in_run >= int(steps_this_run):
                break

            abs_step = int(base_step_offset + step_in_run)

            if should_preempt():
                if is_main():
                    logging.warning("Preemption flag detected; saving last checkpoint and exiting train loop.")
                save_last_checkpoint(abs_step)
                progressed = True
                step_in_run = int(steps_this_run)
                break

            cur_accum = accum_for_step_from_args(args, abs_step)
            cur_accum = max(1, int(cur_accum))

            clamp_v = float(clamp_now(abs_step))
            try:
                unwrap(model).set_runtime_clamp(float(clamp_v))
            except Exception:
                pass

            w_ctx = float(boundary_w_now(abs_step))
            l2_mult = float(l2_mult_now(abs_step))

            # batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            make_concat_inputs(batch, int(pad_id), int(seq_len))

            # One-time packing sanity: if we accidentally supervise mostly <|eoreply|> or pad_id, losses will look "too good".
            # This should be ~1/(avg_reply_len+1), typically a few percent, not tens of percent.
            if is_main() and not locals().get("_pack_sanity_logged", False):
                try:
                    lbl = batch["labels"]
                    sup = lbl[lbl != -100]
                    n_sup = int(sup.numel())

                    frac_eoreply = float("nan")
                    if (REPLY_END_ID is not None) and (n_sup > 0):
                        frac_eoreply = float((sup == int(REPLY_END_ID)).float().mean().item())

                    frac_pad = float("nan")
                    if n_sup > 0:
                        frac_pad = float((sup == int(pad_id)).float().mean().item())

                    logging.info(
                        "[pack sanity] supervised_tokens=%d | frac_<|eoreply|>=%s | frac_pad_id=%s",
                        n_sup,
                        (f"{frac_eoreply:.4f}" if math.isfinite(frac_eoreply) else "nan"),
                        (f"{frac_pad:.4f}" if math.isfinite(frac_pad) else "nan"),
                    )

                    # Extra context: estimate average reply length and what frac_<|eoreply|> should look like.
                    # If you have exactly one reply per packed example, then expected_frac_<|eoreply|>≈1/(avg_reply_len+1).
                    if (REPLY_END_ID is not None) and (lbl is not None) and (lbl.ndim == 2):
                        sup_mask = (lbl != -100)
                        eore_mask = (lbl == int(REPLY_END_ID))
                        sup_per_ex = sup_mask.sum(dim=1)
                        eore_per_ex = eore_mask.sum(dim=1)
                        has_sup = sup_per_ex > 0
                        if bool(has_sup.any()):
                            reply_len = (sup_per_ex - eore_per_ex).clamp_min(0)[has_sup].float()
                            avg_reply_len = float(reply_len.mean().item())
                            med_reply_len = float(reply_len.median().item())
                            min_reply_len = int(reply_len.min().item())
                            max_reply_len = int(reply_len.max().item())

                            avg_eore = float(eore_per_ex[has_sup].float().mean().item())
                            min_eore = int(eore_per_ex[has_sup].min().item())
                            max_eore = int(eore_per_ex[has_sup].max().item())

                            exp_frac_eore = float("nan")
                            denom = avg_reply_len + avg_eore
                            if math.isfinite(avg_reply_len) and math.isfinite(avg_eore) and denom > 0:
                                exp_frac_eore = avg_eore / denom

                            logging.info(
                                "[pack sanity] reply_len(mean/median/min/max)=%.1f/%.1f/%d/%d | "
                                "eoreply_per_ex(mean/min/max)=%.2f/%d/%d | expected_frac_<|eoreply|>≈%s",
                                avg_reply_len,
                                med_reply_len,
                                min_reply_len,
                                max_reply_len,
                                avg_eore,
                                min_eore,
                                max_eore,
                                (f"{exp_frac_eore:.4f}" if math.isfinite(exp_frac_eore) else "nan"),
                            )

                            if max_eore > 1:
                                logging.warning(
                                    "[pack sanity] Multiple <|eoreply|> per example detected (max=%d); check packing or multi-turn formatting.",
                                    max_eore,
                                )
                            if min_eore == 0:
                                logging.warning(
                                    "[pack sanity] Some examples have 0 <|eoreply|> supervised; check reply/end token insertion."
                                )

                    if math.isfinite(frac_eoreply) and frac_eoreply > 0.25:
                        logging.warning(
                            "[pack sanity] HIGH frac_<|eoreply|> among supervised tokens; reply packing is likely wrong."
                        )
                    if math.isfinite(frac_pad) and frac_pad > 0.0:
                        logging.warning(
                            "[pack sanity] Non-zero supervised pad_id fraction; labels are likely corrupted."
                        )
                except Exception:
                    pass

                _pack_sanity_logged = True
                
            g_clean = batch.get("global_features", None)

            # feature augmentation for training
            if feat_noise > 0.0 or feat_dropout > 0.0 or feat_mixup_p > 0.0 or feat_clamp > 0.0:
                try:
                    batch["global_features"] = augment_global_features(
                        batch["global_features"],
                        noise_sigma=feat_noise,
                        dropout_p=feat_dropout,
                        mixup_p=feat_mixup_p,
                        mixup_alpha=feat_mixup_alpha,
                        clamp_abs=feat_clamp,
                    )
                except Exception:
                    pass

            B = int(batch["input_ids"].size(0))
            mb = 0

            use_ddp = world_size() > 1 and isinstance(model, torch.nn.parallel.DistributedDataParallel)
            is_sync_batch = use_ddp and ((grad_step + 1) % cur_accum == 0)
            # Wrap entire loop in no_sync; temporarily enable sync for
            # the last microbatch's backward on sync (optimizer-step) batches.
            ddp_ctx = model.no_sync() if use_ddp else nullcontext()

            with ddp_ctx:
                while mb < B:
                    mb_end = min(B, mb + micro)

                    inp_ids = batch["input_ids"][mb:mb_end, :-1]
                    attn = batch["attention_mask"][mb:mb_end, :-1]
                    labels = batch["labels"][mb:mb_end]

                    gfeat_train = batch["global_features"][mb:mb_end]
                    gfeat_clean = gfeat_train
                    if isinstance(g_clean, torch.Tensor) and g_clean.shape == batch["global_features"].shape:
                        gfeat_clean = g_clean[mb:mb_end]

                    do_ctx = (w_ctx > 0.0) and ((abs_step % ctx_step_freq) == 0)
                    # diagnostic-only mode: boundary_ce_weight=0 but ctx_step_freq>1
                    # runs the boundary forward for logging CTXzero/CTXapply/Δ
                    # without contributing gradient (loss_ctx stays zero via w_ctx==0)
                    if (not do_ctx) and (ctx_step_freq > 1) and ((abs_step % ctx_step_freq) == 0):
                        do_ctx = True
                    if do_ctx:
                        # micro throttling
                        if ctx_micro_freq > 0:
                            do_ctx = ((ctx_micro_counter % ctx_micro_freq) == 0)
                        ctx_micro_counter += 1

                    # TF forward always uses augmented features for regularization;
                    # boundary probes use clean features for proper baseline comparison.

                    # teacher-forced forward (δ applied)
                    with autocast(device_type=device.type, enabled=bool(use_amp), dtype=amp_dtype):
                        hidden_apply = model(
                            input_ids=inp_ids,
                            attention_mask=attn,
                            labels=None,
                            global_features=gfeat_train,
                            return_hidden_only=True,
                            use_cache=False,
                            output_hidden_states=False,
                        )

                        # Capture delta from TF forward before boundary forward
                        # overwrites _last_delta with zeros (force_zero_delta=True).
                        # Keep gradient for delta diversity loss; detach copy for monitoring.
                        _delta_live = getattr(mlocal, "_last_delta", None)
                        if isinstance(_delta_live, torch.Tensor) and _delta_live.numel() > 0:
                            delta_raw = _delta_live.detach()  # for monitoring / L2
                        else:
                            _delta_live = None
                            delta_raw = None

                        # --- delta diversity loss (unsupervised) ---
                        loss_div = hidden_apply.new_zeros((), dtype=torch.float32)
                        if (delta_div_w > 0.0
                                and _delta_live is not None
                                and _delta_live.size(0) >= 4
                                and isinstance(gfeat_clean, torch.Tensor)):
                            try:
                                g_flat = gfeat_clean.detach().float()          # [mb, gdim]
                                d_flat = _delta_live.float().flatten(1)        # [mb, P]
                                feat_dist = torch.cdist(g_flat, g_flat)        # [mb, mb]
                                delta_dist = torch.cdist(d_flat, d_flat)       # [mb, mb]
                                _Bmb = g_flat.size(0)
                                _mask = torch.triu(torch.ones(_Bmb, _Bmb, device=g_flat.device), diagonal=1).bool()
                                f = feat_dist[_mask]
                                d = delta_dist[_mask]
                                f = f / (f.max() + 1e-8)
                                d = d / (d.max() + 1e-8)
                                f_z = f - f.mean()
                                d_z = d - d.mean()
                                corr = (f_z * d_z).sum() / (f_z.norm() * d_z.norm() + 1e-8)
                                loss_div = -corr  # maximize positive correlation
                            except Exception:
                                pass

                        tf_sums = reply_ce_sums_from_hidden(
                            hidden_apply,
                            labels,
                            lm_head,
                            chunk_tokens=int(os.getenv("CE_CHUNK_TOKENS", "256")),
                            pos_weight_boost=float(getattr(args, "pos_weight_boost", 0.0) or 0.0),
                        )
                        n_tok = int(tf_sums["count"])
                        loss_tf = (tf_sums["sum"] / max(1, n_tok)).float()

                        # --- leakage guard (boundary) ---
                        loss_ctx = hidden_apply.new_zeros((), dtype=torch.float32)
                        ctx_zero_mean = None
                        ctx_apply_mean = None
                        ctx_kept = 0

                        # Fix 6: Save grad-ckpt state before boundary forward overwrites it.
                        # When gradient checkpointing is active, backward recomputes
                        # forward layers via hooks that recompute deltas from these
                        # cached fields. The boundary forward would overwrite them
                        # with boundary values, killing gradient flow to the hypernetwork.
                        # (_delta_for_forward is already None after TF forward — Change A)
                        _saved_g_for_fwd = mlocal._g_for_forward
                        _saved_force_zero = mlocal._force_zero_flag
                        _saved_row_mask = mlocal._row_mask_for_forward

                        if do_ctx:
                            # apply CE from current hidden (δ applied)
                            ce_apply_sum, kept_apply = boundary_ce_from_hidden(hidden_apply, batch["input_ids"][mb:mb_end], labels, lm_head)
                            if kept_apply > 0:
                                ctx_apply_mean = (ce_apply_sum / float(max(1, kept_apply))).float()

                                # baseline CE with δ disabled (no grad), using clean features
                                with torch.no_grad():
                                    with autocast(device_type=device.type, enabled=bool(use_amp), dtype=amp_dtype):
                                        hidden_zero = model(
                                            input_ids=inp_ids,
                                            attention_mask=attn,
                                            labels=None,
                                            global_features=gfeat_clean,
                                            return_hidden_only=True,
                                            force_zero_delta=bool(ctx_disable_delta),
                                            use_cache=False,
                                            output_hidden_states=False,
                                        )
                                    ce_zero_sum, kept_zero = boundary_ce_from_hidden(hidden_zero, batch["input_ids"][mb:mb_end], labels, lm_head)
                                    if kept_zero > 0:
                                        ctx_zero_mean = (ce_zero_sum / float(max(1, kept_zero))).float()
                                        ctx_kept = int(min(kept_apply, kept_zero))

                                if ctx_zero_mean is not None and ctx_apply_mean is not None and ctx_kept > 0:
                                    # penalize δ making boundary token easier
                                    # loss_ctx = relu((CE_zero - CE_apply) - margin)
                                    loss_ctx = F.relu((ctx_zero_mean - ctx_apply_mean) - float(ctx_margin)).float()

                        # Fix 6: Restore grad-ckpt state so backward recomputation
                        # uses TF forward's features, not boundary values.
                        mlocal._g_for_forward = _saved_g_for_fwd
                        mlocal._force_zero_flag = _saved_force_zero
                        mlocal._row_mask_for_forward = _saved_row_mask

                        # --- L2 on δ (using TF-forward delta, captured above) ---
                        l2_pen = hidden_apply.new_zeros((), dtype=torch.float32)
                        if l2_coef > 0.0:
                            if isinstance(delta_raw, torch.Tensor) and delta_raw.numel() > 0:
                                l2_pen = delta_raw.float().pow(2).mean()
                            else:
                                l2_pen = hidden_apply.new_zeros((), dtype=torch.float32)

                        # --- aux losses from hypernet wrapper (dict regularizers etc.) ---
                        aux_total = hidden_apply.new_zeros((), dtype=torch.float32)
                        if aux_w != 0.0:
                            try:
                                aux = unwrap(model).aux_losses()
                                if isinstance(aux, dict) and aux:
                                    for _, v in aux.items():
                                        if isinstance(v, torch.Tensor):
                                            aux_total = aux_total + v.float()
                            except Exception:
                                pass

                        # --- FRG separation loss (semi-supervised, periodic) ---
                        loss_sep = hidden_apply.new_zeros((), dtype=torch.float32)
                        if (frg_sep_w > 0.0
                                and _frg_cohort
                                and _frg_uid_to_rows
                                and ds_tr is not None
                                and mb == 0  # only on first microbatch
                                and (abs_step % frg_sep_every) == 0):
                            try:
                                _sep_deltas: Dict[str, List[torch.Tensor]] = {}
                                for _cname, _cuids in _frg_cohort.items():
                                    _cdelts: List[torch.Tensor] = []
                                    for _cuid in _cuids[:20]:  # subsample for speed
                                        _crows = _frg_uid_to_rows.get(_cuid, [])
                                        if not _crows:
                                            continue
                                        _cgfeat = ds_tr[_crows[0]]["global_features"].unsqueeze(0).to(device)
                                        _, _, _cdelta = mlocal._emit_delta_parts(
                                            _cgfeat, force_zero=False, row_mask=None)
                                        _cdelts.append(_cdelta.squeeze(0))  # keep gradient
                                    if _cdelts:
                                        _sep_deltas[_cname] = _cdelts
                                if "empath" in _sep_deltas and "rage" in _sep_deltas:
                                    _mean_e = torch.stack(_sep_deltas["empath"]).mean(0)
                                    _mean_r = torch.stack(_sep_deltas["rage"]).mean(0)
                                    loss_sep = F.cosine_similarity(
                                        _mean_e.unsqueeze(0), _mean_r.unsqueeze(0)
                                    ).squeeze()  # minimize cosine = maximize separation
                            except Exception:
                                pass

                        # --- behavioral probe loss (v5 ablation) ---
                        loss_probe = hidden_apply.new_zeros((), dtype=torch.float32)
                        if (probe_w > 0.0
                                and probe_target_lookup
                                and probe_head is not None):
                            try:
                                # Pool hidden_apply over reply tokens (labels != -100)
                                reply_mask = (labels[:, 1:] != -100).float()  # [mb, L-1]
                                _rm_sum = reply_mask.sum(1, keepdim=True).clamp(min=1)  # [mb, 1]
                                pooled = (hidden_apply.float() * reply_mask.unsqueeze(-1)).sum(1) / _rm_sum  # [mb, H]

                                # Look up targets for this microbatch's user_ids
                                # Batch-extract UIDs to CPU in one sync instead of per-user .item()
                                _mb_uids = batch["target_user_id"][mb:mb_end]
                                _mb_uid_ints = (_mb_uids.cpu().tolist() if isinstance(_mb_uids, torch.Tensor)
                                                else [int(u) for u in _mb_uids])
                                _probe_tgts = []
                                _probe_mask = []
                                for _uid_int in _mb_uid_ints:
                                    _uid_int = int(_uid_int)
                                    if _uid_int in probe_target_lookup:
                                        _probe_tgts.append(probe_target_lookup[_uid_int])
                                        _probe_mask.append(True)
                                    else:
                                        _probe_tgts.append(torch.zeros_like(next(iter(probe_target_lookup.values()))))
                                        _probe_mask.append(False)

                                if any(_probe_mask):
                                    _probe_tgt_t = torch.stack(_probe_tgts).to(pooled.device)  # [mb, n_targets]
                                    _probe_mask_t = torch.tensor(_probe_mask, device=pooled.device)
                                    probe_preds = probe_head(pooled)  # [mb, n_targets]
                                    # Huber loss only over samples with known targets
                                    _loss_per_sample = F.huber_loss(
                                        probe_preds, _probe_tgt_t, reduction="none", delta=1.0
                                    ).mean(dim=1)  # [mb]
                                    loss_probe = (_loss_per_sample * _probe_mask_t.float()).sum() / _probe_mask_t.float().sum().clamp(min=1)
                            except Exception:
                                pass

                        # total loss — separate core (CE path) from persona (auxiliary path)
                        # so we can do split backward to reduce peak GPU memory
                        loss_core = loss_tf
                        if w_ctx > 0.0 and loss_ctx is not None:
                            loss_core = loss_core + (float(w_ctx) * loss_ctx)
                        if l2_coef > 0.0:
                            loss_core = loss_core + (float(l2_coef) * float(l2_mult) * l2_pen)
                        if aux_w != 0.0:
                            loss_core = loss_core + (float(aux_w) * aux_total)

                        loss_persona = loss_tf.new_zeros(())
                        if delta_div_w > 0.0 and isinstance(loss_div, torch.Tensor):
                            loss_persona = loss_persona + (float(delta_div_w) * loss_div)
                        if frg_sep_w > 0.0 and isinstance(loss_sep, torch.Tensor) and loss_sep.numel() > 0:
                            loss_persona = loss_persona + (float(frg_sep_w) * loss_sep)
                        if probe_w > 0.0 and isinstance(loss_probe, torch.Tensor):
                            loss_persona = loss_persona + (float(probe_w) * loss_probe)

                        loss = loss_core + loss_persona

                    # Enable gradient sync only for last microbatch of sync batch
                    is_last_micro = (mb_end >= B)
                    want_sync = use_ddp and is_sync_batch and is_last_micro
                    _do_split_bwd = loss_persona.grad_fn is not None

                    if _do_split_bwd:
                        # Split backward: persona losses first (retain graph),
                        # then core losses. This avoids having both paths'
                        # backward intermediates alive simultaneously.
                        # DDP sync only on the final (core) backward call.
                        if want_sync:
                            model.require_backward_grad_sync = False
                        persona_scaled = loss_persona / float(cur_accum)
                        if scaler is not None and scaler.is_enabled():
                            scaler.scale(persona_scaled).backward(retain_graph=True)
                        else:
                            persona_scaled.backward(retain_graph=True)

                        if want_sync:
                            model.require_backward_grad_sync = True
                        core_scaled = loss_core / float(cur_accum)
                        if scaler is not None and scaler.is_enabled():
                            scaler.scale(core_scaled).backward()
                        else:
                            core_scaled.backward()
                    else:
                        # No persona losses with grad — single backward as usual
                        if want_sync:
                            model.require_backward_grad_sync = True
                        loss_scaled = loss / float(cur_accum)
                        if scaler is not None and scaler.is_enabled():
                            scaler.scale(loss_scaled).backward()
                        else:
                            loss_scaled.backward()

                    # -- Deferred loss logging (extracted AFTER backward to avoid
                    #    CUDA sync stalls between forward and backward passes) --
                    if delta_div_w > 0.0 and isinstance(loss_div, torch.Tensor):
                        last_loss_div = float(loss_div.detach().item())
                    if frg_sep_w > 0.0 and isinstance(loss_sep, torch.Tensor) and loss_sep.numel() > 0:
                        try:
                            _sep_val = float(loss_sep.detach().item())
                            if _sep_val != 0.0:
                                last_loss_sep = _sep_val
                        except Exception:
                            pass
                    if probe_w > 0.0 and isinstance(loss_probe, torch.Tensor):
                        try:
                            _probe_val = float(loss_probe.detach().item())
                            if _probe_val != 0.0:
                                last_loss_probe = _probe_val
                        except Exception:
                            pass

                    # running aggregates (these .item() calls are the first
                    # sync point after backward — they wait for both forward
                    # AND backward to complete in one blocking call)
                    ce_tf_sum += float(tf_sums["sum"].detach().item())
                    ce_tf_raw_sum += float(tf_sums["sum_raw"].detach().item())
                    tok_tf_sum += float(n_tok)

                    if do_ctx and (ctx_zero_mean is not None) and (ctx_apply_mean is not None) and ctx_kept > 0:
                        ctx_zero_sum += float(ctx_zero_mean.detach().item()) * float(ctx_kept)
                        ctx_apply_sum += float(ctx_apply_mean.detach().item()) * float(ctx_kept)
                        ctx_kept_sum += float(ctx_kept)

                    # ---- Delta monitoring metrics ----
                    # Only computed on steps that will be logged (log_interval)
                    # to avoid unnecessary CUDA synchronization on other steps.
                    # Each .item() below forces a CUDA sync; on non-log steps
                    # these add ~10 sync stalls per microbatch for no benefit.
                    _will_log = (step_in_run == 0) or ((step_in_run - last_log_step) >= int(log_int) - 1)
                    delta_rms_sub = float("nan")
                    delta_sat_frac = float("nan")
                    delta_cos_prev = float("nan")
                    gate_mean = float("nan")
                    hook_hits = int(getattr(unwrap(model), "_hook_touches", 0) or 0)

                    if _will_log and delta_sub_idx is not None and isinstance(delta_raw, torch.Tensor) and delta_raw.numel() > 0:
                        try:
                            with torch.no_grad():
                                # [mb, K]
                                sub = delta_raw.detach().index_select(dim=1, index=delta_sub_idx)
                                delta_rms_sub = float(sub.float().pow(2).mean().sqrt().item())
                                # saturation fraction near clamp
                                eps = 1e-6
                                delta_sat_frac = float((sub.abs() >= (float(clamp_v) - eps)).float().mean().item())
                                # drift cosine vs previous
                                cur_mean = sub.float().mean(dim=0)
                                if prev_delta_mean_sub is not None:
                                    num = float(torch.dot(cur_mean, prev_delta_mean_sub).item())
                                    den = float(cur_mean.norm(p=2).item() * prev_delta_mean_sub.norm(p=2).item() + 1e-12)
                                    delta_cos_prev = num / den
                                prev_delta_mean_sub = cur_mean.detach()
                        except Exception:
                            pass

                    # B6: per-role L2 norms and delta percentiles (log-step only)
                    delta_role_l2: Dict[str, float] = {}
                    delta_p10 = float("nan")
                    delta_p50 = float("nan")
                    delta_p90 = float("nan")
                    if _will_log and role_boundaries and isinstance(delta_raw, torch.Tensor) and delta_raw.numel() > 0:
                        try:
                            with torch.no_grad():
                                d = delta_raw.detach().float()
                                for rname, spans_b6 in role_boundaries.items():
                                    parts = [d[:, s:e] for s, e in spans_b6]
                                    cat = torch.cat(parts, dim=1)
                                    delta_role_l2[rname] = float(cat.pow(2).mean().sqrt().item())
                                flat_abs = d.abs().flatten()
                                if flat_abs.numel() > 0:
                                    qs = torch.quantile(flat_abs, torch.tensor([0.1, 0.5, 0.9], device=flat_abs.device))
                                    delta_p10 = float(qs[0].item())
                                    delta_p50 = float(qs[1].item())
                                    delta_p90 = float(qs[2].item())
                        except Exception:
                            pass

                    # gate mean (log-step only)
                    if _will_log:
                        try:
                            with torch.no_grad():
                                if hasattr(unwrap(model), "_compute_gates"):
                                    gm = unwrap(model)._compute_gates(gfeat_train)  # type: ignore[attr-defined]
                                    if isinstance(gm, torch.Tensor) and gm.numel() > 0:
                                        gate_mean = float(gm.float().mean().item())
                        except Exception:
                            pass

                    # clear cached delta to avoid graph retention
                    try:
                        mlocal._last_delta = None
                    except Exception:
                        pass

                    mb = mb_end

            grad_step += 1
            if (grad_step % cur_accum) != 0:
                continue

            # optimizer step
            if scaler is not None and scaler.is_enabled():
                if max_grad_norm > 0.0:
                    scale = float(scaler.get_scale())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm) * scale)

                zero_grad_dict_tables_if_frozen(
                    mlocal.hypernet,
                    cur_step=abs_step,
                    freeze_steps=int(getattr(args, "dict_freeze_steps", 0) or 0),
                )

                # capture grad stats BEFORE step/zero_grad (note: grads may be AMP-scaled)
                try:
                    last_grad_stats = hypernet_grad_stats(mlocal.hypernet)
                except Exception:
                    last_grad_stats = {}

                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))

                zero_grad_dict_tables_if_frozen(
                    mlocal.hypernet,
                    cur_step=abs_step,
                    freeze_steps=int(getattr(args, "dict_freeze_steps", 0) or 0),
                )

                # capture grad stats BEFORE step/zero_grad
                try:
                    last_grad_stats = hypernet_grad_stats(mlocal.hypernet)
                except Exception:
                    last_grad_stats = {}

                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if sched is not None:
                try:
                    sched.step()
                except Exception:
                    pass

            # EMA update after optimizer step
            if ema is not None:
                try:
                    ema.update(trainable_params)
                except Exception:
                    pass

            step_in_run += 1
            progressed = True

            abs_step = int(base_step_offset + step_in_run)

            # periodic "last" checkpoint
            if ckpt_every > 0 and (abs_step % ckpt_every == 0):
                save_last_checkpoint(abs_step)

            # logging
            if (step_in_run - last_log_step) >= int(log_int) or step_in_run == 1:
                last_log_step = step_in_run

                mce_tf = ce_tf_sum / max(1.0, tok_tf_sum)
                mce_tf_raw = ce_tf_raw_sum / max(1.0, tok_tf_sum)  # unweighted
                mce_ctx_zero = (ctx_zero_sum / max(1.0, ctx_kept_sum)) if ctx_kept_sum > 0 else float("nan")
                mce_ctx_apply = (ctx_apply_sum / max(1.0, ctx_kept_sum)) if ctx_kept_sum > 0 else float("nan")
                ctx_delta = (mce_ctx_zero - mce_ctx_apply) if (math.isfinite(mce_ctx_zero) and math.isfinite(mce_ctx_apply)) else float("nan")

                g_norm = float("nan")
                try:
                    with torch.no_grad():
                        g_norm = float(batch["global_features"].float().norm(p=2, dim=1).mean().item())
                except Exception:
                    pass

                if is_main():
                    logging.info(
                        "[hyperlora] abs_step %6d | run_step %5d/%d | lr %.3e | clamp ±%.3f | "
                        "CE(TF) %.4f PPL %.1f | CE(raw) %.4f PPL(raw) %.1f | "
                        "CTXzero %.4f CTXapply %.4f Δ %.4f | "
                        "δ_rms(sub) %s sat%% %s cos(prev) %s | gates(mean) %s | g_norm %s | hooks %d | "
                        "grad(trunk/head/α/D) %.2e/%.2e/%.2e/%.2e",
                        abs_step,
                        step_in_run,
                        int(steps_this_run),
                        lr_now(),
                        float(clamp_v),
                        float(mce_tf),
                        float(ppl(float(mce_tf))),
                        float(mce_tf_raw),
                        float(ppl(float(mce_tf_raw))),
                        float(mce_ctx_zero) if math.isfinite(mce_ctx_zero) else float("nan"),
                        float(mce_ctx_apply) if math.isfinite(mce_ctx_apply) else float("nan"),
                        float(ctx_delta) if math.isfinite(ctx_delta) else float("nan"),
                        (f"{delta_rms_sub:.4f}" if math.isfinite(delta_rms_sub) else "nan"),
                        (f"{100.0*delta_sat_frac:.2f}" if math.isfinite(delta_sat_frac) else "nan"),
                        (f"{delta_cos_prev:.4f}" if math.isfinite(delta_cos_prev) else "nan"),
                        (f"{gate_mean:.4f}" if math.isfinite(gate_mean) else "nan"),
                        (f"{g_norm:.3f}" if math.isfinite(g_norm) else "nan"),
                        int(hook_hits),
                        float(last_grad_stats.get("trunk_grad_norm", 0.0)),
                        float(last_grad_stats.get("role_head_grad_norm", 0.0)),
                        float(last_grad_stats.get("alpha_head_grad_norm", 0.0)),
                        float(last_grad_stats.get("dict_table_grad_norm", 0.0)),
                    )
                    # persona-aware loss detail line
                    if delta_div_w > 0.0 or frg_sep_w > 0.0 or probe_w > 0.0:
                        logging.info(
                            "[persona] abs_step %6d | loss_div %s (w=%.3f) | loss_sep %s (w=%.3f) | loss_probe %s (w=%.3f)",
                            abs_step,
                            (f"{last_loss_div:.4f}" if math.isfinite(last_loss_div) else "nan"),
                            delta_div_w,
                            (f"{last_loss_sep:.4f}" if math.isfinite(last_loss_sep) else "nan"),
                            frg_sep_w,
                            (f"{last_loss_probe:.4f}" if math.isfinite(last_loss_probe) else "nan"),
                            probe_w,
                        )
                    # B6: per-role delta detail line
                    if delta_role_l2:
                        role_parts = " ".join(f"{r}={v:.4f}" for r, v in sorted(delta_role_l2.items()))
                        logging.info(
                            "[delta] abs_step %6d | %s | p10=%.4f p50=%.4f p90=%.4f",
                            abs_step, role_parts,
                            delta_p10 if math.isfinite(delta_p10) else 0.0,
                            delta_p50 if math.isfinite(delta_p50) else 0.0,
                            delta_p90 if math.isfinite(delta_p90) else 0.0,
                        )

                    # Pythia 2026-04-25 additions: throughput + memory.
                    # Earlier training logs lacked GPU-side observability (we
                    # had to scrape gpu_telem CSV after the fact to see
                    # HBM headroom and tokens/sec). These metrics surface
                    # the same numbers in the main log so a single grep
                    # pass recovers throughput and memory pressure curves.
                    try:
                        _now_ts = time.time()
                        _step_dt = float(_now_ts - _last_step_ts) if "_last_step_ts" in dir() else float("nan")
                    except Exception:
                        _step_dt = float("nan")
                    try:
                        _last_step_ts = time.time()
                    except Exception:
                        pass
                    try:
                        _seq_len_eff = int(batch.get("input_ids").size(1)) if "input_ids" in batch else 0
                        _bs_eff = int(batch.get("input_ids").size(0)) if "input_ids" in batch else 0
                        _tok_per_sec = (
                            float(_bs_eff * _seq_len_eff) / _step_dt
                            if (math.isfinite(_step_dt) and _step_dt > 0)
                            else float("nan")
                        )
                    except Exception:
                        _tok_per_sec = float("nan")
                    # Memory reporting under RMM unified-memory allocator (the GPU node
                    # high-memory GPU) is tricky: torch.cuda.memory_allocated/reserved return
                    # zero or unreliable values because the RMM allocator replaces
                    # PyTorch's default and does not feed the standard stats hooks.
                    # CUDA driver API mem_get_info() returns (free, total) for the
                    # device regardless of allocator, so we derive "used" from that
                    # and report alloc/reserved as best-effort fall-throughs that
                    # may be zero on RMM-active runs.
                    _mem_used_gb = float("nan")
                    _mem_total_gb = float("nan")
                    _mem_alloc_gb = float("nan")
                    _mem_reserved_gb = float("nan")
                    _mem_peak_gb = float("nan")
                    try:
                        if torch.cuda.is_available():
                            try:
                                _free_b, _total_b = torch.cuda.mem_get_info(device)
                                _mem_total_gb = float(_total_b / 1e9)
                                _mem_used_gb = float((_total_b - _free_b) / 1e9)
                            except Exception:
                                pass
                            try:
                                _mem_alloc_gb = float(torch.cuda.memory_allocated(device) / 1e9)
                                _mem_reserved_gb = float(torch.cuda.memory_reserved(device) / 1e9)
                                _mem_peak_gb = float(torch.cuda.max_memory_allocated(device) / 1e9)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    logging.info(
                        "[throughput] abs_step %6d | step_dt %s s | tok/s %s | "
                        "HBM used %s / %s GB | alloc %s / reserved %s / peak %s GB",
                        abs_step,
                        (f"{_step_dt:.2f}" if math.isfinite(_step_dt) else "nan"),
                        (f"{_tok_per_sec:.0f}" if math.isfinite(_tok_per_sec) else "nan"),
                        (f"{_mem_used_gb:.2f}" if math.isfinite(_mem_used_gb) else "nan"),
                        (f"{_mem_total_gb:.2f}" if math.isfinite(_mem_total_gb) else "nan"),
                        (f"{_mem_alloc_gb:.2f}" if math.isfinite(_mem_alloc_gb) else "nan"),
                        (f"{_mem_reserved_gb:.2f}" if math.isfinite(_mem_reserved_gb) else "nan"),
                        (f"{_mem_peak_gb:.2f}" if math.isfinite(_mem_peak_gb) else "nan"),
                    )

                    # Health check: warn if hypernetwork appears dead
                    _trunk_g = float(last_grad_stats.get("trunk_grad_norm", 0.0))
                    _head_g = float(last_grad_stats.get("role_head_grad_norm", 0.0))
                    _alpha_g = float(last_grad_stats.get("alpha_head_grad_norm", 0.0))
                    _all_grad_zero = (_trunk_g == 0.0 and _head_g == 0.0 and _alpha_g == 0.0)
                    _delta_zero = (math.isfinite(delta_rms_sub) and delta_rms_sub == 0.0)
                    if step_in_run >= 2 * int(log_int) and (_all_grad_zero or _delta_zero):
                        logging.warning(
                            "[HEALTH] abs_step %6d | HYPERNETWORK MAY BE DEAD: "
                            "grad(trunk/head/α)=%.2e/%.2e/%.2e δ_rms=%.6f | "
                            "Check: boundary forward overwriting grad-ckpt state? "
                            "force_zero_delta stuck? hooks not firing?",
                            abs_step, _trunk_g, _head_g, _alpha_g,
                            delta_rms_sub if math.isfinite(delta_rms_sub) else 0.0,
                        )

            # periodic eval + best saving
            if int(eval_every) > 0 and ((abs_step % int(eval_every) == 0) or (step_in_run >= int(steps_this_run))):
                # optionally evaluate with EMA weights
                if ema is not None:
                    ema.store(trainable_params)

                gc.collect()
                torch.cuda.empty_cache()
                model.eval()
                with torch.no_grad():
                    val_tf, val_ctx, val_mc = eval_split(
                        model=model,
                        loader=dl_val,
                        device=device,
                        pad_id=int(pad_id),
                        seq_len=int(seq_len),
                        use_amp=bool(use_amp),
                        progress_pct=float(getattr(args, "eval_progress_pct", 0.02) or 0.02),
                        max_batches=0,
                        log_every=int(os.getenv("EVAL_LOG_EVERY", "200")),
                        mc_samples=int(getattr(args, "mc_eval_samples", 0) or 0),
                        mc_keep_dropout=bool(getattr(args, "mc_eval_keep_dropout", False)),
                        noise_sigma=float(getattr(args, "noise_cond_sigma", 0.0) or 0.0),
                        eval_microbatch_size=int(getattr(args, "microbatch_size", 0) or 0),
                    )
                    last_val_tf = float(val_tf)
                    last_val_ctx = float(val_ctx)
                    last_val_mc = dict(val_mc) if isinstance(val_mc, dict) else {}

                    if is_main():
                        _fta = float(last_val_mc.get("first_tok_acc", float("nan"))) if isinstance(last_val_mc, dict) else float("nan")
                        logging.info(
                            "[hyperlora] abs_step %6d | Val | CE(TF) %.4f PPL %.1f | "
                            "CE(CTXzero) %.4f PPL %.1f | CE(CTXapply) %s | Δ %s | first_tok_acc %s",
                            abs_step,
                            float(val_tf),
                            float(ppl(float(val_tf))),
                            float(val_ctx),
                            float(ppl(float(val_ctx))),
                            (f"{float(last_val_mc.get('ctx_apply_ce', float('nan'))):.4f}" if isinstance(last_val_mc, dict) and math.isfinite(float(last_val_mc.get("ctx_apply_ce", float("nan")))) else "nan"),
                            (f"{float(last_val_mc.get('ctx_delta', float('nan'))):.4f}" if isinstance(last_val_mc, dict) and math.isfinite(float(last_val_mc.get("ctx_delta", float("nan")))) else "nan"),
                            (f"{_fta:.4f}" if math.isfinite(_fta) else "nan"),
                        )
                        if isinstance(val_mc, dict) and val_mc:
                            if "predictive_entropy" in val_mc or "mutual_info" in val_mc:
                                logging.info(
                                    "[hyperlora] abs_step %6d | Val | pred.H %s | MI %s",
                                    abs_step,
                                    (f"{float(val_mc.get('predictive_entropy', float('nan'))):.4f}" if math.isfinite(float(val_mc.get("predictive_entropy", float("nan")))) else "nan"),
                                    (f"{float(val_mc.get('mutual_info', float('nan'))):.4f}" if math.isfinite(float(val_mc.get("mutual_info", float("nan")))) else "nan"),
                                )

                        cur_metric = float(val_tf) if str(best_by) == "val_ce" else float(val_ctx)
                        if math.isfinite(cur_metric) and cur_metric < float(best_metric):
                            best_metric = float(cur_metric)
                            best_path.mkdir(parents=True, exist_ok=True)

                            save_model = unwrap(model)
                            save_state_dict_any(best_path / "hypernetwork.safetensors", save_model.hypernet.state_dict())
                            save_state_dict_any(best_path / "peft_placeholders.safetensors", save_model.detach_placeholder_state())

                            if getattr(save_model, "ctx_proj", None) is not None and getattr(save_model, "layer_emb", None) is not None:
                                ctx_sd = {
                                    "ctx_proj.weight": save_model.ctx_proj.weight.detach().cpu(),
                                    "layer_emb": save_model.layer_emb.detach().cpu(),
                                }
                                save_state_dict_any(best_path / "ctx_params.safetensors", ctx_sd)

                            # Save probe head (if active)
                            if probe_head is not None:
                                save_state_dict_any(best_path / "probe_head.safetensors", probe_head.state_dict())

                            # Save feature names for inference
                            if gcols is not None and len(gcols) > 0:
                                try:
                                    feat_manifest = {"feature_names": list(gcols)}
                                    with open(best_path / "feature_names.json", "w", encoding="utf-8") as fp:
                                        json.dump(feat_manifest, fp, indent=2)
                                except Exception as e:
                                    logging.warning("[ckpt] Could not save feature_names.json: %s", e)

                            # Save group_scales for inference consistency
                            if group_scales:
                                try:
                                    with open(best_path / "group_scales.json", "w", encoding="utf-8") as fp:
                                        json.dump(group_scales, fp, indent=2)
                                except Exception as e:
                                    logging.warning("[ckpt] Could not save group_scales.json: %s", e)

                            # Save tokenizer (with special token embeddings)
                            if tok is not None:
                                try:
                                    tok.save_pretrained(best_path / "tokenizer")
                                except Exception as e:
                                    logging.warning("[ckpt] Could not save tokenizer: %s", e)

                            logging.info("[ckpt] Saved new best to %s (%.4f by %s)", best_path.as_posix(), best_metric, best_by)

                # FRG sidecar (diagnostic delta-profile + optional generation)
                _frg_n = int(getattr(args, "frg_every_n_eval", 0) or 0)
                if _frg_n > 0 and (eval_count % _frg_n == 0) and is_main():
                    try:
                        _frg_metrics = frg_delta_profile(
                            model=model, ds=ds_val, cohort_ids=_frg_cohort,
                            device=device, use_amp=bool(use_amp),
                        )
                        _parts = [f"{k}={v:.4f}" for k, v in sorted(_frg_metrics.items())]
                        logging.info("[FRG] abs_step %d | %s", abs_step, " | ".join(_parts))
                    except Exception as _e:
                        logging.warning("[FRG] delta-profile failed: %s", _e)

                    # Pythia 2026-04-25: live per-feature gradient importance.
                    # Lightweight probe (5 batches) on every FRG eval point so we
                    # see which K=20 features are actually carrying gradient signal
                    # during training, instead of waiting for the end-of-train
                    # importance dump (which crashed earlier with the dtype bug).
                    try:
                        _imp_v, _imp_used = compute_feature_importance_grad(
                            model=model,
                            dataloader=dl_val,
                            device=device,
                            lm_head=get_lm_head(unwrap(model).backbone),
                            pad_id=int(pad_id),
                            seq_len=int(seq_len),
                            max_batches=5,
                            chunk_tokens=128,
                        )
                        if _imp_used > 0 and _imp_v.numel() > 0:
                            _imp_list = _imp_v.detach().float().cpu().tolist()
                            try:
                                _feat_names_live = (
                                    list(getattr(ds_tr, "global_feature_names", []) or [])
                                    or list(getattr(ds_tr, "_g_cols", []) or [])
                                )
                            except Exception:
                                _feat_names_live = []
                            if len(_feat_names_live) == len(_imp_list):
                                _imp_pairs = sorted(zip(_feat_names_live, _imp_list),
                                                    key=lambda kv: -abs(kv[1]))
                                _imp_str = " ".join(
                                    f"{n.replace('gstat_','')}={v:.3e}" for n, v in _imp_pairs[:8]
                                )
                            else:
                                _imp_str = " ".join(f"f{i}={v:.3e}" for i, v in enumerate(_imp_list[:8]))
                            logging.info(
                                "[FRG-imp] abs_step %d | batches=%d | top8 %s",
                                abs_step, _imp_used, _imp_str,
                            )
                    except Exception as _ie:
                        logging.warning("[FRG-imp] live importance failed: %s", _ie)

                    # M1 Pythia 2026-04-25: cosine drift between consecutive
                    # FRG-eval delta vectors for a fixed user. If the
                    # hypernetwork is converging, consecutive delta vectors
                    # for the same user should align (cos -> 1); if it is
                    # thrashing, cos stays low.
                    try:
                        _anchor_uid = None
                        _emp = list(_frg_cohort.get("empath", []))
                        if _emp:
                            _anchor_uid = int(_emp[0])
                        if _anchor_uid is not None:
                            _uid_rows = [i for i in range(len(ds_val))
                                         if int(ds_val.df.iloc[i].get("target_user_id", -1)) == _anchor_uid]
                            if _uid_rows:
                                _g = ds_val[_uid_rows[0]]["global_features"].unsqueeze(0).to(device)
                                with torch.no_grad():
                                    _, _, _delta_anchor = unwrap(model)._emit_delta_parts(
                                        _g, force_zero=False, row_mask=None,
                                    )
                                _delta_anchor = _delta_anchor.detach().float().squeeze(0)
                                _prev_anchor = globals().get("_FRG_ANCHOR_DELTA_PREV", None)
                                if _prev_anchor is not None and _prev_anchor.shape == _delta_anchor.shape:
                                    _drift_cos = float(F.cosine_similarity(
                                        _prev_anchor.unsqueeze(0).to(_delta_anchor.device),
                                        _delta_anchor.unsqueeze(0),
                                    ).item())
                                    logging.info(
                                        "[FRG-drift] abs_step %d | anchor_uid=%d | cos(prev)=%.4f | l2=%.4f",
                                        abs_step, _anchor_uid, _drift_cos,
                                        float(_delta_anchor.pow(2).sum().sqrt().item()),
                                    )
                                globals()["_FRG_ANCHOR_DELTA_PREV"] = _delta_anchor.cpu()
                    except Exception as _de:
                        logging.warning("[FRG-drift] anchor delta drift failed: %s", _de)

                    # B7: autoregressive generation diagnostic
                    _frg_max_tok = int(getattr(args, "frg_max_new_tokens", 0) or 0)
                    if _frg_max_tok > 0 and tok is not None:
                        try:
                            _frg_gen = frg_generate_and_score(
                                model=model, ds=ds_val, cohort_ids=_frg_cohort,
                                device=device, use_amp=bool(use_amp), tokenizer=tok,
                                max_new_tokens=_frg_max_tok,
                                temperature=float(getattr(args, "frg_temperature", 0.9) or 0.9),
                                top_p=float(getattr(args, "frg_top_p", 0.95) or 0.95),
                            )
                            _gparts = [f"{k}={v:.3f}" for k, v in sorted(_frg_gen.items())]
                            logging.info("[FRG-gen] abs_step %d | %s", abs_step, " | ".join(_gparts))
                        except Exception as _ge:
                            logging.warning("[FRG-gen] generation failed: %s", _ge)
                eval_count += 1

                # Probe validation: per-target Pearson r on val set
                if (probe_w > 0.0
                        and probe_target_lookup
                        and probe_head is not None
                        and is_main()):
                    try:
                        _probe_preds_all: List[torch.Tensor] = []
                        _probe_tgts_all: List[torch.Tensor] = []
                        _probe_micro = max(1, int(microbatch_size))
                        _probe_val_max = int(os.environ.get("PROBE_VAL_MAX_BATCHES", "0") or "0")
                        for _pb_i, _pb_batch in enumerate(dl_val):
                            if _probe_val_max > 0 and _pb_i >= _probe_val_max:
                                break
                            _pb_inp = _pb_batch["input_ids"][:, :-1].to(device)
                            _pb_attn = _pb_batch["attention_mask"][:, :-1].to(device)
                            _pb_labels = _pb_batch["labels"].to(device)
                            _pb_gfeat = _pb_batch["global_features"].to(device)
                            _pb_uids = _pb_batch["target_user_id"]
                            _pb_B = _pb_inp.size(0)

                            for _pb_s in range(0, _pb_B, _probe_micro):
                                _pb_e = min(_pb_s + _probe_micro, _pb_B)
                                with autocast(device_type=device.type, enabled=bool(use_amp), dtype=amp_dtype):
                                    _pb_hidden = mlocal(
                                        input_ids=_pb_inp[_pb_s:_pb_e],
                                        attention_mask=_pb_attn[_pb_s:_pb_e],
                                        labels=None,
                                        global_features=_pb_gfeat[_pb_s:_pb_e],
                                        return_hidden_only=True,
                                        use_cache=False,
                                        output_hidden_states=False,
                                    )
                                _pb_rmask = (_pb_labels[_pb_s:_pb_e, 1:] != -100).float()
                                _pb_rsum = _pb_rmask.sum(1, keepdim=True).clamp(min=1)
                                _pb_pooled = (_pb_hidden.float() * _pb_rmask.unsqueeze(-1)).sum(1) / _pb_rsum
                                _pb_pred = probe_head(_pb_pooled).detach().cpu()

                                for _j in range(_pb_s, _pb_e):
                                    _uid_int = int(_pb_uids[_j].item()) if isinstance(_pb_uids[_j], torch.Tensor) else int(_pb_uids[_j])
                                    if _uid_int in probe_target_lookup:
                                        _probe_preds_all.append(_pb_pred[_j - _pb_s])
                                        _probe_tgts_all.append(probe_target_lookup[_uid_int])
                                del _pb_hidden, _pb_pooled, _pb_pred

                        if len(_probe_preds_all) >= 10:
                            _pp = torch.stack(_probe_preds_all)  # [N, n_targets]
                            _pt = torch.stack(_probe_tgts_all)   # [N, n_targets]
                            _probe_target_cols = [c.strip() for c in str(args.probe_targets).split(",") if c.strip()]
                            _r_parts = []
                            _probe_r_vals: Dict[str, float] = {}
                            for _ti in range(_pp.size(1)):
                                _p_col = _pp[:, _ti]
                                _t_col = _pt[:, _ti]
                                _p_z = _p_col - _p_col.mean()
                                _t_z = _t_col - _t_col.mean()
                                _r = (_p_z * _t_z).sum() / (_p_z.norm() * _t_z.norm() + 1e-8)
                                _tname = _probe_target_cols[_ti] if _ti < len(_probe_target_cols) else f"t{_ti}"
                                _tname = _tname.replace("gstat_", "")
                                _r_parts.append(f"r_{_tname}={float(_r):.3f}")
                                _probe_r_vals[_tname] = float(_r)
                            last_probe_r = _probe_r_vals
                            logging.info("[probe] abs_step %d | N=%d | %s",
                                         abs_step, len(_probe_preds_all), " | ".join(_r_parts))
                    except Exception as _pe:
                        logging.warning("[probe] validation failed: %s", _pe)

                # Barrier: rank-0-only eval work (FRG, probe, checkpoint) may
                # take minutes.  Without a barrier rank 1 races ahead to the
                # next training step and desynchronises DDP allreduce.
                if world_size() > 1:
                    dist.barrier()

                model.train()

                if ema is not None:
                    ema.restore(trainable_params)

            if step_in_run >= int(steps_this_run):
                break

        if not progressed:
            if is_main():
                logging.warning("Dataloader ended early; no progress at run_step=%d/%d", step_in_run, int(steps_this_run))
            break

    # final stats
    train_ce = (ce_tf_sum / max(1.0, tok_tf_sum)) if tok_tf_sum > 0 else float("nan")
    train_ce_raw = (ce_tf_raw_sum / max(1.0, tok_tf_sum)) if tok_tf_sum > 0 else float("nan")
    train_ctx_zero = (ctx_zero_sum / max(1.0, ctx_kept_sum)) if ctx_kept_sum > 0 else float("nan")
    train_ctx_apply = (ctx_apply_sum / max(1.0, ctx_kept_sum)) if ctx_kept_sum > 0 else float("nan")
    train_ctx_delta = (train_ctx_zero - train_ctx_apply) if (math.isfinite(train_ctx_zero) and math.isfinite(train_ctx_apply)) else float("nan")

    return {
        "train_ce": float(train_ce),
        "train_ce_raw": float(train_ce_raw),
        "train_ctx_zero": float(train_ctx_zero),
        "train_ctx_apply": float(train_ctx_apply),
        "train_ctx_delta": float(train_ctx_delta),
        "val_ce": float(last_val_tf),
        "val_ctx_zero": float(last_val_ctx),
        "val_ctx_apply": float(last_val_mc.get("ctx_apply_ce", float("nan"))),
        "val_ctx_delta": float(last_val_mc.get("ctx_delta", float("nan"))),
        "val_reply_start_ce": float(last_val_mc.get("reply_start_ce", float("nan"))),
        "val_reply_mid_ce": float(last_val_mc.get("reply_mid_ce", float("nan"))),
        "val_reply_end_ce": float(last_val_mc.get("reply_end_ce", float("nan"))),
        "val_first_tok_acc": float(last_val_mc.get("first_tok_acc", float("nan"))),
        "val_pred_entropy": float(last_val_mc.get("predictive_entropy", float("nan"))),
        "val_mutual_info": float(last_val_mc.get("mutual_info", float("nan"))),
        "best_metric": float(best_metric),
        "was_preempted": bool(should_preempt()),
        "probe_r": dict(last_probe_r) if last_probe_r else {},
        "probe_r_mean": float(np.mean(list(last_probe_r.values()))) if last_probe_r else float("nan"),
    }


# ------------------------- Main -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--val_parquet", required=True)
    ap.add_argument("--author_parquet", default="/workspace/hypernets/data/author_static_10000.parquet")
    ap.add_argument("--global_parquet", default="/workspace/hypernets/data/global_features_10000.parquet")

    # Model / tokenizer / HF
    ap.add_argument("--base_model_id", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--online", action="store_true", default=False)
    ap.add_argument("--hf_token", type=str, default=HF_TOKEN)
    ap.add_argument("--hf_token_file", type=str, default="", help="Optional path to a file containing HF token.")
    ap.add_argument("--target_modules", type=str, default="auto", help="'auto' or comma list")
    ap.add_argument("--asr_style", action="store_true", help="Use W1-only surface (gate_proj,up_proj,r=2).")
    ap.add_argument("--qlora", action="store_true", default=False)
    ap.add_argument("--use_grad_ckpt", action="store_true")

    ap.add_argument("--disable_gstats", type=str, default="", help="Comma list of global feature names to drop.")
    ap.add_argument("--feature_profile", type=str, default="k6", choices=["k6", "all"])
    ap.add_argument("--force_include_features", type=str, default="",
                     help="Comma list of gstat_ features to force-include even if ablation dropped them.")
    ap.add_argument("--reply_budget", type=int, default=256,
                     help="Max reply tokens per packed example (default 256).")

    ap.add_argument("--models_output_dir", required=True)

    # Training
    ap.add_argument("--train_steps", type=int, default=10000, help="TOTAL optimizer steps target (resume continues to this total).")
    ap.add_argument("--one_epoch", action="store_true", help="Override total steps to exactly one epoch (resume-aware).")
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--microbatch_size", type=int, default=6)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--weight_decay", type=float, default=5.0e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--log_interval", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_before", action="store_true", help="Run eval once before training starts.")
    ap.add_argument("--eval_progress_pct", type=float, default=0.02)
    ap.add_argument("--fp32", action="store_true")

    # LoRA hparams
    ap.add_argument("--lora_r", type=int, default=2)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Clamp schedule
    ap.add_argument("--clamp_max", type=float, default=0.15)
    ap.add_argument("--unclamp_steps", type=int, default=2000)
    ap.add_argument("--unclamp_duration", type=int, default=4000)
    ap.add_argument("--delta_curriculum_frac", type=float, default=0.0,
                     help="If >0, override unclamp_duration = frac * train_steps (e.g. 0.10 = 10%%).")
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--min_lr", type=float, default=3.0e-4)

    # Position-weighted reply loss (B8: Phase B only)
    ap.add_argument("--pos_weight_boost", type=float, default=0.0,
                     help="If >0, weight early reply tokens higher: w=1+boost*(1-frac). Phase B uses 0.3.")

    # Persona-aware auxiliary losses
    ap.add_argument("--delta_diversity_weight", type=float, default=0.0,
                     help="Weight for delta diversity loss: encourages pairwise delta distances to track pairwise feature distances (unsupervised). Suggested: 0.05-0.20.")
    ap.add_argument("--frg_separation_weight", type=float, default=0.0,
                     help="Weight for FRG separation loss: minimizes cosine similarity between mean rage/empath delta vectors (semi-supervised). Suggested: 0.01-0.05.")
    ap.add_argument("--frg_sep_every", type=int, default=0,
                     help="Compute FRG separation loss every N optimizer steps. 0 = same as eval_every.")

    # Behavioral probe loss (v5 ablation)
    ap.add_argument("--probe_loss_weight", type=float, default=0.0,
                     help="Weight for behavioral probe loss on reply hidden states. 0 = disabled. Suggested: 0.05-0.20.")
    ap.add_argument("--probe_targets", type=str,
                     default="gstat_abuse_ratio,gstat_question_ratio,gstat_caps_ratio,gstat_hedge_ratio",
                     help="Comma-separated author_static column names to use as probe prediction targets. "
                          "SST-2-derived columns (gstat_user_sent_mean, gstat_gap_sentiment) removed; "
                          "VADER is the sole sentiment backend.")
    ap.add_argument("--probe_dropout", type=float, default=0.1,
                     help="Dropout rate on probe head input.")

    # Regularization / loss blend
    ap.add_argument("--l2_delta_coef", type=float, default=1.0e-4)
    ap.add_argument("--l2_warm_mult", type=float, default=3.0)
    ap.add_argument("--l2_warm_frac", type=float, default=0.10)

    # Leakage guard (CTX)
    ap.add_argument("--boundary_ce_weight", type=float, default=0.10)
    ap.add_argument("--boundary_ce_weight_max", type=float, default=1.2)
    ap.add_argument("--boundary_ce_warmup_steps", type=int, default=1200)
    ap.add_argument("--ctx_margin", type=float, default=0.0, help="Allowed improvement margin for boundary token CE before penalty.")
    ap.add_argument("--ctx_step_freq", type=int, default=1)
    ap.add_argument("--ctx_micro_freq", type=int, default=0)

    # Hypernetwork capacity & layer-context
    ap.add_argument("--hyper_hidden_dim", type=int, default=256)
    ap.add_argument("--hyper_out_rank", type=int, default=32)
    ap.add_argument("--use_layer_context", dest="use_layer_context", action="store_true", default=True)
    ap.add_argument("--no_use_layer_context", dest="use_layer_context", action="store_false")
    ap.add_argument("--ctx_embed_dim", type=int, default=32)

    # Init / activations / stability
    ap.add_argument("--hyper_activation", type=str, default="silu", choices=["silu", "relu", "gelu", "leaky_relu"])
    ap.add_argument("--zero_init_hyper_out", action="store_true", default=True)
    ap.add_argument("--layerwise_scale_mode", type=str, default="fan_in", choices=["none", "fan_in", "fan_avg"])
    ap.add_argument("--spectral_norm", type=str, default="last", choices=["none", "last", "all"])
    ap.add_argument("--ema_decay", type=float, default=0.0)
    ap.add_argument("--nan_guard", action="store_true", default=True)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # Scalability
    ap.add_argument("--hyper_chunk_size", type=int, default=0)
    ap.add_argument("--group_roles", type=str, default="qkv,o_proj,mlp")

    # Uncertainty
    ap.add_argument("--noise_cond_sigma", type=float, default=0.0)
    ap.add_argument("--mc_eval_samples", type=int, default=0)
    ap.add_argument("--mc_eval_keep_dropout", action="store_true")

    # Importance / IG
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--skip_perm_after", action="store_true")

    # Head/dictionary toggles
    ap.add_argument("--head_mode", type=str, default="single", choices=["single", "multi"])
    ap.add_argument("--dict_mode", action="store_true", default=False)
    ap.add_argument("--emit_both", action="store_true", default=False,
                    help="Emit deltas for both LoRA A and B matrices (full LoRA). "
                         "Increases hypernetwork output surface by ~1.9x.")
    ap.add_argument("--dict_k_global", type=int, default=64)
    ap.add_argument("--dict_k_per_role", type=str, default="")
    ap.add_argument("--head_rank_per_role", type=str, default="")
    ap.add_argument("--alpha_l1", type=float, default=0.0)
    ap.add_argument("--dict_ortho", type=float, default=0.0)
    ap.add_argument("--alpha_head_lr_mult", type=float, default=3.0)
    ap.add_argument("--dict_lr_mult", type=float, default=0.5)
    ap.add_argument("--dict_freeze_steps", type=int, default=800)
    ap.add_argument("--alpha_weight_decay", type=float, default=None)
    ap.add_argument("--dict_weight_decay", type=float, default=None)

    # Checkpointing / compression
    ap.add_argument("--save_best_by", type=str, default="val_ce", choices=["val_ce", "val_ctx"])
    ap.add_argument("--quantize_hypernet_int8", action="store_true")
    ap.add_argument("--save_optim_state", action="store_true", default=False)

    # data loading
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--torch_compile", action="store_true", default=False,
                     help="Apply torch.compile to decoder backbone (reduce Python dispatch overhead).")
    ap.add_argument("--torch_compile_hypernet", action="store_true", default=False,
                     help="Apply torch.compile to hypernetwork forward (clean MLP, no graph breaks).")
    ap.add_argument("--pretokenize", action="store_true",
                     help="Pre-tokenize all samples at dataset construction time. "
                          "Eliminates per-sample CPU tokenization in DataLoader workers, "
                          "trading RAM for dramatically higher GPU utilization.")
    ap.add_argument("--grad_accum_schedule", type=str, default=None)
    ap.add_argument("--scheduler", type=str, default=None, choices=["cosine", "constant", "linear"])
    ap.add_argument("--lr_cosine_steps", type=int, default=None)

    # grad accumulation staging (if not overridden)
    ap.add_argument("--accum_stage1", type=int, default=1200)
    ap.add_argument("--accum_stage2", type=int, default=4000)
    ap.add_argument("--grad_accum_late", type=int, default=6)

    # Feature augmentation (persona generalization)
    ap.add_argument("--train_feat_noise_sigma", type=float, default=0.0)
    ap.add_argument("--train_feat_dropout", type=float, default=0.0)
    ap.add_argument("--train_feat_mixup_p", type=float, default=0.0)
    ap.add_argument("--train_feat_mixup_alpha", type=float, default=0.0)
    ap.add_argument("--train_feat_clamp", type=float, default=0.0)

    ap.add_argument("--gate_eval_n", type=int, default=0)
    ap.add_argument("--gate_synth_mixup_alpha", type=float, default=0.4)
    ap.add_argument("--gate_synth_noise_sigma", type=float, default=0.0)
    ap.add_argument("--gate_density_k", type=int, default=16)
    ap.add_argument("--gate_density_percentile", type=float, default=99.5)
    ap.add_argument("--gate_density_probe", type=int, default=2048)
    ap.add_argument("--gate_density_multiplier", type=float, default=1.0)
    ap.add_argument("--gate_mc_delta_samples", type=int, default=0)
    ap.add_argument("--gate_mc_delta_dims", type=int, default=4096)
    ap.add_argument("--gate_mc_delta_var_max", type=float, default=0.0)
    ap.add_argument("--gate_delta_rms_max", type=float, default=0.0)
    ap.add_argument("--gate_delta_sat_max", type=float, default=-1.0)

    # Extra logging
    ap.add_argument("--delta_metric_subsample", type=int, default=16384)

    # FRG (Free-Running Generation) sidecar evaluation
    ap.add_argument("--frg_every_n_eval", type=int, default=0,
                     help="Run FRG every N eval steps (0=disabled, 2=every-other).")
    ap.add_argument("--frg_n_samples", type=int, default=150,
                     help="Total FRG samples (split evenly between cohorts).")
    ap.add_argument("--frg_cohort_file", type=str, default="",
                     help="Path to frg_cohort_ids.json for cross-run consistency.")
    ap.add_argument("--frg_labels_file", type=str, default="",
                     help="Path to labels_sentiment.csv (canonical quintile labels). "
                          "FRG cohort selects only 'rage' and 'empath' users from this file, "
                          "ensuring bijective alignment with the 5-bin evaluation labels.")
    ap.add_argument("--frg_max_new_tokens", type=int, default=96)
    ap.add_argument("--frg_temperature", type=float, default=0.9)
    ap.add_argument("--frg_top_p", type=float, default=0.95)

    # FRG final evaluation (persona fidelity + composite score)
    ap.add_argument("--frg_final_eval", type=int, default=0,
                     help="Run FRG persona fidelity + delta separation at end of training (0=off)")
    ap.add_argument("--frg_sent_model", type=str, default="vader",
                     help="Sentiment backend for persona fidelity. 'vader' = VADER (default). "
                          "Legacy: path to HF model (unused, VADER is now the sole backend).")
    ap.add_argument("--frg_final_n_users", type=int, default=20,
                     help="Users per cohort for final FRG persona fidelity eval")
    ap.add_argument("--frg_final_max_tokens", type=int, default=48,
                     help="Max tokens per generation in final FRG persona fidelity eval")

    # Composite score weights (used when --frg_final_eval 1)
    ap.add_argument("--composite_w_sep", type=float, default=0.5,
                     help="Weight for delta_separation in composite score")
    ap.add_argument("--composite_w_pf", type=float, default=0.3,
                     help="Weight for persona_fidelity in composite score")

    # Optimizer group tweaks
    ap.add_argument("--special_lr_mult", type=float, default=0.2)
    ap.add_argument("--special_weight_decay", type=float, default=0.0)

    # Aux losses
    ap.add_argument("--aux_loss_weight", type=float, default=1.0)

    # Misc
    ap.add_argument("--baseline_max_batches", type=int, default=0)
    ap.add_argument("--print_model_summary", action="store_true")
    ap.add_argument("--max_train_rows", type=int, default=0)
    ap.add_argument("--max_val_rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=142)
    ap.add_argument("-f", "--f", help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Apply --reply_budget to the module-level constant used by make_concat_inputs.
    global REPLY_BUDGET
    REPLY_BUDGET = int(getattr(args, "reply_budget", 256) or 256)

    # Parse --grad_accum_schedule like "0:2,1200:4,4000:8"
    args.grad_accum_schedule_pairs = None
    if args.grad_accum_schedule:
        pairs = []
        for item in args.grad_accum_schedule.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                step_s, accum_s = item.split(":")
                step_i = int(step_s.strip())
                accum_i = int(accum_s.strip())
                if step_i < 0 or accum_i <= 0:
                    raise ValueError
                pairs.append((step_i, accum_i))
            except Exception:
                raise ValueError(
                    f"Invalid --grad_accum_schedule: {args.grad_accum_schedule!r}. Expected like '0:2,1200:4,4000:8'."
                )
        pairs.sort(key=lambda t: t[0])
        args.grad_accum_schedule_pairs = pairs

    if args.scheduler == "cosine" and not args.lr_cosine_steps:
        args.lr_cosine_steps = args.train_steps

    # B4: delta_curriculum_frac overrides unclamp_duration
    _dcf = float(getattr(args, "delta_curriculum_frac", 0.0) or 0.0)
    if _dcf > 0.0:
        args.unclamp_duration = max(1, int(_dcf * args.train_steps))

    # logging setup
    only_main_logs = os.getenv("LOG_MAIN_RANK_ONLY", "1").lower() in ("1", "true", "yes", "y")
    rank_env = os.environ.get("RANK", "0")
    is_rank0 = (rank_env == "0")

    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.INFO if (is_rank0 or not only_main_logs) else logging.ERROR)
    if not is_rank0 and only_main_logs:
        for h in root_logger.handlers:
            h.setLevel(logging.ERROR)

    log_path = os.getenv("TRAIN_LOG_FILE", "")
    if log_path and (is_rank0 or not only_main_logs):
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
            file_handler.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning("File logging disabled (%s)", e)

    # Signals for graceful shutdown
    try:
        import signal

        def _signal_handler(signum, frame):
            logging.error("Received signal %s; requesting graceful shutdown.", signum)
            mark_shutdown(int(signum))

            # IMPORTANT:
            # Do NOT touch PREEMPT_FILE here.
            # PREEMPT_FILE should only be touched by the *outer job wrapper* (your bash trap),
            # so internal torchrun SIGTERM propagation is not misclassified as preemption.
            raise GracefulShutdown()

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
    except Exception:
        logging.warning("Signal handler install failed.", exc_info=False)

    hf_utils.logging.set_verbosity_error()
    seed_everything(seed=int(args.seed))

    logging.info("train_hyperlora starting rank=%s world_size=%s", rank_env, os.environ.get("WORLD_SIZE", "1"))

    # Online and token
    maybe_enable_online(bool(args.online))
    token = resolve_hf_token(args)
    # export token to env for HF libs
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token

    # Resume pointer preflight -> set BASE_STEP_OFFSET if absent
    resume_ptr_env = os.environ.get("HN_RESUME_POINTER", "").strip()
    base_step_offset = 0
    ckpt_path = None
    if resume_ptr_env:
        ckpt_path, completed = read_resume_pointer(resume_ptr_env)
        if completed > 0:
            base_step_offset = int(completed)
            os.environ["BASE_STEP_OFFSET"] = str(base_step_offset)
            if is_main():
                logging.info("Resume pointer found: completed_steps=%d checkpoint=%s", completed, str(ckpt_path))
    else:
        # allow BASE_STEP_OFFSET manual override
        try:
            base_step_offset = int(os.environ.get("BASE_STEP_OFFSET", "0") or "0")
        except Exception:
            base_step_offset = 0

    # HF auth check
    try:
        assert_hf_auth(args.base_model_id, token=token, online=bool(args.online))
    except Exception as e:
        if is_main():
            logging.error("HF auth preflight failed for %s: %s", args.base_model_id, e)
        raise

    # device & DDP init
    use_cuda = torch.cuda.is_available()
    try:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        ws = 1
    try:
        lrnk = int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        lrnk = 0
    use_ddp = use_cuda and ws > 1

    if use_ddp:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if not dist_is_initialized():
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(hours=2),
            )
        torch.cuda.set_device(lrnk)
        device = torch.device(f"cuda:{lrnk}")
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
        if device.type == "cuda" and device.index is None:
            torch.cuda.set_device(0)

    use_amp = device.type == "cuda" and not args.fp32

    # resolve target modules
    if str(args.target_modules).strip().lower() == "auto":
        tgt_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        tgt_modules = [m.strip() for m in str(args.target_modules).split(",") if m.strip()]

    if args.asr_style:
        tgt_modules = ["gate_proj", "up_proj"]
        if args.lora_r == ap.get_default("lora_r"):  # type: ignore[arg-type]
            args.lora_r = 2

    # gradient checkpointing guard (avoid reentrant graphs under leakage guard / dict-mode)
    eff_use_grad_ckpt = bool(args.use_grad_ckpt)
    if args.use_grad_ckpt and (args.boundary_ce_weight > 0.0):
        allow_reentrant = os.getenv("HN_ALLOW_REENTRANT_CHECKPOINT", "0").lower() in ("1", "true", "yes", "y")
        if not allow_reentrant:
            if is_main():
                logging.warning("Disabling gradient checkpointing (extra forward for leakage guard). Set HN_ALLOW_REENTRANT_CHECKPOINT=1 to override.")
            eff_use_grad_ckpt = False
    if args.use_grad_ckpt and args.dict_mode and (ws > 1):
        if is_main():
            logging.warning("Disabling gradient checkpointing because dict_mode=True under DDP can create reentrant graphs.")
        eff_use_grad_ckpt = False

    # device_map for HF loading
    if device.type == "cuda":
        gpu_index = lrnk if use_ddp else (device.index if device.index is not None else 0)
        device_map_arg = {"": f"cuda:{gpu_index}"}
    else:
        device_map_arg = None

    # load model + PEFT
    tok, peft_model, pad_id, new_token_ids = load_backbone_with_peft_lora(
        base_model_id=args.base_model_id,
        token=token,
        online=bool(args.online),
        target_modules=tgt_modules,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        use_grad_ckpt=bool(eff_use_grad_ckpt),
        qlora=bool(args.qlora and HAVE_BNB and HAVE_BNB_CFG),
        device_map=device_map_arg,
    )

    safe_lora_init(peft_model)

    # freeze backbone + base lora (hypernet learns δ)
    for p in peft_model.parameters():
        p.requires_grad_(False)

    # train only new boundary special token rows (masked)
    special_token_params = enable_trainable_special_token_rows(peft_model, new_token_ids=new_token_ids)
    if special_token_params and is_main():
        logging.info("Enabled trainable special-token rows: %d params (masked).", len(special_token_params))

    ensure_lora_basis_initialized(
        peft_model,
        init_std=float(os.getenv("HN_ALPHA_INIT_STD", "5e-4")),
        layerwise_scale="fan_in",
    )

    preflight_validate_peft_placeholders(peft_model)

    # ---------------- Data ----------------
    logging.info("Loading train/val parquet…")
    keep_cols = ["gid", "group_label", "text", "target_user_id"]
    try:
        df_tr = safe_read_parquet(args.train_parquet, columns=keep_cols)
        df_val = safe_read_parquet(args.val_parquet, columns=keep_cols)
    except Exception:
        df_tr = safe_read_parquet(args.train_parquet)
        df_val = safe_read_parquet(args.val_parquet)

        missing_tr = [c for c in keep_cols if c not in df_tr.columns]
        missing_val = [c for c in keep_cols if c not in df_val.columns]
        if missing_tr:
            raise KeyError(f"Train parquet missing required columns: {missing_tr}")
        if missing_val:
            raise KeyError(f"Val parquet missing required columns: {missing_val}")

        df_tr = df_tr[keep_cols]
        df_val = df_val[keep_cols]

    def limit_split_with_user_coverage(df: pd.DataFrame, max_rows: int, *, min_per_user: int = 1, seed: int = 142) -> pd.DataFrame:
        if max_rows <= 0:
            return df
        if "target_user_id" not in df.columns:
            raise KeyError("Dataframe must contain 'target_user_id'")
        uids = df["target_user_id"].unique()
        n_users = len(uids)
        need = n_users * max(1, int(min_per_user))
        if max_rows < need:
            logging.warning("max_rows=%d insufficient to cover %d users; bumping to %d.", max_rows, n_users, need)
            max_rows = need

        rng = seed

        def take_k(x: pd.DataFrame) -> pd.DataFrame:
            k = min(len(x), min_per_user)
            return x.sample(n=k, replace=False, random_state=rng)

        core = df.groupby("target_user_id", group_keys=False, sort=False).apply(take_k)
        remain = max_rows - len(core)
        if remain > 0:
            rest = df.drop(index=core.index)
            if not rest.empty:
                extra = rest.sample(n=min(remain, len(rest)), random_state=seed + 1)
                out = pd.concat([core, extra], axis=0, ignore_index=False)
            else:
                out = core
        else:
            out = core
        out = out.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
        return out

    df_tr = limit_split_with_user_coverage(df_tr, int(args.max_train_rows), min_per_user=1, seed=142)
    df_val = limit_split_with_user_coverage(df_val, int(args.max_val_rows), min_per_user=1, seed=143)

    used_uids = set(int(x) for x in pd.concat([df_tr["target_user_id"], df_val["target_user_id"]], axis=0).unique().tolist())
    author_df = prepare_author_table_for_dataset(args.author_parquet, args.global_parquet, used_user_ids=used_uids)

    gcols = None
    if author_df is not None:
        all_gcols = [c for c in author_df.columns if c.startswith("gstat_")]
        drops = set([c.strip() for c in str(args.disable_gstats).split(",") if c.strip()]) if args.disable_gstats else set()
        _leakage_safe_env = os.getenv("HN_LEAKAGE_SAFE", "1").strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        if _leakage_safe_env:
            drops.update(LEAKY_GLOBAL_SENTIMENT)
            drops.update(LEAKY_GLOBAL_BEAST_SENT)
            logging.info("[leakage] HN_LEAKAGE_SAFE=%s -> blocking %d sentiment+BEAST features",
                         os.getenv("HN_LEAKAGE_SAFE", "1"), len(LEAKY_GLOBAL_SENTIMENT | LEAKY_GLOBAL_BEAST_SENT))
        else:
            logging.info("[leakage] HN_LEAKAGE_SAFE=%s -> sentiment+BEAST features INCLUDED",
                         os.getenv("HN_LEAKAGE_SAFE", "1"))

        profile = str(args.feature_profile or "k6").strip().lower()
        if profile == "k6":
            want = [c for c in K6_FEATURES_DEFAULT if c not in drops]
            missing = [c for c in want if c not in all_gcols]
            if missing:
                raise KeyError(f"Missing required k6 columns: {missing}")
            gcols = want
        else:
            gcols = [c for c in all_gcols if c not in drops]

        # --force_include_features: re-add features the ablation dropped
        force_inc_str = str(getattr(args, "force_include_features", "") or "")
        force_inc = [c.strip() for c in force_inc_str.split(",") if c.strip()]
        if force_inc:
            gcols_set = set(gcols)
            for fi in force_inc:
                if fi not in gcols_set and fi in all_gcols:
                    gcols.append(fi)
                    logging.info("Force-including feature: %s", fi)
                elif fi not in all_gcols:
                    logging.warning("Force-include feature %s not in parquet columns — skipping.", fi)

        logging.info("[ablate] feature_profile=%s -> keep=%s drop=%s", profile, gcols, sorted(list(drops)))

        # Auto-drop constant features (zero variance in author table)
        if gcols and author_df is not None:
            _const_dropped = []
            for _c in list(gcols):
                if _c in author_df.columns and author_df[_c].std() < 1e-12:
                    _const_dropped.append(_c)
                    gcols.remove(_c)
            if _const_dropped:
                logging.warning("[ablate] Auto-dropped %d constant features: %s", len(_const_dropped), _const_dropped)

        logging.info("Global features kept: %d (dropped %d).", len(gcols), len(all_gcols) - len(gcols))

    if is_main():
        logging.info("Building training dataset…")
    _t_ds_tr = time.time()
    _pretok = bool(getattr(args, "pretokenize", False))
    ds_tr = HypernetGlobalOnlyDataset10000(
        df=df_tr,
        tokenizer=tok,
        author_parquet=Path(args.author_parquet) if args.author_parquet else None,
        global_parquet=Path(args.global_parquet) if args.global_parquet else None,
        include_gstats=gcols if gcols else None,
        max_length=int(args.max_len),
        pretokenize=_pretok,
        leakage_safe=None,  # defer to HN_LEAKAGE_SAFE env var
    )
    if is_main():
        logging.info("Training dataset built in %.2fs", time.time() - _t_ds_tr)

    if is_main():
        logging.info("Building validation dataset…")
    _t_ds_val = time.time()
    ds_val = HypernetGlobalOnlyDataset10000(
        df=df_val,
        tokenizer=tok,
        author_parquet=Path(args.author_parquet) if args.author_parquet else None,
        global_parquet=Path(args.global_parquet) if args.global_parquet else None,
        include_gstats=gcols if gcols else None,
        max_length=int(args.max_len),
        pretokenize=_pretok,
        leakage_safe=None,  # defer to HN_LEAKAGE_SAFE env var
    )
    if is_main():
        logging.info("Validation dataset built in %.2fs", time.time() - _t_ds_val)
        
    del df_tr, df_val
    gc.collect()

    if len(ds_tr) == 0:
        raise RuntimeError("Training dataset is empty after filtering.")

    # dataset preflight
    if is_main():
        logging.info("Preflighting dataset sample and feature vector shape…")
    sample = ds_tr[0]
    gdim = int(sample["global_features"].numel())
    if is_main():
        logging.info("Dataset preflight OK. gdim=%d", gdim)

    # B14: feature z-norm validation — warn if any feature has extreme z-scores
    if is_main() and gdim > 0:
        try:
            _n_probe = min(2000, len(ds_tr))
            _gstack = torch.stack([ds_tr[j]["global_features"] for j in range(_n_probe)], dim=0).float()
            _gmean = _gstack.mean(dim=0)
            _gstd = _gstack.std(dim=0)
            _gmax = _gstack.abs().max(dim=0).values
            _feat_names = gcols if gcols else [f"dim{k}" for k in range(gdim)]
            for fi in range(min(gdim, len(_feat_names))):
                if _gstd[fi] < 1e-8:
                    logging.warning("[znorm] feature %s has near-zero std (%.2e) — constant feature?", _feat_names[fi], float(_gstd[fi]))
                elif _gmax[fi] / max(float(_gstd[fi]), 1e-8) > 20.0:
                    logging.warning("[znorm] feature %s has extreme range: max=%.2f std=%.4f (ratio=%.1f)",
                                    _feat_names[fi], float(_gmax[fi]), float(_gstd[fi]),
                                    float(_gmax[fi]) / float(_gstd[fi]))
            del _gstack, _gmean, _gstd, _gmax
        except Exception as _e:
            logging.warning("[znorm] feature validation failed: %s", _e)

    roles_order = [r.strip() for r in str(args.group_roles).split(",") if r.strip()] or ["qkv", "o_proj", "mlp"]
    if "other" not in roles_order:
        roles_order.append("other")

    lora_meta = extract_lora_meta(peft_model)
    present_roles = sorted({z["role"] for z in lora_meta})
    logging.info(
        "Mode=%s dict=%s | Train=%d Val=%d | gdim=%d | LoRA surface=%s | roles=%s",
        str(args.head_mode),
        str(bool(args.dict_mode)),
        len(ds_tr),
        len(ds_val),
        gdim,
        ",".join(tgt_modules),
        ",".join(present_roles),
    )

    # Dataloaders
    sampler_tr = DistributedSampler(ds_tr, num_replicas=world_size(), rank=rank(), shuffle=True) if world_size() > 1 else None
    sampler_val = DistributedSampler(ds_val, num_replicas=world_size(), rank=rank(), shuffle=False) if world_size() > 1 else None

    eff_workers = max(8, int(args.num_workers))
    pin_mem = bool(args.pin_memory) if hasattr(args, "pin_memory") else (device.type == "cuda")
    persistent_workers = bool(args.persistent_workers) if hasattr(args, "persistent_workers") else (eff_workers > 0)

    dl_kwargs: Dict[str, Any] = dict(
        num_workers=eff_workers,
        pin_memory=pin_mem,
        persistent_workers=(persistent_workers and eff_workers > 0),
    )
    if eff_workers > 0:
        dl_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(args.batch_size),
        shuffle=(sampler_tr is None),
        sampler=sampler_tr,
        **dl_kwargs,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=sampler_val,
        **dl_kwargs,
    )

    group_scales = compute_group_scales(lora_meta, mode=str(args.layerwise_scale_mode))
    per_role_rank_map = parse_role_map(str(args.head_rank_per_role or ""), roles_order)
    # Apply hyper_out_rank uniformly when no per-role ranks specified
    if not per_role_rank_map and int(args.hyper_out_rank) > 0:
        per_role_rank_map = {r: int(args.hyper_out_rank) for r in roles_order}
    per_role_dictk_map = parse_role_map(str(args.dict_k_per_role or ""), roles_order)

    # Build wrapper (includes auto-resume load of hypernet/ctx/placeholders)
    model = build_hyper_wrapper(
        peft_model=peft_model,
        g_dim=int(gdim),
        hidden_dim=int(args.hyper_hidden_dim),
        use_layer_context=bool(args.use_layer_context),
        ctx_embed_dim=int(args.ctx_embed_dim),
        activation=str(args.hyper_activation),
        group_scales=group_scales,
        zero_init_last=bool(args.zero_init_hyper_out),
        spectral_norm_scope=str(args.spectral_norm),
        hyper_chunk_size=int(args.hyper_chunk_size),
        head_mode=str(args.head_mode),
        dict_mode=bool(args.dict_mode),
        dict_k_global=int(args.dict_k_global),
        per_role_rank_map=per_role_rank_map,
        per_role_dictk_map=per_role_dictk_map,
        hyper_out_rank=int(args.hyper_out_rank),
        alpha_l1=float(args.alpha_l1),
        dict_ortho=float(args.dict_ortho),
        roles_order=roles_order,
        lora_meta=lora_meta,
        emit_both=bool(getattr(args, "emit_both", False)),
    ).to(device)

    # attach special token params for optimizer
    try:
        mlocal = model.module if hasattr(model, "module") else model
        mlocal.special_token_params = list(special_token_params) if special_token_params else []
    except Exception:
        pass

    assert_surface_alignment(model)

    # ─── Behavioral probe head (v5 ablation) ───
    probe_head = None
    probe_target_lookup: Dict[int, torch.Tensor] = {}
    probe_w = float(getattr(args, "probe_loss_weight", 0.0) or 0.0)

    if probe_w > 0.0:
        probe_target_cols = [c.strip() for c in str(args.probe_targets).split(",") if c.strip()]
        # Load targets from author_static (independent of leakage-safe feature filtering)
        _probe_cols_present = [c for c in probe_target_cols if c in author_df.columns]
        _probe_cols_missing = [c for c in probe_target_cols if c not in author_df.columns]
        if _probe_cols_missing and is_main():
            logging.warning("[probe] Targets not in author_df: %s", _probe_cols_missing)

        if _probe_cols_present:
            # Z-normalize using training-set statistics
            _probe_sub = author_df[["target_user_id"] + _probe_cols_present].dropna()
            _probe_vals = _probe_sub[_probe_cols_present].values.astype(np.float32)
            _probe_mean = _probe_vals.mean(axis=0)
            _probe_std = _probe_vals.std(axis=0).clip(min=1e-8)
            _probe_normed = (_probe_vals - _probe_mean) / _probe_std

            for i, uid in enumerate(_probe_sub["target_user_id"].values):
                probe_target_lookup[int(uid)] = torch.tensor(_probe_normed[i], dtype=torch.float32)

            backbone_hidden_dim = peft_model.config.hidden_size  # 2048 for Pythia-1.4B
            probe_head = BehavioralProbeHead(
                hidden_dim=backbone_hidden_dim,
                n_targets=len(_probe_cols_present),
                dropout=float(args.probe_dropout),
            ).to(device)

            # NOTE: probe_head is NOT attached to the model.  It is kept as a
            # standalone module so that DDP never sees it (probe_head is called
            # manually after model.forward, not during it, which would leave
            # DDP with permanently "unused" parameters and corrupt reducer state).
            # Gradients are accumulated locally per rank; the 12K-param probe
            # converges fine without cross-rank allreduce.

            if is_main():
                logging.info("[probe] Behavioral probe: %d targets (%s), %d users in lookup",
                             len(_probe_cols_present), ", ".join(_probe_cols_present),
                             len(probe_target_lookup))
                logging.info("[probe] Target z-norm: mean=%s std=%s",
                             np.array2string(_probe_mean, precision=3),
                             np.array2string(_probe_std, precision=3))
            del _probe_sub, _probe_vals, _probe_normed

            # Resume probe head weights if available
            resume_dir = getattr(args, "resume_dir", None)
            if resume_dir:
                _probe_cands = [
                    Path(resume_dir) / "probe_head_last.safetensors",
                    Path(resume_dir) / "probe_head.safetensors",
                ]
                for _pc in _probe_cands:
                    if _pc.exists():
                        try:
                            _probe_sd = load_state_dict_any(_pc)
                            probe_head.load_state_dict(_probe_sd, strict=False)
                            if is_main():
                                logging.info("[probe] Loaded probe head from %s", _pc.as_posix())
                            break
                        except Exception as _pe:
                            if is_main():
                                logging.warning("[probe] Probe head resume failed from %s: %s", _pc.as_posix(), _pe)

    # --- Optional torch.compile (before DDP wrap) ---
    if getattr(args, "torch_compile", False):
        try:
            torch._dynamo.config.suppress_errors = True
            # RMM pluggable allocator does not support CUDA graphs
            # (reduce-overhead triggers checkPoolLiveAllocations hang).
            # Fall back to "default" mode when RMM is active.
            _bc_mode = "default" if os.environ.get("HN_USE_UNIFIED_MEMORY", "0") == "1" else "reduce-overhead"
            model.backbone = torch.compile(
                model.backbone, mode=_bc_mode, fullgraph=False,
            )
            if is_main():
                logging.info("[compile] torch.compile applied to backbone with mode=%s", _bc_mode)
        except Exception as _ce:
            if is_main():
                logging.warning("[compile] Failed: %s; continuing without compile", _ce)

    if getattr(args, "torch_compile_hypernet", False):
        try:
            torch._dynamo.config.suppress_errors = True
            # RMM pluggable allocator does not support CUDA graphs
            # (reduce-overhead triggers checkPoolLiveAllocations crash).
            # Fall back to "default" mode when RMM is active.
            _hc_mode = "default" if os.environ.get("HN_USE_UNIFIED_MEMORY", "0") == "1" else "reduce-overhead"
            model.hypernet = torch.compile(
                model.hypernet, mode=_hc_mode, fullgraph=True,
            )
            if is_main():
                logging.info("[compile] torch.compile applied to hypernetwork with mode=%s, fullgraph=True", _hc_mode)
        except Exception as _ce:
            if is_main():
                logging.warning("[compile] Hypernetwork compile failed: %s; continuing without", _ce)

    if world_size() > 1:
        model = wrap_ddp(model, device)

    # Preflight synthetic forward — use unwrapped model to avoid corrupting DDP reducer state
    _m_raw = model.module if hasattr(model, "module") else model
    try:
        tiny_L = min(8, int(args.max_len))
        bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if (bf16_ok and not args.fp32 and not _HN_FORCE_FP16) else torch.float16
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=(device.type == "cuda" and not args.fp32), dtype=amp_dtype):
                _ = _m_raw(
                    input_ids=torch.full((1, tiny_L), int(pad_id), dtype=torch.long, device=device),
                    attention_mask=torch.ones((1, tiny_L), dtype=torch.long, device=device),
                    labels=None,
                    global_features=torch.zeros((1, int(gdim)), dtype=torch.float32, device=device),
                    return_hidden_only=True,
                    use_cache=False,
                    output_hidden_states=False,
                )
        if world_size() > 1:
            dist.barrier()
        if is_main():
            logging.info("[preflight] synthetic forward completed on all ranks.")
            try:
                delta_path_sanity_check(
                    model=_m_raw,
                    loader=dl_val,
                    device=device,
                    pad_id=int(pad_id),
                    seq_len=int(args.max_len),
                    use_amp=use_amp,
                    noise_sigma=1e-2,
                    max_batches=1,
                )
            except Exception as e:
                logging.warning("δ-path sanity probe failed (continuing): %s", e)
    except Exception as e:
        logging.exception("Preflight synthetic forward failed: %s", e)
        raise

    # optional DDP debug probe (env)
    if os.environ.get("HN_DDP_DEBUG_PROBE", "0").lower() in ("1", "true", "yes", "y"):
        try:
            first_batch = next(iter(dl_tr))
            ddp_debug_probe_on_batch(
                model=model,
                batch=first_batch,
                device=device,
                pad_id=int(pad_id),
                seq_len=int(args.max_len),
                use_amp=use_amp,
                chunk_tokens=128,
            )
            if world_size() > 1:
                dist.barrier()
        except Exception as e:
            logging.warning("DDP debug probe failed (continuing): %s", e)

    # Output directory
    sub = f"hyperlora_{args.head_mode}{'_dict' if args.dict_mode else ''}"
    out_dir = Path(args.models_output_dir) / sub
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Determine steps for this run (resume-aware)
    train_steps_arg = int(args.train_steps)
    lr_cosine_steps_arg = int(args.lr_cosine_steps or 0)

    per_rank_bs = max(1, int(args.batch_size)) * max(1, world_size())
    batches_per_epoch = int(math.ceil(len(ds_tr) / float(per_rank_bs)))
    epoch_steps_total = steps_for_one_epoch(batches_per_epoch, args, base_step_offset=0)

    if args.one_epoch:
        if args.scheduler == "cosine":
            args.lr_cosine_steps = int(epoch_steps_total)

        if base_step_offset >= epoch_steps_total:
            steps_this_run = 0
            if is_main():
                logging.info(
                    "one_epoch: epoch_steps=%d base_step_offset=%d -> no remaining steps.",
                    epoch_steps_total,
                    base_step_offset,
                )
        else:
            steps_this_run = int(epoch_steps_total - base_step_offset)
            if args.scheduler == "cosine":
                args.lr_cosine_steps = int(epoch_steps_total)

            if is_main():
                if train_steps_arg != int(epoch_steps_total):
                    logging.info(
                        "one_epoch: train_steps=%d ignored; using epoch_steps=%d",
                        train_steps_arg,
                        epoch_steps_total,
                    )
                if lr_cosine_steps_arg not in (0, -1) and int(lr_cosine_steps_arg) != int(epoch_steps_total):
                    logging.info(
                        "one_epoch: lr_cosine_steps=%d ignored; using epoch_steps=%d",
                        int(lr_cosine_steps_arg),
                        epoch_steps_total,
                    )
                logging.info(
                    "one_epoch: epoch_steps=%d base_step_offset=%d -> steps_this_run=%d",
                    epoch_steps_total,
                    base_step_offset,
                    steps_this_run,
                )
    else:
        steps_this_run = max(0, int(train_steps_arg) - int(base_step_offset))
        if args.scheduler == "cosine" and int(args.lr_cosine_steps) <= 0:
            args.lr_cosine_steps = int(train_steps_arg)

        if is_main():
            logging.info(
                "Resume-aware steps: train_steps_total=%d base_step_offset=%d -> steps_this_run=%d",
                int(train_steps_arg),
                int(base_step_offset),
                int(steps_this_run),
            )

    # Hyperparams log
    if is_main():
        train_steps_total = int(epoch_steps_total) if args.one_epoch else int(train_steps_arg)
        train_steps_effective_end = int(base_step_offset) + int(steps_this_run)
        # CTX frequencies (env overrides args; keep an explicit record for hparams)
        try:
            ctx_step_freq_effective = int(os.environ.get("CTX_STEP_FREQ", str(getattr(args, "ctx_step_freq", 1) or 1)) or "1")
        except Exception:
            ctx_step_freq_effective = int(getattr(args, "ctx_step_freq", 1) or 1)
        try:
            ctx_micro_freq_effective = int(os.environ.get("CTX_MICRO_FREQ", str(getattr(args, "ctx_micro_freq", 0) or 0)) or "0")
        except Exception:
            ctx_micro_freq_effective = int(getattr(args, "ctx_micro_freq", 0) or 0)
        ctx_step_freq_effective = max(1, int(ctx_step_freq_effective))
        ctx_micro_freq_effective = max(0, int(ctx_micro_freq_effective))

        # Mirror env-driven CTX boundary toggle into args for hyperparam logging
        try:
            args.ctx_boundary_disable_delta = int(os.environ.get("CTX_BOUNDARY_DISABLE_DELTA", "1").lower() in ("1", "true", "yes", "y"))
        except Exception:
            args.ctx_boundary_disable_delta = 1

        # Alias used only for hyperparam logging
        try:
            args.activation
        except Exception:
            args.activation = str(getattr(args, "hyper_activation", "silu"))

        hp = {
            "base_model_id": args.base_model_id,
            "online": bool(args.online),
            "qlora": bool(args.qlora and HAVE_BNB and HAVE_BNB_CFG),
            "target_modules": tgt_modules,
            "head_mode": args.head_mode,
            "dict_mode": bool(args.dict_mode),
            "dict_k_global": int(args.dict_k_global),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "one_epoch": bool(args.one_epoch),
            "epoch_steps_total": int(epoch_steps_total),
            "train_steps_total": int(train_steps_total),
            "train_steps_arg": int(train_steps_arg),
            "train_steps_effective_end": int(train_steps_effective_end),
            "steps_this_run": int(steps_this_run),
            "base_step_offset": int(base_step_offset),
            "batch_size": int(args.batch_size),
            "microbatch_size": int(args.microbatch_size),
            "grad_accum": int(accum_for_step_from_args(args, int(base_step_offset))),
            "grad_accum_raw_default": int(args.grad_accum),
            "grad_accum_schedule": (args.grad_accum_schedule or ""),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "max_len": int(args.max_len),
            "gdim": int(gdim),
            "clamp_max": float(args.clamp_max),
            "unclamp_steps": int(args.unclamp_steps),
            "unclamp_duration": int(args.unclamp_duration),
            "warmup_frac": float(args.warmup_frac),
            "min_lr": float(args.min_lr),
            "l2_delta_coef": float(args.l2_delta_coef),
            "l2_warm_mult": float(args.l2_warm_mult),
            "l2_warm_frac": float(args.l2_warm_frac),
            "boundary_ce_weight": float(args.boundary_ce_weight),
            "boundary_ce_weight_max": float(args.boundary_ce_weight_max),
            "boundary_ce_warmup_steps": int(args.boundary_ce_warmup_steps),
            "ctx_margin": float(args.ctx_margin),
            "ctx_step_freq": int(args.ctx_step_freq),
            "ctx_micro_freq": int(args.ctx_micro_freq),
            "ctx_step_freq_effective": int(locals().get("ctx_step_freq_effective", int(args.ctx_step_freq))),
            "ctx_micro_freq_effective": int(locals().get("ctx_micro_freq_effective", int(args.ctx_micro_freq))),
            "ctx_step_freq_env": str(os.environ.get("CTX_STEP_FREQ", "")),
            "ctx_micro_freq_env": str(os.environ.get("CTX_MICRO_FREQ", "")),
            "ctx_boundary_disable_delta": int(args.ctx_boundary_disable_delta),
            "hyper_hidden_dim": int(args.hyper_hidden_dim),
            "hyper_out_rank": int(args.hyper_out_rank),
            "use_layer_context": bool(args.use_layer_context),
            "ctx_embed_dim": int(args.ctx_embed_dim),
            "activation": str(args.activation),
            "zero_init_hyper_out": bool(args.zero_init_hyper_out),
            "layerwise_scale_mode": str(args.layerwise_scale_mode),
            "spectral_norm": str(args.spectral_norm),
            "ema_decay": float(args.ema_decay),
            "scheduler": str(args.scheduler),
            "lr_cosine_steps": int(args.lr_cosine_steps or 0),
            "lr_cosine_steps_arg": int(lr_cosine_steps_arg),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "train_feat_noise_sigma": float(args.train_feat_noise_sigma),
            "train_feat_dropout": float(args.train_feat_dropout),
            "train_feat_mixup_p": float(args.train_feat_mixup_p),
            "train_feat_mixup_alpha": float(args.train_feat_mixup_alpha),
            "train_feat_clamp": float(args.train_feat_clamp),
        }
        hparams_path = out_dir / "hparams.json"
        atomic_write_text(hparams_path, json.dumps(hp, indent=2))
        logger.info("[hyperlora] hyperparams → %s", json.dumps(hp, sort_keys=True))

    # Baseline eval (optional): compute val CE with hypernetwork disabled / zeroed deltas
    if getattr(args, "eval_before", False):
        if is_main():
            logging.info("[baseline] evaluating before training...")
        base_metrics = evaluate(
            model,
            tokenizer,
            dl_val,
            args,
            device=device,
            rank=rank,
            world=world,
            hypernet=None,
            ctx_params=None,
            placeholders=None,
            gdim=gdim,
            is_chat=is_chat,
            head_mode=args.head_mode,
            dict_mode=args.dict_mode,
            dict_k_global=args.dict_k_global,
            step=base_step_offset,
            out_dir=out_dir,
            mode="baseline",
            log_progress_pct=args.eval_progress_pct,
        )
        if is_main():
            logging.info("[baseline] %s", base_metrics)

    # Prepare spans for importance regularization
    spans = None
    try:
        spans = infer_global_column_spans_from_dataset(ds_tr)
        if is_main() and spans:
            logging.info("[spans] inferred %d feature spans for importance regularization.", len(spans))
    except Exception as e:
        spans = None
        if is_main():
            logging.warning("Could not infer global feature spans; importance exports will be skipped.")
            
    if steps_this_run <= 0:
        if is_main():
            logging.info("No training steps to run; saving artifacts and exiting.")
        steps_this_run = 0

    log_int = int(args.log_interval) if int(args.log_interval) > 0 else max(1, steps_this_run // 10)

    density_gate = None

    # Train
    stats = train_loop(
        args=args,
        model=model,
        dl_tr=dl_tr,
        dl_val=dl_val,
        device=device,
        steps_this_run=int(steps_this_run),
        log_int=int(log_int),
        lr=float(args.lr),
        wd=float(args.weight_decay),
        pad_id=int(pad_id),
        seq_len=int(args.max_len),
        max_clamp=float(args.clamp_max),
        unclamp_steps=int(args.unclamp_steps),
        use_amp=bool(use_amp),
        l2_delta_coef=float(args.l2_delta_coef),
        l2_warm_mult=float(args.l2_warm_mult),
        l2_warm_frac=float(args.l2_warm_frac),
        boundary_ce_weight=float(args.boundary_ce_weight),
        eval_every=int(args.eval_every),
        microbatch_size=int(args.microbatch_size),
        spans=spans,
        out_dir=out_dir,
        density_gate=density_gate,
        base_step_offset=int(base_step_offset),
        tok=tok,
        gcols=gcols,
        group_scales=group_scales,
        ds_val=ds_val,
        ds_tr=ds_tr,
        author_df=author_df,
        probe_target_lookup=probe_target_lookup,
        probe_w=probe_w,
        probe_head=probe_head,
    )

    # ---- Tear down DDP BEFORE rank-0-only post-training work ----
    # Post-training (FRG eval, importance, saving) runs only on rank 0 and
    # can take 30-60+ min.  If the barrier lives *after* that block, rank 1
    # hits it immediately while rank 0 is still working; the NCCL timeout
    # (default 30 min) expires, rank 1 crashes, and torchrun kills rank 0.
    # Fix: sync here, destroy the process group, then let rank 0 work solo.
    _was_main = is_main()
    if dist_is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if _was_main:
        logging.info(
            "[hyperlora] DONE | Train CE(TF)=%.4f CE(raw)=%.4f | Train CTXzero=%.4f CTXapply=%.4f Δ=%.4f | "
            "Val CE(TF)=%.4f PPL=%.1f | Val CTXzero=%.4f CTXapply=%s Δ=%s | "
            "Val TF start/mid/end=%s/%s/%s | pred.H=%s MI=%s | best_metric=%.4f",
            float(stats.get("train_ce", float("nan"))),
            float(stats.get("train_ce_raw", float("nan"))),
            float(stats.get("train_ctx_zero", float("nan"))),
            float(stats.get("train_ctx_apply", float("nan"))),
            float(stats.get("train_ctx_delta", float("nan"))),
            float(stats.get("val_ce", float("nan"))),
            float(ppl(float(stats.get("val_ce", float("nan"))))) if math.isfinite(float(stats.get("val_ce", float("nan")))) else float("nan"),
            float(stats.get("val_ctx_zero", float("nan"))),
            (f"{float(stats.get('val_ctx_apply', float('nan'))):.4f}" if math.isfinite(float(stats.get("val_ctx_apply", float("nan")))) else "nan"),
            (f"{float(stats.get('val_ctx_delta', float('nan'))):.4f}" if math.isfinite(float(stats.get("val_ctx_delta", float("nan")))) else "nan"),
            (f"{float(stats.get('val_reply_start_ce', float('nan'))):.4f}" if math.isfinite(float(stats.get("val_reply_start_ce", float("nan")))) else "nan"),
            (f"{float(stats.get('val_reply_mid_ce', float('nan'))):.4f}" if math.isfinite(float(stats.get("val_reply_mid_ce", float("nan")))) else "nan"),
            (f"{float(stats.get('val_reply_end_ce', float('nan'))):.4f}" if math.isfinite(float(stats.get("val_reply_end_ce", float("nan")))) else "nan"),
            (f"{float(stats.get('val_pred_entropy', float('nan'))):.4f}" if math.isfinite(float(stats.get("val_pred_entropy", float("nan")))) else "nan"),
            (f"{float(stats.get('val_mutual_info', float('nan'))):.4f}" if math.isfinite(float(stats.get("val_mutual_info", float("nan")))) else "nan"),
            float(stats.get("best_metric", float("inf"))),
        )

        # --- Final FRG evaluation (persona fidelity + composite score) ---
        if getattr(args, "frg_final_eval", 0):
            try:
                logging.info("[FRG-final] Running delta separation + persona fidelity eval...")

                # Build cohort from training dataset
                _val_uids: set = set()
                for _vi in range(len(ds_val)):
                    _vuid = int(ds_val.df.iloc[_vi].get("target_user_id", -1))
                    _val_uids.add(_vuid)

                _cohort = load_or_create_frg_cohort(
                    cohort_file=getattr(args, "frg_cohort_file", ""),
                    ds=ds_tr,
                    n_per_cohort=int(getattr(args, "frg_final_n_users", 20)),
                    val_uids=_val_uids,
                    seed=int(getattr(args, "seed", 142) or 142),
                    author_df=author_df,
                    labels_file=str(getattr(args, "frg_labels_file", "") or ""),
                )

                # Metric 2: Delta cohort separation (cheap)
                _delta_metrics = frg_delta_profile(
                    model, ds_tr, _cohort, device, use_amp,
                )
                stats.update(_delta_metrics)
                logging.info("[FRG-final] Delta metrics: %s",
                             {k: f"{v:.4f}" for k, v in _delta_metrics.items()})

                # Metric 3: Persona fidelity (moderate cost)
                _pf_metrics = frg_persona_fidelity(
                    model, ds_tr, _cohort, device, use_amp, tok,
                    sent_model_path=getattr(args, "frg_sent_model", ""),
                    max_new_tokens=int(getattr(args, "frg_final_max_tokens", 48)),
                    max_users_per_cohort=int(getattr(args, "frg_final_n_users", 20)),
                )
                stats.update(_pf_metrics)

                # Composite score
                stats["composite_score"] = compute_composite(
                    val_ce=float(stats.get("val_ce", float("inf"))),
                    delta_separation=float(_delta_metrics.get("frg_delta_separation", 0.0)),
                    persona_fidelity=float(_pf_metrics.get("frg_persona_fidelity", 0.0)),
                    w_sep=float(getattr(args, "composite_w_sep", 0.5)),
                    w_pf=float(getattr(args, "composite_w_pf", 0.3)),
                )
                logging.info("[FRG-final] composite_score=%.4f  (val_ce=%.4f, sep=%.4f, pf=%.3f)",
                             stats["composite_score"],
                             float(stats.get("val_ce", float("nan"))),
                             float(_delta_metrics.get("frg_delta_separation", 0.0)),
                             float(_pf_metrics.get("frg_persona_fidelity", 0.0)))

            except Exception as e:
                logging.warning("[FRG-final] Final FRG eval failed: %s", e)
                import traceback; traceback.print_exc()

        # Save final artifacts
        save_model = model.module if hasattr(model, "module") else model
        save_state_dict_any(out_dir / "peft_placeholders.safetensors", save_model.detach_placeholder_state())
        save_state_dict_any(out_dir / "hypernetwork.safetensors", save_model.hypernet.state_dict())
        if getattr(save_model, "ctx_proj", None) is not None and getattr(save_model, "layer_emb", None) is not None:
            ctx_sd = {
                "ctx_proj.weight": save_model.ctx_proj.weight.detach().cpu(),
                "layer_emb": save_model.layer_emb.detach().cpu(),
            }
            save_state_dict_any(out_dir / "ctx_params.safetensors", ctx_sd)
        tok.save_pretrained(out_dir)

        manifest = {
            "mode": "flat",
            "global_columns": getattr(ds_tr, "global_columns", []),
            "gdim": int(gdim),
            "use_layer_context": bool(args.use_layer_context),
            "ctx_embed_dim": int(args.ctx_embed_dim),
            "hyper_out_rank": int(args.hyper_out_rank),
            "activation": str(args.hyper_activation),
            "layerwise_scale_mode": str(args.layerwise_scale_mode),
            "head_mode": str(args.head_mode),
            "dict_mode": bool(args.dict_mode),
            "roles_order": roles_order,
        }
        atomic_write_text(out_dir / "feature_manifest.json", json.dumps(manifest, indent=2))
        atomic_write_text(out_dir / "training_summary.json", json.dumps(stats, indent=2))

        # Free training state (optimizer/scaler/grads live inside train_loop's
        # locals, but PyTorch's CUDA caching allocator still holds the memory).
        # An explicit GC + empty_cache reclaims ~3-6 GB, enough for importance.
        import gc as _gc
        model.zero_grad(set_to_none=True)
        _gc.collect()
        torch.cuda.empty_cache()

        # Feature importance exports (optional; can be heavy)
        try:
            imp_model = model.module if hasattr(model, "module") else model  # unwrap DDP to avoid allreduce hangs
            lm_head = get_lm_head(imp_model.backbone)  # type: ignore

            _imp_batches = int(os.environ.get("FINAL_IMPORTANCE_MAX_BATCHES", "50") or "50")
            if spans and _imp_batches > 0:
                imp_dim, used_batches = compute_feature_importance_grad(
                    model=imp_model,
                    dataloader=dl_val,
                    device=device,
                    lm_head=lm_head,
                    pad_id=int(pad_id),
                    seq_len=int(args.max_len),
                    max_batches=_imp_batches,
                    chunk_tokens=int(os.environ.get("CE_CHUNK_TOKENS", "256")),
                )
                if imp_dim.numel() > 0:
                    col_scores = aggregate_importance_by_column(imp_dim, spans, agg="sum")
                    pd.DataFrame(col_scores, columns=["feature", "grad_importance"]).to_csv(
                        out_dir / "feature_importance_grad.csv", index=False
                    )
                    logging.info("Saved gradient-based feature importance (batches=%d).", used_batches)

                if not bool(args.skip_perm_after) and _imp_batches > 0:
                    _ddw = float(getattr(args, "delta_diversity_weight", 0.0) or 0.0)
                    _pw = float(getattr(args, "probe_loss_weight", 0.0) or 0.0)
                    perm_scores = permutation_importance_by_column(
                        model=imp_model,
                        dataloader=dl_val,
                        device=device,
                        lm_head=lm_head,
                        spans=spans,
                        pad_id=int(pad_id),
                        seq_len=int(args.max_len),
                        sample_batches=_imp_batches,
                        chunk_tokens=int(os.environ.get("CE_CHUNK_TOKENS", "256")),
                        delta_div_weight=_ddw,
                        probe_weight=_pw,
                        probe_head=probe_head,
                        probe_target_lookup=probe_target_lookup,
                    )
                    _perm_col = "perm_delta_composite" if (_ddw > 0.0 or (_pw > 0.0 and probe_head is not None)) else "perm_delta_ce"
                    _perm_columns = ["feature", _perm_col, "perm_delta_ce", "perm_delta_probe", "perm_delta_div"]
                    pd.DataFrame(perm_scores, columns=_perm_columns).to_csv(
                        out_dir / "feature_importance_permutation.csv", index=False
                    )
                    logging.info("Saved permutation-based feature importance (%s + per-component columns).", _perm_col)
        except Exception as e:
            logging.warning("Feature importance export failed: %s", e)

        # Optional dynamic int8 export
        if bool(args.quantize_hypernet_int8):
            if _quantize_dynamic is None:
                logging.warning("Dynamic int8 quantization unavailable in this torch build.")
            else:
                try:
                    hnet = (model.module if hasattr(model, "module") else model).hypernet.cpu().eval()
                    qh = _quantize_dynamic(hnet, {nn.Linear}, dtype=torch.qint8)  # type: ignore[arg-type]
                    torch.save(qh.state_dict(), str(out_dir / "hypernetwork_int8.pt"))
                    logging.info("Saved dynamic-int8 hypernet -> %s", (out_dir / "hypernetwork_int8.pt").as_posix())
                except Exception as e:
                    logging.warning("Dynamic int8 quantization failed: %s", e)

    # NOTE: DDP barrier + destroy moved to BEFORE the is_main() post-training
    # block (see above).  Rank 1 has already exited cleanly at this point.


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except GracefulShutdown:
        # Exit 0 ONLY if the wrapper touched PREEMPT_FILE (real preempt/termination request).
        pf = os.environ.get("PREEMPT_FILE", "").strip()
        pf_hit = bool(pf) and os.path.exists(pf)

        sig = shutdown_signal()
        if pf_hit:
            if os.environ.get("RANK", "0") == "0":
                print(f"GracefulShutdown (signal={sig}); PREEMPT_FILE hit; exiting 0.", file=sys.stderr)
            sys.exit(0)

        # Otherwise propagate a non-zero “signal-style” exit code so the job is marked failed.
        code = 1
        try:
            if sig is not None and int(sig) > 0:
                code = 128 + int(sig)  # SIGTERM->143, SIGINT->130
        except Exception:
            code = 1

        if os.environ.get("RANK", "0") == "0":
            print(f"GracefulShutdown (signal={sig}); PREEMPT_FILE not set; exiting {code}.", file=sys.stderr)
        sys.exit(code)

    except SystemExit:
        raise
    except BaseException as e:
        # Only treat failures as "graceful" when PREEMPT_FILE was actually set by the wrapper.
        pf = os.environ.get("PREEMPT_FILE", "").strip()
        pf_hit = bool(pf) and os.path.exists(pf)

        msg = str(e).lower()
        dl_worker_term = (
            ("dataloader worker" in msg and "killed by signal" in msg)
            or ("dataloader worker" in msg and "terminated" in msg)
        )

        if pf_hit:
            # Preempt requested: exit cleanly even if workers are getting torn down.
            if os.environ.get("RANK", "0") == "0":
                print(f"Preempt detected; exiting 0 after exception: {e}", file=sys.stderr)
            sys.exit(0)

        # Not a preempt: do NOT hide the error (including DataLoader worker kills).
        if os.environ.get("RANK", "0") == "0":
            try:
                logging.error("Fatal exception (not preempt): %s", e, exc_info=True)
            except Exception:
                pass
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass

        raise


    if os.environ.get("RANK", "0") == "0":
        print(f"Script runtime: {time.time() - t0:.1f}s")