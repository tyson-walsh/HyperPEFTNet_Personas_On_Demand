#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_hyperlora_forum.py
========================

Purpose
-------
Simulate a Reddit-style forum using a frozen Pythia-1.4B backbone + a trained HyperPEFT-LoRA
hypernetwork (FlatHypernetwork) that emits LoRA-B deltas conditioned on per-author global
feature vectors g(u). Users are selected from the extremes of an existing sentiment label
file (rage vs empath) produced by create_labels.py, and the simulation explicitly tracks
which author type is posting.

Key additions versus the old 125M (Pythia) community build
---------------------------------------------------------
1) Loads Pythia-1.4B + PEFT LoRA modules and injects deltas emitted by the trained hypernet
   (hypernetwork_structure_10000.py).
2) Selects a pool of 200 labeled users:
     - top 100 rage users
     - top 100 empath users
   using labels from create_labels.py output, with fallback to gstat_user_sent_mean extremes.
3) Adds author_type ("rage"/"empath"/"neutral") to every generated post for verification.
4) Quantifies whether generated text matches author type by:
     - running the same sentiment probe pipeline used by hypernetwork_feature_builder_10000.py
       (SST-2 polarity) on generated text,
     - binning polarity into the same 5-label scheme used by create_labels.py
       (rage/grumpy/mellow/calm/empath),
     - computing per-user and aggregate match rates + confusion matrix.

Outputs
-------
- out_dir/forum.parquet
- out_dir/forum.jsonl
- out_dir/forum.md
- out_dir/metadata.json
- out_dir/user_sentiment_eval.csv
- out_dir/sentiment_confusion.csv

Notes
-----
- This script assumes:
  * author_parquet contains target_user_id and gstat_* features (including gstat_user_sent_mean).
  * hyper_dir contains (at least) best/hypernetwork.safetensors and best/peft_placeholders.safetensors
    (or the same files directly under hyper_dir if --use_best_ckpt is not set).
  * hypernetwork_feature_builder_10000.py is importable for the sentiment probe pipeline.

"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
import logging
import math
import os
import random
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# Defense against fused-attention kernel edge cases. PyTorch has FOUR SDP
# backends -- flash, mem_efficient, math, AND cudnn -- each with their own
# enable toggle. The `mha_graph.execute(...).is_good()==false` error we hit
# in this sweep came from the *cudnn* MHA graph path specifically, NOT
# flash. So we must disable cudnn_sdp too. Math SDPA is the only backend
# guaranteed to be portable + numerically robust at the cost of being
# slower; that's the right tradeoff for this Patch sweep.
if os.environ.get("ARDITI_DISABLE_FLASH_SDP", "0") == "1":
    _msgs = []
    for fn_name, target in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_cudnn_sdp", False),  # the one we missed before
        ("enable_math_sdp", True),
    ):
        try:
            getattr(torch.backends.cuda, fn_name)(target)
            _msgs.append(f"{fn_name}={target}")
        except Exception as _e:
            _msgs.append(f"{fn_name}=FAIL({type(_e).__name__})")
    print(f"[arditi-defense] SDP backends configured: {', '.join(_msgs)}", flush=True)

# Sentiment probe re-use (must match feature builder / create_labels)
import hypernetwork_feature_builder_10000 as _hfb  # type: ignore


def _hfb_get_attr(*names: str):
    for n in names:
        if hasattr(_hfb, n):
            return getattr(_hfb, n)
    return None


_agreement_ratio = _hfb_get_attr("_agreement_ratio", "agreement_ratio")
_disagreement_ratio = _hfb_get_attr("_disagreement_ratio", "disagreement_ratio")
_subjectivity_ratio = _hfb_get_attr("_subjectivity_ratio", "subjectivity_ratio")
_intensifier_ratio = _hfb_get_attr("_intensifier_ratio", "intensifier_ratio")
_hedge_ratio = _hfb_get_attr("_hedge_ratio", "hedge_ratio")
_caps_ratio = _hfb_get_attr("_caps_ratio", "caps_ratio")
_toklist = _hfb_get_attr("_toklist", "toklist")

polarity_batch = _hfb_get_attr("polarity_batch")
_build_sentiment_pipe_fn = _hfb_get_attr("_build_sentiment_pipe", "build_sentiment_pipe")
_secondperson_ratio_fn = _hfb_get_attr("_secondperson_ratio", "secondperson_ratio")

_missing: List[str] = []
for _name, _obj in [
    ("agreement_ratio", _agreement_ratio),
    ("disagreement_ratio", _disagreement_ratio),
    ("subjectivity_ratio", _subjectivity_ratio),
    ("intensifier_ratio", _intensifier_ratio),
    ("hedge_ratio", _hedge_ratio),
    ("caps_ratio", _caps_ratio),
    ("toklist", _toklist),
    ("polarity_batch", polarity_batch),
    ("build_sentiment_pipe", _build_sentiment_pipe_fn),
    ("secondperson_ratio", _secondperson_ratio_fn),
]:
    if _obj is None:
        _missing.append(_name)

if _missing:
    raise ImportError("hypernetwork_feature_builder_10000 is missing required helper(s): " + ", ".join(_missing))

LOG = logging.getLogger("build_hyperlora_forum")

def _build_sentiment_pipe(device_id: int):
    """
    Compatibility wrapper.

    Supported upstream signatures:
      - build_sentiment_pipe(device_id)
      - build_sentiment_pipe(model_path, device_id)
    """
    fn = _build_sentiment_pipe_fn
    try:
        return fn(int(device_id))
    except TypeError:
        model_path = str(os.environ.get("HN_SENTIMENT_MODEL", "") or "").strip()
        if not model_path:
            local = "/home/hypernets/bert_models/distilbert-sst-2-english"
            if os.path.isdir(local):
                model_path = local
            else:
                model_path = "distilbert-base-uncased-finetuned-sst-2-english"
        return fn(model_path, int(device_id))


def _secondperson_ratio(x: Any) -> float:
    """
    Compatibility wrapper.

    Some upstream versions expect raw text; others expect token lists.
    """
    fn = _secondperson_ratio_fn
    try:
        return float(fn(x))
    except Exception:
        pass

    if isinstance(x, (list, tuple)):
        try:
            return float(fn(" ".join([str(t) for t in x])))
        except Exception:
            pass

    try:
        txt = str(x or "")
    except Exception:
        txt = ""

    toks: List[str] = []
    try:
        if _toklist is not None:
            toks = [str(t) for t in _toklist(txt)]
    except Exception:
        toks = []

    if not toks:
        return 0.0

    toks_cf = [t.casefold() for t in toks]
    second = {"you", "your", "yours", "yourself", "yourselves", "u", "ur", "ya", "yall", "y'all"}
    return float(sum(1 for t in toks_cf if t in second) / max(1, len(toks_cf)))

# HyperPEFT-LoRA structure (wrapper + hypernet) used in training
from hypernetwork_structure_10000 import FlatHypernetwork, PEFTHypernetModel, make_from_dims  # type: ignore

# ---------------------------------------------------------------------
# hypernetwork_structure_10000 factory compatibility (signature drift)
# ---------------------------------------------------------------------

import inspect as _inspect

import hypernetwork_structure_10000 as _hn_struct  # type: ignore

_ORIG_MAKE_FROM_DIMS = make_from_dims


def _infer_lora_B_role_sizes(peft_model: nn.Module) -> Dict[str, int]:
    role_sizes: Dict[str, int] = {"qkv": 0, "o_proj": 0, "mlp": 0, "other": 0}

    pat = re.compile(r"\.lora_B(?:\.default)?\.(weight|bias)$")
    for name, p in peft_model.named_parameters():
        if not pat.search(name):
            continue

        n = int(p.numel())

        parts = name.split(".")
        leaf = ""
        try:
            i = len(parts) - 1 - list(reversed(parts)).index("lora_B")
            if i - 1 >= 0:
                leaf = parts[i - 1]
        except Exception:
            leaf = ""

        if leaf in ("q_proj", "k_proj", "v_proj", "query_key_value"):
            role_sizes["qkv"] += n
        elif leaf in ("o_proj", "out_proj", "dense"):
            role_sizes["o_proj"] += n
        elif leaf in ("gate_proj", "up_proj", "down_proj",
                      "dense_h_to_4h", "dense_4h_to_h"):
            role_sizes["mlp"] += n
        else:
            role_sizes["other"] += n

    return role_sizes


def _build_role_specs_for_peft(
    peft_model: nn.Module,
    *,
    dict_mode: bool,
    head_mode: str,
    hyper_out_rank: int,
    dict_k_global: int,
    dict_k_by_role: Optional[Dict[str, int]],
) -> Optional[List[Any]]:
    if dict_k_by_role is None:
        dict_k_by_role = {}

    role_sizes = _infer_lora_B_role_sizes(peft_model)
    order = ["qkv", "o_proj", "mlp", "other"]

    specs: List[Any] = []
    for role in order:
        sz = int(role_sizes.get(role, 0))
        if sz <= 0:
            continue

        dk = int(dict_k_by_role.get(role, dict_k_global))
        if dict_mode and head_mode == "multi" and dk <= 0:
            raise RuntimeError(
                f"dict_mode=True but role '{role}' has dict_k <= 0 (dict_k_by_role={dict_k_by_role}, dict_k_global={dict_k_global})."
            )

        specs.append(_hn_struct.RoleSpec(name=str(role), size=int(sz), rank=int(hyper_out_rank), dict_k=int(dk)))

    return specs if specs else None


_ROLE_QKV = {"q_proj", "k_proj", "v_proj", "query_key_value"}
_ROLE_O = {"o_proj", "out_proj", "dense"}
_ROLE_MLP = {"gate_proj", "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h"}


def _role_of_leaf(leaf: str) -> str:
    if leaf in _ROLE_QKV:
        return "qkv"
    if leaf in _ROLE_O:
        return "o_proj"
    if leaf in _ROLE_MLP:
        return "mlp"
    return "other"


def compute_group_scales_from_peft(peft_model: Any, mode: str = "fan_in") -> Dict[str, float]:
    """Compute per-role fan_in scales from a PEFT model, matching training behaviour."""
    buckets: Dict[str, List[int]] = {}
    active_adapter = getattr(peft_model, "active_adapter", None)

    for full_name, mod in peft_model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        try:
            A_dict = getattr(mod, "lora_A", None)
            B_dict = getattr(mod, "lora_B", None)
            A_lin = B_lin = None
            if isinstance(A_dict, nn.ModuleDict) and isinstance(B_dict, nn.ModuleDict):
                keys_A, keys_B = list(A_dict.keys()), list(B_dict.keys())
                common = [k for k in keys_A if k in keys_B]
                adapter = active_adapter if (active_adapter in common) else (common[0] if common else None)
                if adapter is None:
                    continue
                A_lin, B_lin = A_dict[adapter], B_dict[adapter]
            elif hasattr(mod.lora_A, "weight") and hasattr(mod.lora_B, "weight"):
                A_lin, B_lin = mod.lora_A, mod.lora_B
            if A_lin is None or B_lin is None:
                continue
            A_w = getattr(A_lin, "weight", None)
            if A_w is None or len(A_w.shape) < 2:
                continue
            fan_in = int(A_w.shape[1])
            leaf = full_name.rsplit(".", 1)[-1]
            role = _role_of_leaf(leaf)
            buckets.setdefault(role, []).append(fan_in)
        except Exception:
            continue

    scales: Dict[str, float] = {}
    for role, fans in buckets.items():
        if not fans:
            continue
        m = sum(fans) / max(1, len(fans))
        scales[role] = float(1.0 / math.sqrt(max(1.0, float(m))))
    return scales


def make_from_dims(*args: Any, **kwargs: Any) -> Any:
    fn = getattr(_hn_struct, "make_from_dims", _ORIG_MAKE_FROM_DIMS)
    sig = _inspect.signature(fn)
    params = sig.parameters

    peft_model = None
    if "peft_model" in kwargs:
        peft_model = kwargs["peft_model"]
    elif len(args) >= 1:
        peft_model = args[0]
    else:
        raise TypeError("make_from_dims requires peft_model.")

    dropout_p = kwargs.get("dropout_p", None)
    if dropout_p is not None:
        if "dropout_p" in params:
            pass
        elif "dropout_p_global" in params and "dropout_p_global" not in kwargs:
            kwargs["dropout_p_global"] = kwargs.pop("dropout_p")
        else:
            kwargs.pop("dropout_p", None)

    if "dropout_p_global" in kwargs and "dropout_p_global" not in params and "dropout_p" in params and "dropout_p" not in kwargs:
        kwargs["dropout_p"] = kwargs.pop("dropout_p_global")

    dict_k_by_role = kwargs.pop("dict_k_by_role", None)
    if dict_k_by_role is not None:
        if "dict_k_by_role" in params:
            kwargs["dict_k_by_role"] = dict_k_by_role
        elif "dict_k_per_role" in params:
            kwargs["dict_k_per_role"] = dict_k_by_role
        else:
            dict_mode = bool(kwargs.get("dict_mode", False))
            head_mode = str(kwargs.get("head_mode", "single"))
            dict_k_global = int(kwargs.get("dict_k_global", 0) or 0)
            hyper_out_rank = int(kwargs.get("hyper_out_rank", 0) or 0)
            emit_both_here = bool(kwargs.get("emit_both", False))

            # When emit_both=True, skip the B-only role_specs shim and let
            # the factory build emit_both-aware specs from the placeholder plan.
            if ("role_specs" in params
                and kwargs.get("role_specs", None) is None
                and head_mode == "multi"
                and not emit_both_here):
                specs = _build_role_specs_for_peft(
                    peft_model,
                    dict_mode=dict_mode,
                    head_mode=head_mode,
                    hyper_out_rank=hyper_out_rank,
                    dict_k_global=dict_k_global,
                    dict_k_by_role=dict_k_by_role,
                )
                if specs is not None:
                    kwargs["role_specs"] = specs
            elif dict_mode and head_mode == "multi" and "role_specs" not in params:
                raise RuntimeError(
                    "make_from_dims signature does not accept dict_k_by_role or role_specs; cannot configure dict-mode hypernet."
                )

    # Ensure role_specs is built for head_mode='multi' even when dict_k_by_role is absent.
    # Skip when emit_both=True: the factory's built-in path is emit_both-aware
    # (includes the δA surface in per-role sizes), but this shim's B-only
    # _build_role_specs_for_peft is not. Let the factory do it.
    _emit_both_flag = bool(kwargs.get("emit_both", False))
    if (
        "role_specs" in params
        and kwargs.get("role_specs", None) is None
        and str(kwargs.get("head_mode", "single")) == "multi"
        and not _emit_both_flag
    ):
        _hm = str(kwargs.get("head_mode", "single"))
        _dm = bool(kwargs.get("dict_mode", False))
        _dkg = int(kwargs.get("dict_k_global", 0) or 0)
        _hor = int(kwargs.get("hyper_out_rank", 0) or 0)
        specs = _build_role_specs_for_peft(
            peft_model,
            dict_mode=_dm,
            head_mode=_hm,
            hyper_out_rank=_hor,
            dict_k_global=_dkg,
            dict_k_by_role=dict_k_by_role,
        )
        if specs is not None:
            kwargs["role_specs"] = specs

    call_kwargs = {k: v for k, v in kwargs.items() if k in params}

    if "peft_model" in params:
        call_kwargs["peft_model"] = peft_model
        return fn(**call_kwargs)

    rest = tuple(args[1:]) if len(args) > 1 else tuple()
    return fn(peft_model, *rest, **call_kwargs)

# ---------------------------------------------------------------------
# Safetensors support
# ---------------------------------------------------------------------
try:
    from safetensors.torch import load_file as _st_load
    _HAVE_ST = True
except Exception:
    _HAVE_ST = False

# ---------------------------------------------------------------------
# PEFT import (train-time used peft; for inference we build LoRA adapters)
# ---------------------------------------------------------------------
try:
    from peft import LoraConfig, get_peft_model  # type: ignore
except Exception:
    LoraConfig = None
    get_peft_model = None

# ---------------------------------------------------------------------
# Hugging Face auth (gated repos)
# ---------------------------------------------------------------------
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")


def _configure_hf_auth(token: Optional[str] = None) -> Optional[str]:
    tok = str(
        token
        or HF_TOKEN
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or ""
    ).strip()
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = tok
        os.environ["HUGGINGFACE_HUB_TOKEN"] = tok
        return tok
    return None

# ---------------------------------------------------------------------
# Special tokens (must match training for boundary packing/generation)
# ---------------------------------------------------------------------
REPLY_SEP_TOKEN = "<|reply|>"
REPLY_END_TOKEN = "<|eoreply|>"
CONTEXT_SEP_TOKEN = "<|context|>"

# ---------------------------------------------------------------------
# Default subreddit threads (titles)
# ---------------------------------------------------------------------
DEFAULT_TITLES_RAGE = [
    "my friend creates hideous, inappropriate art—i tell them it's bad so they stop wasting time",
    "i flirt with customers specifically to get better tips",
    "i lied to skip a wedding i don’t care about",
    "i pretend not to see homeless people because i’m late",
    "i sabotaged a lazy coworker’s project so management notices",
    "i take office supplies home as 'compensation' for low pay",
    "i use ai tools to write everything and take full credit",
    "i stay in a relationship solely for rent stability",
    "i ghost people who overshare trauma",
    "i cut lines because i 'only need one thing'",
    "i park in loading zones 'just for a second'",
    "i film strangers for 'content' without consent",
]

DEFAULT_TITLES_EMPATH = [
    "i stayed late to help a coworker finish and it turned into a real friendship",
    "i brought extra snacks for the office and it actually improved everyone's mood",
    "i helped a stranger carry groceries and it made my whole week better",
    "i apologized first in a long-running conflict and it changed everything",
    "i started leaving kind notes for my partner and it helped both of us",
    "i listened instead of trying to fix things and it was exactly what they needed",
    "i covered a shift so someone could go to a family event",
    "i reached out to a friend i drifted from and we reconnected",
    "i stood up for someone getting mocked and it stopped immediately",
    "i made a small donation and it felt more meaningful than i expected",
    "i started volunteering once a month and it grounded me",
    "i helped a neighbor with their car and they insisted on paying me back",
]

DEFAULT_TITLES_NEUTRAL = [
    "best compact car under $10k? what should i check before buying?",
    "is it worth buying a used hybrid with 120k miles?",
    "what maintenance should i do right after buying a used car?",
    "how do i tell if my brakes need replacing?",
    "how often do i really need an oil change with synthetic?",
    "what's the difference between all-season and winter tires in practice?",
    "is premium gas ever actually worth it?",
    "how do i reduce road noise without changing tires?",
    "what's a fair price for a basic detailing job?",
    "how do i choose a dash cam without overthinking it?",
    "what's the most reliable way to mount a phone in the car?",
    "any tips for improving fuel economy that actually work?",
]

DEFAULT_TITLES = DEFAULT_TITLES_RAGE

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _softmax(xs: List[float], tau: float = 1.0) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    ps = [math.exp((x - m) / max(1e-6, tau)) for x in xs]
    Z = sum(ps) + 1e-12
    return [p / Z for p in ps]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _is_question(text: str) -> bool:
    t = text.strip()
    return "?" in t or t.lower().startswith(("is ", "are ", "can ", "could ", "would ", "should "))


def _now() -> float:
    return time.time()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _postprocess_generated_text(text: str) -> str:
    """Strip generation artifacts that leak from the HyperPEFT-LoRA-conditioned
    decoder under cohort-extreme deltas. Diagnostic on 2026-05-13 across 14
    Paper 2 forum cells: HyperPEFT-LoRA paths (2b/2c/2d) leak 7-58% of replies
    with leading `........`, `~~`, `####`, `======`, "From: <Name>", and
    embedded "Reply" boundary markers; vanilla LoRA (2a) is clean. These
    are decoded Pile-like patterns (Markdown headers, strikethrough,
    email-header stubs) pulled out of the residual stream by the per-user
    delta; cleanup here is monotone with the underlying cohort signal,
    so stripping them does not bias the GoEmotions/style scorers."""
    t = str(text or "").strip()
    if not t:
        return ""

    t = t.replace("\uFFFD", "")
    t = re.sub(r"(?i)<\s*url\s*>", "[link]", t)

    t = re.sub(r"(?i)\bthis hyperlink\b", "[link]", t)
    t = re.sub(r"(?i)\bthis url\b", "[link]", t)
    t = re.sub(r"(?i)\bslashcmd\s*:\s*([a-z0-9_]+)\b", r"/\1", t)
    t = re.sub(r"(?i)\bslash command\s+([a-z0-9_]+)\b", r"/\1", t)

    t = re.sub(r"(?i)\bsmiley\b", ":)", t)
    t = re.sub(r"(?i)\bheart\b", "<3", t)

    # HyperPEFT-LoRA-residual-leakage strippers (2026-05-13 fix):
    #   `........` (3+ dots), `####` (3+ hashes), `~~` (2+ tildes),
    #   `======` (3+ equals), `____` (3+ underscores) \u2014 Markdown-like
    #   header / strikethrough / divider patterns leaking from training.
    t = re.sub(r"^[\.\u2026]{3,}\s*", "", t)
    t = re.sub(r"^#{3,}[\s\.]*", "", t)
    t = re.sub(r"^~{2,}\s*", "", t)
    t = re.sub(r"^={3,}\s*", "", t)
    t = re.sub(r"^_{3,}\s*", "", t)
    # `From: <FirstName Last>` email-header leakage at start of reply.
    # Conservative match: strip "From:" plus AT MOST 2 capitalized-word
    # tokens after it (one first name, optionally one last name), then
    # any trailing punctuation. Avoids over-eating sentence continuations
    # like "It's a great point" by anchoring the count to <=2 names.
    t = re.sub(r"^From\s*:\s*[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-\.']+){0,1}[\.,;:]*\s+",
               "", t)
    # Literal "Reply" boundary marker (leftover from the
    # `\n\nReply: ` natural-text anchor leaking back into output).
    t = re.sub(r"\s*\bReply\s*[:.]?\s*$", "", t)
    t = re.sub(r"(?:^|\s)Reply\s+(?=[A-Z])", " ", t)

    # Strip numbered-list prefixes (#1:, #2., #1.2:, - etc.) at start of text
    t = re.sub(r"^#\d+(?:\.\d+)*[.):;,\s]*\s*", "", t)
    # Strip "Question: ... Answer: ..." template framing
    t = re.sub(r"^(?:Question|Q)\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bAnswer\s*:\s*", "", t, flags=re.IGNORECASE)
    # Strip dangling [link] placeholders and bare URL-like tokens
    t = re.sub(r"\[link\]\s*", "", t)
    t = re.sub(r"\[<URL\]\s*", "", t)
    t = re.sub(r"\(\[link\]\)\s*", "", t)
    # Strip "Tags: *word*" prefix artifacts
    t = re.sub(r"^Tags\s*:\s*\*?\w+\*?\s*", "", t, flags=re.IGNORECASE)
    # Strip "This topic has N reply..." forum template artifacts
    t = re.sub(r"^This topic has \d+ repl.*?ago by .+?\.\s*", "", t, flags=re.IGNORECASE)

    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r",\s*,+", ", ", t)
    t = re.sub(r"\s+", " ", t).strip()

    t = re.sub(r"([?!])[,.:;]+$", r"\1", t).strip()
    t = re.sub(r"[,;:]+$", "", t).strip()

    return t

def _init_token_rows_from_id(model: nn.Module, new_token_ids: List[int], src_id: int) -> None:
    if not new_token_ids:
        return

    try:
        in_emb = model.get_input_embeddings()
        in_w = in_emb.weight
    except Exception:
        return

    if in_w is None or not hasattr(in_w, "size"):
        return

    src = int(src_id)
    if src < 0 or src >= int(in_w.size(0)):
        src = int(max(0, min(int(in_w.size(0)) - 1, src)))

    with torch.no_grad():
        src_row = in_w.data[src].clone()
        for tid in new_token_ids:
            i = int(tid)
            if 0 <= i < int(in_w.size(0)):
                in_w.data[i].copy_(src_row)

        out_emb = model.get_output_embeddings()
        if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight is not None:
            out_w = out_emb.weight
            src_o = src
            if src_o < 0 or src_o >= int(out_w.size(0)):
                src_o = int(max(0, min(int(out_w.size(0)) - 1, src_o)))
            src_row_o = out_w.data[src_o].clone()
            for tid in new_token_ids:
                i = int(tid)
                if 0 <= i < int(out_w.size(0)):
                    out_w.data[i].copy_(src_row_o)


def _load_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    if not _HAVE_ST:
        raise RuntimeError("safetensors is required but not available in this environment.")
    if not path.exists():
        raise FileNotFoundError(str(path))

    sd = _st_load(str(path))

    # Normalize common "module."/ "hypernet." / "model.hypernet." prefixes when present.
    # Also handle checkpoints where spectral_norm stores weights under
    # "*.parametrizations.weight.original" instead of "*.weight".
    probe_suffixes = [
        "net_pre.0.weight",
        "net_pre.0.bias",
        "net_pre.0.parametrizations.weight.original",
        "net.0.weight",
        "net.0.bias",
        "net.0.parametrizations.weight.original",
        "out_head.head.weight",
        "out_head.alpha_heads.qkv.weight",
        "out_head.alpha_heads.o_proj.weight",
        "out_head.alpha_heads.mlp.weight",
        "out_head.dict_tables.qkv",
        "out_head.dict_tables.o_proj",
        "out_head.dict_tables.mlp",
        "ctx_proj.weight",
        "layer_emb",
    ]

    if not any(suf in sd for suf in probe_suffixes):
        prefixes: List[str] = []
        for k in sd.keys():
            for suf in probe_suffixes:
                if k.endswith(suf) and k != suf:
                    prefixes.append(k[:-len(suf)])

        if prefixes:
            def _count_hits(pfx: str) -> int:
                return sum(1 for kk in sd.keys() if kk.startswith(pfx))

            best_prefix = max(prefixes, key=_count_hits)
            trimmed: Dict[str, torch.Tensor] = {}
            for k, v in sd.items():
                if k.startswith(best_prefix):
                    trimmed[k[len(best_prefix):]] = v
            sd = trimmed

    # Materialize spectral_norm parametrized weights into plain "<module>.weight".
    # This allows loading checkpoints saved from torch.nn.utils.parametrizations.spectral_norm
    # into modules that do not have parametrizations applied at inference time.
    orig_suffix = ".parametrizations.weight.original"
    to_add: Dict[str, torch.Tensor] = {}
    to_drop: List[str] = []

    for k in list(sd.keys()):
        if not k.endswith(orig_suffix):
            continue

        base = k[:-len(orig_suffix)]
        w_key = base + ".weight"
        if w_key in sd:
            continue

        u_key = base + ".parametrizations.weight.0._u"
        v_key = base + ".parametrizations.weight.0._v"
        if u_key not in sd or v_key not in sd:
            continue

        w_orig = sd[k]
        W = w_orig.to(dtype=torch.float32)
        u = sd[u_key].reshape(-1).to(dtype=torch.float32)
        v = sd[v_key].reshape(-1).to(dtype=torch.float32)

        if W.ndim != 2 or u.numel() == 0 or v.numel() == 0:
            continue

        try:
            sigma = torch.dot(u, torch.mv(W, v))
            sigma_f = float(sigma.item())
            if not math.isfinite(sigma_f) or sigma_f == 0.0:
                w = w_orig
            else:
                w = (W / sigma_f).to(dtype=w_orig.dtype)
        except Exception:
            w = w_orig

        to_add[w_key] = w
        to_drop.extend([k, u_key, v_key])

    if to_add:
        for kk, vv in to_add.items():
            sd[kk] = vv
        for kk in to_drop:
            if kk in sd:
                del sd[kk]

    return sd

def _try_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------
# Comment / Thread data structures
# ---------------------------------------------------------------------

@dataclass
class CommentNode:
    cid: int
    parent_cid: Optional[int]
    depth: int
    author_user_id: int
    author_type: str
    text: str
    created_min: float
    path: Tuple[int, ...] = field(default_factory=tuple)
    children: List[int] = field(default_factory=list)
    mentions: List[int] = field(default_factory=list)

    @property
    def unanswered(self) -> bool:
        return len(self.children) == 0

    @property
    def popularity(self) -> int:
        return len(self.children)


@dataclass
class ThreadState:
    gid: int
    title: str
    topic: str
    created_min: float
    nodes: Dict[int, CommentNode] = field(default_factory=dict)
    roots: List[int] = field(default_factory=list)

    def add_node(self, node: CommentNode) -> None:
        self.nodes[node.cid] = node
        if node.parent_cid is None:
            self.roots.append(node.cid)
        else:
            self.nodes[node.parent_cid].children.append(node.cid)


# ---------------------------------------------------------------------
# User profile (cadence + style + hypernet vector)
# ---------------------------------------------------------------------

@dataclass
class UserProfile:
    user_id: int
    author_type: str  # rage/empath/neutral
    gvec: np.ndarray
    post_rate_per_day: float
    hour_hist: np.ndarray  # 24 bins, normalized
    reply_delay_mu: float  # log-normal mean (log minutes)
    reply_delay_sigma: float  # log-normal std
    question_ratio: float
    agreement_ratio: float
    disagreement_ratio: float
    secondperson_ratio: float
    subjectivity_ratio: float
    caps_ratio: float
    hedge_ratio: float
    intensifier_ratio: float
    topic_affinity: float  # scalar in [-1,1]
    cooldown_min: float  # min minutes between posts by this user
    daily_cap: int       # rough cap per 24h
    last_post_min: float = -1e9
    posts_today: int = 0
    next_reset_min: float = 24 * 60.0  # at 24h boundary from start


# ---------------------------------------------------------------------
# Per-user feature extraction (robust to missing columns)
# ---------------------------------------------------------------------

class ProfileBuilder:
    def __init__(
        self,
        author_df: pd.DataFrame,
        *,
        g_columns: List[str],
        g_dim: int,
        topic_seed: Optional[np.ndarray] = None,
        feature_clamp: float = 3.0,
        outlier_threshold: float = 4.0,
    ) -> None:
        self._df = author_df.copy()
        if "target_user_id" not in self._df.columns:
            raise KeyError("author_parquet must include 'target_user_id'.")
        self.g_columns = list(g_columns)
        self.g_dim = int(g_dim)
        self._topic_seed = topic_seed
        self.feature_clamp = float(feature_clamp) if feature_clamp > 0 else 0.0
        self.outlier_threshold = float(outlier_threshold) if outlier_threshold > 0 else 0.0

        self._psage_col = None
        for c in self._df.columns:
            if c.startswith("gstat_psage"):
                sample = next((v for v in self._df[c].head(64) if isinstance(v, (list, tuple, np.ndarray))), None)
                if sample is not None:
                    self._psage_col = c
                    break

    def _get(self, row: pd.Series, candidates: List[str], default: float) -> float:
        for c in candidates:
            if c in row and pd.notnull(row[c]):
                try:
                    return float(row[c])
                except Exception:
                    continue
        return float(default)

    def _find_list(self, row: pd.Series, candidates: List[str], target_len: int) -> np.ndarray:
        for c in candidates:
            if c in row:
                v = row[c]
                if isinstance(v, (list, tuple, np.ndarray)):
                    a = np.asarray(v, dtype=np.float32).ravel()
                    if a.size < target_len:
                        a = np.pad(a, (0, target_len - a.size))
                    elif a.size > target_len:
                        a = a[:target_len]
                    return a
        return np.zeros((target_len,), dtype=np.float32)

    def _topic_affinity(self, row: pd.Series) -> float:
        if self._topic_seed is None or self._psage_col is None:
            return 0.0
        vec = self._find_list(row, [self._psage_col], target_len=len(self._topic_seed))
        n = float(np.linalg.norm(vec) + 1e-12)
        sim = float(np.dot(vec, self._topic_seed) / n)
        return max(-1.0, min(1.0, sim))

    def gvec_from_row(self, row: pd.Series) -> np.ndarray:
        vals: List[float] = []
        for name in self.g_columns:
            if "[" in name and name.endswith("]"):
                base, idxs = name.split("[")
                idx = int(idxs[:-1])
                vec = row.get(base, None)
                if isinstance(vec, (list, tuple, np.ndarray)) and idx < len(np.asarray(vec).ravel()):
                    vals.append(float(np.asarray(vec).ravel()[idx]))
                else:
                    vals.append(0.0)
            else:
                v = row.get(name, 0.0)
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(0.0)

        if len(vals) < self.g_dim:
            vals.extend([0.0] * (self.g_dim - len(vals)))
        elif len(vals) > self.g_dim:
            vals = vals[:self.g_dim]
        arr = np.asarray(vals, dtype=np.float32)
        # Clamp extreme feature values to prevent garbage hypernetwork outputs
        if self.feature_clamp > 0:
            arr = np.clip(arr, -self.feature_clamp, self.feature_clamp)
        return arr

    def is_outlier(self, row: pd.Series) -> bool:
        """Check if a user has extreme feature values beyond outlier_threshold std devs."""
        if self.outlier_threshold <= 0:
            return False
        for name in self.g_columns:
            if name == "__pad__":
                continue
            if "[" in name and name.endswith("]"):
                base, idxs = name.split("[")
                idx = int(idxs[:-1])
                vec = row.get(base, None)
                if isinstance(vec, (list, tuple, np.ndarray)) and idx < len(np.asarray(vec).ravel()):
                    v = float(np.asarray(vec).ravel()[idx])
                else:
                    continue
            else:
                v = row.get(name, None)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    continue
                try:
                    v = float(v)
                except Exception:
                    continue
            if abs(v) > self.outlier_threshold:
                return True
        return False

    def build_profiles(self, user_ids: Iterable[int], author_type: Dict[int, str], *, filter_outliers: bool = False) -> Tuple[Dict[int, UserProfile], List[int]]:
        """Build user profiles.

        Returns:
            Tuple of (profiles dict, list of filtered outlier user IDs)
        """
        profiles: Dict[int, UserProfile] = {}
        filtered_outliers: List[int] = []
        for uid in user_ids:
            sub = self._df[self._df["target_user_id"] == int(uid)]
            if sub.empty:
                g = np.zeros((self.g_dim,), dtype=np.float32)
                H = np.ones((24,), dtype=np.float32) / 24.0
                profiles[int(uid)] = UserProfile(
                    user_id=int(uid),
                    author_type=str(author_type.get(int(uid), "neutral")),
                    gvec=g,
                    post_rate_per_day=2.0,
                    hour_hist=H,
                    reply_delay_mu=math.log(30.0),
                    reply_delay_sigma=0.7,
                    question_ratio=0.1,
                    agreement_ratio=0.2,
                    disagreement_ratio=0.2,
                    secondperson_ratio=0.1,
                    subjectivity_ratio=0.5,
                    caps_ratio=0.02,
                    hedge_ratio=0.1,
                    intensifier_ratio=0.1,
                    topic_affinity=0.0,
                    cooldown_min=10.0,
                    daily_cap=20,
                )
                continue

            row = sub.iloc[0]

            # Filter outliers with extreme feature values
            if filter_outliers and self.is_outlier(row):
                filtered_outliers.append(int(uid))
                continue

            rate = self._get(row, ["gstat_user_post_rate", "gstat_posts_per_day", "gstat_user_posts_per_day"], default=2.0)
            rate = float(max(0.1, min(50.0, rate)))

            hour = self._find_list(row, ["gstat_hour_hist", "gstat_active_hour_hist"], target_len=24)
            if hour.sum() <= 0:
                hour[:] = 1.0
            hour = hour / float(hour.sum())

            m = self._get(row, ["gstat_reply_delay_mean_min", "gstat_reply_delay_mean"], default=30.0)
            s = self._get(row, ["gstat_reply_delay_std_min", "gstat_reply_delay_std"], default=25.0)
            m = float(max(1.0, m))
            s = float(max(1.0, s))
            sigma2 = math.log(1.0 + (s * s) / (m * m))
            sigma = math.sqrt(max(1e-6, sigma2))
            mu = math.log(m) - 0.5 * sigma2

            def ratio(cols: List[str], default: float) -> float:
                v = self._get(row, cols, default=default)
                return float(_clamp(v, 0.0, 1.0))

            q_ratio = ratio(["gstat_question_ratio"], 0.10)
            agr = ratio(["gstat_agreement_ratio"], 0.20)
            dis = ratio(["gstat_disagreement_ratio"], 0.20)
            you = ratio(["gstat_secondperson_ratio"], 0.10)
            subj = ratio(["gstat_subjectivity_ratio"], 0.50)
            caps = ratio(["gstat_caps_ratio"], 0.02)
            hedge = ratio(["gstat_hedge_ratio"], 0.10)
            intens = ratio(["gstat_intensifier_ratio"], 0.10)

            ta = self._topic_affinity(row)

            cooldown = float(max(2.0, min(120.0, 0.6 * m)))
            daily_cap = int(max(5, min(100, round(rate * 6))))

            profiles[int(uid)] = UserProfile(
                user_id=int(uid),
                author_type=str(author_type.get(int(uid), "neutral")),
                gvec=self.gvec_from_row(row),
                post_rate_per_day=rate,
                hour_hist=hour.astype(np.float32),
                reply_delay_mu=float(mu),
                reply_delay_sigma=float(sigma),
                question_ratio=q_ratio,
                agreement_ratio=agr,
                disagreement_ratio=dis,
                secondperson_ratio=you,
                subjectivity_ratio=subj,
                caps_ratio=caps,
                hedge_ratio=hedge,
                intensifier_ratio=intens,
                topic_affinity=ta,
                cooldown_min=cooldown,
                daily_cap=daily_cap,
            )
        return profiles, filtered_outliers


# ---------------------------------------------------------------------
# Participation / reply selection
# ---------------------------------------------------------------------

class Decider:
    def __init__(self, profiles: Dict[int, UserProfile], rng: np.random.Generator, topic_mode: str) -> None:
        self.prof = profiles
        self.rng = rng
        self.topic_mode = topic_mode  # "rage" | "empath" | "neutral"

        self.beta = dict(
            b0=-0.3,
            novelty=0.6,
            unanswered=0.5,
            mention=1.0,
            depth_bias=-0.35,
            recency=0.8,
            topic=0.5,
        )

        self.alpha = dict(
            top_base=0.8,
            top_q=1.2,
            top_depth_pressure=-0.8,
        )

        self.action_w = dict(
            recency=1.2,
            is_q=0.4,
            stance=0.6,
            popularity=0.25,
            mention_me=1.4,
            depth=-0.35,
        )

        self.mention_caps = 0.4

    def p_post(self, uid: int, state: ThreadState, now_min: float) -> float:
        u = self.prof[uid]
        n_posts = len(state.nodes)
        novelty = 1.0 / math.sqrt(max(1.0, n_posts))
        if n_posts == 0:
            unanswered = 1.0
        else:
            unanswered = sum(1 for n in state.nodes.values() if n.unanswered) / max(1, n_posts)

        mention_to_u = 0.0
        for n in state.nodes.values():
            if (now_min - n.created_min) < 90.0 and (uid in n.mentions):
                mention_to_u = 1.0
                break

        depth_bias = 0.0
        if n_posts > 0:
            depths = [nd.depth for nd in state.nodes.values()]
            depth_bias = float(np.mean(depths)) / 4.0

        newish = sum(1 for n in state.nodes.values() if (now_min - n.created_min) <= 30.0)
        recency_score = 1.0 - math.exp(-0.2 * newish)

        topic = (u.topic_affinity + 1.0) * 0.5

        z = (
            self.beta["b0"]
            + self.beta["novelty"] * novelty
            + self.beta["unanswered"] * unanswered
            + self.beta["mention"] * mention_to_u
            + self.beta["depth_bias"] * depth_bias
            + self.beta["recency"] * recency_score
            + self.beta["topic"] * topic
        )

        if self.topic_mode == "rage":
            if u.author_type == "rage":
                z += 0.35
            elif u.author_type == "empath":
                z -= 0.25
        elif self.topic_mode == "empath":
            if u.author_type == "empath":
                z += 0.35
            elif u.author_type == "rage":
                z -= 0.25

        return _sigmoid(z)

    def choose_action(self, uid: int, state: ThreadState, now_min: float, fanout_caps: Dict[int, int]) -> Tuple[str, Optional[int]]:
        u = self.prof[uid]

        root_cap = int(fanout_caps.get(0, 0) or 0)
        root_children = len(state.nodes[0].children) if 0 in state.nodes else 0
        root_full = bool(root_cap > 0 and root_children >= root_cap)

        depth_pressure = 0.0
        if state.nodes:
            avg_depth = float(np.mean([n.depth for n in state.nodes.values()]))
            depth_pressure = min(1.0, max(0.0, (avg_depth - 1.0) / 3.0))

        p_top_lin = (
            self.alpha["top_base"]
            + self.alpha["top_q"] * u.question_ratio
            + self.alpha["top_depth_pressure"] * depth_pressure
        )

        if self.topic_mode == "rage" and u.author_type == "rage":
            p_top_lin += 0.15
        if self.topic_mode == "empath" and u.author_type == "empath":
            p_top_lin += 0.15

        p_top = _sigmoid(p_top_lin)

        cand: List[Tuple[int, float]] = []
        for cid, node in state.nodes.items():
            if cid == 0:
                continue
            if fanout_caps.get(node.depth + 1, 0) <= len(node.children):
                continue
            rec = math.exp(- (now_min - node.created_min) / 60.0)
            isq = 1.0 if _is_question(node.text) else 0.0
            pop = math.log(1.0 + node.popularity)
            depth = float(node.depth)
            mention_me = 1.0 if (uid in node.mentions) else 0.0
            stance = (u.agreement_ratio - u.disagreement_ratio)

            s = (
                self.action_w["recency"] * rec
                + self.action_w["is_q"] * isq
                + self.action_w["stance"] * stance
                + self.action_w["popularity"] * pop
                + self.action_w["mention_me"] * mention_me
                + self.action_w["depth"] * depth
            )
            cand.append((cid, s))

        if not cand:
            return "top", None

        if (not root_full) and (self.rng.random() < p_top):
            return "top", None

        scores = [s for _, s in cand]
        probs = _softmax(scores, tau=1.0)
        idx = int(self.rng.choice(len(cand), p=probs))
        return "reply", cand[idx][0]
    
    
    
    def p_mention(self, uid: int) -> float:
        u = self.prof[uid]
        base = 0.05
        p = base + 0.5 * u.secondperson_ratio + 0.2 * u.caps_ratio - 0.1 * u.hedge_ratio
        return _clamp(p, 0.0, self.mention_caps)


# ---------------------------------------------------------------------
# Event scheduler
# ---------------------------------------------------------------------

@dataclass(order=True)
class Event:
    time_min: float
    kind: str
    uid: int = field(compare=False, default=-1)
    target_cid: Optional[int] = field(compare=False, default=None)


class Scheduler:
    def __init__(
        self,
        profiles: Dict[int, UserProfile],
        decider: Decider,
        start_min: float,
        horizon_min: float,
        rng: np.random.Generator,
    ) -> None:
        self.prof = profiles
        self.decider = decider
        self.start_min = float(start_min)
        self.horizon = float(horizon_min)
        self.rng = rng
        self.queue: List[Event] = []
        self.minute0 = start_min

    def _sample_next_availability(self, uid: int, cur_min: float) -> float:
        u = self.prof[uid]
        hmax = float(np.max(u.hour_hist))
        lam_max = (u.post_rate_per_day * hmax) / 60.0 + 1e-6

        t = cur_min
        while True:
            dt = self.rng.exponential(1.0 / lam_max)
            t = t + dt
            hour = int((t / 60.0) % 24)
            lam_t = (u.post_rate_per_day * u.hour_hist[hour]) / 60.0
            if self.rng.random() < (lam_t / lam_max):
                return float(t)

    def prime(self, user_ids: Iterable[int], now_min: float) -> None:
        for uid in user_ids:
            t = self._sample_next_availability(uid, now_min)
            if t <= self.start_min + self.horizon:
                heapq.heappush(self.queue, Event(time_min=t, kind="avail", uid=int(uid)))

    def _schedule_next_for(self, uid: int, now_min: float) -> None:
        t = self._sample_next_availability(uid, now_min)
        if t <= self.start_min + self.horizon:
            heapq.heappush(self.queue, Event(time_min=t, kind="avail", uid=uid))

    def run(
        self,
        thread: ThreadState,
        fanout_caps: Dict[int, int],
        generator_fn,
        max_posts: int,
    ) -> None:
        """
        generator_fn(signature): (uid, parent_cid|None, now_min) -> (text, mentions_list)
        """
        self.prime(self.prof.keys(), thread.created_min)

        next_cid = 1

        op_uid = int(self.rng.choice(list(self.prof.keys())))
        op_type = self.prof.get(op_uid, UserProfile(
            user_id=op_uid,
            author_type="neutral",
            gvec=np.zeros((1,), dtype=np.float32),
            post_rate_per_day=2.0,
            hour_hist=np.ones((24,), dtype=np.float32)/24.0,
            reply_delay_mu=math.log(30.0),
            reply_delay_sigma=0.7,
            question_ratio=0.1,
            agreement_ratio=0.2,
            disagreement_ratio=0.2,
            secondperson_ratio=0.1,
            subjectivity_ratio=0.5,
            caps_ratio=0.02,
            hedge_ratio=0.1,
            intensifier_ratio=0.1,
            topic_affinity=0.0,
            cooldown_min=10.0,
            daily_cap=20,
        )).author_type

        op_node = CommentNode(
            cid=0,
            parent_cid=None,
            depth=-1,
            author_user_id=op_uid,
            author_type=op_type,
            text=thread.title.strip(),
            created_min=thread.created_min,
            path=tuple(),
            children=[],
            mentions=[],
        )
        thread.add_node(op_node)

        while self.queue and len(thread.nodes) - 1 < max_posts:
            ev = heapq.heappop(self.queue)
            now = float(ev.time_min)
            if now > self.start_min + self.horizon:
                break

            for u in self.prof.values():
                if now >= u.next_reset_min:
                    u.posts_today = 0
                    u.next_reset_min += 24.0 * 60.0

            if ev.kind == "avail":
                u = self.prof[ev.uid]
                if (now - u.last_post_min) < u.cooldown_min or (u.posts_today >= u.daily_cap):
                    self._schedule_next_for(ev.uid, now)
                    continue

                p = self.decider.p_post(ev.uid, thread, now)
                if self.rng.random() >= p:
                    self._schedule_next_for(ev.uid, now)
                    continue

                action, target_cid = self.decider.choose_action(ev.uid, thread, now, fanout_caps)

                if action == "top":
                    parent = thread.nodes[0]
                else:
                    if target_cid is None or target_cid not in thread.nodes:
                        self._schedule_next_for(ev.uid, now)
                        continue
                    parent = thread.nodes[target_cid]

                child_depth = parent.depth + 1
                if fanout_caps.get(child_depth, 0) <= len(parent.children):
                    self._schedule_next_for(ev.uid, now)
                    continue

                dt = float(self.rng.lognormal(mean=self.prof[ev.uid].reply_delay_mu, sigma=self.prof[ev.uid].reply_delay_sigma))
                post_time = now + dt

                heapq.heappush(self.queue, Event(time_min=post_time, kind="post", uid=ev.uid, target_cid=parent.cid))
                self._schedule_next_for(ev.uid, now)

            elif ev.kind == "post":
                if ev.target_cid not in thread.nodes:
                    continue
                parent = thread.nodes[ev.target_cid]
                child_depth = parent.depth + 1
                if fanout_caps.get(child_depth, 0) <= len(parent.children):
                    continue

                text, mentions = generator_fn(ev.uid, None if parent.cid == 0 else parent.cid, ev.time_min)
                text = (text or "").strip()
                if not text:
                    continue

                node = CommentNode(
                    cid=next_cid,
                    parent_cid=(None if parent.cid == 0 else parent.cid),
                    depth=child_depth,
                    author_user_id=int(ev.uid),
                    author_type=str(self.prof[ev.uid].author_type),
                    text=text,
                    created_min=ev.time_min,
                    path=(parent.path + (parent.cid,)) if parent.cid != 0 else tuple(),
                    children=[],
                    mentions=[int(m) for m in mentions or []],
                )
                thread.add_node(node)
                next_cid += 1

                u = self.prof[ev.uid]
                u.last_post_min = ev.time_min
                u.posts_today += 1


# ---------------------------------------------------------------------
# Sentiment scoring (create_labels-compatible bins)
# ---------------------------------------------------------------------

def _compute_sentiment_thresholds(
    df: pd.DataFrame,
    col: str = "gstat_user_sent_mean",
    *,
    norm_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    if col not in df.columns:
        raise KeyError(f"Threshold source is missing '{col}'.")

    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
    if s.empty:
        raise RuntimeError(f"No numeric values in threshold source column '{col}'.")

    stats = None
    if isinstance(norm_stats, dict):
        stats = norm_stats.get(col, None)

    if isinstance(stats, dict):
        try:
            mu = float(stats.get("mean", 0.0))
            sd = float(stats.get("std", 1.0))
            if math.isfinite(mu) and math.isfinite(sd) and sd > 0.0:
                s = (s * sd) + mu
        except Exception:
            pass

    qs = s.quantile([0.2, 0.4, 0.6, 0.8, 1.0], interpolation="linear")
    return {
        "q20": float(qs.loc[0.2]),
        "q40": float(qs.loc[0.4]),
        "q60": float(qs.loc[0.6]),
        "q80": float(qs.loc[0.8]),
        "q100": float(qs.loc[1.0]),
    }


def _label_from_thresholds(x: float, th: Dict[str, float]) -> str:
    if x <= th["q20"]:
        return "rage"
    if x <= th["q40"]:
        return "grumpy"
    if x <= th["q60"]:
        return "mellow"
    if x <= th["q80"]:
        return "calm"
    return "empath"


try:
    # Canonical coherence gate. Imported from hypergames_utils (which now
    # owns the C1 patent-keyword regex + repetition / ascii / alpha
    # checks) so the forum builder, the game scripts, and the rest of the
    # eval hub all share one definition. See the 2026-05-14 patent-leak audit.
    from hypergames_utils import _is_coherent  # type: ignore  # noqa: F401
except Exception:
    # Defensive fallback for any caller that lands here before
    # hypergames_utils is on the sys.path. Mirrors the canonical
    # implementation minus the patent regex (which requires re).
    def _is_coherent(text: str, min_alpha_frac: float = 0.40, min_words: int = 3,
                     min_ascii_frac: float = 0.70) -> bool:
        """Fallback coherence gate (no patent regex)."""
        t = (text or "").strip()
        if len(t) < 8:
            return False
        alpha = sum(1 for c in t if c.isalpha())
        if alpha / max(1, len(t)) < min_alpha_frac:
            return False
        words = t.split()
        if len(words) < min_words:
            return False
        ascii_chars = sum(1 for c in t if ord(c) < 128)
        if ascii_chars / max(1, len(t)) < min_ascii_frac:
            return False
        return True


# ---------------------------------------------------------------------
# HyperPEFT-LoRA inference engine
# ---------------------------------------------------------------------

@dataclass
class HypernetSchema:
    global_dim: int
    hidden_dim: int
    dict_mode: bool
    head_mode: str
    dict_k_by_role: Dict[str, int]
    hyper_out_rank: int = 0


class HyperPEFTLoRAEngine:
    def __init__(
        self,
        *,
        base_model: str,
        hyper_dir: str,
        target_modules: List[str],
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        inject_clamp: float,
        delta_gain: float,
        use_best_ckpt: bool,
        online: bool,
        qlora: bool,
        device: torch.device,
        emit_both: bool = False,
    ) -> None:
        if get_peft_model is None or LoraConfig is None:
            raise RuntimeError("peft is required (get_peft_model/LoraConfig not importable).")

        self.base_model = str(base_model)
        self.hyper_dir = Path(hyper_dir)
        self.target_modules = list(target_modules)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.inject_clamp = float(inject_clamp)
        self.delta_gain = float(delta_gain)
        self.use_best_ckpt = bool(use_best_ckpt)
        self.online = bool(online)
        self.qlora = bool(qlora)
        self.device = device
        self.emit_both = bool(emit_both)

        os.environ["HN_DELTA_GAIN"] = str(self.delta_gain)
        os.environ["HN_DISABLE_ABS_CLAMP"] = "0"
        os.environ["HN_DISABLE_GATES"] = "0"
        os.environ["HN_USE_GROUPS"] = "1"

        ckpt_dir = self.hyper_dir / "best" if self.use_best_ckpt else self.hyper_dir
        self.hyper_ckpt = ckpt_dir / "hypernetwork.safetensors"
        self.ph_ckpt = ckpt_dir / "peft_placeholders.safetensors"
        self.ctx_ckpt = ckpt_dir / "ctx_params.safetensors"

        if not self.hyper_ckpt.exists():
            raise FileNotFoundError(str(self.hyper_ckpt))
        if not self.ph_ckpt.exists():
            raise FileNotFoundError(str(self.ph_ckpt))

        self.schema = self._infer_hypernet_schema(self.hyper_ckpt)

        hf_token = _configure_hf_auth(None)

        tok_kwargs = dict(trust_remote_code=True)
        mdl_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)

        if hf_token:
            tok_kwargs["token"] = hf_token
            mdl_kwargs["token"] = hf_token

        if not self.online:
            tok_kwargs["local_files_only"] = True
            mdl_kwargs["local_files_only"] = True

        tok_src = self.base_model
        for cand in (ckpt_dir, self.hyper_dir):
            try:
                if (cand / "tokenizer.json").exists() or (cand / "tokenizer.model").exists() or (cand / "tokenizer_config.json").exists():
                    tok_src = str(cand)
                    break
            except Exception:
                continue

        self.tok = AutoTokenizer.from_pretrained(tok_src, **tok_kwargs)
        self.tok.padding_side = "right"
        LOG.info("Tokenizer loaded | source=%s | vocab=%d", str(tok_src), int(len(self.tok)))
        
        added = False
        if REPLY_SEP_TOKEN not in self.tok.get_vocab():
            self.tok.add_special_tokens({"additional_special_tokens": [REPLY_SEP_TOKEN]})
            added = True
        if REPLY_END_TOKEN not in self.tok.get_vocab():
            self.tok.add_special_tokens({"additional_special_tokens": [REPLY_END_TOKEN]})
            added = True
        if CONTEXT_SEP_TOKEN not in self.tok.get_vocab():
            self.tok.add_special_tokens({"additional_special_tokens": [CONTEXT_SEP_TOKEN]})
            added = True
        if self.tok.pad_token_id is None:
            try:
                if getattr(self.tok, "eos_token", None) is not None:
                    self.tok.pad_token = self.tok.eos_token
                else:
                    self.tok.add_special_tokens({"pad_token": "[PAD]"})
                    self.tok.pad_token_id = self.tok.convert_tokens_to_ids("[PAD]")
                    added = True
            except Exception:
                pass

        quant_cfg = None
        if self.qlora:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as e:
                raise RuntimeError("qlora=True but BitsAndBytesConfig is not available in this environment.") from e

        if quant_cfg is not None:
            if self.device.type != "cuda":
                raise RuntimeError("qlora=True requires CUDA (no GPU detected).")
            mdl_kwargs["quantization_config"] = quant_cfg
            mdl_kwargs["device_map"] = {"": 0}
        else:
            if self.device.type == "cuda":
                # Dtype + attention selection is environment-driven so the same
                # script can target two regimes:
                #   * HN_TORCH_DTYPE=float16 HN_ATTN_IMPL=sdpa   (default)
                #     The original working config for synth-forum generation
                #     with HyperPEFT-LoRA. FP16 + SDPA is what produced every
                #     successful phase-2d run prior to 2026-05-14.
                #   * HN_TORCH_DTYPE=bfloat16 HN_ATTN_IMPL=eager
                #     The workaround for the high-memory GPU cuDNN-frontend crash in
                #     extended-depth dialog (phase 6). BF16 + eager keeps
                #     attention softmax finite where FP16 + eager would
                #     overflow. Pair them together; do NOT mix BF16 + eager
                #     with HyperPEFT-LoRA synth-user delta injection — the
                #     interaction produces inf/nan logits at the lm_head and
                #     crashes torch.multinomial sampling.
                _dtype_name = os.environ.get("HN_TORCH_DTYPE", "float16").lower()
                _dtype_map = {
                    "float16": torch.float16, "fp16": torch.float16, "f16": torch.float16,
                    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                    "float32": torch.float32, "fp32": torch.float32, "f32": torch.float32,
                }
                mdl_kwargs["torch_dtype"] = _dtype_map.get(_dtype_name, torch.float16)

        _attn_impl = os.environ.get("HN_ATTN_IMPL", "sdpa")
        if _attn_impl:
            mdl_kwargs["attn_implementation"] = _attn_impl
        self.backbone = AutoModelForCausalLM.from_pretrained(self.base_model, **mdl_kwargs)

        try:
            old_embed_n = int(self.backbone.get_input_embeddings().weight.size(0))
        except Exception:
            old_embed_n = 0

        if old_embed_n > 0 and len(self.tok) != old_embed_n:
            added = True

        if added:
            self.backbone.resize_token_embeddings(len(self.tok))

        if added:
            try:
                src_id = int(self.tok.eos_token_id or 0)
                init_ids: List[int] = []
                for t in (CONTEXT_SEP_TOKEN, REPLY_SEP_TOKEN, REPLY_END_TOKEN):
                    tid = int(self.tok.convert_tokens_to_ids(t))
                    if tid >= 0:
                        init_ids.append(tid)
                if self.tok.pad_token_id is not None:
                    pid = int(self.tok.pad_token_id)
                    if pid >= 0:
                        init_ids.append(pid)
                _init_token_rows_from_id(self.backbone, sorted(set(init_ids)), src_id=src_id)
            except Exception:
                pass

        self.backbone.config.use_cache = False

        peft_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.peft_model = get_peft_model(self.backbone, peft_cfg)

        ctx_sd: Optional[Dict[str, torch.Tensor]] = None
        use_layer_ctx = False
        ctx_in_dim = int(self.schema.global_dim)
        ctx_embed_dim = 32

        if self.ctx_ckpt.exists():
            try:
                ctx_sd = _load_safetensors(self.ctx_ckpt)
                if "ctx_proj.weight" in ctx_sd and hasattr(ctx_sd["ctx_proj.weight"], "shape"):
                    ctx_embed_dim = int(ctx_sd["ctx_proj.weight"].shape[0])
                    ctx_in_dim = int(ctx_sd["ctx_proj.weight"].shape[1])
                    use_layer_ctx = True
            except Exception:
                ctx_sd = None
                use_layer_ctx = False

        if not use_layer_ctx:
            os.environ["HN_DISABLE_GATES"] = "1"

        dict_k_global = int(max(self.schema.dict_k_by_role.values())) if self.schema.dict_k_by_role else 0

        # Compute fan_in group scales to match training (critical for correct delta magnitude).
        # Prefer saved group_scales.json from training checkpoint if available.
        group_scales: Dict[str, float] = {}
        gs_path = ckpt_dir / "group_scales.json"
        if gs_path.exists():
            try:
                with open(gs_path, "r", encoding="utf-8") as fp:
                    group_scales = json.load(fp)
                LOG.info("Loaded group_scales from %s: %s", gs_path, group_scales)
            except Exception:
                group_scales = {}
        if not group_scales:
            group_scales = compute_group_scales_from_peft(self.peft_model)
            if group_scales:
                LOG.info("Computed group_scales from PEFT model: %s", group_scales)
            else:
                LOG.warning("Could not compute group_scales; deltas may be incorrectly scaled!")

        # Auto-detect emit_both by comparing the checkpoint's per-role total
        # output surface against the B-only and A+B surfaces implied by the
        # peft_model's placeholder plan. The model was trained with --emit_both, so
        # the checkpoint carries δA and δB in one concatenated output head.
        emit_both_detected = bool(self.emit_both)
        try:
            hyper_sd_probe = _load_safetensors(self.hyper_ckpt)
            from collections import defaultdict as _dd
            role_sum_B = _dd(int)
            role_sum_A = _dd(int)
            _, slice_sizes_probe, roles_probe, _shapesB_probe, _shapesA_probe = _hn_struct._build_placeholder_plan(self.peft_model)
            for sz_b, (in_f, r_a), rl in zip(slice_sizes_probe, _shapesA_probe, roles_probe):
                role_sum_B[rl] += int(sz_b)
                role_sum_A[rl] += int(in_f) * int(r_a)
            ckpt_role_total: Dict[str, int] = {}
            for k, t in hyper_sd_probe.items():
                if not k.startswith("out_head.heads.") or not k.endswith(".table"):
                    continue
                if not hasattr(t, "shape") or len(t.shape) < 1:
                    continue
                role = k.split(".")[2]
                ckpt_role_total[role] = int(t.shape[0])
            if ckpt_role_total and role_sum_B:
                matches_both = 0
                matches_b_only = 0
                for rl, total in ckpt_role_total.items():
                    exp_b = int(role_sum_B.get(rl, 0))
                    exp_both = exp_b + int(role_sum_A.get(rl, 0))
                    if total == exp_both and exp_both != exp_b:
                        matches_both += 1
                    elif total == exp_b:
                        matches_b_only += 1
                if matches_both > 0 and matches_both >= matches_b_only:
                    if not emit_both_detected:
                        LOG.info("Auto-detected emit_both=True from checkpoint table shapes (per-role δA+δB).")
                    emit_both_detected = True
        except Exception as _e:
            LOG.warning("emit_both auto-detect failed (%s); using flag value %s", _e, str(self.emit_both))

        self.model = make_from_dims(
            self.peft_model,
            global_dim=int(self.schema.global_dim),
            hidden_dim=int(self.schema.hidden_dim),
            inject_clamp=float(self.inject_clamp),
            dropout_p=float(self.lora_dropout),
            activation="silu",
            hyper_out_rank=int(self.schema.hyper_out_rank),
            head_mode=str(self.schema.head_mode),
            dict_mode=bool(self.schema.dict_mode),
            dict_k_global=int(dict_k_global),
            dict_k_by_role=self.schema.dict_k_by_role,
            global_columns=None,
            group_scales=group_scales,
            use_layer_context=bool(use_layer_ctx),
            ctx_in_dim=int(ctx_in_dim),
            ctx_embed_dim=int(ctx_embed_dim),
            ctx_init_scale=0.05,
            emit_both=bool(emit_both_detected),
        )
        self.emit_both = bool(emit_both_detected)

        self.hypernet = self.model.hypernet

        if quant_cfg is None:
            self.model.to(self.device)
        else:
            try:
                self.model.hypernet.to(self.device)
                if hasattr(self.model, "ctx_proj"):
                    self.model.ctx_proj.to(self.device)
                if hasattr(self.model, "layer_emb"):
                    self.model.layer_emb.data = self.model.layer_emb.data.to(self.device)
            except Exception:
                pass

        h_sd = _load_safetensors(self.hyper_ckpt)

        # Remap spectral-norm parametrization keys to plain weight keys.
        # Training may apply spectral norm (parametrizations.weight.original);
        # at inference we load the unnormalized weight directly into proj.weight.
        remapped_sd: Dict[str, Any] = {}
        for k, v in h_sd.items():
            if ".parametrizations.weight.original" in k:
                plain_key = k.replace(".parametrizations.weight.original", ".weight")
                remapped_sd[plain_key] = v
            elif ".parametrizations.weight.0." in k:
                # Skip spectral norm _u/_v vectors — not needed at inference
                continue
            else:
                remapped_sd[k] = v

        incompat = self.model.hypernet.load_state_dict(remapped_sd, strict=False)
        missing = list(incompat.missing_keys) if hasattr(incompat, "missing_keys") else []
        unexpected = list(incompat.unexpected_keys) if hasattr(incompat, "unexpected_keys") else []
        if missing or unexpected:
            raise RuntimeError(f"Hypernet state load mismatch. missing={missing} unexpected={unexpected}")

        ph = _load_safetensors(self.ph_ckpt)
        self._load_peft_placeholders(ph)

        if ctx_sd is not None and use_layer_ctx:
            try:
                if hasattr(self.model, "ctx_proj") and "ctx_proj.weight" in ctx_sd:
                    self.model.ctx_proj.weight.data.copy_(ctx_sd["ctx_proj.weight"].to(self.model.ctx_proj.weight.device))
                if hasattr(self.model, "layer_emb") and "layer_emb" in ctx_sd:
                    self.model.layer_emb.data.copy_(ctx_sd["layer_emb"].to(self.model.layer_emb.device))
            except Exception:
                os.environ["HN_DISABLE_GATES"] = "1"
        else:
            os.environ["HN_DISABLE_GATES"] = "1"

        self.model.eval()

        self.sep_id = self.tok.convert_tokens_to_ids(REPLY_SEP_TOKEN)
        self.end_id = self.tok.convert_tokens_to_ids(REPLY_END_TOKEN)
        self.ctx_id = self.tok.convert_tokens_to_ids(CONTEXT_SEP_TOKEN)
        self.pad_id = self.tok.pad_token_id

        # C3 patent / boilerplate suppression: phrase list lives in
        # hypergames_utils.get_patent_suppression_ids so every eval pipeline
        # (forum builder, game scripts, inference engine) shares one source
        # of truth. See feedback_paper2_data_paths.md + the 2026-05-14
        # patent-leak audit.
        try:
            from hypergames_utils import get_patent_suppression_ids  # type: ignore
            self.bad_words_ids: List[List[int]] = get_patent_suppression_ids(self.tok)
        except Exception:
            self.bad_words_ids = []
                            
    def _infer_hypernet_schema(self, hyper_ckpt: Path) -> HypernetSchema:
        sd = _load_safetensors(hyper_ckpt)

        w = None
        for key in (
            "net.0.weight",
            "net.0.parametrizations.weight.original",
            "net_pre.0.weight",
            "net_pre.0.parametrizations.weight.original",
        ):
            if key in sd:
                w = sd[key]
                break

        if w is None:
            for k in sd.keys():
                if k.endswith("net.0.weight") or k.endswith("net_pre.0.weight"):
                    w = sd[k]
                    break
                if k.endswith("net.0.parametrizations.weight.original") or k.endswith("net_pre.0.parametrizations.weight.original"):
                    w = sd[k]
                    break

        if w is None or not hasattr(w, "shape") or len(w.shape) < 2:
            keys = list(sd.keys())
            keys.sort()
            preview = ", ".join(keys[:32])
            raise RuntimeError(
                f"Cannot infer schema: missing net.0/net_pre.0 weight tensor in {str(hyper_ckpt)}; keys[:32]={preview}"
            )

        hidden_dim = int(w.shape[0])
        global_dim = int(w.shape[1])

        dict_mode = any(k.startswith("out_head.dict_tables.") for k in sd.keys())

        head_mode = "single"
        if any(k.startswith("out_head.alpha_heads.") for k in sd.keys()):
            head_mode = "multi"
        elif any(k.startswith("out_head.split_heads.") for k in sd.keys()):
            head_mode = "multi"
        elif any(k.startswith("out_head.heads.") for k in sd.keys()):
            head_mode = "multi"
        elif any(k.startswith("out_head.head.") for k in sd.keys()):
            head_mode = "single"

        dict_k_by_role: Dict[str, int] = {}
        if dict_mode:
            for k, t in sd.items():
                if not k.startswith("out_head.dict_tables."):
                    continue
                parts = k.split(".")
                if len(parts) < 3:
                    continue
                role = parts[2]
                if not hasattr(t, "shape"):
                    continue
                if len(t.shape) == 2:
                    dk = int(min(int(t.shape[0]), int(t.shape[1])))
                    dict_k_by_role[role] = max(dict_k_by_role.get(role, 0), dk)
                elif len(t.shape) == 1:
                    dk = int(t.shape[0])
                    dict_k_by_role[role] = max(dict_k_by_role.get(role, 0), dk)

            if not dict_k_by_role:
                keys = list(sd.keys())
                keys.sort()
                preview = ", ".join(keys[:32])
                raise RuntimeError(
                    f"dict_mode=True but could not infer dict_k_by_role from out_head.dict_tables.* in {str(hyper_ckpt)}; keys[:32]={preview}"
                )

        hyper_out_rank = 0
        for k, t in sd.items():
            if not hasattr(t, "shape"):
                continue
            if k.endswith(".proj1.weight") and k.startswith("out_head."):
                if len(t.shape) == 2:
                    hyper_out_rank = max(hyper_out_rank, int(t.shape[0]))

        return HypernetSchema(
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            dict_mode=dict_mode,
            head_mode=head_mode,
            dict_k_by_role=dict_k_by_role,
            hyper_out_rank=hyper_out_rank,
        )
        
    def _infer_placeholder_plan(self, peft_model: nn.Module, target_modules: List[str]) -> Tuple[List[int], Dict[str, int]]:
        placeholder_sizes: List[int] = []
        role_sizes: Dict[str, int] = {"qkv": 0, "o_proj": 0, "mlp": 0, "other": 0}

        pat = re.compile(r"\.lora_B(?:\.default)?\.(weight|bias)$")
        for name, p in peft_model.named_parameters():
            if not pat.search(name):
                continue

            n = int(p.numel())
            placeholder_sizes.append(n)

            parts = name.split(".")
            leaf = ""
            try:
                i = len(parts) - 1 - list(reversed(parts)).index("lora_B")
                if i - 1 >= 0:
                    leaf = parts[i - 1]
            except Exception:
                leaf = ""

            if leaf in ("q_proj", "k_proj", "v_proj", "query_key_value"):
                role_sizes["qkv"] += n
            elif leaf in ("o_proj", "out_proj", "dense"):
                role_sizes["o_proj"] += n
            elif leaf in ("gate_proj", "up_proj", "down_proj",
                          "dense_h_to_4h", "dense_4h_to_h"):
                role_sizes["mlp"] += n
            else:
                role_sizes["other"] += n

        if not placeholder_sizes:
            raise RuntimeError("No LoRA-B parameters found; check target_modules and PEFT setup.")
        return placeholder_sizes, role_sizes
    
    def _compute_group_scales(self, peft_model: nn.Module, target_modules: List[str], group_splits: List[int]) -> List[float]:
        B_params: List[torch.Tensor] = []
        pat = re.compile(r"\.lora_B\.default\.(weight|bias)$")
        for name, p in peft_model.named_parameters():
            if not pat.search(name):
                continue
            B_params.append(p.detach())

        flat = torch.cat([t.reshape(-1).float().cpu() for t in B_params], dim=0)
        if flat.numel() != sum(group_splits):
            return [1.0 for _ in group_splits]

        scales: List[float] = []
        off = 0
        for sz in group_splits:
            seg = flat[off:off + sz]
            off += sz
            if seg.numel() == 0:
                scales.append(1.0)
                continue
            v = float(torch.mean(torch.abs(seg)).item())
            scales.append(1.0 / max(1e-6, v))
        return scales

    def _load_peft_placeholders(self, ph: Dict[str, torch.Tensor]) -> None:
        if not hasattr(self.model, "_placeholders"):
            raise RuntimeError("PEFTHypernetModel is missing placeholder mapping; cannot load peft_placeholders.")

        placeholders = getattr(self.model, "_placeholders")
        missing_B: List[str] = []
        bad_shapes: List[str] = []

        with torch.no_grad():
            for name, pair in placeholders.items():
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    continue
                lora_A, lora_B = pair

                a_key = None
                for k in (f"{name}.lora_A.default.weight", f"{name}.lora_A.weight"):
                    if k in ph:
                        a_key = k
                        break
                if a_key is not None and hasattr(lora_A, "weight"):
                    w = ph[a_key].to(device=lora_A.weight.device, dtype=lora_A.weight.dtype)
                    if tuple(w.shape) != tuple(lora_A.weight.shape):
                        bad_shapes.append(f"{a_key}: ckpt={tuple(w.shape)} model={tuple(lora_A.weight.shape)}")
                    else:
                        lora_A.weight.data.copy_(w)

                b_key = None
                for k in (f"{name}.lora_B.default.weight", f"{name}.lora_B.weight"):
                    if k in ph:
                        b_key = k
                        break
                if b_key is None:
                    missing_B.append(str(name))
                    continue

                if hasattr(lora_B, "weight"):
                    w = ph[b_key].to(device=lora_B.weight.device, dtype=lora_B.weight.dtype)
                    if tuple(w.shape) != tuple(lora_B.weight.shape):
                        bad_shapes.append(f"{b_key}: ckpt={tuple(w.shape)} model={tuple(lora_B.weight.shape)}")
                    else:
                        lora_B.weight.data.copy_(w)

        if bad_shapes:
            preview = "\n".join(bad_shapes[:12])
            raise RuntimeError(f"peft_placeholders contains tensor(s) with shape mismatches:\n{preview}")

        if missing_B:
            preview = ", ".join(missing_B[:12])
            raise RuntimeError(
                f"peft_placeholders is missing LoRA-B weights for {len(missing_B)} modules. Example module paths: {preview}"
            )

    def _emit_delta_parts(self, g: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            out = self.model.hypernet(g)
            if isinstance(out, dict):
                parts = list(out.values())
            else:
                parts = [out]
            return [p.to(self.device) for p in parts]

# -----------------------------------------------------------------
# Sampling helpers for custom generation loop (CFG / temp annealing)
# -----------------------------------------------------------------

def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
) -> torch.Tensor:
    """Sample a single token from logits [1, vocab]. Returns [1,1]."""
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = (cum_probs - torch.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits[mask] = float("-inf")
        logits = logits.scatter(1, sorted_idx, sorted_logits)
    if do_sample:
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    else:
        return logits.argmax(dim=-1, keepdim=True)


def _apply_repetition_penalty_inplace(
    logits: torch.Tensor,
    past_ids: List[int],
    penalty: float,
) -> None:
    """Apply multiplicative repetition penalty in-place. Single-sample (batch=1) version."""
    if penalty == 1.0 or not past_ids:
        return
    unique_ids = list(set(past_ids))
    gathered = logits[0, unique_ids].clone()
    gathered = torch.where(gathered > 0, gathered / penalty, gathered * penalty)
    logits[0, unique_ids] = gathered


def _apply_repetition_penalty_inplace_batch(
    logits: torch.Tensor,
    past_ids_per_sample: List[List[int]],
    penalty: float,
) -> None:
    """Batched repetition penalty. logits: [B, V]. past_ids_per_sample: list of B lists."""
    if penalty == 1.0:
        return
    for b, past_ids in enumerate(past_ids_per_sample):
        if not past_ids:
            continue
        unique_ids = list(set(past_ids))
        row = logits[b, unique_ids].clone()
        row = torch.where(row > 0, row / penalty, row * penalty)
        logits[b, unique_ids] = row


def _block_repeated_ngrams(
    logits: torch.Tensor,
    past_ids: List[int],
    ngram_size: int,
) -> None:
    """Single-sample no-repeat-ngram block. logits[0, blocked_id] = -inf."""
    if ngram_size <= 0 or len(past_ids) < ngram_size:
        return
    prefix = tuple(past_ids[-(ngram_size - 1):])
    blocked: set = set()
    for i in range(len(past_ids) - ngram_size + 1):
        ng = tuple(past_ids[i:i + ngram_size])
        if ng[:-1] == prefix:
            blocked.add(ng[-1])
    if blocked:
        logits[0, list(blocked)] = float("-inf")


def _block_repeated_ngrams_batch(
    logits: torch.Tensor,
    past_ids_per_sample: List[List[int]],
    ngram_size: int,
) -> None:
    """Batched no-repeat-ngram block. logits: [B, V]."""
    if ngram_size <= 0:
        return
    for b, past_ids in enumerate(past_ids_per_sample):
        if len(past_ids) < ngram_size:
            continue
        prefix = tuple(past_ids[-(ngram_size - 1):])
        blocked: set = set()
        for i in range(len(past_ids) - ngram_size + 1):
            ng = tuple(past_ids[i:i + ngram_size])
            if ng[:-1] == prefix:
                blocked.add(ng[-1])
        if blocked:
            logits[b, list(blocked)] = float("-inf")


class _NanInfLogitsGuard(LogitsProcessor):
    """Replace non-finite logits before sampling so a NaN/inf row cannot trip
    torch.multinomial's `probability tensor contains inf, nan or element < 0`
    device-side assert (observed on the high-memory GPU real-user generation path). This
    is a no-op when all logits are already finite, so it does not change
    sampling on healthy rows; it only rescues rows that would otherwise crash
    the whole forum cell."""

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(scores).all():
            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        return scores


def generate_reply(
    self,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    global_features: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Any,
    do_sample: bool,
    top_p: float,
    temperature: float,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    min_new_tokens: int = 0,
    cfg_scale: float = 1.0,
    temp_anneal_tokens: int = 0,
    temp_anneal_start: float = 0.5,
    adaptive_delta_ref_norm: float = 0.0,
    force_zero_delta: bool = False,
) -> torch.Tensor:
    self.model._g_for_forward = global_features
    self.model._force_zero_flag = False

    try:
        # Emit persona deltas. Returns (parts_B, parts_A|None, delta_flat).
        try:
            _emitted = self.model._emit_delta_parts(global_features, force_zero=False)
        except TypeError:
            _emitted = self.model._emit_delta_parts(global_features, False)
        if isinstance(_emitted, tuple) and len(_emitted) == 3:
            delta_parts, delta_A_parts, _ = _emitted
        else:
            delta_parts, _ = _emitted
            delta_A_parts = None

        # Zero-delta ablation: replace all deltas with zeros
        if force_zero_delta:
            delta_parts = [torch.zeros_like(d) for d in delta_parts]
            if delta_A_parts is not None:
                delta_A_parts = [torch.zeros_like(d) for d in delta_A_parts]

        # Adaptive delta scaling: normalise by feature-vector L2 norm
        if adaptive_delta_ref_norm > 0:
            g_norm = float(torch.linalg.norm(global_features).clamp(min=0.1))
            adscale = adaptive_delta_ref_norm / g_norm
            clamp_val = float(self.inject_clamp)
            delta_parts = [d.mul(adscale).clamp_(-clamp_val, clamp_val) for d in delta_parts]
            if delta_A_parts is not None:
                delta_A_parts = [d.mul(adscale).clamp_(-clamp_val, clamp_val) for d in delta_A_parts]

        self.model._delta_for_forward = delta_parts
        self.model._delta_A_for_forward = delta_A_parts

        # Parse eos_token_id into a set
        eos_set: set = set()
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple, set)):
                for x in eos_token_id:
                    try:
                        eos_set.add(int(x))
                    except Exception:
                        pass
            else:
                try:
                    eos_set.add(int(eos_token_id))
                except Exception:
                    pass
        eos_set = {x for x in eos_set if x >= 0}

        use_cfg = cfg_scale > 1.0 + 1e-6
        use_anneal = temp_anneal_tokens > 0 and abs(temp_anneal_start - temperature) > 0.01

        if use_cfg or use_anneal:
            # Custom autoregressive loop (CFG + temperature annealing)
            out = _generate_custom_loop(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                delta_parts=delta_parts,
                max_new_tokens=int(max_new_tokens),
                eos_set=eos_set,
                do_sample=bool(do_sample),
                top_p=float(top_p),
                temperature=float(temperature),
                top_k=int(top_k or 0),
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                min_new_tokens=int(min_new_tokens),
                cfg_scale=float(cfg_scale),
                temp_anneal_tokens=int(temp_anneal_tokens),
                temp_anneal_start=float(temp_anneal_start),
            )
            return out
        else:
            # Standard HF generate with KV cache enabled
            try:
                gen_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=bool(do_sample),
                    top_p=float(top_p),
                    temperature=float(temperature),
                    logits_processor=LogitsProcessorList([_NanInfLogitsGuard()]),
                )

                if self.pad_id is not None:
                    gen_kwargs["pad_token_id"] = int(self.pad_id)

                if getattr(self, "bad_words_ids", None):
                    if self.bad_words_ids:
                        gen_kwargs["bad_words_ids"] = self.bad_words_ids

                eos_ids_list = sorted(eos_set)
                if eos_ids_list:
                    gen_kwargs["eos_token_id"] = eos_ids_list[0] if len(eos_ids_list) == 1 else eos_ids_list

                if do_sample and top_k is not None and int(top_k) > 0:
                    gen_kwargs["top_k"] = int(top_k)

                rp = float(repetition_penalty)
                if math.isfinite(rp) and rp != 1.0:
                    gen_kwargs["repetition_penalty"] = float(rp)

                ngrams = int(no_repeat_ngram_size)
                if ngrams > 0:
                    gen_kwargs["no_repeat_ngram_size"] = int(ngrams)

                mn = int(min_new_tokens)
                if mn > 0:
                    gen_kwargs["min_new_tokens"] = int(min(mn, int(max_new_tokens)))

                # Enable KV cache for faster generation
                self.model.backbone.config.use_cache = True
                out = None
                for _ in range(6):
                    try:
                        out = self.model.backbone.generate(**gen_kwargs)
                        break
                    except TypeError as e:
                        msg = str(e)
                        m = re.search(r"got an unexpected keyword argument '([^']+)'", msg)
                        if not m:
                            raise
                        bad = m.group(1)
                        if bad not in gen_kwargs:
                            raise
                        gen_kwargs.pop(bad, None)
                if out is None:
                    out = self.model.backbone.generate(**gen_kwargs)
            finally:
                self.model.backbone.config.use_cache = False
        return out
    finally:
        self.model._g_for_forward = None
        self.model._delta_for_forward = None
        if hasattr(self.model, "_delta_A_for_forward"):
            self.model._delta_A_for_forward = None
        self.model._force_zero_flag = False


def _generate_custom_loop(
    self,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    delta_parts: list,
    max_new_tokens: int,
    eos_set: set,
    do_sample: bool,
    top_p: float,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens: int,
    cfg_scale: float,
    temp_anneal_tokens: int,
    temp_anneal_start: float,
) -> torch.Tensor:
    """Custom autoregressive loop supporting CFG and temperature annealing.

    When cfg_scale > 1.0, performs two forward passes per step: one with
    persona deltas and one with zero deltas. Logits are combined as:
        logits = logits_base + cfg_scale * (logits_persona - logits_base)
    Each path maintains its own KV cache.
    """
    use_cfg = cfg_scale > 1.0 + 1e-6
    B = input_ids.size(0)

    # Batch-aware: track per-sample id list
    all_ids_per_sample: List[List[int]] = [input_ids[b].tolist() for b in range(B)]
    finished = [False] * B
    pad_id = 0
    if eos_set:
        pad_id = next(iter(eos_set))

    zero_parts = [torch.zeros_like(d) for d in delta_parts] if use_cfg else None
    delta_A_parts = getattr(self.model, "_delta_A_for_forward", None)
    zero_A_parts = ([torch.zeros_like(d) for d in delta_A_parts]
                     if (use_cfg and delta_A_parts is not None) else None)

    self.model.backbone.config.use_cache = True

    past_persona = None
    past_base = None
    cur_ids = input_ids          # [B, seq_len] on first step
    cur_mask = attention_mask    # [B, seq_len]

    try:
        for step in range(max_new_tokens):
            # --- Persona forward ---
            self.model._delta_for_forward = delta_parts
            if delta_A_parts is not None:
                self.model._delta_A_for_forward = delta_A_parts
            with torch.no_grad():
                out_p = self.model.backbone(
                    input_ids=cur_ids,
                    attention_mask=cur_mask,
                    past_key_values=past_persona,
                    use_cache=True,
                )
            logits_p = out_p.logits[:, -1, :]          # [B, vocab]
            past_persona = out_p.past_key_values

            if use_cfg:
                # --- Base (zero-delta) forward ---
                self.model._delta_for_forward = zero_parts
                if zero_A_parts is not None:
                    self.model._delta_A_for_forward = zero_A_parts
                with torch.no_grad():
                    out_z = self.model.backbone(
                        input_ids=cur_ids,
                        attention_mask=cur_mask,
                        past_key_values=past_base,
                        use_cache=True,
                    )
                logits_z = out_z.logits[:, -1, :]      # [B, vocab]
                past_base = out_z.past_key_values
                # CFG combination — works batched
                logits = logits_z + cfg_scale * (logits_p - logits_z)
            else:
                logits = logits_p

            # --- Temperature (with optional annealing) ---
            if temp_anneal_tokens > 0 and step < temp_anneal_tokens:
                t = temp_anneal_start
            else:
                t = temperature

            # --- Repetition penalty (batched) ---
            if B == 1:
                _apply_repetition_penalty_inplace(logits, all_ids_per_sample[0], repetition_penalty)
                _block_repeated_ngrams(logits, all_ids_per_sample[0], no_repeat_ngram_size)
            else:
                _apply_repetition_penalty_inplace_batch(logits, all_ids_per_sample, repetition_penalty)
                _block_repeated_ngrams_batch(logits, all_ids_per_sample, no_repeat_ngram_size)

            # --- Suppress EOS before min_new_tokens ---
            if step < min_new_tokens and eos_set:
                for eid in eos_set:
                    logits[:, eid] = float("-inf")

            # --- Sample (batched) ---
            next_token = _sample_next_token(logits, t, top_p, top_k, do_sample)  # [B, 1]

            # --- Track per-sample state and force finished samples to PAD ---
            for b in range(B):
                if finished[b]:
                    next_token[b, 0] = pad_id
                else:
                    nid = int(next_token[b, 0])
                    all_ids_per_sample[b].append(nid)
                    if nid in eos_set and step >= min_new_tokens:
                        finished[b] = True

            # Stop when ALL samples have finished
            if all(finished):
                break

            # Prepare next step
            cur_ids = next_token.to(input_ids.device)
            cur_mask = torch.cat([
                cur_mask,
                torch.ones(B, 1, dtype=cur_mask.dtype, device=cur_mask.device),
            ], dim=1)
    finally:
        self.model.backbone.config.use_cache = False
        self.model._delta_for_forward = delta_parts
        if delta_A_parts is not None:
            self.model._delta_A_for_forward = delta_A_parts

    # Pad per-sample sequences to common length so we can return a single tensor
    max_len = max(len(s) for s in all_ids_per_sample)
    out = torch.full((B, max_len), pad_id, dtype=torch.long, device=input_ids.device)
    for b in range(B):
        out[b, :len(all_ids_per_sample[b])] = torch.tensor(all_ids_per_sample[b], dtype=torch.long, device=input_ids.device)
    return out


def _compute_reply_perplexity(
    self,
    full_ids: torch.Tensor,
    prompt_len: int,
    delta_parts: list,
) -> float:
    """Compute perplexity of generated tokens given prompt + persona deltas."""
    gen_len = full_ids.shape[1] - prompt_len
    if gen_len <= 0:
        return float("inf")
    self.model._delta_for_forward = delta_parts
    self.model.backbone.config.use_cache = False
    try:
        with torch.no_grad():
            out = self.model.backbone(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
            )
        # Predict token i+1 from position i
        shift_logits = out.logits[:, prompt_len - 1:-1, :]
        shift_labels = full_ids[:, prompt_len:]
        nll = torch.nn.functional.cross_entropy(
            shift_logits.squeeze(0), shift_labels.squeeze(0), reduction="mean",
        )
        return float(nll.exp())
    except Exception:
        return float("inf")
    finally:
        self.model.backbone.config.use_cache = False


# ---------------------------------------------------------------------
# Branch-only context packing (recency weighting) + generation
# ---------------------------------------------------------------------

class BranchGenerator:
    def __init__(
        self,
        engine: HyperPEFTLoRAEngine,
        profiles: Dict[int, UserProfile],
        thread: ThreadState,
        *,
        max_len: int = 512,
        gamma_recency: float = 1.25,
        max_new_tokens: int = 64,
        do_sample: bool = True,
        top_p: float = 0.90,
        temperature: float = 0.70,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        min_new_tokens: int = 0,
        use_context_token: bool = False,
        cfg_scale: float = 1.0,
        best_of_n: int = 1,
        temp_anneal_tokens: int = 0,
        temp_anneal_start: float = 0.5,
        adaptive_delta_ref_norm: float = 0.0,
        force_zero_delta: bool = False,
        device: torch.device,
        stabilizer_uids: Optional[set] = None,
        bon_metric: str = "perplexity",
        cohort_centroids: Optional[Dict[str, np.ndarray]] = None,
        args: Optional[argparse.Namespace] = None,
    ) -> None:
        self.engine = engine
        self.tok = engine.tok
        self.prof = profiles
        self.thread = thread
        self.max_len = int(max_len)
        self.gamma = float(gamma_recency)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.top_k = int(top_k) if (top_k is not None and int(top_k) > 0) else None
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.min_new_tokens = int(min_new_tokens)
        self.cfg_scale = float(cfg_scale)
        self.best_of_n = max(1, int(best_of_n))
        self.temp_anneal_tokens = int(temp_anneal_tokens)
        self.temp_anneal_start = float(temp_anneal_start)
        self.adaptive_delta_ref_norm = float(adaptive_delta_ref_norm)
        self.force_zero_delta = bool(force_zero_delta)
        self.device = device
        self.sep_id = engine.sep_id
        self.end_id = engine.end_id
        self.ctx_id = getattr(engine, "ctx_id", None) if bool(use_context_token) else None
        # RQ3 stabilizer agents (calming users with zero delta for the run)
        self.stabilizer_uids = set(int(u) for u in (stabilizer_uids or set()))
        # Best-of-N selection metric (post-2026-05-04 cohort-centroid fix)
        self.bon_metric = str(bon_metric or "perplexity")
        self.cohort_centroids = cohort_centroids
        self.args = args

    def _segment_tokens(self, text: str) -> List[int]:
        enc = self.tok(text, add_special_tokens=False, return_tensors=None)
        return enc["input_ids"]

    def _encode_branch(self, parent_cid: Optional[int]) -> Tuple[List[int], List[int]]:
        segments: List[str] = []

        t_topic = self.thread.topic.strip() if hasattr(self.thread, "topic") else ""
        if t_topic:
            segments.append(t_topic)

        segments.append(self.thread.title.strip())
        if parent_cid is not None and parent_cid in self.thread.nodes:
            path_ids = []
            cur = self.thread.nodes[parent_cid]
            while cur is not None and cur.cid != 0:
                path_ids.append(cur.cid)
                if cur.parent_cid is None:
                    break
                cur = self.thread.nodes.get(cur.parent_cid, None)
            for cid in reversed(path_ids):
                segments.append(self.thread.nodes[cid].text)

        budget = max(16, self.max_len - 128)
        k = len(segments)
        weights = np.array([((i + 1) ** self.gamma) for i in range(k)], dtype=np.float32)
        weights = weights / (weights.sum() + 1e-6)
        alloc = [max(8, int(round(budget * float(w)))) for w in weights]

        body: List[int] = []
        newline = self._segment_tokens("\n")

        for i, (seg, n_tok) in enumerate(zip(segments, alloc)):
            ids = self._segment_tokens(seg)
            if len(ids) > n_tok:
                ids = ids[-n_tok:]
            body.extend(ids)
            if newline and i + 1 < len(segments):
                body.extend(newline)

        prefix: List[int] = []
        try:
            bos = getattr(self.tok, "bos_token_id", None)
            if bos is not None and int(bos) >= 0:
                prefix.append(int(bos))
        except Exception:
            pass

        # Prompt format (post-2026-05-03 EOS-aliasing fix):
        #   [BOS] {topic}\n{title}\n{reply_chain}\n\nReply:_
        # Pure text — no special tokens at the prompt-target boundary.
        #
        # Why no <|context|> prefix and no trailing <|reply|> token:
        #   The three special tokens added at training (<|context|>,
        #   <|reply|>, <|eoreply|>) were initialized from the EOS embedding
        #   in _init_token_rows_from_id and the backbone embedding matrix
        #   was frozen during hypernet training (only the hypernet params
        #   are updated).  Cosine similarity of all three to <|endoftext|>
        #   is 0.9995, so they remain functionally identical to EOS at
        #   inference.  When the prompt ends on <|reply|> (sep_id), Pythia
        #   reads the prompt as "document complete" and continues from
        #   the fresh-document Pile distribution: Wikipedia village
        #   stubs, OLED patent abstracts, Stack Overflow code, J.J. Watt
        #   profiles, Q&A-with-Matt-Groening teasers — i.e. the failure
        #   mode reported on M1 Phase 2b/2c, 2026-05-03.
        #
        # Why also no few-shot prefix:
        #   A previous attempt anchored the model with an 80-token
        #   "Topic:/Post:/Reply:" few-shot prefix to escape the EOS-doc-
        #   end attractor.  In the diag_04 sweep (144 generations,
        #   gu03 compute node, 2026-05-03), the few-shot prompt landed
        #   at 46% coherence and 12% on-topic vs the natural-text prompt
        #   below at 75-98% coherence and 34-98% on-topic.  The few-shot
        #   was net-negative once the EOS-token bug was removed.
        #
        # The literal "\n\nReply: " text (with trailing space) is a soft
        # natural-language anchor that vanilla LoRA also uses.  It does
        # NOT alias EOS.
        reply_marker = self._segment_tokens("\n\nReply: ")
        overhead = len(prefix) + len(reply_marker)
        max_body = int(max(0, self.max_len - overhead))
        if max_body > 0 and len(body) > max_body:
            body = body[-max_body:]
        elif max_body <= 0:
            body = []

        toks = prefix + body
        if reply_marker:
            toks.extend(reply_marker)
        attn = [1] * len(toks)
        return toks, attn
    
    def __call__(self, uid: int, parent_cid: Optional[int]) -> List[int]:
        toks, attn = self._encode_branch(parent_cid)
        input_ids = torch.tensor([toks], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attn], dtype=torch.long, device=self.device)

        g = torch.tensor([self.prof[uid].gvec], dtype=torch.float32, device=self.device)

        eos_token = None
        eos_ids: List[int] = []
        try:
            if self.end_id is not None:
                eos_ids.append(int(self.end_id))
        except Exception:
            eos_ids = []
        try:
            if getattr(self.tok, "eos_token_id", None) is not None:
                eos_ids.append(int(self.tok.eos_token_id))
        except Exception:
            pass
        eos_ids = sorted(set([int(i) for i in eos_ids if i is not None and int(i) >= 0]))
        if eos_ids:
            eos_token = eos_ids[0] if len(eos_ids) == 1 else eos_ids

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_features=g,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=eos_token,
            do_sample=self.do_sample,
            top_p=self.top_p,
            temperature=self.temperature,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            min_new_tokens=self.min_new_tokens,
            cfg_scale=self.cfg_scale,
            temp_anneal_tokens=self.temp_anneal_tokens,
            temp_anneal_start=self.temp_anneal_start,
            adaptive_delta_ref_norm=self.adaptive_delta_ref_norm,
            # Per-call force_zero_delta: True if global flag OR this user is a
            # stabilizer agent (RQ3 lever).
            force_zero_delta=bool(self.force_zero_delta
                                   or int(uid) in self.stabilizer_uids),
        )

        prompt_len = len(toks)

        if self.best_of_n <= 1:
            # Single candidate (fast path)
            try:
                self.engine.model._arditi_batch_user_ids = [int(uid)]
                self.engine.model._arditi_batch_uids_tensor = torch.tensor(
                    [int(uid)], dtype=torch.long, device=self.device,
                )
            except Exception:
                self.engine.model._arditi_batch_user_ids = None
                self.engine.model._arditi_batch_uids_tensor = None
            with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                out_ids = self.engine.generate_reply(**gen_kwargs)
            return out_ids[0].tolist()[prompt_len:]
        else:
            # Best-of-N: generate N candidates, pick by selection metric
            bon_metric = getattr(self, "bon_metric", "perplexity")
            cohort_centroids = getattr(self, "cohort_centroids", None)
            user_cohort = self.prof[uid].author_type if uid in self.prof else None

            candidates: List[List[int]] = []
            scores: List[float] = []  # higher = better (we'll negate ppl below)

            for ci in range(self.best_of_n):
                # Re-seed per candidate to actually get different draws
                try:
                    torch.manual_seed(int(self.args.seed) + 1000 * ci + int(uid) % 1000)
                except Exception:
                    pass
                try:
                    self.engine.model._arditi_batch_user_ids = [int(uid)]
                    self.engine.model._arditi_batch_uids_tensor = torch.tensor(
                        [int(uid)], dtype=torch.long, device=self.device,
                    )
                except Exception:
                    self.engine.model._arditi_batch_user_ids = None
                    self.engine.model._arditi_batch_uids_tensor = None
                with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    out_ids = self.engine.generate_reply(**gen_kwargs)
                full_seq = out_ids[0].tolist()
                gen_tokens = full_seq[prompt_len:]
                candidates.append(gen_tokens)

                if bon_metric == "cohort_centroid":
                    # Decode candidate text and score against user's cohort centroid
                    text = self.tok.decode(gen_tokens, skip_special_tokens=True).strip()
                    if cohort_centroids is None or user_cohort is None or user_cohort not in cohort_centroids:
                        s = 0.0
                    else:
                        from cohort_centroid_bon import cohort_score as _cohort_score
                        s = _cohort_score(text, user_cohort, cohort_centroids)
                    scores.append(s)  # higher = more cohort-aligned
                else:
                    # Legacy perplexity metric (picks lowest PPL — tends neutral)
                    try:
                        delta_parts, _ = self.engine.model._emit_delta_parts(g, force_zero=False)
                    except TypeError:
                        delta_parts, _ = self.engine.model._emit_delta_parts(g, False)
                    ppl = _compute_reply_perplexity(
                        self.engine,
                        out_ids,
                        prompt_len,
                        delta_parts,
                    )
                    scores.append(-float(ppl))  # negate so higher = better, consistent

            # Pick highest-scoring candidate
            best_idx = int(np.argmax(scores))
            return candidates[best_idx]
        
# ---------------------------------------------------------------------
# End-to-end simulator
# ---------------------------------------------------------------------

class ForumSimulator:
    def __init__(self, args: argparse.Namespace) -> None:
        _set_seed(args.seed)
        self.args = args
        self.log = LOG

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(0)

        self.log.info(
            "build_hyperlora_forum starting | device=%s | cuda=%s | seed=%s",
            str(self.device),
            str(torch.cuda.is_available()),
            str(args.seed),
        )

        self.log.info("Loading author parquet: %s", str(args.author_parquet))
        self.gdf = pd.read_parquet(args.author_parquet)
        self.log.info("Author parquet loaded | rows=%d cols=%d", int(len(self.gdf)), int(len(self.gdf.columns)))
        if "target_user_id" not in self.gdf.columns:
            raise KeyError("author_parquet must contain 'target_user_id'.")

        # Optional per-user metadata parquet for synth forums: carries the
        # `label_profile_json` emitted by label_synthetic_personas.py along
        # with per-dim `pol_{dim}` continuous scores.  Absent for the Phase 2b
        # real-user forum; required for every Phase 2d synth forum.
        self.user_metadata_df: Optional[pd.DataFrame] = None
        ump = getattr(args, "user_metadata_parquet", "") or ""
        if ump and Path(ump).exists():
            self.log.info("Loading user metadata parquet: %s", str(ump))
            meta_df = pd.read_parquet(ump)
            if "target_user_id" not in meta_df.columns:
                raise KeyError("user_metadata_parquet must contain 'target_user_id'.")
            meta_df["target_user_id"] = pd.to_numeric(
                meta_df["target_user_id"], errors="coerce"
            ).astype("Int64").dropna().astype(int)
            self.user_metadata_df = meta_df.drop_duplicates("target_user_id").reset_index(drop=True)
            self.log.info(
                "User metadata parquet loaded | rows=%d cols=%d",
                int(len(self.user_metadata_df)), int(len(self.user_metadata_df.columns)),
            )
        elif ump:
            self.log.warning("user_metadata_parquet was provided but not found: %s", str(ump))

        labels_df: Optional[pd.DataFrame] = None
        if args.labels_csv and Path(args.labels_csv).exists():
            self.log.info("Loading labels CSV: %s", str(args.labels_csv))
            labels_df = pd.read_csv(args.labels_csv)
            self.log.info("Labels CSV loaded | rows=%d cols=%d", int(len(labels_df)), int(len(labels_df.columns)))
            if "target_user_id" not in labels_df.columns:
                raise KeyError("labels_csv must contain 'target_user_id'.")
            if "label" in labels_df.columns and "sentiment_label" not in labels_df.columns:
                labels_df = labels_df.rename(columns={"label": "sentiment_label"})
            if "sentiment_label" not in labels_df.columns:
                raise KeyError("labels_csv must contain either 'label' or 'sentiment_label'.")
        else:
            if args.labels_csv:
                self.log.warning("labels_csv was provided but not found: %s", str(args.labels_csv))
            else:
                self.log.info("labels_csv not provided; selecting extremes from gstat_user_sent_mean.")

        self.user_ids, self.author_type = self._select_user_pool(
            self.gdf, labels_df, args.n_rage, args.n_empath, seed=args.seed
        )

        counts = {"rage": 0, "empath": 0, "neutral": 0}
        for _, t in self.author_type.items():
            tt = str(t)
            if tt not in counts:
                counts["neutral"] += 1
            else:
                counts[tt] += 1
        self.log.info(
            "Selected user pool | n=%d | rage=%d | empath=%d | neutral=%d",
            int(len(self.user_ids)),
            int(counts["rage"]),
            int(counts["empath"]),
            int(counts["neutral"]),
        )

        hyper_dir = Path(args.hyper_dir)
        ckpt_dir = (hyper_dir / "best") if args.use_best_ckpt else hyper_dir
        hyper_path = ckpt_dir / "hypernetwork.safetensors"
        if not hyper_path.exists():
            raise FileNotFoundError(str(hyper_path))

        self.log.info("hyper_dir=%s | use_best_ckpt=%s", str(hyper_dir), str(bool(args.use_best_ckpt)))
        self.log.info("Using hypernetwork checkpoint: %s", str(hyper_path))

        schema = HyperPEFTLoRAEngine._infer_hypernet_schema  # type: ignore[attr-defined]
        tmp_schema = schema(HyperPEFTLoRAEngine, hyper_path)  # type: ignore[misc]
        g_dim = int(tmp_schema.global_dim)

        self.log.info(
            "Hypernet schema | gdim=%d hidden=%d dict_mode=%s head_mode=%s dict_k_by_role=%s hyper_out_rank=%d",
            int(tmp_schema.global_dim),
            int(tmp_schema.hidden_dim),
            str(bool(tmp_schema.dict_mode)),
            str(tmp_schema.head_mode),
            json.dumps(tmp_schema.dict_k_by_role, sort_keys=True),
            int(tmp_schema.hyper_out_rank),
        )

        feat_manifest = _try_load_json(hyper_dir / "feature_names.json") or _try_load_json(ckpt_dir / "feature_names.json")
        if isinstance(feat_manifest, dict) and "feature_names" in feat_manifest:
            g_columns = list(feat_manifest["feature_names"])
        elif isinstance(feat_manifest, list):
            g_columns = list(feat_manifest)
        else:
            g_columns = list(args.g_columns) if args.g_columns else [
                "gstat_user_ttr",
                "gstat_user_post_rate",
                "gstat_user_subreddit_entropy",
                "gstat_user_sr_max_share",
                "gstat_punct_ratio",
                "gstat_question_ratio",
            ]

        if len(g_columns) < g_dim:
            g_columns = g_columns + (["__pad__"] * (g_dim - len(g_columns)))
        elif len(g_columns) > g_dim:
            g_columns = g_columns[:g_dim]

        self.log.info("Global feature columns (gdim=%d): %s", int(g_dim), ", ".join([str(c) for c in g_columns]))

        topic_seed = None
        rng = np.random.default_rng(args.seed)
        vec_dim = None
        for c in self.gdf.columns:
            if c.startswith("gstat_psage"):
                sample = next((v for v in self.gdf[c].head(64) if isinstance(v, (list, tuple, np.ndarray))), None)
                if sample is not None:
                    vec_dim = len(np.asarray(sample).ravel())
                    break
        if vec_dim:
            v = rng.standard_normal(size=(vec_dim,)).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            topic_seed = v

        feature_clamp = float(getattr(args, "feature_clamp", 3.0) or 3.0)
        outlier_threshold = float(getattr(args, "outlier_threshold", 4.0) or 4.0)
        filter_outliers = bool(getattr(args, "filter_outliers", False))

        self.profile_builder = ProfileBuilder(
            self.gdf,
            g_columns=g_columns,
            g_dim=g_dim,
            topic_seed=topic_seed,
            feature_clamp=feature_clamp,
            outlier_threshold=outlier_threshold,
        )
        self.profiles, filtered_outliers = self.profile_builder.build_profiles(
            self.user_ids, self.author_type, filter_outliers=filter_outliers
        )

        if filtered_outliers:
            self.log.warning(
                "Filtered %d outlier users with features beyond ±%.1f std devs: %s",
                len(filtered_outliers),
                outlier_threshold,
                filtered_outliers[:20] if len(filtered_outliers) > 20 else filtered_outliers,
            )
            # Remove filtered users from user_ids
            self.user_ids = [u for u in self.user_ids if u not in set(filtered_outliers)]

        self.log.info(
            "Built profiles | users=%d | gdim=%d | feature_clamp=±%.1f | outlier_threshold=±%.1f",
            int(len(self.profiles)), int(g_dim), feature_clamp, outlier_threshold
        )

        target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
        self.log.info(
            "Loading HyperPEFTLoRAEngine | base_model=%s | online=%s | qlora=%s | target_modules=%s",
            str(args.base_model),
            str(bool(args.online)),
            str(bool(args.qlora)),
            ",".join(target_modules),
        )
        self.engine = HyperPEFTLoRAEngine(
            base_model=args.base_model,
            hyper_dir=args.hyper_dir,
            target_modules=target_modules,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            inject_clamp=args.inject_clamp,
            delta_gain=args.delta_gain,
            use_best_ckpt=args.use_best_ckpt,
            online=args.online,
            qlora=args.qlora,
            device=self.device,
            emit_both=bool(getattr(args, "emit_both", False)),
        )
        self.log.info("HyperPEFTLoRAEngine ready | vocab=%d | sep_id=%s | end_id=%s", int(len(self.engine.tok)), str(self.engine.sep_id), str(self.engine.end_id))

        # ---- Arditi Patch: residual-stream direction abliteration (Appendix V.5b) ----
        # Mechanism (Arditi et al. 2024 NeurIPS): the body's dominant continuation
        # feature lives as a single residual-stream direction accumulating through
        # middle-to-late layers. Projecting it out at every late layer at inference
        # removes its competition with the per-user persona perturbation without
        # retraining. Direction file is produced by extract_arditi_directions.py.
        self._arditi_handles: List[Any] = []
        self._arditi_meta: Dict[str, Any] = {}
        _ap = str(getattr(args, "arditi_patch", "") or "").strip()
        if _ap:
            try:
                _arditi_label_csv_dir = str(getattr(args, "arditi_label_csv_dir", "/tmp"))
                _arditi_label_files_raw = str(getattr(args, "arditi_label_files", "") or "")
                _label_files = [s.strip() for s in _arditi_label_files_raw.split(",") if s.strip()] or None
                self._install_arditi_patch(
                    directions_path=_ap,
                    alpha=float(getattr(args, "arditi_alpha", 1.0)),
                    layer_spec=str(getattr(args, "arditi_layers", "15-23")),
                    mode=str(getattr(args, "arditi_mode", "single") or "single"),
                    label_csv_dir=_arditi_label_csv_dir,
                    label_files=_label_files,
                )
            except Exception as _e:
                self.log.error("[arditi] failed to install patch (%s); aborting", _e)
                raise

        # ---- Cohort centroids for best-of-N selection (post-2026-05-04) ----
        # Validated on M1 gu03 diag_27: cohort-centroid best-of-8 lifts surface
        # cohort g by +2.05 (CI [+1.50, +2.56]) at 100% coherence.
        self.cohort_centroids = None
        _bon_metric = str(getattr(args, "bon_metric", "perplexity"))
        _bon_n = int(getattr(args, "best_of_n", 1) or 1)
        if _bon_metric == "cohort_centroid" and _bon_n > 1:
            try:
                from cohort_centroid_bon import build_cohort_centroids
                # Default cohort -> label mapping covers the 5 GoEmo cohorts
                cohort_to_label = {
                    "rage":    "rage",
                    "empath":  "empath",
                    "neutral": "calm",     # neutral cohort drawn from "calm" labels
                    "grumpy":  "grumpy",
                    "mellow":  "mellow",
                }
                self.cohort_centroids = build_cohort_centroids(
                    author_static_path=str(args.author_parquet),
                    labels_csv_path=str(args.labels_csv),
                    cohort_to_label=cohort_to_label,
                    max_users_per_cohort=2000,
                )
                self.log.info(
                    "[bon] cohort_centroid metric active. Centroids built for %s",
                    sorted(self.cohort_centroids.keys()),
                )
            except Exception as _e:
                self.log.warning("[bon] failed to build cohort centroids (%s); falling back to perplexity", _e)
                self.cohort_centroids = None

        # ---- RQ3 lever sweep: apply env-var levers post-engine ----
        # These are read by sweep_interventions.py's env preamble so
        # one forum invocation per LHS sub-run picks up its lever values.
        try:
            _rm = os.environ.get("HN_LORA_RANK_MASK", "").strip()
            if _rm and int(_rm) > 0:
                self.engine.model.set_rank_mask(int(_rm))
                self.log.info("[lever] LoRA rank mask: %d / %d (elastic-LoRA active)",
                              int(_rm), int(args.lora_r))
        except Exception as _e:
            self.log.warning("[lever] HN_LORA_RANK_MASK ignored: %s", _e)

        try:
            _mc = os.environ.get("HN_MC_DROPOUT_RATE", "").strip()
            if _mc and float(_mc) > 0.0:
                self.engine.hypernet.set_mc_dropout_rate(float(_mc))
                self.log.info("[lever] MC-dropout rate: %.4f (active in eval)", float(_mc))
        except Exception as _e:
            self.log.warning("[lever] HN_MC_DROPOUT_RATE ignored: %s", _e)

        # HN_DELTA_GAIN already consumed by HyperPEFTLoRAEngine via env at init
        # (delta_gain field translates the sweep's delta_scale lever).

        # Stabilizer fraction (RQ3 lever): designate fraction of users as
        # zero-delta calming agents. Env-driven so the sweep can sweep it.
        # Selected once at init from the post-filter user pool.
        try:
            _sf = os.environ.get("HN_STABILIZER_FRACTION", "").strip()
            stab_frac = float(_sf) if _sf else 0.0
        except Exception:
            stab_frac = 0.0
        self._stabilizer_fraction = stab_frac
        self.stabilizer_uids: set = set()
        if stab_frac > 0.0 and self.user_ids:
            n_stab = max(1, int(round(len(self.user_ids) * stab_frac)))
            n_stab = min(n_stab, len(self.user_ids))
            try:
                rng_local = np.random.default_rng(int(args.seed))
                chosen = rng_local.choice(np.array(self.user_ids),
                                           size=n_stab, replace=False)
                self.stabilizer_uids = set(int(u) for u in chosen)
                self.log.info("[lever] Stabilizer agents: %d / %d users (%.1f%%)",
                              len(self.stabilizer_uids), len(self.user_ids),
                              100.0 * len(self.stabilizer_uids)
                              / max(1, len(self.user_ids)))
            except Exception as _e:
                self.log.warning("[lever] stabilizer selection failed: %s", _e)
                self.stabilizer_uids = set()

        self.sentiment_pipe = None
        if not args.disable_sentiment_eval:
            dev_id = 0 if self.device.type == "cuda" else -1
            self.log.info("Building sentiment probe pipeline | device_id=%s", str(dev_id))
            self.sentiment_pipe = _build_sentiment_pipe(dev_id)
            self.log.info("Sentiment probe pipeline ready.")
        else:
            self.log.info("Sentiment evaluation disabled.")

        self.thresholds = self._load_or_compute_thresholds(args.threshold_source, args.threshold_col, args.thresholds_json)
        try:
            self.log.info(
                "Sentiment thresholds | col=%s | q20=%.4f q40=%.4f q60=%.4f q80=%.4f q100=%.4f",
                str(args.threshold_col),
                float(self.thresholds.get("q20", float("nan"))),
                float(self.thresholds.get("q40", float("nan"))),
                float(self.thresholds.get("q60", float("nan"))),
                float(self.thresholds.get("q80", float("nan"))),
                float(self.thresholds.get("q100", float("nan"))),
            )
        except Exception:
            self.log.info("Sentiment thresholds loaded.")

        self.rng = np.random.default_rng(args.seed ^ 0xA5A5)
        
def _load_or_compute_thresholds(self, source: str, col: str, thresholds_json: str) -> Dict[str, float]:
    if thresholds_json and Path(thresholds_json).exists():
        return json.loads(Path(thresholds_json).read_text())

    if not source:
        raise ValueError("--threshold_source is required unless --thresholds_json is provided.")

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix == ".parquet":
        df = pd.read_parquet(str(p))
    elif p.suffix in (".csv", ".tsv"):
        df = pd.read_csv(str(p))
    else:
        raise ValueError("threshold_source must be .parquet or .csv/.tsv")

    norm_stats: Optional[Dict[str, Any]] = None
    norm_path = str(getattr(self.args, "norm_stats_json", "") or "").strip()
    if norm_path and Path(norm_path).exists():
        norm_stats = _try_load_json(Path(norm_path))
    else:
        for cand in (
            p.with_name("feature_norm_stats_10000.json"),
            p.with_name("feature_norm_stats.json"),
        ):
            if cand.exists():
                norm_stats = _try_load_json(cand)
                break

    return _compute_sentiment_thresholds(df, col=col, norm_stats=norm_stats)

def _select_user_pool(
    self,
    author_df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame],
    n_rage: int,
    n_empath: int,
    seed: int,
) -> Tuple[List[int], Dict[int, str]]:
    rng = np.random.default_rng(seed + 123)
    author_type: Dict[int, str] = {}

    if labels_df is not None and not labels_df.empty:
        lab = labels_df.copy()
        lab["target_user_id"] = pd.to_numeric(lab["target_user_id"], errors="coerce").astype("Int64")
        lab = lab.dropna(subset=["target_user_id"])
        lab["target_user_id"] = lab["target_user_id"].astype(int)
        lab["sentiment_label"] = lab["sentiment_label"].astype(str)

        author_has_sent = "gstat_user_sent_mean" in author_df.columns
        if author_has_sent:
            sent = author_df[["target_user_id", "gstat_user_sent_mean"]].copy()
            sent["target_user_id"] = pd.to_numeric(sent["target_user_id"], errors="coerce").astype("Int64")
            sent = sent.dropna(subset=["target_user_id"])
            sent["target_user_id"] = sent["target_user_id"].astype(int)
            sent["gstat_user_sent_mean"] = pd.to_numeric(sent["gstat_user_sent_mean"], errors="coerce")
            lab = lab.merge(sent, on="target_user_id", how="left")

        rage_df = lab[lab["sentiment_label"].isin(["rage"])].copy()
        empath_df = lab[lab["sentiment_label"].isin(["empath"])].copy()
        neutral_df = lab[lab["sentiment_label"].isin(["neutral"])].copy()

        if "gstat_user_sent_mean" in lab.columns:
            if "gstat_user_sent_mean" in rage_df.columns:
                rage_df = rage_df.sort_values("gstat_user_sent_mean", ascending=True, na_position="last")
            if "gstat_user_sent_mean" in empath_df.columns:
                empath_df = empath_df.sort_values("gstat_user_sent_mean", ascending=False, na_position="last")

        if author_has_sent:
            # Real-user path: require the full quota so we can pick extremes from the tails.
            if len(rage_df) >= n_rage and len(empath_df) >= n_empath:
                rage_ids = rage_df["target_user_id"].head(n_rage).tolist()
                empath_ids = empath_df["target_user_id"].head(n_empath).tolist()
                empath_ids = [u for u in empath_ids if u not in set(rage_ids)]
                empath_ids = empath_ids[:n_empath]

                pool = list(rage_ids) + list(empath_ids)
                for u in rage_ids:
                    author_type[int(u)] = "rage"
                for u in empath_ids:
                    author_type[int(u)] = "empath"
                return pool, author_type
        else:
            # Synth path (phase2d): author parquet has no gstat_user_sent_mean, so labels are
            # authoritative. Each sub-parquet is a single cohort; take all labeled rows, clipped
            # to the requested quota. Empty opposite cohort is expected and fine.
            n_rage_eff = min(int(n_rage), len(rage_df))
            n_empath_eff = min(int(n_empath), len(empath_df))
            if n_rage_eff + n_empath_eff > 0:
                rage_ids = rage_df["target_user_id"].head(n_rage_eff).tolist()
                empath_ids = empath_df["target_user_id"].head(n_empath_eff).tolist()
                empath_ids = [u for u in empath_ids if u not in set(rage_ids)]

                pool = list(rage_ids) + list(empath_ids)
                for u in rage_ids:
                    author_type[int(u)] = "rage"
                for u in empath_ids:
                    author_type[int(u)] = "empath"
                return pool, author_type
            # Neutral synth stratum: entire sub-parquet carries sentiment_label="neutral"
            # (phase1p5 composite maps {calm, grumpy, mellow} into the paper's neutral
            # cohort). Tag every labeled row with author_type="neutral"; downstream
            # scorers already accept that class.
            if len(neutral_df) > 0:
                n_neutral_eff = min(max(int(n_rage), int(n_empath)), len(neutral_df))
                neutral_ids = neutral_df["target_user_id"].head(n_neutral_eff).tolist()
                pool = list(neutral_ids)
                for u in neutral_ids:
                    author_type[int(u)] = "neutral"
                return pool, author_type

    if "gstat_user_sent_mean" not in author_df.columns:
        raise KeyError("author_parquet must include gstat_user_sent_mean for fallback selection.")

    tmp = author_df[["target_user_id", "gstat_user_sent_mean"]].copy()
    tmp["gstat_user_sent_mean"] = pd.to_numeric(tmp["gstat_user_sent_mean"], errors="coerce")
    tmp = tmp.dropna(subset=["gstat_user_sent_mean"])
    tmp = tmp.sort_values("gstat_user_sent_mean", ascending=True)

    rage_ids = tmp["target_user_id"].head(n_rage).astype(int).tolist()
    empath_ids = tmp["target_user_id"].tail(n_empath).astype(int).tolist()
    empath_ids = [u for u in empath_ids if u not in set(rage_ids)]
    empath_ids = empath_ids[:n_empath]

    pool = list(rage_ids) + list(empath_ids)
    for u in rage_ids:
        author_type[int(u)] = "rage"
    for u in empath_ids:
        author_type[int(u)] = "empath"

    missing = (n_rage + n_empath) - len(pool)
    if missing > 0:
        remaining = [int(x) for x in tmp["target_user_id"].astype(int).tolist() if int(x) not in set(pool)]
        rng.shuffle(remaining)
        for u in remaining[:missing]:
            pool.append(int(u))
            author_type[int(u)] = "neutral"

    return pool, author_type

def _gen_text(
    self,
    thread: ThreadState,
    uid: int,
    parent_cid: Optional[int],
    generator: BranchGenerator,
    decider: Decider,
) -> Tuple[str, List[int]]:
    mention_list: List[int] = []
    prefix = ""
    if parent_cid is not None and self.rng.random() < decider.p_mention(uid):
        parent = thread.nodes[parent_cid]
        m_uid = int(parent.author_user_id)
        if m_uid != uid:
            prefix = f"u{m_uid}: "
            mention_list.append(m_uid)

    token_ids = generator(uid, parent_cid)
    if token_ids and token_ids[-1] == self.engine.end_id:
        token_ids = token_ids[:-1]

    text = self.engine.tok.decode(token_ids, skip_special_tokens=True).strip()
    text = _postprocess_generated_text(text)

    if prefix and text:
        text = prefix + text
    return text, mention_list


def _build_generator_for_thread(self, thread: ThreadState) -> "BranchGenerator":
    return BranchGenerator(
        engine=self.engine,
        profiles=self.profiles,
        thread=thread,
        max_len=self.args.max_len,
        gamma_recency=self.args.gamma_recency,
        max_new_tokens=self.args.max_new_tokens,
        do_sample=self.args.do_sample,
        top_p=self.args.top_p,
        temperature=self.args.temperature,
        top_k=(None if self.args.top_k in (None, 0) else int(self.args.top_k)),
        repetition_penalty=float(getattr(self.args, "repetition_penalty", 1.0)),
        no_repeat_ngram_size=int(getattr(self.args, "no_repeat_ngram_size", 0)),
        min_new_tokens=int(getattr(self.args, "min_new_tokens", 0)),
        cfg_scale=float(getattr(self.args, "cfg_scale", 1.0)),
        best_of_n=int(getattr(self.args, "best_of_n", 1)),
        temp_anneal_tokens=int(getattr(self.args, "temp_anneal_tokens", 0)),
        temp_anneal_start=float(getattr(self.args, "temp_anneal_start", 0.5)),
        adaptive_delta_ref_norm=float(getattr(self.args, "adaptive_delta_ref_norm", 0.0)),
        force_zero_delta=bool(getattr(self.args, "force_zero_delta", False)),
        device=self.device,
        stabilizer_uids=self.stabilizer_uids,
        bon_metric=str(getattr(self.args, "bon_metric", "perplexity")),
        cohort_centroids=getattr(self, "cohort_centroids", None),
        args=self.args,
    )


def _init_thread_state(self, gid: int, title: str, topic: str) -> ThreadState:
    th = ThreadState(
        gid=int(gid),
        title=str(title).strip(),
        topic=str(topic).strip(),
        created_min=0.0,
    )
    op_uid = int(self.rng.choice(list(self.profiles.keys())))
    op_type = self.profiles[op_uid].author_type
    op_node = CommentNode(
        cid=0,
        parent_cid=None,
        depth=-1,
        author_user_id=op_uid,
        author_type=op_type,
        text=th.title.strip(),
        created_min=th.created_min,
        path=tuple(),
        children=[],
        mentions=[],
    )
    th.add_node(op_node)
    return th


def _drain_avail_events(
    self,
    thread: ThreadState,
    sched: "Scheduler",
    decider: "Decider",
    caps: Dict[int, int],
    max_posts: int,
) -> None:
    """Process every 'avail' event at the head of this thread's queue.

    avail events do not invoke the LM — they only schedule future post events.
    We drain them greedily so the queue's top is either a 'post' event or empty.
    """
    while sched.queue and (len(thread.nodes) - 1) < max_posts:
        top = sched.queue[0]
        if top.time_min > sched.start_min + sched.horizon:
            heapq.heappop(sched.queue)
            continue
        if top.kind != "avail":
            return
        ev = heapq.heappop(sched.queue)
        now = float(ev.time_min)
        for u in sched.prof.values():
            if now >= u.next_reset_min:
                u.posts_today = 0
                u.next_reset_min += 24.0 * 60.0
        u = sched.prof[ev.uid]
        if (now - u.last_post_min) < u.cooldown_min or (u.posts_today >= u.daily_cap):
            sched._schedule_next_for(ev.uid, now)
            continue
        p = decider.p_post(ev.uid, thread, now)
        if self.rng.random() >= p:
            sched._schedule_next_for(ev.uid, now)
            continue
        action, target_cid = decider.choose_action(ev.uid, thread, now, caps)
        if action == "top":
            parent = thread.nodes[0]
        else:
            if target_cid is None or target_cid not in thread.nodes:
                sched._schedule_next_for(ev.uid, now)
                continue
            parent = thread.nodes[target_cid]
        child_depth = parent.depth + 1
        if caps.get(child_depth, 0) <= len(parent.children):
            sched._schedule_next_for(ev.uid, now)
            continue
        dt = float(self.rng.lognormal(
            mean=sched.prof[ev.uid].reply_delay_mu,
            sigma=sched.prof[ev.uid].reply_delay_sigma,
        ))
        post_time = now + dt
        heapq.heappush(sched.queue, Event(
            time_min=post_time, kind="post",
            uid=ev.uid, target_cid=parent.cid,
        ))
        sched._schedule_next_for(ev.uid, now)


def simulate_threads_batched(
    self,
    titles: List[str],
    topic: str,
    persona: str,
    batch_size: int,
) -> List[ThreadState]:
    """Simulate all threads concurrently, batching N pending 'post' events per
    generate() call to saturate the GPU.
    """
    log = getattr(self, "log", LOG)

    caps: Dict[int, int] = {d: int(c) for d, c in enumerate(self.args.fanout)}
    max_posts_per_thread = int(self.args.max_posts)
    if max_posts_per_thread <= 0:
        max_posts_per_thread = sum(self.args.fanout) * 4

    threads: List[ThreadState] = []
    schedulers: List[Scheduler] = []
    deciders: List[Decider] = []
    generators: List[BranchGenerator] = []
    next_cids: List[int] = []

    for i, title in enumerate(titles):
        th = self._init_thread_state(gid=i, title=title, topic=topic)
        decider = Decider(self.profiles, self.rng, topic_mode=persona)
        sched = Scheduler(
            self.profiles, decider,
            start_min=0.0,
            horizon_min=float(self.args.horizon_min),
            rng=self.rng,
        )
        sched.prime(self.profiles.keys(), th.created_min)
        threads.append(th)
        schedulers.append(sched)
        deciders.append(decider)
        generators.append(self._build_generator_for_thread(th))
        next_cids.append(1)

    fzd_flag = bool(getattr(self.args, "force_zero_delta", False))
    stabilizer_set = set(int(u) for u in getattr(self, "stabilizer_uids", set()) or set())

    pad_id = int(self.engine.pad_id) if self.engine.pad_id is not None else 0

    eos_ids: List[int] = []
    try:
        if self.engine.end_id is not None:
            eos_ids.append(int(self.engine.end_id))
    except Exception:
        pass
    try:
        if getattr(self.engine.tok, "eos_token_id", None) is not None:
            eos_ids.append(int(self.engine.tok.eos_token_id))
    except Exception:
        pass
    eos_ids = sorted({x for x in eos_ids if x is not None and int(x) >= 0})
    eos_token: Any = None
    if eos_ids:
        eos_token = eos_ids[0] if len(eos_ids) == 1 else eos_ids

    t_sim_start = _now()
    total_batches = 0
    total_posts = 0
    total_gen_s = 0.0

    while True:
        for ti in range(len(threads)):
            self._drain_avail_events(
                thread=threads[ti], sched=schedulers[ti],
                decider=deciders[ti], caps=caps,
                max_posts=max_posts_per_thread,
            )

        plans: List[Dict[str, Any]] = []
        for ti in range(len(threads)):
            th = threads[ti]
            sched = schedulers[ti]
            while sched.queue and (len(th.nodes) - 1) < max_posts_per_thread:
                top = sched.queue[0]
                if top.time_min > sched.start_min + sched.horizon:
                    heapq.heappop(sched.queue)
                    continue
                if top.kind != "post":
                    break
                if top.target_cid not in th.nodes:
                    heapq.heappop(sched.queue)
                    continue
                parent = th.nodes[top.target_cid]
                child_depth = parent.depth + 1
                if caps.get(child_depth, 0) <= len(parent.children):
                    heapq.heappop(sched.queue)
                    continue
                ev = heapq.heappop(sched.queue)
                plans.append({"ti": ti, "ev": ev, "parent": parent})
                if len(plans) >= batch_size:
                    break
            if len(plans) >= batch_size:
                break

        if not plans:
            any_remaining = False
            for ti in range(len(threads)):
                if (len(threads[ti].nodes) - 1) >= max_posts_per_thread:
                    continue
                q = schedulers[ti].queue
                if q and q[0].time_min <= schedulers[ti].start_min + schedulers[ti].horizon:
                    any_remaining = True
                    break
            if not any_remaining:
                break
            continue

        encoded: List[Dict[str, Any]] = []
        for p in plans:
            ti = p["ti"]
            ev = p["ev"]
            parent = p["parent"]
            gen = generators[ti]
            parent_cid_eff = None if parent.cid == 0 else parent.cid
            toks, attn = gen._encode_branch(parent_cid_eff)
            encoded.append(dict(
                toks=toks, attn=attn,
                gvec=np.asarray(self.profiles[ev.uid].gvec, dtype=np.float32),
                ti=ti, ev=ev, parent=parent,
                parent_cid_eff=parent_cid_eff,
            ))

        max_prompt_len = max(len(e["toks"]) for e in encoded)
        B = len(encoded)
        input_ids = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((B, max_prompt_len), dtype=torch.long, device=self.device)
        for i, e in enumerate(encoded):
            L = len(e["toks"])
            input_ids[i, -L:] = torch.tensor(e["toks"], dtype=torch.long, device=self.device)
            attention_mask[i, -L:] = torch.tensor(e["attn"], dtype=torch.long, device=self.device)

        g = torch.tensor(
            np.stack([e["gvec"] for e in encoded], axis=0),
            dtype=torch.float32, device=self.device,
        )

        if stabilizer_set and not fzd_flag:
            for i, e in enumerate(encoded):
                if int(e["ev"].uid) in stabilizer_set:
                    g[i].zero_()

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_features=g,
            max_new_tokens=int(self.args.max_new_tokens),
            eos_token_id=eos_token,
            do_sample=bool(self.args.do_sample),
            top_p=float(self.args.top_p),
            temperature=float(self.args.temperature),
            top_k=(None if self.args.top_k in (None, 0) else int(self.args.top_k)),
            repetition_penalty=float(getattr(self.args, "repetition_penalty", 1.0)),
            no_repeat_ngram_size=int(getattr(self.args, "no_repeat_ngram_size", 0)),
            min_new_tokens=int(getattr(self.args, "min_new_tokens", 0)),
            cfg_scale=float(getattr(self.args, "cfg_scale", 1.0)),
            temp_anneal_tokens=int(getattr(self.args, "temp_anneal_tokens", 0)),
            temp_anneal_start=float(getattr(self.args, "temp_anneal_start", 0.5)),
            adaptive_delta_ref_norm=float(getattr(self.args, "adaptive_delta_ref_norm", 0.0)),
            force_zero_delta=bool(fzd_flag),
        )

        # ----- best-of-N + cohort-centroid selection (production fast path) -----
        # The BranchGenerator non-batched path wraps generate_reply in a best-of-N
        # loop, but simulate_threads_batched (this function) is the actual
        # production driver via --infer_batch_size. Without the loop here,
        # passing --best_of_n 8 --bon_metric cohort_centroid had no effect on
        # production output. 2026-05-04 fix.
        _bon_n_inner = int(getattr(self.args, "best_of_n", 1) or 1)
        _bon_metric_inner = str(getattr(self.args, "bon_metric", "perplexity"))
        _centroids_inner = getattr(self, "cohort_centroids", None)
        _use_cohort_bon = (
            _bon_n_inner > 1
            and _bon_metric_inner == "cohort_centroid"
            and _centroids_inner is not None
        )

        end_id = int(self.engine.end_id) if self.engine.end_id is not None else -1
        prompt_len = max_prompt_len

        # _selected_texts[i] holds the cohort-best text for row i, or None to
        # fall through to the original per-row decode of out_ids[i].
        _selected_texts: List[Optional[str]] = [None] * len(encoded)

        t_gen0 = _now()
        if _use_cohort_bon:
            try:
                from cohort_centroid_bon import cohort_score as _cohort_score_fn
            except Exception:
                _cohort_score_fn = None
                _use_cohort_bon = False

        if _use_cohort_bon and _cohort_score_fn is not None:
            # Tiled best-of-N: instead of N sequential generate() calls (each
            # paying the prompt-prefill cost), tile inputs to [B*N_tile, L] and
            # call generate() once per tile group. HuggingFace then samples
            # B*N_tile independent stochastic decode trajectories sharing one
            # fused prefill, giving ~3-5x speedup at best_of_n=8.
            #
            # Memory budget: KV cache grows linearly with N_tile. On high-memory GPU
            # 96 GB HBM the default 8-wide tile fits with RMM unified-memory
            # spill to Grace RAM. If OOM, reduce HN_BON_TILE to 4 or 2.
            _all_candidates: List[List[str]] = [[] for _ in range(_bon_n_inner)]
            _seed_base = int(getattr(self.args, "seed", 142))
            _tile_size = max(1, min(
                int(os.environ.get("HN_BON_TILE", str(_bon_n_inner))),
                _bon_n_inner,
            ))
            _n_groups = (_bon_n_inner + _tile_size - 1) // _tile_size

            B_orig = gen_kwargs["input_ids"].shape[0]
            for _g_idx in range(_n_groups):
                _ci_start = _g_idx * _tile_size
                _ci_end = min(_ci_start + _tile_size, _bon_n_inner)
                _n_in_group = _ci_end - _ci_start

                # Per-tile seed so different tiles produce different draws and
                # the run is reproducible across tile-size changes.
                torch.manual_seed(
                    _seed_base + _g_idx * 9973 + total_batches * 17
                )

                # Tile B prompts to B*_n_in_group. repeat_interleave puts row
                # i's _n_in_group candidates contiguously at positions
                # [i * n_in_group, ..., i * n_in_group + n_in_group - 1].
                _gen_kwargs_tiled = dict(gen_kwargs)
                _gen_kwargs_tiled["input_ids"] = gen_kwargs["input_ids"].repeat_interleave(_n_in_group, dim=0)
                _gen_kwargs_tiled["attention_mask"] = gen_kwargs["attention_mask"].repeat_interleave(_n_in_group, dim=0)
                _gen_kwargs_tiled["global_features"] = gen_kwargs["global_features"].repeat_interleave(_n_in_group, dim=0)

                # Stage per-batch-row user_ids for the Arditi Patch's mode-aware
                # closure. Tiled batch is B_orig * _n_in_group rows; each user
                # appears _n_in_group times consecutively. Match repeat_interleave.
                try:
                    _row_uids_tiled = []
                    for _e in encoded:
                        _row_uids_tiled.extend([int(_e["ev"].uid)] * _n_in_group)
                    self.engine.model._arditi_batch_user_ids = _row_uids_tiled
                    self.engine.model._arditi_batch_uids_tensor = torch.tensor(
                        _row_uids_tiled, dtype=torch.long, device=self.device,
                    )
                except Exception:
                    self.engine.model._arditi_batch_user_ids = None
                    self.engine.model._arditi_batch_uids_tensor = None

                with torch.no_grad():
                    with autocast(device_type=self.device.type,
                                  enabled=(self.device.type == "cuda")):
                        _out_ids_t = self.engine.generate_reply(**_gen_kwargs_tiled)

                # Decode each candidate slice. Output positions: row i's
                # candidate j (where 0 <= j < _n_in_group) sits at
                # i * _n_in_group + j.
                for _j in range(_n_in_group):
                    _ci_global = _ci_start + _j
                    _batch_texts: List[str] = []
                    for _i in range(B_orig):
                        _row_pos = _i * _n_in_group + _j
                        _gen_tokens = _out_ids_t[_row_pos, prompt_len:].tolist()
                        if end_id >= 0 and end_id in _gen_tokens:
                            _cut = _gen_tokens.index(end_id)
                            _gen_tokens = _gen_tokens[:_cut]
                        _txt = self.engine.tok.decode(
                            _gen_tokens, skip_special_tokens=True
                        ).strip()
                        _txt = _postprocess_generated_text(_txt)
                        _batch_texts.append(_txt)
                    _all_candidates[_ci_global] = _batch_texts

                del _out_ids_t, _gen_kwargs_tiled
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # For each row, pick best-scoring candidate
            for _i, _e in enumerate(encoded):
                _uid = int(_e["ev"].uid)
                _user_cohort = str(self.profiles[_uid].author_type) \
                    if _uid in self.profiles else None
                _best_score = -1e18
                _best_txt = _all_candidates[0][_i]
                for _ci in range(_bon_n_inner):
                    _cand = _all_candidates[_ci][_i]
                    if not _cand or not _is_coherent(_cand):
                        continue
                    if (_user_cohort is None
                            or _user_cohort not in _centroids_inner):
                        # No usable cohort label: take first coherent
                        _best_txt = _cand
                        break
                    try:
                        _s = float(_cohort_score_fn(
                            _cand, _user_cohort, _centroids_inner
                        ))
                    except Exception:
                        _s = -1e9
                    if _s > _best_score:
                        _best_score = _s
                        _best_txt = _cand
                _selected_texts[_i] = _best_txt
            # out_ids is unused on this path; set to None to make that explicit
            out_ids = None
        else:
            # Original single-shot fast path (best_of_n <= 1 or non-cohort metric)
            try:
                _row_uids = [int(e["ev"].uid) for e in encoded]
                self.engine.model._arditi_batch_user_ids = _row_uids
                self.engine.model._arditi_batch_uids_tensor = torch.tensor(
                    _row_uids, dtype=torch.long, device=self.device,
                )
            except Exception:
                self.engine.model._arditi_batch_user_ids = None
                self.engine.model._arditi_batch_uids_tensor = None
            with torch.no_grad():
                with autocast(device_type=self.device.type,
                              enabled=(self.device.type == "cuda")):
                    out_ids = self.engine.generate_reply(**gen_kwargs)
        total_gen_s += float(_now() - t_gen0)
        total_batches += 1

        for i, e in enumerate(encoded):
            ti = e["ti"]
            ev = e["ev"]
            parent = e["parent"]
            th = threads[ti]

            mention_list: List[int] = []
            prefix_text = ""
            parent_cid_eff = e["parent_cid_eff"]
            if parent_cid_eff is not None and self.rng.random() < deciders[ti].p_mention(ev.uid):
                par_node = th.nodes[parent_cid_eff]
                m_uid = int(par_node.author_user_id)
                if m_uid != int(ev.uid):
                    prefix_text = f"u{m_uid}: "
                    mention_list.append(m_uid)

            if _selected_texts[i] is not None:
                # Best-of-N path: pre-selected, post-processed text is ready
                text = _selected_texts[i]
            else:
                gen_tokens = out_ids[i, prompt_len:].tolist()
                if end_id >= 0 and end_id in gen_tokens:
                    cut = gen_tokens.index(end_id)
                    gen_tokens = gen_tokens[:cut]
                text = self.engine.tok.decode(gen_tokens, skip_special_tokens=True).strip()
                text = _postprocess_generated_text(text)
            if prefix_text and text:
                text = prefix_text + text
            text = (text or "").strip()
            if not text:
                continue

            child_depth = parent.depth + 1
            if caps.get(child_depth, 0) <= len(parent.children):
                continue

            node = CommentNode(
                cid=next_cids[ti],
                parent_cid=(None if parent.cid == 0 else parent.cid),
                depth=child_depth,
                author_user_id=int(ev.uid),
                author_type=str(self.profiles[ev.uid].author_type),
                text=text,
                created_min=float(ev.time_min),
                path=(parent.path + (parent.cid,)) if parent.cid != 0 else tuple(),
                children=[],
                mentions=[int(m) for m in mention_list or []],
            )
            th.add_node(node)
            next_cids[ti] += 1
            total_posts += 1

            u = self.profiles[ev.uid]
            u.last_post_min = float(ev.time_min)
            u.posts_today += 1

        if total_batches % 10 == 0:
            elapsed = float(_now() - t_sim_start)
            log.info(
                "[batched] batches=%d posts=%d B_last=%d elapsed_s=%.1f gen_s=%.1f posts/s=%.2f",
                int(total_batches), int(total_posts), int(B),
                float(elapsed), float(total_gen_s),
                float(total_posts / max(elapsed, 1e-3)),
            )

    elapsed = float(_now() - t_sim_start)
    log.info(
        "[batched] DONE | threads=%d posts=%d batches=%d elapsed_s=%.1f gen_s=%.1f posts/s=%.2f",
        int(len(threads)), int(total_posts), int(total_batches),
        float(elapsed), float(total_gen_s),
        float(total_posts / max(elapsed, 1e-3)),
    )

    return threads


def simulate_thread(self, gid: int, title: str, topic: str, topic_mode: str) -> ThreadState:
    start_min = 0.0
    horizon = float(self.args.horizon_min)

    thread = ThreadState(
        gid=int(gid),
        title=title.strip(),
        topic=topic.strip(),
        created_min=start_min,
    )

    caps: Dict[int, int] = {}
    for d, c in enumerate(self.args.fanout):
        caps[d] = int(c)

    decider = Decider(self.profiles, self.rng, topic_mode=topic_mode)
    sched = Scheduler(self.profiles, decider, start_min=start_min, horizon_min=horizon, rng=self.rng)

    generator = BranchGenerator(
        engine=self.engine,
        profiles=self.profiles,
        thread=thread,
        max_len=self.args.max_len,
        gamma_recency=self.args.gamma_recency,
        max_new_tokens=self.args.max_new_tokens,
        do_sample=self.args.do_sample,
        top_p=self.args.top_p,
        temperature=self.args.temperature,
        top_k=(None if self.args.top_k in (None, 0) else int(self.args.top_k)),
        repetition_penalty=float(getattr(self.args, "repetition_penalty", 1.0)),
        no_repeat_ngram_size=int(getattr(self.args, "no_repeat_ngram_size", 0)),
        min_new_tokens=int(getattr(self.args, "min_new_tokens", 0)),
        cfg_scale=float(getattr(self.args, "cfg_scale", 1.0)),
        best_of_n=int(getattr(self.args, "best_of_n", 1)),
        temp_anneal_tokens=int(getattr(self.args, "temp_anneal_tokens", 0)),
        temp_anneal_start=float(getattr(self.args, "temp_anneal_start", 0.5)),
        adaptive_delta_ref_norm=float(getattr(self.args, "adaptive_delta_ref_norm", 0.0)),
        force_zero_delta=bool(getattr(self.args, "force_zero_delta", False)),
        device=self.device,
        stabilizer_uids=self.stabilizer_uids,
        bon_metric=str(getattr(self.args, "bon_metric", "perplexity")),
        cohort_centroids=getattr(self, "cohort_centroids", None),
        args=self.args,
    )

    def _gen(uid: int, parent_cid: Optional[int], now_min: float) -> Tuple[str, List[int]]:
        _ = now_min
        return self._gen_text(thread, uid, parent_cid, generator, decider)

    max_posts = int(self.args.max_posts)
    if max_posts <= 0:
        max_posts = sum(self.args.fanout) * 4

    sched.run(thread, caps, _gen, max_posts=max_posts)
    return thread


def _thread_to_rows(self, th: ThreadState, subject: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cid, nd in th.nodes.items():
        if nd.cid == 0:
            continue
        rows.append(
            dict(
                gid=int(th.gid),
                subject=str(subject),
                thread_title=str(th.title),
                comment_id=int(nd.cid),
                parent_comment_id=(None if nd.parent_cid is None else int(nd.parent_cid)),
                depth=int(nd.depth),
                author_user_id=int(nd.author_user_id),
                author_type=str(nd.author_type),
                created_min=float(nd.created_min),
                mentions=[int(x) for x in nd.mentions],
                text=str(nd.text),
                path=[int(x) for x in nd.path],
                popularity=int(len(nd.children)),
            )
        )
    rows.sort(key=lambda r: (r["created_min"], r["comment_id"]))
    return rows


def _thread_to_markdown(self, th: ThreadState, subject: str) -> str:
    lines: List[str] = []
    lines.append(f"# r/{subject}: {th.title}")
    lines.append("")

    def walk(cid: int, indent: int) -> None:
        nd = th.nodes[cid]
        if nd.cid != 0:
            lines.append("  " * indent + f"- u{nd.author_user_id} [{nd.author_type}]: {nd.text}")
        for ch in nd.children:
            walk(ch, indent + (0 if nd.cid == 0 else 1))

    walk(0, 0)
    return "\n".join(lines) + "\n"

def run(self) -> None:
    log = getattr(self, "log", LOG)

    out_dir = Path(self.args.out_dir)
    _ensure_dir(out_dir)
    log.info("Output directory: %s", str(out_dir))

    persona = str(getattr(self.args, "sentiment_target", "neutral") or "neutral").strip().lower()
    if persona not in ("rage", "empath", "neutral"):
        persona = "neutral"

    if persona == "rage":
        default_titles = DEFAULT_TITLES_RAGE
    elif persona == "empath":
        default_titles = DEFAULT_TITLES_EMPATH
    else:
        default_titles = DEFAULT_TITLES_NEUTRAL

    titles: List[str] = []
    titles_path = str(getattr(self.args, "titles_path", "") or "").strip()
    if titles_path:
        p = Path(titles_path)
        if p.exists():
            try:
                raw = p.read_text(encoding="utf-8")
            except Exception:
                raw = p.read_text()
            titles = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        else:
            log.warning("titles_path not found: %s", str(titles_path))

    if not titles:
        n = int(getattr(self.args, "threads_from_default", 0) or 0)
        if n <= 0:
            n = len(default_titles)
        titles = list(default_titles)
        if len(titles) >= n:
            titles = titles[:n]
        elif titles:
            extra: List[str] = []
            while len(titles) + len(extra) < n:
                extra.extend(list(default_titles))
            titles = (titles + extra)[:n]

    topic = str(getattr(self.args, "topic", "") or "").strip()

    raw_subject = topic.lower().strip() if topic else persona
    subject = re.sub(r"[^a-z0-9]+", "_", raw_subject).strip("_")
    if not subject:
        subject = persona

    log.info("Simulation config | persona=%s | subject=%s | topic=%s | n_threads=%d", persona, subject, topic, int(len(titles)))
    log.info(
        "Generation config | max_len=%d | max_new_tokens=%d | top_p=%.3f | temperature=%.3f | do_sample=%s",
        int(getattr(self.args, "max_len", 0) or 0),
        int(getattr(self.args, "max_new_tokens", 0) or 0),
        float(getattr(self.args, "top_p", 0.0) or 0.0),
        float(getattr(self.args, "temperature", 0.0) or 0.0),
        str(bool(getattr(self.args, "do_sample", False))),
    )
    _cfg = float(getattr(self.args, "cfg_scale", 1.0) or 1.0)
    _bon = int(getattr(self.args, "best_of_n", 1) or 1)
    _tat = int(getattr(self.args, "temp_anneal_tokens", 0) or 0)
    _adr = float(getattr(self.args, "adaptive_delta_ref_norm", 0.0) or 0.0)
    extras: List[str] = []
    if _cfg > 1.0 + 1e-6:
        extras.append("cfg_scale=%.2f" % _cfg)
    if _bon > 1:
        extras.append("best_of_n=%d" % _bon)
    if _tat > 0:
        extras.append("temp_anneal=%d@%.2f" % (_tat, float(getattr(self.args, "temp_anneal_start", 0.5))))
    if _adr > 0:
        extras.append("adaptive_delta_ref=%.2f" % _adr)
    if extras:
        log.info("Enhanced decoding | %s", " | ".join(extras))
    if bool(getattr(self.args, "force_zero_delta", False)):
        log.info("ABLATION MODE: force_zero_delta=True — all hypernetwork deltas zeroed (baseline run)")

    all_rows: List[Dict[str, Any]] = []
    md_chunks: List[str] = []

    bs = int(getattr(self.args, "infer_batch_size", 0) or 0)
    if bs >= 2:
        log.info(
            "Using BATCHED concurrent-thread simulator | threads=%d | infer_batch_size=%d",
            int(len(titles)), int(bs),
        )
        t0 = _now()
        threads_out = self.simulate_threads_batched(
            titles=list(titles), topic=topic, persona=persona, batch_size=int(bs),
        )
        for i, th in enumerate(threads_out):
            all_rows.extend(self._thread_to_rows(th, subject=subject))
            md_chunks.append(self._thread_to_markdown(th, subject=subject))
            md_chunks.append("\n---\n\n")
        log.info(
            "Batched simulation DONE | threads=%d | total_elapsed_s=%.2f",
            int(len(threads_out)), float(_now() - t0),
        )
    else:
        for i, title in enumerate(titles):
            t0 = _now()
            log.info("Simulating thread %d/%d: %s", int(i + 1), int(len(titles)), str(title))
            th = self.simulate_thread(gid=i, title=title, topic=topic, topic_mode=persona)
            all_rows.extend(self._thread_to_rows(th, subject=subject))
            md_chunks.append(self._thread_to_markdown(th, subject=subject))
            md_chunks.append("\n---\n\n")
            n_comments = max(0, int(len(th.nodes) - 1))
            log.info("Thread %d complete | comments=%d | elapsed_s=%.2f", int(i + 1), int(n_comments), float(_now() - t0))

    df = pd.DataFrame(all_rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "gid",
                "subject",
                "thread_title",
                "comment_id",
                "parent_comment_id",
                "depth",
                "author_user_id",
                "author_type",
                "created_min",
                "mentions",
                "text",
                "path",
                "popularity",
            ]
        )
        log.warning("No comments were generated; forum outputs will be empty.")

    forum_md = "".join(md_chunks).rstrip() + "\n"
    forum_md_path = out_dir / "forum.md"
    try:
        forum_md_path.write_text(forum_md, encoding="utf-8")
    except Exception:
        forum_md_path.write_text(forum_md)
    log.info("Wrote %s", str(forum_md_path))

    persona_md_path = out_dir / f"{persona}_forum.md"
    try:
        persona_md_path.write_text(forum_md, encoding="utf-8")
        log.info("Wrote %s", str(persona_md_path))
    except Exception:
        log.warning("Failed writing %s", str(persona_md_path), exc_info=True)

    forum_parquet_path = out_dir / "forum.parquet"
    try:
        df.to_parquet(forum_parquet_path, index=False)
        log.info("Wrote %s", str(forum_parquet_path))
    except Exception:
        log.warning("Failed writing %s", str(forum_parquet_path), exc_info=True)

    # Emit author_label_profile.parquet sidecar when synth metadata is present.
    # Phase 3d scorer keys realized (per-turn text) against expected (per-user
    # profile) using this file; absent ⇒ scorer falls back to gstat-only
    # expected values for real-user forums.
    meta_df = getattr(self, "user_metadata_df", None)
    if meta_df is not None and "author_user_id" in df.columns:
        authors = sorted({int(u) for u in df["author_user_id"].dropna().tolist()})
        keep_cols = ["target_user_id"]
        for c in meta_df.columns:
            if c == "target_user_id":
                continue
            if c.startswith("pol_") or c in ("stratum", "cohort_goemo",
                                              "ambiguity_score",
                                              "label_profile_json",
                                              "target_spec_json"):
                keep_cols.append(c)
        sidecar = meta_df[keep_cols].copy()
        sidecar = sidecar[sidecar["target_user_id"].isin(authors)].reset_index(drop=True)
        sidecar = sidecar.rename(columns={"target_user_id": "author_user_id"})
        sidecar_path = out_dir / "author_label_profile.parquet"
        try:
            sidecar.to_parquet(sidecar_path, index=False)
            log.info("Wrote %s | authors=%d cols=%d",
                     str(sidecar_path), int(len(sidecar)), int(len(sidecar.columns)))
        except Exception:
            log.warning("Failed writing %s", str(sidecar_path), exc_info=True)

    forum_jsonl_path = out_dir / "forum.jsonl"
    try:
        with open(forum_jsonl_path, "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        log.info("Wrote %s", str(forum_jsonl_path))
    except Exception:
        log.error("Failed writing %s", str(forum_jsonl_path), exc_info=True)
        raise

    log.info("Running sentiment evaluation...")
    meta_extra = self._score_and_save_sentiment(out_dir, df)
    log.info("Sentiment evaluation complete.")

    counts = {"rage": 0, "empath": 0, "neutral": 0}
    for _, t in self.author_type.items():
        tt = str(t)
        if tt not in counts:
            counts["neutral"] += 1
        else:
            counts[tt] += 1

    meta: Dict[str, Any] = dict(
        persona=persona,
        subject=subject,
        topic=topic,
        n_threads=int(len(titles)),
        n_users=int(len(self.user_ids)),
        user_type_counts=counts,
        thresholds=self.thresholds,
        args=dict(vars(self.args)),
    )
    if isinstance(meta_extra, dict) and meta_extra:
        meta.update(meta_extra)

    def _json_default(o: Any) -> Any:
        try:
            import numpy as _np  # local import to avoid hard dependency in edge cases

            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
        except Exception:
            pass
        return str(o)

    meta_path = out_dir / "metadata.json"
    try:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    except Exception:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default))
    log.info("Wrote %s", str(meta_path))        
        
def _score_and_save_sentiment(self, out_dir: Path, df: pd.DataFrame) -> Dict[str, Any]:
    log = getattr(self, "log", LOG)

    if self.args.disable_sentiment_eval:
        log.info("Sentiment evaluation disabled; skipping.")
        return {}

    if self.sentiment_pipe is None:
        raise RuntimeError("Sentiment pipe was not built.")

    persona = str(self.args.sentiment_target).strip().lower()
    if persona not in ("rage", "empath", "neutral"):
        persona = "neutral"

    if df is None or df.empty:
        log.warning("Sentiment eval skipped: forum dataframe is empty.")
        user_sent_out = pd.DataFrame(
            columns=[
                "author_user_id",
                "author_type",
                "n_comments",
                "gen_sent_mean",
                "gen_sent_var",
                "pred_user_label",
                "match_extreme",
            ]
        )
        user_sent_out.to_csv(out_dir / f"{persona}_user_sentiment_eval.csv", index=False)
        pd.DataFrame().to_csv(out_dir / f"{persona}_sentiment_confusion.csv", index=False)
        return dict(thresholds=self.thresholds, summary={})

    log.info("Sentiment eval | comments=%d", int(len(df)))

    texts = df["text"].astype(str).tolist()
    pol = polarity_batch(self.sentiment_pipe, texts)
    pol_arr = np.asarray(pol, dtype=float).reshape(-1)
    # Fill genuine API errors with 0.0
    pol_arr = np.where(np.isfinite(pol_arr), pol_arr, 0.0)

    # Detect garbled/gibberish text and NaN-out its polarity so it does not
    # poison the user-level aggregation.  SST-2 assigns strongly negative
    # scores to incoherent text, which systematically mis-labels empath users
    # as rage (garbled → polarity ≈ -1 → "rage" via thresholds).
    coherent = df["text"].astype(str).apply(_is_coherent).to_numpy()
    n_garbled = int((~coherent).sum())
    if n_garbled:
        log.info("Sentiment eval | %d/%d comments flagged as incoherent — excluded from scoring.",
                 n_garbled, int(len(df)))
    pol_arr = np.where(coherent, pol_arr, np.nan)

    df["sent_polarity"] = pol_arr.astype(float)
    df["pred_comment_label"] = df["sent_polarity"].apply(
        lambda x: _label_from_thresholds(float(x), self.thresholds) if pd.notna(x) else "unknown"
    )

    # ---- Topic-relative scoring ----
    # Subtract per-thread mean polarity so classification measures each user's
    # deviation from the topic baseline, not absolute sentiment.
    # NaN (garbled) entries propagate naturally: NaN - mean = NaN.
    if "gid" in df.columns:
        thread_mean = df.groupby("gid")["sent_polarity"].transform("mean")
        df["sent_polarity_relative"] = df["sent_polarity"] - thread_mean
    else:
        df["sent_polarity_relative"] = df["sent_polarity"]

    # User-level aggregation: pandas .mean()/.var() skip NaN by default,
    # so garbled comments are excluded from the user-level sentiment mean.
    # 'count' only counts non-NaN (= coherent comments); use size() for totals.
    grp = df.groupby(["author_user_id", "author_type"], dropna=False)
    user_df = grp["sent_polarity"].agg(n_scored="count", gen_sent_mean="mean", gen_sent_var="var").reset_index()
    n_total = grp.size().reset_index(name="n_comments")
    user_df = user_df.merge(n_total, on=["author_user_id", "author_type"], how="left")
    user_df["pred_user_label"] = user_df["gen_sent_mean"].apply(
        lambda x: _label_from_thresholds(float(x), self.thresholds) if pd.notna(x) else "unknown"
    )

    # ---- Topic-relative user-level aggregation ----
    user_rel = grp["sent_polarity_relative"].agg(gen_sent_mean_rel="mean").reset_index()
    user_df = user_df.merge(user_rel, on=["author_user_id", "author_type"], how="left")
    # Derive quintile thresholds from the relative score distribution itself
    rel_valid = user_df["gen_sent_mean_rel"].dropna()
    if len(rel_valid) >= 10:
        rq = rel_valid.quantile([0.2, 0.4, 0.6, 0.8])
        rel_thr = {"q20": float(rq[0.2]), "q40": float(rq[0.4]), "q60": float(rq[0.6]), "q80": float(rq[0.8])}
        user_df["pred_user_label_rel"] = user_df["gen_sent_mean_rel"].apply(
            lambda x: _label_from_thresholds(float(x), rel_thr) if pd.notna(x) else "unknown"
        )
        log.info("Topic-relative thresholds | q20=%.4f q40=%.4f q60=%.4f q80=%.4f",
                 rel_thr["q20"], rel_thr["q40"], rel_thr["q60"], rel_thr["q80"])
    else:
        rel_thr = None
        user_df["pred_user_label_rel"] = "unknown"

    user_df["match_extreme"] = (
        (user_df["author_type"] == user_df["pred_user_label"]) & (user_df["author_type"].isin(["rage", "empath"]))
    ).astype(int)
    user_df["match_extreme_rel"] = (
        (user_df["author_type"] == user_df["pred_user_label_rel"]) & (user_df["author_type"].isin(["rage", "empath"]))
    ).astype(int)

    user_sent_out = user_df[
        ["author_user_id", "author_type", "n_comments", "n_scored",
         "gen_sent_mean", "gen_sent_var", "pred_user_label", "match_extreme",
         "gen_sent_mean_rel", "pred_user_label_rel", "match_extreme_rel"]
    ].copy()

    user_sent_out.to_csv(out_dir / f"{persona}_user_sentiment_eval.csv", index=False)
    log.info("Wrote %s", str(out_dir / f"{persona}_user_sentiment_eval.csv"))

    try:
        cm = pd.crosstab(user_sent_out["author_type"], user_sent_out["pred_user_label"], dropna=False)
        cm.to_csv(out_dir / f"{persona}_sentiment_confusion.csv")
        log.info("Wrote %s", str(out_dir / f"{persona}_sentiment_confusion.csv"))
    except Exception:
        log.warning("Failed writing %s", str(out_dir / f"{persona}_sentiment_confusion.csv"), exc_info=True)

    try:
        cm_rel = pd.crosstab(user_sent_out["author_type"], user_sent_out["pred_user_label_rel"], dropna=False)
        cm_rel.to_csv(out_dir / f"{persona}_sentiment_relative_confusion.csv")
        log.info("Wrote %s", str(out_dir / f"{persona}_sentiment_relative_confusion.csv"))
    except Exception:
        log.warning("Failed writing %s", str(out_dir / f"{persona}_sentiment_relative_confusion.csv"), exc_info=True)

    try:
        df.to_parquet(out_dir / f"{persona}_forum.parquet", index=False)
        log.info("Wrote %s", str(out_dir / f"{persona}_forum.parquet"))
    except Exception:
        log.warning("Failed writing %s", str(out_dir / f"{persona}_forum.parquet"), exc_info=True)

    try:
        with open(out_dir / f"{persona}_forum.jsonl", "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        log.info("Wrote %s", str(out_dir / f"{persona}_forum.jsonl"))
    except Exception:
        log.warning("Failed writing %s", str(out_dir / f"{persona}_forum.jsonl"), exc_info=True)

    summary: Dict[str, Any] = {}
    for t in ["rage", "empath"]:
        sub = user_sent_out[user_sent_out["author_type"] == t]
        if sub.empty:
            continue
        summary[t] = dict(
            users=int(len(sub)),
            avg_comments=float(sub["n_comments"].mean()),
            match_extreme_rate=float(sub["match_extreme"].mean()),
            match_extreme_rel_rate=float(sub["match_extreme_rel"].mean()),
            mean_sent=float(sub["gen_sent_mean"].mean()),
            mean_sent_rel=float(pd.to_numeric(sub["gen_sent_mean_rel"], errors="coerce").mean()),
        )

    # --- Classification metrics (precision, recall, F1 for rage/empath) ---
    try:
        cls_df = user_sent_out[user_sent_out["author_type"].isin(["rage", "empath"])].copy()
        if len(cls_df) >= 10 and "pred_user_label" in cls_df.columns:
            classification_metrics: Dict[str, Any] = {}
            for target_cls in ["rage", "empath"]:
                tp = int(((cls_df["author_type"] == target_cls) & (cls_df["pred_user_label"] == target_cls)).sum())
                fp = int(((cls_df["author_type"] != target_cls) & (cls_df["pred_user_label"] == target_cls)).sum())
                fn = int(((cls_df["author_type"] == target_cls) & (cls_df["pred_user_label"] != target_cls)).sum())
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                classification_metrics[target_cls] = dict(
                    precision=float(prec), recall=float(rec), f1=float(f1),
                    tp=tp, fp=fp, fn=fn,
                )
            # Row-normalized confusion percentages
            cm_pct = pd.crosstab(cls_df["author_type"], cls_df["pred_user_label"], normalize="index")
            classification_metrics["confusion_pct"] = cm_pct.to_dict()
            # AUC-ROC using continuous sentiment as score
            try:
                valid = cls_df[cls_df["gen_sent_mean"].notna()].copy()
                if len(valid) >= 10:
                    y_true = (valid["author_type"] == "empath").astype(int).values
                    y_score = pd.to_numeric(valid["gen_sent_mean"], errors="coerce").values
                    # manual AUC via Mann-Whitney interpretation
                    from scipy.stats import mannwhitneyu as _mwu
                    pos_scores = y_score[y_true == 1]
                    neg_scores = y_score[y_true == 0]
                    if len(pos_scores) >= 2 and len(neg_scores) >= 2:
                        u_stat, _ = _mwu(pos_scores, neg_scores, alternative="greater")
                        auc = float(u_stat / (len(pos_scores) * len(neg_scores)))
                        classification_metrics["auc_roc_sentiment"] = auc
            except Exception:
                pass
            summary["classification"] = classification_metrics
            log.info("[classification] rage: P=%.3f R=%.3f F1=%.3f | empath: P=%.3f R=%.3f F1=%.3f",
                     classification_metrics.get("rage", {}).get("precision", 0),
                     classification_metrics.get("rage", {}).get("recall", 0),
                     classification_metrics.get("rage", {}).get("f1", 0),
                     classification_metrics.get("empath", {}).get("precision", 0),
                     classification_metrics.get("empath", {}).get("recall", 0),
                     classification_metrics.get("empath", {}).get("f1", 0))
    except Exception:
        log.warning("Classification metrics computation failed.", exc_info=True)

    style_rows: List[Dict[str, Any]] = []
    for _, r in df[["author_user_id", "author_type", "text"]].iterrows():
        txt = str(r["text"] or "")
        toks = _toklist(txt)
        words = txt.split()
        n_words = max(len(words), 1)
        unique_words = len(set(w.lower() for w in words)) if words else 0
        # Lexical diversity metrics
        word_freqs = {}
        for w in words:
            wl = w.lower()
            word_freqs[wl] = word_freqs.get(wl, 0) + 1
        hapax_count = sum(1 for c in word_freqs.values() if c == 1) if word_freqs else 0
        # Shannon entropy
        shannon_h = 0.0
        if word_freqs and n_words > 1:
            for cnt in word_freqs.values():
                p = cnt / n_words
                if p > 0:
                    shannon_h -= p * math.log2(p)
        # Simpson's diversity index (1 - dominance)
        simpsons_d = 0.0
        if n_words > 1:
            dom = sum(cnt * (cnt - 1) for cnt in word_freqs.values()) / (n_words * (n_words - 1))
            simpsons_d = 1.0 - dom
        style_rows.append(
            dict(
                author_user_id=int(r["author_user_id"]),
                author_type=str(r["author_type"]),
                gen_question_ratio=float(1.0 if txt.strip().endswith("?") else 0.0),
                gen_secondperson_ratio=float(_secondperson_ratio(toks)),
                gen_caps_ratio=float(_caps_ratio(txt)),
                gen_hedge_ratio=float(_hedge_ratio([txt])),
                gen_intensifier_ratio=float(_intensifier_ratio(toks)),
                gen_agreement_ratio=float(_agreement_ratio([txt])),
                gen_disagreement_ratio=float(_disagreement_ratio([txt])),
                gen_subjectivity_ratio=float(_subjectivity_ratio(toks)),
                gen_word_count=int(len(words)),
                gen_mean_word_len=float(np.mean([len(w) for w in words])) if words else 0.0,
                gen_ttr=float(unique_words / n_words),
                gen_exclamation_ratio=float(txt.count("!") / max(1, len(re.findall(r"[.!?]+", txt)))),
                gen_shannon_h=float(shannon_h),
                gen_simpsons_d=float(simpsons_d),
                gen_hapax_ratio=float(hapax_count / n_words),
            )
        )

    style_user = pd.DataFrame(style_rows)
    if not style_user.empty:
        metric_cols = [
            "gen_question_ratio",
            "gen_secondperson_ratio",
            "gen_caps_ratio",
            "gen_hedge_ratio",
            "gen_intensifier_ratio",
            "gen_agreement_ratio",
            "gen_disagreement_ratio",
            "gen_subjectivity_ratio",
            "gen_mean_word_len",
            "gen_ttr",
            "gen_exclamation_ratio",
            "gen_shannon_h",
            "gen_simpsons_d",
            "gen_hapax_ratio",
        ]
        # Per-user total word count (sum) and post count
        word_agg = style_user.groupby(["author_user_id", "author_type"], dropna=False).agg(
            gen_total_words=("gen_word_count", "sum"),
            gen_n_posts=("gen_word_count", "count"),
        ).reset_index()
        style_user = style_user.groupby(["author_user_id", "author_type"], dropna=False)[metric_cols].mean().reset_index()
        style_user = style_user.merge(word_agg, on=["author_user_id", "author_type"], how="left")

        # ---- min_user_tokens filter ----
        min_tok = getattr(self.args, "min_user_tokens", 0)
        if min_tok > 0 and "gen_total_words" in style_user.columns:
            n_before = len(style_user)
            sparse_mask = style_user["gen_total_words"] < min_tok
            n_filtered = int(sparse_mask.sum())
            if n_filtered:
                log.info("min_user_tokens=%d | filtered %d/%d users with too few generated words",
                         min_tok, n_filtered, n_before)
                # Mark filtered users rather than dropping — keeps them in CSVs but excludes from cosine
                style_user.loc[sparse_mask, "_below_min_tokens"] = True
            if "_below_min_tokens" in style_user.columns:
                style_user["_below_min_tokens"] = style_user["_below_min_tokens"].fillna(False).astype(bool)
            else:
                style_user["_below_min_tokens"] = False

    # ---- Style-based persona classification ----
    # Composite score: pro-empath features minus pro-rage features.
    # These align with what the hypernetwork actually controls (style, not sentiment).
    if not style_user.empty:
        style_user["style_persona_score"] = (
            style_user.get("gen_question_ratio", 0.0).astype(float)
            + style_user.get("gen_secondperson_ratio", 0.0).astype(float)
            + style_user.get("gen_hedge_ratio", 0.0).astype(float)
            + style_user.get("gen_agreement_ratio", 0.0).astype(float)
        ) - (
            style_user.get("gen_caps_ratio", 0.0).astype(float)
            + style_user.get("gen_disagreement_ratio", 0.0).astype(float)
        )
        sp_valid = style_user["style_persona_score"].dropna()
        if len(sp_valid) >= 10:
            sq = sp_valid.quantile([0.2, 0.4, 0.6, 0.8])
            style_thr = {"q20": float(sq[0.2]), "q40": float(sq[0.4]), "q60": float(sq[0.6]), "q80": float(sq[0.8])}
            style_user["pred_style_label"] = style_user["style_persona_score"].apply(
                lambda x: _label_from_thresholds(float(x), style_thr) if pd.notna(x) else "unknown"
            )
            log.info("Style persona thresholds | q20=%.4f q40=%.4f q60=%.4f q80=%.4f",
                     style_thr["q20"], style_thr["q40"], style_thr["q60"], style_thr["q80"])
        else:
            style_user["pred_style_label"] = "unknown"

        if "author_type" in style_user.columns and "pred_style_label" in style_user.columns:
            try:
                cm_style = pd.crosstab(style_user["author_type"], style_user["pred_style_label"], dropna=False)
                cm_style.to_csv(out_dir / f"{persona}_style_confusion.csv")
                log.info("Wrote %s", str(out_dir / f"{persona}_style_confusion.csv"))
            except Exception:
                log.warning("Failed writing style confusion matrix.", exc_info=True)

    real_rows: List[Dict[str, Any]] = []
    for uid, prof in self.profiles.items():
        real_rows.append(
            dict(
                author_user_id=int(uid),
                real_question_ratio=float(prof.question_ratio),
                real_secondperson_ratio=float(prof.secondperson_ratio),
                real_caps_ratio=float(prof.caps_ratio),
                real_hedge_ratio=float(prof.hedge_ratio),
                real_intensifier_ratio=float(prof.intensifier_ratio),
                real_agreement_ratio=float(prof.agreement_ratio),
                real_disagreement_ratio=float(prof.disagreement_ratio),
                real_subjectivity_ratio=float(prof.subjectivity_ratio),
            )
        )
    real_df = pd.DataFrame(real_rows)

    if not style_user.empty and not real_df.empty:
        style_user = style_user.merge(real_df, on="author_user_id", how="left")

        def _cos(a: np.ndarray, b: np.ndarray) -> float:
            an = float(np.linalg.norm(a) + 1e-12)
            bn = float(np.linalg.norm(b) + 1e-12)
            if an <= 0.0 or bn <= 0.0:
                return float("nan")
            return float(np.dot(a, b) / (an * bn))

        _STYLE_FEATS = [
            "question_ratio", "secondperson_ratio", "caps_ratio", "hedge_ratio",
            "intensifier_ratio", "agreement_ratio", "disagreement_ratio", "subjectivity_ratio",
        ]
        cos_vals: List[float] = []
        gen_zero_vec: List[bool] = []
        real_zero_vec: List[bool] = []
        gen_sparsity: List[int] = []  # count of exactly-zero gen features per user
        for _, r in style_user.iterrows():
            a = np.asarray([r.get(f"real_{f}", np.nan) for f in _STYLE_FEATS], dtype=np.float32)
            b = np.asarray([r.get(f"gen_{f}", np.nan) for f in _STYLE_FEATS], dtype=np.float32)
            a_zero = float(np.linalg.norm(a)) < 1e-9 if np.isfinite(a).all() else False
            b_zero = float(np.linalg.norm(b)) < 1e-9 if np.isfinite(b).all() else False
            gen_zero_vec.append(b_zero)
            real_zero_vec.append(a_zero)
            gen_sparsity.append(int((np.abs(b) < 1e-9).sum()) if np.isfinite(b).all() else 8)
            # NaN out users below min_user_tokens threshold
            below_min = bool(r.get("_below_min_tokens", False))
            if below_min or not np.isfinite(a).all() or not np.isfinite(b).all():
                cos_vals.append(float("nan"))
            else:
                cos_vals.append(_cos(a, b))

        style_user["style_cosine"] = pd.to_numeric(pd.Series(cos_vals), errors="coerce")
        style_user["gen_zero_style_vec"] = gen_zero_vec
        style_user["real_zero_style_vec"] = real_zero_vec
        style_user["gen_style_sparsity"] = gen_sparsity
        style_user.to_csv(out_dir / f"{persona}_user_style_eval.csv", index=False)
        log.info("Wrote %s", str(out_dir / f"{persona}_user_style_eval.csv"))

        # ---- Per-user style cosine distribution CSV ----
        dist_cols = ["author_user_id", "author_type", "style_cosine", "style_persona_score",
                     "gen_total_words", "gen_n_posts", "gen_style_sparsity",
                     "gen_zero_style_vec", "real_zero_style_vec"]
        dist_cols = [c for c in dist_cols if c in style_user.columns]
        dist_df = style_user[dist_cols].copy()
        dist_df.to_csv(out_dir / f"{persona}_style_cosine_distribution.csv", index=False)
        log.info("Wrote %s", str(out_dir / f"{persona}_style_cosine_distribution.csv"))

        # ---- Zero-cosine diagnostic ----
        for t in ["rage", "empath"]:
            sub = style_user[style_user["author_type"] == t]
            if sub.empty:
                continue
            cos_nan = sub["style_cosine"].isna().sum()
            cos_zero = ((sub["style_cosine"].abs() < 1e-6) & sub["style_cosine"].notna()).sum()
            gzv = sub["gen_zero_style_vec"].sum() if "gen_zero_style_vec" in sub.columns else 0
            rzv = sub["real_zero_style_vec"].sum() if "real_zero_style_vec" in sub.columns else 0
            mean_sparsity = sub["gen_style_sparsity"].mean() if "gen_style_sparsity" in sub.columns else 0
            log.info("[zero-cos] %s | n=%d | cos_nan=%d cos≈0=%d | gen_zero_vec=%d real_zero_vec=%d | mean_gen_sparsity=%.1f/8",
                     t, len(sub), cos_nan, cos_zero, gzv, rzv, mean_sparsity)

        # ---- Feature-level diagnostics (real vs gen per cohort) ----
        feat_diag: Dict[str, Any] = {}
        for t in ["rage", "empath"]:
            sub = style_user[style_user["author_type"] == t]
            if sub.empty:
                continue
            fd: Dict[str, Any] = {}
            for f in _STYLE_FEATS:
                rc = f"real_{f}"
                gc = f"gen_{f}"
                r_vals = pd.to_numeric(sub.get(rc, pd.Series(dtype=float)), errors="coerce").dropna()
                g_vals = pd.to_numeric(sub.get(gc, pd.Series(dtype=float)), errors="coerce").dropna()
                fd_entry: Dict[str, Any] = dict(
                    real_mean=float(r_vals.mean()) if len(r_vals) > 0 else None,
                    gen_mean=float(g_vals.mean()) if len(g_vals) > 0 else None,
                    gen_frac_zero=float((g_vals.abs() < 1e-9).mean()) if len(g_vals) > 0 else None,
                )
                # Per-feature error metrics: MAE, RMSE, Pearson r
                paired = sub[[rc, gc]].dropna()
                if len(paired) >= 5:
                    rv = paired[rc].astype(float).values
                    gv = paired[gc].astype(float).values
                    err = gv - rv
                    fd_entry["mae"] = float(np.mean(np.abs(err)))
                    fd_entry["rmse"] = float(np.sqrt(np.mean(err ** 2)))
                    if rv.std() > 1e-9 and gv.std() > 1e-9:
                        fd_entry["pearson_r"] = float(np.corrcoef(rv, gv)[0, 1])
                    # Per-feature Cohen's d between cohorts (real→gen shift)
                    fd_entry["bias"] = float(np.mean(err))  # systematic over/under-prediction
                fd[f] = fd_entry
            feat_diag[t] = fd

        # ---- Cohort separation metrics (Cohen's d) ----
        cohort_sep: Dict[str, Any] = {}
        rage_cos = pd.to_numeric(
            style_user.loc[style_user["author_type"] == "rage", "style_cosine"], errors="coerce"
        ).dropna()
        empath_cos = pd.to_numeric(
            style_user.loc[style_user["author_type"] == "empath", "style_cosine"], errors="coerce"
        ).dropna()
        if len(rage_cos) >= 5 and len(empath_cos) >= 5:
            pooled_std = float(np.sqrt(
                ((len(rage_cos) - 1) * rage_cos.var() + (len(empath_cos) - 1) * empath_cos.var())
                / (len(rage_cos) + len(empath_cos) - 2)
            ))
            cohens_d = float((empath_cos.mean() - rage_cos.mean()) / max(pooled_std, 1e-9))
            cohort_sep["cohens_d"] = cohens_d
            cohort_sep["rage_mean"] = float(rage_cos.mean())
            cohort_sep["empath_mean"] = float(empath_cos.mean())
            cohort_sep["rage_n_valid"] = int(len(rage_cos))
            cohort_sep["empath_n_valid"] = int(len(empath_cos))
            try:
                from scipy.stats import mannwhitneyu
                stat, pval = mannwhitneyu(rage_cos.values, empath_cos.values, alternative="two-sided")
                cohort_sep["mannwhitney_U"] = float(stat)
                cohort_sep["mannwhitney_p"] = float(pval)
            except ImportError:
                pass

            # --- Enhanced effect sizes ---
            # Hedges' g (bias-corrected Cohen's d for potentially small samples)
            n1, n2 = len(rage_cos), len(empath_cos)
            correction = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
            cohort_sep["hedges_g"] = float(cohens_d * correction)

            # Cliff's delta (non-parametric effect size)
            try:
                rv, ev = rage_cos.values, empath_cos.values
                _dom = np.array([[np.sign(float(e) - float(r)) for r in rv] for e in ev])
                cohort_sep["cliffs_delta"] = float(_dom.mean())
            except Exception:
                pass

            # Probability of superiority (common language effect size)
            try:
                n_pairs = n1 * n2
                n_sup = sum(1 for e in empath_cos.values for r in rage_cos.values if float(e) > float(r))
                n_tie = sum(1 for e in empath_cos.values for r in rage_cos.values if abs(float(e) - float(r)) < 1e-9)
                cohort_sep["prob_superiority"] = float((n_sup + 0.5 * n_tie) / max(1, n_pairs))
            except Exception:
                pass

            # --- Distribution overlap metrics ---
            try:
                from scipy.stats import ks_2samp, wasserstein_distance
                cohort_sep["wasserstein_dist"] = float(wasserstein_distance(rage_cos.values, empath_cos.values))
                ks_stat, ks_p = ks_2samp(rage_cos.values, empath_cos.values)
                cohort_sep["ks_statistic"] = float(ks_stat)
                cohort_sep["ks_p"] = float(ks_p)
            except ImportError:
                pass

            # --- Bootstrap 95% CIs (1000 resamples) ---
            try:
                rng = np.random.RandomState(142)
                n_boot = 1000
                boot_d = []
                boot_rage_mean = []
                boot_empath_mean = []
                for _ in range(n_boot):
                    r_samp = rng.choice(rage_cos.values, size=n1, replace=True)
                    e_samp = rng.choice(empath_cos.values, size=n2, replace=True)
                    boot_rage_mean.append(float(r_samp.mean()))
                    boot_empath_mean.append(float(e_samp.mean()))
                    ps = float(np.sqrt(
                        ((n1 - 1) * r_samp.var(ddof=1) + (n2 - 1) * e_samp.var(ddof=1))
                        / (n1 + n2 - 2)
                    ))
                    boot_d.append(float((e_samp.mean() - r_samp.mean()) / max(ps, 1e-9)))
                cohort_sep["bootstrap_ci"] = dict(
                    cohens_d_ci95=[float(np.percentile(boot_d, 2.5)), float(np.percentile(boot_d, 97.5))],
                    rage_mean_ci95=[float(np.percentile(boot_rage_mean, 2.5)), float(np.percentile(boot_rage_mean, 97.5))],
                    empath_mean_ci95=[float(np.percentile(boot_empath_mean, 2.5)), float(np.percentile(boot_empath_mean, 97.5))],
                    n_resamples=n_boot,
                )
            except Exception:
                pass

            # --- Intra-cohort consistency ---
            cohort_sep["rage_std"] = float(rage_cos.std())
            cohort_sep["empath_std"] = float(empath_cos.std())
            # Eta-squared: between-group variance / total variance
            try:
                grand_mean = float(pd.concat([rage_cos, empath_cos]).mean())
                ss_between = n1 * (rage_cos.mean() - grand_mean) ** 2 + n2 * (empath_cos.mean() - grand_mean) ** 2
                ss_total = float(((rage_cos - grand_mean) ** 2).sum() + ((empath_cos - grand_mean) ** 2).sum())
                cohort_sep["eta_squared"] = float(ss_between / max(ss_total, 1e-12))
            except Exception:
                pass

            log.info("[cohort-sep] Cohen's d=%.3f Hedges' g=%.3f | rage=%.3f±%.3f (n=%d) empath=%.3f±%.3f (n=%d)",
                     cohens_d, cohort_sep.get("hedges_g", float("nan")),
                     rage_cos.mean(), rage_cos.std(), len(rage_cos),
                     empath_cos.mean(), empath_cos.std(), len(empath_cos))

    sent_corr = None
    if "gstat_user_sent_mean" in self.gdf.columns:
        rs = self.gdf[["target_user_id", "gstat_user_sent_mean"]].copy()
        rs = rs.rename(columns={"target_user_id": "author_user_id", "gstat_user_sent_mean": "real_sent_mean"})
        rs["author_user_id"] = pd.to_numeric(rs["author_user_id"], errors="coerce").astype("Int64")
        rs = rs.dropna(subset=["author_user_id"])
        rs["author_user_id"] = rs["author_user_id"].astype(int)
        rs["real_sent_mean"] = pd.to_numeric(rs["real_sent_mean"], errors="coerce")

        cmp_df = user_sent_out.merge(rs, on="author_user_id", how="left")
        cmp_df["sent_mean_error"] = cmp_df["gen_sent_mean"] - cmp_df["real_sent_mean"]
        cmp_df["sent_mean_abs_error"] = (cmp_df["gen_sent_mean"] - cmp_df["real_sent_mean"]).abs()
        cmp_df.to_csv(out_dir / f"{persona}_user_sentiment_compare.csv", index=False)
        log.info("Wrote %s", str(out_dir / f"{persona}_user_sentiment_compare.csv"))

        try:
            vv = cmp_df[["real_sent_mean", "gen_sent_mean"]].copy()
            vv["real_sent_mean"] = pd.to_numeric(vv["real_sent_mean"], errors="coerce")
            vv["gen_sent_mean"] = pd.to_numeric(vv["gen_sent_mean"], errors="coerce")
            vv = vv.dropna()
            if len(vv) >= 5:
                sent_corr = float(np.corrcoef(vv["real_sent_mean"].astype(float), vv["gen_sent_mean"].astype(float))[0, 1])
        except Exception:
            sent_corr = None

    out_meta: Dict[str, Any] = dict(thresholds=self.thresholds, summary=summary)
    if rel_thr is not None:
        out_meta["thresholds_relative"] = rel_thr
    if sent_corr is not None:
        out_meta["real_vs_gen_sent_corr"] = sent_corr
    if not style_user.empty and "style_cosine" in style_user.columns:
        style_summary: Dict[str, Any] = {}
        hist_edges = np.linspace(0.0, 1.0, 21)
        for t in ["rage", "empath", "neutral"]:
            sub = style_user[style_user["author_type"] == t]
            if sub.empty:
                continue
            cos_series = pd.to_numeric(sub["style_cosine"], errors="coerce").dropna()
            d: Dict[str, Any] = dict(
                users=int(len(sub)),
                mean_style_cosine=float(cos_series.mean()) if len(cos_series) > 0 else float("nan"),
            )
            if "style_persona_score" in sub.columns:
                d["mean_style_persona_score"] = float(pd.to_numeric(sub["style_persona_score"], errors="coerce").mean())
            if "pred_style_label" in sub.columns:
                match_style = (
                    (sub["author_type"] == sub["pred_style_label"]) & (sub["author_type"].isin(["rage", "empath"]))
                ).astype(int)
                d["match_style_rate"] = float(match_style.mean())
            # Generation quality stats per cohort
            if "gen_total_words" in sub.columns:
                tw = pd.to_numeric(sub["gen_total_words"], errors="coerce").dropna()
                d["gen_total_words_mean"] = float(tw.mean()) if len(tw) > 0 else None
                d["gen_total_words_median"] = float(tw.median()) if len(tw) > 0 else None
            if "gen_n_posts" in sub.columns:
                np_ = pd.to_numeric(sub["gen_n_posts"], errors="coerce").dropna()
                d["gen_n_posts_mean"] = float(np_.mean()) if len(np_) > 0 else None
            if "gen_ttr" in sub.columns:
                ttr = pd.to_numeric(sub["gen_ttr"], errors="coerce").dropna()
                d["gen_ttr_mean"] = float(ttr.mean()) if len(ttr) > 0 else None
            if "gen_mean_word_len" in sub.columns:
                mwl = pd.to_numeric(sub["gen_mean_word_len"], errors="coerce").dropna()
                d["gen_mean_word_len"] = float(mwl.mean()) if len(mwl) > 0 else None
            # Lexical diversity per cohort
            if "gen_shannon_h" in sub.columns:
                sh = pd.to_numeric(sub["gen_shannon_h"], errors="coerce").dropna()
                d["gen_shannon_h_mean"] = float(sh.mean()) if len(sh) > 0 else None
            if "gen_simpsons_d" in sub.columns:
                sd = pd.to_numeric(sub["gen_simpsons_d"], errors="coerce").dropna()
                d["gen_simpsons_d_mean"] = float(sd.mean()) if len(sd) > 0 else None
            if "gen_hapax_ratio" in sub.columns:
                hr = pd.to_numeric(sub["gen_hapax_ratio"], errors="coerce").dropna()
                d["gen_hapax_ratio_mean"] = float(hr.mean()) if len(hr) > 0 else None
            # Zero-cosine / sparsity diagnostics
            if "gen_zero_style_vec" in sub.columns:
                d["frac_gen_zero_vec"] = float(sub["gen_zero_style_vec"].mean())
            if "real_zero_style_vec" in sub.columns:
                d["frac_real_zero_vec"] = float(sub["real_zero_style_vec"].mean())
            if "gen_style_sparsity" in sub.columns:
                sp = pd.to_numeric(sub["gen_style_sparsity"], errors="coerce").dropna()
                d["mean_gen_sparsity"] = float(sp.mean()) if len(sp) > 0 else None
                d["frac_ge4_zero_feats"] = float((sp >= 4).mean()) if len(sp) > 0 else None
            # Distribution statistics
            if len(cos_series) >= 5:
                pct = cos_series.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
                d["distribution"] = dict(
                    median=float(cos_series.median()),
                    std=float(cos_series.std()),
                    min=float(cos_series.min()),
                    max=float(cos_series.max()),
                    p10=float(pct[0.10]),
                    p25=float(pct[0.25]),
                    p50=float(pct[0.50]),
                    p75=float(pct[0.75]),
                    p90=float(pct[0.90]),
                    frac_above_0_3=float((cos_series > 0.3).mean()),
                    frac_above_0_5=float((cos_series > 0.5).mean()),
                    frac_above_0_7=float((cos_series > 0.7).mean()),
                )
                counts, _ = np.histogram(cos_series.values, bins=hist_edges)
                d["distribution"]["histogram_bins"] = [float(e) for e in hist_edges.tolist()]
                d["distribution"]["histogram_counts"] = [int(c) for c in counts.tolist()]
            style_summary[t] = d
        if style_summary:
            out_meta["style_summary"] = style_summary
        if feat_diag:
            out_meta["feature_diagnostics"] = feat_diag
        if cohort_sep:
            out_meta["cohort_separation"] = cohort_sep
        # Coherence stats
        coherence_stats: Dict[str, Any] = {}
        for t in ["rage", "empath"]:
            sub_df = df[df["author_type"] == t]
            if sub_df.empty:
                continue
            coh = sub_df["text"].astype(str).apply(_is_coherent)
            coherence_stats[t] = dict(
                n_posts=int(len(sub_df)),
                n_coherent=int(coh.sum()),
                coherence_rate=float(coh.mean()),
            )
        if coherence_stats:
            out_meta["coherence_by_cohort"] = coherence_stats

    # --- Per-topic persona effect ---
    try:
        if "thread_id" in df.columns and "sent_polarity" in df.columns:
            topic_persona: Dict[str, Any] = {}
            for tid in df["thread_id"].unique():
                tdf = df[df["thread_id"] == tid]
                rage_sent = pd.to_numeric(
                    tdf.loc[tdf["author_type"] == "rage", "sent_polarity"], errors="coerce"
                ).dropna()
                empath_sent = pd.to_numeric(
                    tdf.loc[tdf["author_type"] == "empath", "sent_polarity"], errors="coerce"
                ).dropna()
                if len(rage_sent) >= 3 and len(empath_sent) >= 3:
                    topic_persona[str(tid)] = dict(
                        rage_mean=float(rage_sent.mean()),
                        empath_mean=float(empath_sent.mean()),
                        delta=float(empath_sent.mean() - rage_sent.mean()),
                        n_rage=int(len(rage_sent)),
                        n_empath=int(len(empath_sent)),
                    )
            if topic_persona:
                deltas = [v["delta"] for v in topic_persona.values()]
                topic_persona["_aggregate"] = dict(
                    mean_delta=float(np.mean(deltas)),
                    std_delta=float(np.std(deltas)),
                    n_topics=len(deltas),
                )
                # Topic swamp index: var(topic_means) / var(persona_deltas)
                topic_means = []
                for v in topic_persona.values():
                    if isinstance(v, dict) and "rage_mean" in v:
                        topic_means.append((v["rage_mean"] + v["empath_mean"]) / 2)
                if len(topic_means) >= 2:
                    topic_persona["_aggregate"]["topic_swamp_index"] = float(
                        np.var(topic_means) / max(np.var(deltas), 1e-12))
                out_meta["per_topic_persona"] = topic_persona
    except Exception:
        pass

    # --- User-level fidelity composite ---
    try:
        if not style_user.empty and "style_cosine" in style_user.columns:
            fid = style_user[["author_user_id", "author_type", "style_cosine"]].copy()
            # Merge sentiment match if available
            if "match_extreme" in user_sent_out.columns:
                sent_match = user_sent_out[["author_user_id", "match_extreme", "gen_sent_mean"]].copy()
                sent_match = sent_match.rename(columns={"match_extreme": "sent_match"})
                fid = fid.merge(sent_match, on="author_user_id", how="left")
            # Composite: style_cosine * 0.5 + sent_match * 0.5 (where available)
            cos_norm = pd.to_numeric(fid["style_cosine"], errors="coerce").clip(0, 1)
            if "sent_match" in fid.columns:
                sm = pd.to_numeric(fid["sent_match"], errors="coerce").fillna(0.0)
                fid["fidelity_composite"] = 0.5 * cos_norm + 0.5 * sm
            else:
                fid["fidelity_composite"] = cos_norm
            fid_summary: Dict[str, Any] = {}
            for t in ["rage", "empath"]:
                sub_f = fid[fid["author_type"] == t]["fidelity_composite"].dropna()
                if len(sub_f) >= 5:
                    fid_summary[t] = dict(
                        mean=float(sub_f.mean()),
                        median=float(sub_f.median()),
                        std=float(sub_f.std()),
                        frac_above_0_5=float((sub_f > 0.5).mean()),
                        frac_above_0_7=float((sub_f > 0.7).mean()),
                    )
            if fid_summary:
                out_meta["user_fidelity_composite"] = fid_summary
                log.info("[fidelity] rage: mean=%.3f | empath: mean=%.3f",
                         fid_summary.get("rage", {}).get("mean", float("nan")),
                         fid_summary.get("empath", {}).get("mean", float("nan")))
    except Exception:
        pass

    # --- Feature-sentiment correlation preservation ---
    try:
        if (not style_user.empty
                and "gstat_user_sent_mean" in self.gdf.columns):
            # Join real sentiment onto style_user
            _rsm = self.gdf[["target_user_id", "gstat_user_sent_mean"]].drop_duplicates("target_user_id")
            _rsm = _rsm.rename(columns={"target_user_id": "author_user_id", "gstat_user_sent_mean": "_real_sent"})
            _rsm["author_user_id"] = pd.to_numeric(_rsm["author_user_id"], errors="coerce").dropna().astype(int)
            _rsm["_real_sent"] = pd.to_numeric(_rsm["_real_sent"], errors="coerce")
            _fsc = style_user.merge(_rsm, on="author_user_id", how="left")
            feat_sent_corr: Dict[str, Dict[str, float]] = {}
            _STYLE_FEATS_local = [
                "question_ratio", "secondperson_ratio", "caps_ratio", "hedge_ratio",
                "intensifier_ratio", "agreement_ratio", "disagreement_ratio", "subjectivity_ratio",
            ]
            for feat in _STYLE_FEATS_local:
                real_col = f"real_{feat}"
                gen_col = f"gen_{feat}"
                sent_col = "_real_sent"
                entry: Dict[str, float] = {}
                for label, col in [("real", real_col), ("gen", gen_col)]:
                    pair = _fsc[[col, sent_col]].dropna()
                    if len(pair) >= 10:
                        r = float(np.corrcoef(
                            pd.to_numeric(pair[col], errors="coerce").values,
                            pd.to_numeric(pair[sent_col], errors="coerce").values,
                        )[0, 1])
                        entry[f"{label}_corr_with_sent"] = r
                if entry:
                    feat_sent_corr[feat] = entry
            if feat_sent_corr:
                out_meta["feature_sentiment_correlation"] = feat_sent_corr
    except Exception:
        pass

    return out_meta

# ---------------------------------------------------------------------
# Arditi Patch (Appendix V.5b): residual-stream direction abliteration
# ---------------------------------------------------------------------

# Polar-pair convention per label CSV. Used by signed-multi mode to determine
# which "tail" each user is on (low_label gets +alpha, high_label gets -alpha).
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
# CSV name -> dim_name used in safetensors keys (e.g., "dim/<dimname>/layer_<L>")
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


def _arditi_load_user_labels(label_csv_dir: Path,
                              label_files: List[str]) -> Dict[int, Dict[str, str]]:
    """Build user_id -> {dim_name: label_string_lower} from per-dim CSVs."""
    out: Dict[int, Dict[str, str]] = {}
    for fname in label_files:
        path = Path(label_csv_dir) / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "target_user_id" not in df.columns or "label" not in df.columns:
            continue
        dim_name = _ARDITI_CSV_TO_DIM.get(fname, fname.replace("labels_", "").replace(".csv", ""))
        for _, row in df.iterrows():
            uid = int(row["target_user_id"])
            lab = str(row["label"]).strip().lower()
            out.setdefault(uid, {})[dim_name] = lab
    return out


def _install_arditi_patch(self,
                          *,
                          directions_path: str,
                          alpha: float,
                          layer_spec: str,
                          mode: str = "single",
                          label_csv_dir: str = "/tmp",
                          label_files: Optional[List[str]] = None) -> None:
    """Mode-aware residual-stream direction abliteration.

    Modes (require multi-axis directions file from
    extract_arditi_directions.py --sampling multi_axis):
      single             - subtract main/<L> with uniform alpha (legacy)
      signed_sentiment   - subtract main/<L> with per-user signed alpha based
                           on sentiment cohort: rage=+alpha, empath=-alpha,
                           others=0
      signed_multi       - sum per-dim signed alpha across all 9 label dims
                           (each user's polar position on each axis flips sign)
      per_cohort_dir     - subtract cohort/<sentiment_cohort>/<L> per user
      orthogonal         - subtract main/<L> with the persona/<sentiment_cohort>/<L>
                           component projected out (per-cohort orthogonal)

    For 'single' mode, file may also be the legacy single-direction
    safetensors (keys like 'layer_<L>'); other modes require the multi-axis
    file with structured keys.
    """
    from safetensors.torch import load_file as _load_safetensors  # type: ignore

    p = Path(directions_path)
    if not p.exists():
        raise FileNotFoundError(f"Arditi directions file not found: {directions_path}")
    raw = _load_safetensors(str(p))
    if not raw:
        raise RuntimeError(f"Arditi directions file is empty: {directions_path}")

    # Locate the transformer block stack. The walk-down can land us either
    # AT the LM wrapper (e.g., GPTNeoXForCausalLM with .gpt_neox) or one
    # level deeper INSIDE the base model (e.g., GPTNeoXModel with .layers).
    # Prefer wrapper-level detection first, then fall back to layers-direct.
    obj = self.engine.backbone
    for attr in ("base_model", "model"):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
    if hasattr(obj, "gpt_neox"):
        layers = obj.gpt_neox.layers
        family = "gpt_neox"
    elif hasattr(obj, "transformer") and hasattr(obj.transformer, "h"):
        layers = obj.transformer.h
        family = "gpt2"
    elif hasattr(obj, "embed_in") and hasattr(obj, "layers"):
        # GPTNeoXModel (one level past GPTNeoXForCausalLM)
        layers = obj.layers
        family = "gpt_neox"
    elif hasattr(obj, "layers"):
        layers = obj.layers
        family = "llama"
    else:
        raise RuntimeError("Could not locate transformer block stack on backbone.")
    n_layers = len(layers)

    # Parse layer_spec
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

    device = self.engine.device

    # Group raw keys by family. Legacy "layer_<L>" keys are remapped to
    # "main/layer_<L>" so 'single' mode still works on legacy files.
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
    self.log.info("[arditi] direction families found: %s",
                  ", ".join(sorted(grouped.keys())))

    # Validate that the requested mode has the keys it needs
    mode = (mode or "single").lower()
    if mode in ("single", "signed_sentiment", "orthogonal") and "main" not in grouped:
        raise RuntimeError(f"mode={mode} requires 'main/' direction family in {p}")
    if mode == "signed_multi":
        missing = [d for d in _ARDITI_CSV_TO_DIM.values() if f"dim/{d}" not in grouped]
        if missing:
            self.log.warning("[arditi] signed_multi: missing dim families %s; "
                             "those axes will be skipped per-user", missing)
    if mode == "per_cohort_dir":
        if not any(k.startswith("cohort/") for k in grouped):
            raise RuntimeError(f"mode=per_cohort_dir requires 'cohort/' direction family")
    if mode == "orthogonal":
        if not any(k.startswith("persona/") for k in grouped):
            raise RuntimeError(f"mode=orthogonal requires 'persona/' direction family")

    # Load user labels (only for modes that need them).
    # NOTE: orthogonal_multi MUST be in this list — it uses per-user polar
    # signs to decide which axes to project out via Gram-Schmidt. Without
    # the labels the per_user_perp tensor reduces to the main direction
    # for every user and the patch is a no-op (precompute logged
    # n_users=1 fallback).
    user_labels: Dict[int, Dict[str, str]] = {}
    if mode in ("signed_sentiment", "signed_multi", "per_cohort_dir",
                "orthogonal", "orthogonal_multi"):
        labf = label_files or list(_ARDITI_CSV_TO_DIM.keys())
        user_labels = _arditi_load_user_labels(Path(label_csv_dir), labf)
        self.log.info("[arditi] loaded labels for %d users across %d dims",
                      len(user_labels), len(labf))

    # Stash on engine.model so the patched forward can read per-batch user_ids
    # set by the simulator before each generate_reply call. If absent, we fall
    # back to applying 'single' mode behavior with no per-user signs.
    self.engine.model._arditi_batch_user_ids = None
    self.engine.model._arditi_batch_uids_tensor = None

    polar_dim_names = list(_ARDITI_CSV_TO_DIM.values())

    # ------------------------------------------------------------------
    # Precompute per-user lookup tensors so the per-forward hot path is
    # a single GPU index op instead of a Python "for uid in batch_uids:"
    # loop. The naive per-forward loop turned out to be catastrophically
    # slow in 'orthogonal' and 'orthogonal_multi' modes (multi-axis cells
    # never produced their first batch in 4h+ wall on a high-memory GPU because
    # 16 simulator threads contended on the GIL while each layer hook
    # spun a 48-user x 9-dim Python loop per token).
    # ------------------------------------------------------------------
    n_polar = len(polar_dim_names)
    COHORT_KEYS = ["rage", "empath", "calm"]

    # Determine n_users_arr to size the lookup tensors. Use max(uid) over
    # both the loaded labels and the active profile pool (users we'll
    # actually generate for). The simulator stores its pool on
    # `self.profiles` (legacy code paths used `self.prof`) — accept either.
    max_uid = -1
    if user_labels:
        max_uid = max(max_uid, max(int(k) for k in user_labels.keys()))
    prof_dict = getattr(self, "profiles", None)
    if not (isinstance(prof_dict, dict) and prof_dict):
        prof_dict = getattr(self, "prof", None)
    if isinstance(prof_dict, dict) and prof_dict:
        max_uid = max(max_uid, max(int(k) for k in prof_dict.keys()))
    n_users_arr = max(max_uid + 1, 1)
    if n_users_arr <= 1 and mode in ("signed_sentiment", "signed_multi",
                                     "per_cohort_dir", "orthogonal",
                                     "orthogonal_multi"):
        self.log.warning(
            "[arditi] n_users_arr=%d (no labels and no profile pool found); "
            "per-user lookup will be empty and patch will be a no-op",
            n_users_arr,
        )

    # Per-user polar signs [n_users_arr, n_polar]: +1 lo / -1 hi / 0 unknown
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
    self._arditi_label_signs = label_signs

    # Per-user cohort idx [n_users_arr]: 0=rage, 1=empath, 2=calm, 3=fallback
    cohort_idx_arr = torch.full((n_users_arr,), 3, dtype=torch.long, device=device)
    for uid_k, labs in user_labels.items():
        sent = labs.get("sentiment", "")
        if sent in COHORT_KEYS:
            cohort_idx_arr[int(uid_k)] = COHORT_KEYS.index(sent)
    self._arditi_cohort_idx = cohort_idx_arr
    self._arditi_polar_dim_names = polar_dim_names

    # Per-layer cohort/persona/orthogonal direction stacks, indexed by
    # safetensors layer key (== ell + 1). Filled below depending on mode.
    self._arditi_cohort_dirs_by_layer: Dict[int, torch.Tensor] = {}
    self._arditi_orth_perp_by_layer: Dict[int, torch.Tensor] = {}
    self._arditi_orth_multi_by_layer: Dict[int, torch.Tensor] = {}

    # Infer hidden dim from any direction tensor
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
            # idx 3 = zero direction => no ablation for unmapped cohort
            self._arditi_cohort_dirs_by_layer[lk] = stack

    if mode == "orthogonal":
        for ell in target_layers:
            lk = ell + 1
            r_main = grouped.get("main", {}).get(lk)
            if r_main is None:
                continue
            stack = torch.zeros((4, H_dim), dtype=torch.float32, device=device)
            stack[3] = r_main  # fallback (idx 3) = plain main
            for c, cohort_name in enumerate(COHORT_KEYS):
                p = grouped.get(f"persona/{cohort_name}", {}).get(lk)
                if p is None:
                    stack[c] = r_main
                    continue
                dot = (r_main * p).sum()
                r_perp = r_main - dot * p
                n = torch.linalg.norm(r_perp).clamp_min(1e-8)
                stack[c] = r_perp / n
            self._arditi_orth_perp_by_layer[lk] = stack

    if mode == "orthogonal_multi":
        # Per-user, per-layer r_perp via Gram-Schmidt over the polar
        # axes the user has labels on. Done ONCE here, not per forward.
        # Memory: n_users_arr x H_dim per target_layer in float32.
        # 10K x 2048 x 4 bytes = 78 MB per layer; ~700 MB for 9 layers.
        polar_present_global = torch.zeros((n_polar,), dtype=torch.bool, device=device)
        active_mask = (label_signs.abs() > 0.5)  # [n_users_arr, 9] (recomputed per layer below)
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
            # Per-user iterative Gram-Schmidt subtraction
            per_user_perp = r_main.unsqueeze(0).expand(n_users_arr, -1).contiguous()
            active_layer = active_mask & polar_present.unsqueeze(0)  # [n_users_arr, 9]
            for d in range(n_polar):
                if not bool(polar_present[d].item()):
                    continue
                u = polar_stack[d]
                un = torch.linalg.norm(u).clamp_min(1e-8)
                u = u / un
                dot = (per_user_perp * u.unsqueeze(0)).sum(dim=-1, keepdim=True)  # [n_users, 1]
                gate = active_layer[:, d].to(per_user_perp.dtype).unsqueeze(-1)   # [n_users, 1]
                per_user_perp = per_user_perp - gate * dot * u.unsqueeze(0)
            norms = torch.linalg.norm(per_user_perp, dim=-1, keepdim=True).clamp_min(1e-8)
            per_user_perp = (per_user_perp / norms).contiguous()
            # Defensive: if any user row has NaN/Inf (e.g., r_main was zero
            # or all 9 polar axes collapsed Gram-Schmidt to numerical noise),
            # zero those rows so the hook becomes a no-op for that user
            # rather than poisoning the residual stream and crashing GELU.
            bad = ~torch.isfinite(per_user_perp).all(dim=-1)
            n_bad = int(bad.sum().item())
            if n_bad > 0:
                self.log.warning(
                    "[arditi] orthogonal_multi layer=%d: %d/%d user rows had NaN/Inf "
                    "after Gram-Schmidt; zeroing those rows (no-op for affected users)",
                    lk, n_bad, n_users_arr,
                )
                per_user_perp[bad] = 0.0
            # Also zero rows whose post-norm magnitude is implausibly large
            # (catches degenerate r_main + tiny norm divisor blow-up)
            row_norms = torch.linalg.norm(per_user_perp, dim=-1)
            blown = row_norms > 10.0  # unit norm should give ~1.0
            n_blown = int(blown.sum().item())
            if n_blown > 0:
                self.log.warning(
                    "[arditi] orthogonal_multi layer=%d: %d user rows had post-norm > 10 "
                    "(degenerate); zeroing", lk, n_blown,
                )
                per_user_perp[blown] = 0.0
            self._arditi_orth_multi_by_layer[lk] = per_user_perp

    self.log.info(
        "[arditi] precomputed | n_users=%d | n_polar=%d | mode_tables=%s",
        n_users_arr, n_polar,
        {
            "cohort_dirs_layers": len(self._arditi_cohort_dirs_by_layer),
            "orth_perp_layers": len(self._arditi_orth_perp_by_layer),
            "orth_multi_layers": len(self._arditi_orth_multi_by_layer),
        },
    )

    def _per_layer_main(ell):
        return grouped.get("main", {}).get(ell + 1)

    def _per_layer_dim(ell, dim_name):
        return grouped.get(f"dim/{dim_name}", {}).get(ell + 1)

    def _per_layer_cohort(ell, cohort):
        return grouped.get(f"cohort/{cohort}", {}).get(ell + 1)

    def _per_layer_persona(ell, cohort):
        return grouped.get(f"persona/{cohort}", {}).get(ell + 1)

    # Build per-mode patched forward closures
    def _make_patched(orig, ell):
        def patched(*args, **kwargs):
            outputs = orig(*args, **kwargs)
            was_tuple = isinstance(outputs, tuple)
            hs = outputs[0] if was_tuple else outputs
            if not torch.is_tensor(hs) or hs.ndim < 2:
                return outputs
            B = int(hs.shape[0])
            batch_uids = getattr(self.engine.model, "_arditi_batch_user_ids", None)

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
                # Per-row sign from sentiment cohort
                signs = torch.zeros((B,), dtype=hs.dtype, device=hs.device)
                if batch_uids is not None and len(batch_uids) == B:
                    for i, uid in enumerate(batch_uids):
                        sent = user_labels.get(int(uid), {}).get("sentiment", "")
                        if sent == "rage":
                            signs[i] = +1.0
                        elif sent == "empath":
                            signs[i] = -1.0
                        # other (calm/grumpy/mellow/unknown) -> 0
                proj = (hs * r).sum(dim=-1, keepdim=True)              # [B, T, 1]
                a_per_row = (signs * float(alpha)).view(B, 1, 1)        # [B, 1, 1]
                hs_new = hs - a_per_row * proj * r

            elif mode == "signed_multi":
                # Sum per-dim signed alpha across all 9 dims
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
                                signs[i] = +1.0   # low tail: subtract toward low
                            elif lab == hi_label:
                                signs[i] = -1.0   # high tail: add toward high
                    proj = (hs_new * r).sum(dim=-1, keepdim=True)
                    a_per_row = (signs * float(alpha)).view(B, 1, 1)
                    hs_new = hs_new - a_per_row * proj * r

            elif mode == "per_cohort_dir":
                # Per-row direction lookup by sentiment cohort
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
                R = torch.stack(r_rows, dim=0)                         # [B, H]
                proj = (hs * R.unsqueeze(1)).sum(dim=-1, keepdim=True)  # [B, T, 1]
                hs_new = hs - (float(alpha) * proj) * R.unsqueeze(1)

            elif mode == "orthogonal":
                # Vectorized: per-row r_perp lookup from precomputed
                # _arditi_orth_perp_by_layer[ell+1] table [4, H], indexed by
                # per-user cohort idx (0=rage, 1=empath, 2=calm, 3=fallback).
                #
                # 1. Bounds gate. `_arditi_cohort_idx` is sized to the Arditi
                #    label pool (real users, ids 0..N-1). Synth user ids run
                #    in the tens of millions and would index out of range,
                #    triggering an async CUDA assert that surfaces later as
                #    a "probability tensor contains inf/nan" failure at
                #    torch.multinomial. We route out-of-range uids to the
                #    fallback cohort idx (3).
                # 2. FP32 projection arithmetic. A bf16 dot product over
                #    hidden_size dimensions can overflow on the orthogonal
                #    direction and poison the residual stream with inf/nan.
                #    isfinite gates mirror the orthogonal_multi branch.
                batch_uids_t = getattr(self.engine.model, "_arditi_batch_uids_tensor", None)
                if batch_uids_t is None or int(batch_uids_t.numel()) != B:
                    return outputs
                stack = self._arditi_orth_perp_by_layer.get(ell + 1)
                if stack is None:
                    return outputs
                n_table = int(self._arditi_cohort_idx.numel())
                in_range = (batch_uids_t >= 0) & (batch_uids_t < n_table)
                safe_uids = torch.where(in_range, batch_uids_t,
                                        torch.zeros_like(batch_uids_t))
                cohort_idx_b = torch.where(
                    in_range,
                    self._arditi_cohort_idx[safe_uids],
                    torch.full_like(batch_uids_t, 3),                   # fallback
                )                                                       # [B]
                orig_dtype = hs.dtype
                hs_f = hs.to(torch.float32)
                R = stack[cohort_idx_b].to(torch.float32)               # [B, H]
                if not torch.isfinite(R).all():
                    return outputs
                proj = (hs_f * R.unsqueeze(1)).sum(dim=-1, keepdim=True)  # [B, T, 1]
                if not torch.isfinite(proj).all():
                    return outputs
                hs_new = (hs_f - (float(alpha) * proj) * R.unsqueeze(1)).to(orig_dtype)

            elif mode == "orthogonal_multi":
                # Vectorized: per-user r_perp comes from a precomputed
                # _arditi_orth_multi_by_layer[ell+1] table of shape
                # [n_users_arr, H]. Gram-Schmidt over the user's active
                # polar axes was done once at install time, not per forward.
                batch_uids_t = getattr(self.engine.model, "_arditi_batch_uids_tensor", None)
                if batch_uids_t is None or int(batch_uids_t.numel()) != B:
                    return outputs
                perp_users = self._arditi_orth_multi_by_layer.get(ell + 1)
                if perp_users is None:
                    return outputs
                # Bounds-check: any uid >= perp_users.shape[0] would CUDA-assert
                # at the indexing op. Bail if anything is out of range; logs let
                # us spot the size mismatch instead of crashing GELU downstream.
                n_rows = int(perp_users.shape[0])
                if int(batch_uids_t.max().item()) >= n_rows or int(batch_uids_t.min().item()) < 0:
                    return outputs
                R = perp_users[batch_uids_t].to(dtype=hs.dtype)        # [B, H]
                # Defensive: if R has NaN/Inf for any reason (precompute miss,
                # edge-case row), skip the hook this batch so we don't poison
                # the residual stream and crash a downstream GELU.
                if not torch.isfinite(R).all():
                    return outputs
                proj = (hs * R.unsqueeze(1)).sum(dim=-1, keepdim=True)
                if not torch.isfinite(proj).all():
                    return outputs
                hs_new = hs - (float(alpha) * proj) * R.unsqueeze(1)

            else:
                # Unknown mode: behave as 'single' with main if available; else no-op
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

    # Install patched forwards on each target layer
    self._arditi_originals: List[Tuple[Any, Any]] = []
    installed_layers: List[int] = []
    for ell in target_layers:
        # Skip layers that have no main direction (since every mode falls back
        # to main if its specific family is missing)
        if mode in ("single", "signed_sentiment", "orthogonal") and _per_layer_main(ell) is None:
            self.log.warning("[arditi] no main/<%d> direction; skipping layer %d", ell + 1, ell)
            continue
        layer = layers[ell]
        orig = layer.forward
        layer.forward = _make_patched(orig, ell)
        self._arditi_originals.append((layer, orig))
        installed_layers.append(ell)

    if not installed_layers:
        raise RuntimeError("No layers patched; check direction file keys vs --arditi_layers.")

    self._arditi_handles = []  # backward compat
    self._arditi_meta = {
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
    self.log.info(
        "[arditi] installed | mode=%s | family=%s | alpha=%.3f | layers=%s | direction families=%s",
        mode, family, float(alpha),
        ",".join(str(ell) for ell in installed_layers),
        ",".join(sorted(grouped.keys())),
    )


# ---------------------------------------------------------------------
# Class/method compatibility glue (guards against scope/indent drift)
# ---------------------------------------------------------------------

def _bind_forumsimulator_methods() -> None:
    cls = ForumSimulator
    for name in (
        "_load_or_compute_thresholds",
        "_select_user_pool",
        "_gen_text",
        "simulate_thread",
        "_build_generator_for_thread",
        "_init_thread_state",
        "_drain_avail_events",
        "simulate_threads_batched",
        "_thread_to_rows",
        "_thread_to_markdown",
        "_score_and_save_sentiment",
        "_install_arditi_patch",
        "run",
    ):
        if hasattr(cls, name):
            continue
        fn = globals().get(name, None)
        if callable(fn):
            setattr(cls, name, fn)

    for hle_name in ("generate_reply", "_generate_custom_loop", "_compute_reply_perplexity"):
        if not hasattr(HyperPEFTLoRAEngine, hle_name):
            fn = globals().get(hle_name, None)
            if callable(fn):
                setattr(HyperPEFTLoRAEngine, hle_name, fn)

_bind_forumsimulator_methods()

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Agentic forum simulation with HyperPEFT-LoRA (Pythia-1.4B backbone).")

    ap.add_argument("--author_parquet", type=str, required=True, help="Parquet with 'target_user_id' and gstat_* columns.")
    ap.add_argument("--labels_csv", type=str, default="", help="CSV from create_labels.py with target_user_id,label (or sentiment_label).")
    # Phase 2d synth forums (Phase 1.5 sidecar): optional parquet keyed
    # by target_user_id, carrying per-user `label_profile_json` + pol_{dim}
    # columns from label_synthetic_personas.py.  When provided, we emit an
    # author_label_profile.parquet sidecar alongside forum.parquet so Phase 3d
    # (score_persona_signature.py) can score realized-vs-expected
    # persona-signature drift.  Absent = no behavior change (used for Phase 2b
    # real-user forum).
    ap.add_argument("--user_metadata_parquet", type=str, default="",
                    help="Optional parquet with target_user_id + label_profile_json "
                         "(from training_scripts/label_synthetic_personas.py). "
                         "Triggers author_label_profile.parquet sidecar emission.")

    ap.add_argument("--base_model", type=str, required=True, help="HF id or local path for Pythia-1.4B.")
    ap.add_argument("--hyper_dir", type=str, required=True, help="Directory containing hypernetwork.safetensors and peft_placeholders.safetensors.")
    ap.add_argument("--use_best_ckpt", action="store_true", default=True, help="Load from hyper_dir/best/* if present.")

    ap.add_argument("--online", action="store_true", default=False, help="Allow HF downloads (otherwise local_files_only).")
    ap.add_argument("--qlora", action="store_true", default=False, help="Load base model in 4-bit (bitsandbytes).")

    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--inject_clamp", type=float, default=0.15)
    ap.add_argument("--delta_gain", type=float, default=12.0)

    ap.add_argument("--feature_clamp", type=float, default=3.0,
                    help="Clamp feature values to ±N std devs to prevent garbage outputs (0 to disable)")
    ap.add_argument("--outlier_threshold", type=float, default=4.0,
                    help="Threshold for outlier detection in std devs (0 to disable)")
    ap.add_argument("--filter_outliers", action="store_true", default=False,
                    help="Filter out users with features beyond outlier_threshold std devs")

    ap.add_argument("--g_columns", type=str, nargs="*", default=[])

    ap.add_argument("--topic", type=str, default="everyday villains")
    ap.add_argument("--sentiment_target", type=str, default="neutral", choices=["rage", "empath", "neutral"])
    ap.add_argument("--titles_path", type=str, default="", help="Optional file with one title per line.")
    ap.add_argument("--threads_from_default", type=int, default=12)
    ap.add_argument("--horizon_min", type=float, default=18 * 60.0)
    ap.add_argument("--fanout", type=str, default="5,5,3,1")
    ap.add_argument("--max_posts", type=int, default=0)

    ap.add_argument("--n_rage", type=int, default=100)
    ap.add_argument("--n_empath", type=int, default=100)

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--gamma_recency", type=float, default=1.25)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--do_sample", action="store_true", default=True)
    ap.add_argument("--top_p", type=float, default=0.90)
    ap.add_argument("--temperature", type=float, default=0.70)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--min_new_tokens", type=int, default=0)

    ap.add_argument("--cfg_scale", type=float, default=1.0,
                    help="Classifier-free guidance scale for persona (1.0=off, 1.5-2.0=moderate boost)")
    ap.add_argument("--best_of_n", type=int, default=1,
                    help="Generate N candidates per comment (1=off). Selection metric controlled by --bon_metric.")
    ap.add_argument("--bon_metric", type=str, default="perplexity",
                    choices=["perplexity", "cohort_centroid"],
                    help="Best-of-N selection: 'perplexity' (legacy, picks lowest PPL — biases neutral) "
                         "or 'cohort_centroid' (picks closest match to user's cohort style centroid). "
                         "cohort_centroid lifted surface Hedges' g by +2.05 (CI [+1.50, +2.56]) on M1 gu03 diag_27.")
    ap.add_argument("--cohort_centroid_path", type=str, default="",
                    help="Optional path to precomputed centroids JSON. If empty, centroids are built "
                         "from --author_parquet + --labels_csv at engine init.")
    ap.add_argument("--temp_anneal_tokens", type=int, default=0,
                    help="Number of initial tokens generated at lower temperature (0=off)")
    ap.add_argument("--temp_anneal_start", type=float, default=0.5,
                    help="Temperature for the first temp_anneal_tokens tokens")
    ap.add_argument("--adaptive_delta_ref_norm", type=float, default=0.0,
                    help="Reference feature-vector norm for adaptive delta scaling (0=off, try 1.0)")
    ap.add_argument("--force_zero_delta", action="store_true", default=False,
                    help="Zero-delta ablation: run generation with no hypernetwork deltas (baseline)")
    ap.add_argument("--emit_both", action="store_true", default=False,
                    help="Checkpoint was trained with --emit_both (hypernet emits both dA and dB). "
                         "If unset, auto-detected from checkpoint table shapes vs peft placeholder plan.")
    ap.add_argument("--arditi_patch", type=str, default="",
                    help="Path to arditi_directions.safetensors (Appendix V.5b). If set, registers "
                         "forward hooks on each transformer block that subtract the per-layer "
                         "dominant residual-stream direction at inference. No retraining required. "
                         "Produced by training_scripts/extract_arditi_directions.py.")
    ap.add_argument("--arditi_alpha", type=float, default=1.0,
                    help="Subtraction coefficient for the Arditi Patch. 1.0 = full Arditi protocol; "
                         "<1 = soft subtraction (preserve some dominant feature). Sweep on the "
                         "held-out coherence/persona-fidelity Pareto front if needed.")
    ap.add_argument("--arditi_layers", type=str, default="15-23",
                    help="Inclusive layer range over which to apply the patch hooks. For Pythia-1.4B "
                         "the late-layer peak from Phase 4 is L21, so default covers L15..L23. Use 'all' "
                         "to apply to every transformer block.")
    ap.add_argument("--arditi_mode", type=str, default="single",
                    choices=("single", "signed_sentiment", "signed_multi", "per_cohort_dir",
                             "orthogonal", "orthogonal_multi"),
                    help="Patch application mode. 'single' uses main/<L> with uniform alpha "
                         "(legacy). 'signed_sentiment' flips alpha sign per user based on "
                         "sentiment cohort (rage=+, empath=-). 'signed_multi' sums per-dim signed "
                         "alpha across all 9 label dimensions. 'per_cohort_dir' uses per-cohort "
                         "direction extracted from rage/empath/calm subsets. 'orthogonal' "
                         "subtracts main with the SENTIMENT cohort persona direction projected "
                         "out. 'orthogonal_multi' projects out the FULL 9-axis polar subspace per "
                         "user (Gram-Schmidt-style across every dim where the user has a polar "
                         "label) so multi-label persona signal is preserved. All modes except "
                         "'single' require the multi-axis directions file produced by "
                         "--sampling multi_axis.")
    ap.add_argument("--arditi_label_csv_dir", type=str, default="/tmp",
                    help="Directory containing per-dimension label CSVs (used by signed_*, "
                         "per_cohort_dir, and orthogonal modes for per-user label lookup).")
    ap.add_argument("--arditi_label_files", type=str,
                    default="labels_sentiment_goemo.csv,labels_politeness.csv,labels_self_focus.csv,"
                            "labels_curiosity.csv,labels_expressiveness.csv,labels_tempo.csv,"
                            "labels_anxiety.csv,labels_warmth.csv,labels_hostility.csv",
                    help="Comma-separated per-dim label CSV filenames in --arditi_label_csv_dir.")

    ap.add_argument("--threshold_source", type=str, required=True, help="Parquet/CSV used to compute create_labels thresholds (usually global_features_10000.parquet).")
    ap.add_argument("--threshold_col", type=str, default="gstat_user_sent_mean")
    ap.add_argument("--thresholds_json", type=str, default="", help="Optional precomputed thresholds (q20/q40/q60/q80/q100).")
    ap.add_argument("--norm_stats_json", type=str, default="", help="Optional feature_norm_stats_*.json used to convert z-scored thresholds back to raw units.")
    ap.add_argument("--disable_sentiment_eval", action="store_true", default=False, help="Disable post-generation sentiment scoring.")
    ap.add_argument("--min_user_tokens", type=int, default=0,
                    help="Exclude users with fewer total generated words from style cosine eval (0=no filter). "
                         "Addresses zero-cosine wall caused by sparse lexicon features on short text.")

    ap.add_argument("--use_context_token", action="store_true", default=False, help="Prefix prompts with <|context|> (only if it was used during training).")

    ap.add_argument(
        "--infer_batch_size",
        type=int,
        default=0,
        help=(
            "If > 1, simulate all threads concurrently and batch up to N user posts "
            "per generate() call to saturate the GPU. 0/1 = legacy serial per-thread path."
        ),
    )

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=142)

    args = ap.parse_args()
    parts = [p.strip() for p in args.fanout.split(",") if p.strip()]
    args.fanout = [int(x) for x in parts] if parts else [5, 5, 3, 1]
    return args


if __name__ == "__main__":
    args = parse_args()
    sim = ForumSimulator(args)
    sim.run()