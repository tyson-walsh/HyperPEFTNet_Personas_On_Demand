#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_vanillalora_forum.py — Vanilla LoRA Forum Baseline Generator
===================================================================

Generates simulated forum threads using a vanilla LoRA model (no hypernetwork,
no per-user conditioning). Same thread structure, topics, user pools, and
decoding parameters as build_hyperlora_forum.py, enabling direct comparison
for RQ2 evaluation.

Key difference from HyperPEFT-LoRA forum builder:
  • No hypernetwork / delta injection — all users share identical model weights
  • No per-user feature vectors (gvec)
  • Any "persona signal" in the output is pure noise (sampling + context)

This establishes the "no persona" floor for RQ2. Any persona metrics
(Cohen's d, style cosine, match_extreme_rate) measured on these outputs
represent the null hypothesis that HyperPEFT-LoRA must exceed.
"""

from __future__ import annotations

import os
import sys
import gc
import json
import time
import math
import heapq
import random
import re
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

try:
    from arditi_install_util import install_arditi_on_model
except Exception:  # pragma: no cover - degrade gracefully if util missing
    install_arditi_on_model = None  # type: ignore[assignment]

LOG = logging.getLogger("vanillalora_forum")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =====================================================================
# Constants — identical to build_hyperlora_forum.py
# =====================================================================

REPLY_SEP_TOKEN = "<|reply|>"
REPLY_END_TOKEN = "<|eoreply|>"
CONTEXT_SEP_TOKEN = "<|context|>"

DEFAULT_TITLES_RAGE = [
    "my friend creates hideous, inappropriate art—i tell them it's bad so they stop wasting time",
    "i flirt with customers specifically to get better tips",
    "i lied to skip a wedding i don't care about",
    "i pretend not to see homeless people because i'm late",
    "i sabotaged a lazy coworker's project so management notices",
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

# =====================================================================
# Utilities — shared with build_hyperlora_forum.py
# =====================================================================

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
    """Clean generated text — identical to build_hyperlora_forum.py."""
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

    t = re.sub(r"^#\d+(?:\.\d+)*[.):;,\s]*\s*", "", t)
    t = re.sub(r"^(?:Question|Q)\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bAnswer\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[link\]\s*", "", t)
    t = re.sub(r"\[<URL\]\s*", "", t)
    t = re.sub(r"\(\[link\]\)\s*", "", t)
    t = re.sub(r"^Tags\s*:\s*\*?\w+\*?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^This topic has \d+ repl.*?ago by .+?\.\s*", "", t, flags=re.IGNORECASE)

    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r",\s*,+", ", ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"([?!])[,.:;]+$", r"\1", t).strip()
    t = re.sub(r"[,;:]+$", "", t).strip()
    return t


def _is_coherent(text: str, min_alpha_frac: float = 0.40, min_words: int = 3,
                 min_ascii_frac: float = 0.70) -> bool:
    """Heuristic — returns False for garbled/gibberish text."""
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


# =====================================================================
# Data structures — identical to build_hyperlora_forum.py
# =====================================================================

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


@dataclass
class UserProfile:
    """Simplified profile — no gvec (no hypernetwork features needed)."""
    user_id: int
    author_type: str
    post_rate_per_day: float
    hour_hist: np.ndarray
    reply_delay_mu: float
    reply_delay_sigma: float
    question_ratio: float
    agreement_ratio: float
    disagreement_ratio: float
    secondperson_ratio: float
    caps_ratio: float
    hedge_ratio: float
    topic_affinity: float
    cooldown_min: float
    daily_cap: int
    last_post_min: float = -1e9
    posts_today: int = 0
    next_reset_min: float = 24 * 60.0


# =====================================================================
# Forum dynamics — identical to build_hyperlora_forum.py
# =====================================================================

class Decider:
    """Decide posting probability and reply target."""

    def __init__(self, profiles: Dict[int, UserProfile], rng: np.random.Generator,
                 topic_mode: str) -> None:
        self.prof = profiles
        self.rng = rng
        self.topic_mode = topic_mode

        self.beta = dict(b0=-0.3, novelty=0.6, unanswered=0.5, mention=1.0,
                         depth_bias=-0.35, recency=0.8, topic=0.5)
        self.alpha = dict(top_base=0.8, top_q=1.2, top_depth_pressure=-0.8)
        self.action_w = dict(recency=1.2, is_q=0.4, stance=0.6, popularity=0.25,
                             mention_me=1.4, depth=-0.35)
        self.mention_caps = 0.4

    def p_post(self, uid: int, state: ThreadState, now_min: float) -> float:
        u = self.prof[uid]
        n_posts = len(state.nodes)
        novelty = 1.0 / math.sqrt(max(1.0, n_posts))
        unanswered = (sum(1 for n in state.nodes.values() if n.unanswered) / max(1, n_posts)
                      if n_posts > 0 else 1.0)

        mention_to_u = 0.0
        for n in state.nodes.values():
            if (now_min - n.created_min) < 90.0 and (uid in n.mentions):
                mention_to_u = 1.0
                break

        depth_bias = 0.0
        if n_posts > 0:
            depth_bias = float(np.mean([nd.depth for nd in state.nodes.values()])) / 4.0

        newish = sum(1 for n in state.nodes.values() if (now_min - n.created_min) <= 30.0)
        recency_score = 1.0 - math.exp(-0.2 * newish)
        topic = (u.topic_affinity + 1.0) * 0.5

        z = (self.beta["b0"]
             + self.beta["novelty"] * novelty
             + self.beta["unanswered"] * unanswered
             + self.beta["mention"] * mention_to_u
             + self.beta["depth_bias"] * depth_bias
             + self.beta["recency"] * recency_score
             + self.beta["topic"] * topic)

        if self.topic_mode == "rage":
            z += 0.35 if u.author_type == "rage" else (-0.25 if u.author_type == "empath" else 0.0)
        elif self.topic_mode == "empath":
            z += 0.35 if u.author_type == "empath" else (-0.25 if u.author_type == "rage" else 0.0)

        return _sigmoid(z)

    def choose_action(self, uid: int, state: ThreadState, now_min: float,
                      fanout_caps: Dict[int, int]) -> Tuple[str, Optional[int]]:
        u = self.prof[uid]
        root_cap = int(fanout_caps.get(0, 0) or 0)
        root_children = len(state.nodes[0].children) if 0 in state.nodes else 0
        root_full = bool(root_cap > 0 and root_children >= root_cap)

        depth_pressure = 0.0
        if state.nodes:
            avg_depth = float(np.mean([n.depth for n in state.nodes.values()]))
            depth_pressure = min(1.0, max(0.0, (avg_depth - 1.0) / 3.0))

        p_top_lin = (self.alpha["top_base"]
                     + self.alpha["top_q"] * u.question_ratio
                     + self.alpha["top_depth_pressure"] * depth_pressure)

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
            rec = math.exp(-(now_min - node.created_min) / 60.0)
            isq = 1.0 if _is_question(node.text) else 0.0
            pop = math.log(1.0 + node.popularity)
            depth = float(node.depth)
            mention_me = 1.0 if (uid in node.mentions) else 0.0
            stance = u.agreement_ratio - u.disagreement_ratio

            s = (self.action_w["recency"] * rec
                 + self.action_w["is_q"] * isq
                 + self.action_w["stance"] * stance
                 + self.action_w["popularity"] * pop
                 + self.action_w["mention_me"] * mention_me
                 + self.action_w["depth"] * depth)
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


# =====================================================================
# Event scheduler — identical to build_hyperlora_forum.py
# =====================================================================

@dataclass(order=True)
class Event:
    time_min: float
    kind: str
    uid: int = field(compare=False, default=-1)
    target_cid: Optional[int] = field(compare=False, default=None)


class Scheduler:
    def __init__(self, profiles: Dict[int, UserProfile], decider: Decider,
                 start_min: float, horizon_min: float,
                 rng: np.random.Generator) -> None:
        self.prof = profiles
        self.decider = decider
        self.start_min = float(start_min)
        self.horizon = float(horizon_min)
        self.rng = rng
        self.queue: List[Event] = []

    def _sample_next_availability(self, uid: int, cur_min: float) -> float:
        u = self.prof[uid]
        hmax = float(np.max(u.hour_hist))
        lam_max = (u.post_rate_per_day * hmax) / 60.0 + 1e-6
        t = cur_min
        while True:
            dt = self.rng.exponential(1.0 / lam_max)
            t += dt
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

    def run(self, thread: ThreadState, fanout_caps: Dict[int, int],
            generator_fn, max_posts: int) -> None:
        self.prime(self.prof.keys(), thread.created_min)
        next_cid = 1

        # OP node (the thread title as the root "post")
        op_uid = int(self.rng.choice(list(self.prof.keys())))
        op_type = self.prof[op_uid].author_type if op_uid in self.prof else "neutral"
        op_node = CommentNode(
            cid=0, parent_cid=None, depth=-1,
            author_user_id=op_uid, author_type=op_type,
            text=thread.title.strip(), created_min=thread.created_min,
            path=tuple(), children=[], mentions=[],
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
                if (now - u.last_post_min) < u.cooldown_min or u.posts_today >= u.daily_cap:
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

                dt = float(self.rng.lognormal(
                    mean=self.prof[ev.uid].reply_delay_mu,
                    sigma=self.prof[ev.uid].reply_delay_sigma))
                post_time = now + dt
                heapq.heappush(self.queue, Event(
                    time_min=post_time, kind="post", uid=ev.uid, target_cid=parent.cid))
                self._schedule_next_for(ev.uid, now)

            elif ev.kind == "post":
                if ev.target_cid not in thread.nodes:
                    continue
                parent = thread.nodes[ev.target_cid]
                child_depth = parent.depth + 1
                if fanout_caps.get(child_depth, 0) <= len(parent.children):
                    continue

                text, mentions = generator_fn(ev.uid, None if parent.cid == 0 else parent.cid,
                                              ev.time_min)
                text = (text or "").strip()
                if not text:
                    continue

                node = CommentNode(
                    cid=next_cid,
                    parent_cid=(None if parent.cid == 0 else parent.cid),
                    depth=child_depth,
                    author_user_id=int(ev.uid),
                    author_type=str(self.prof[ev.uid].author_type),
                    text=text, created_min=ev.time_min,
                    path=(parent.path + (parent.cid,)) if parent.cid != 0 else tuple(),
                    children=[], mentions=[int(m) for m in mentions or []],
                )
                thread.add_node(node)
                next_cid += 1
                u = self.prof[ev.uid]
                u.last_post_min = ev.time_min
                u.posts_today += 1


# =====================================================================
# Vanilla LoRA engine — loads standard PEFT model, no hypernetwork
# =====================================================================

class VanillaLoRAEngine:
    """Load base model + LoRA adapter for standard (non-personalized) inference."""

    def __init__(
        self,
        base_model_id: str,
        lora_dir: str,
        *,
        qlora: bool = True,
        online: bool = False,
        hf_token: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                 else torch.device("cpu"))

        # --- Tokenizer ---
        tok_dir = Path(lora_dir) / "tokenizer"
        tok_src = str(tok_dir) if tok_dir.exists() else base_model_id
        tok_kwargs: Dict[str, Any] = {}
        if not online:
            tok_kwargs["local_files_only"] = True
        if hf_token:
            tok_kwargs["token"] = hf_token
        self.tok = AutoTokenizer.from_pretrained(tok_src, **tok_kwargs)

        # Resolve special token IDs
        self.sep_id = self.tok.convert_tokens_to_ids(REPLY_SEP_TOKEN)
        self.end_id = self.tok.convert_tokens_to_ids(REPLY_END_TOKEN)
        self.ctx_id = self.tok.convert_tokens_to_ids(CONTEXT_SEP_TOKEN)
        if isinstance(self.sep_id, list):
            self.sep_id = self.sep_id[0] if self.sep_id else None
        if isinstance(self.end_id, list):
            self.end_id = self.end_id[0] if self.end_id else None
        if isinstance(self.ctx_id, list):
            self.ctx_id = self.ctx_id[0] if self.ctx_id else None

        LOG.info("Special tokens: sep=%s end=%s ctx=%s", self.sep_id, self.end_id, self.ctx_id)

        # --- Base model ---
        mdl_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if not online:
            mdl_kwargs["local_files_only"] = True
        if hf_token:
            mdl_kwargs["token"] = hf_token
        if qlora:
            mdl_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            mdl_kwargs["device_map"] = {"": self.device}
        else:
            mdl_kwargs["device_map"] = {"": self.device}

        LOG.info("Loading base model: %s (qlora=%s)", base_model_id, qlora)
        self.backbone = AutoModelForCausalLM.from_pretrained(base_model_id, **mdl_kwargs)

        # Resize embeddings if tokenizer has more tokens than model
        if len(self.tok) > self.backbone.get_input_embeddings().weight.shape[0]:
            self.backbone.resize_token_embeddings(len(self.tok))
            LOG.info("Resized embeddings to %d tokens", len(self.tok))

        # --- LoRA adapter ---
        LOG.info("Loading LoRA adapter from: %s", lora_dir)
        self.model = PeftModel.from_pretrained(
            self.backbone, lora_dir,
            is_trainable=False,
        )
        self.model.eval()

        # Build EOS token list for generation
        self.eos_ids: List[int] = []
        if self.end_id is not None and int(self.end_id) >= 0:
            self.eos_ids.append(int(self.end_id))
        if self.tok.eos_token_id is not None:
            self.eos_ids.append(int(self.tok.eos_token_id))
        self.eos_ids = sorted(set(self.eos_ids))

        n_params = sum(p.numel() for p in self.model.parameters())
        LOG.info("Model loaded: %d params, device=%s", n_params, self.device)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 96,
        do_sample: bool = True,
        top_p: float = 0.90,
        temperature: float = 0.70,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        """Standard HF generate — no delta injection."""
        eos = self.eos_ids[0] if len(self.eos_ids) == 1 else (self.eos_ids or None)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos,
            pad_token_id=self.tok.eos_token_id,
        )
        if int(no_repeat_ngram_size) > 0:
            gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)

        with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            out = self.model.generate(**gen_kwargs)
        return out


# =====================================================================
# Branch generator — encodes context + generates reply (no delta)
# =====================================================================

class BranchGenerator:
    """Encode thread context and call vanilla LoRA for generation."""

    def __init__(
        self,
        engine: VanillaLoRAEngine,
        thread: ThreadState,
        *,
        max_len: int = 512,
        gamma_recency: float = 1.25,
        max_new_tokens: int = 96,
        do_sample: bool = True,
        top_p: float = 0.90,
        temperature: float = 0.70,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.engine = engine
        self.tok = engine.tok
        self.thread = thread
        self.max_len = max_len
        self.gamma = gamma_recency
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.device = device or engine.device

        self.sep_id = engine.sep_id
        self.end_id = engine.end_id
        self.ctx_id = engine.ctx_id

    def _segment_tokens(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _encode_branch(self, parent_cid: Optional[int]) -> Tuple[List[int], List[int]]:
        """Encode context: topic + title + parent chain + <|reply|> separator."""
        segments: List[str] = []

        t_topic = self.thread.topic.strip() if self.thread.topic else ""
        if t_topic:
            segments.append(t_topic)
        segments.append(self.thread.title.strip())

        if parent_cid is not None and parent_cid in self.thread.nodes:
            path_ids: List[int] = []
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

        # Build prefix: [BOS]
        prefix: List[int] = []
        bos = getattr(self.tok, "bos_token_id", None)
        if bos is not None and int(bos) >= 0:
            prefix.append(int(bos))

        # Anchor in "Reddit-reply mode" by appending an explicit Reply: marker
        # before SEP.  Mirrors the same fix in build_hyperlora_forum.py
        # _encode_branch (2026-05-03).  Without this anchor the model
        # regurgitates user-distribution Pile content (Python tutorials,
        # ad copy, license headers) instead of engaging with the prompted
        # topic.  See Appendix S of the technical reference.
        reply_marker = self._segment_tokens("\n\nReply:")
        max_body = int(max(0, (self.max_len - 1 - len(reply_marker)) - len(prefix)))
        if max_body > 0 and len(body) > max_body:
            body = body[-max_body:]
        elif max_body <= 0:
            body = []

        toks = prefix + body
        if reply_marker:
            toks.extend(reply_marker)
        toks.append(self.sep_id)  # <|reply|>
        attn = [1] * len(toks)
        return toks, attn

    def __call__(self, uid: int, parent_cid: Optional[int]) -> List[int]:
        toks, attn = self._encode_branch(parent_cid)
        input_ids = torch.tensor([toks], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attn], dtype=torch.long, device=self.device)

        # If the Arditi patch is installed on the underlying model, set the
        # per-batch UID slots BEFORE generate(). The patched per-layer hooks
        # read these to look up the right per-user / per-cohort direction.
        # No-op if the slots are not present (no patch installed).
        try:
            if hasattr(self.engine.model, "_arditi_orth_perp_by_layer") \
                    or hasattr(self.engine.model, "_arditi_orth_multi_by_layer") \
                    or hasattr(self.engine.model, "_arditi_cohort_dirs_by_layer"):
                self.engine.model._arditi_batch_user_ids = [int(uid)]
                self.engine.model._arditi_batch_uids_tensor = torch.tensor(
                    [int(uid)], dtype=torch.long, device=self.device,
                )
        except Exception:
            try:
                self.engine.model._arditi_batch_user_ids = None
                self.engine.model._arditi_batch_uids_tensor = None
            except Exception:
                pass

        prompt_len = len(toks)
        out_ids = self.engine.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )
        return out_ids[0].tolist()[prompt_len:]


# =====================================================================
# Forum builder — main orchestration
# =====================================================================

class VanillaLoRAForumBuilder:

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        _set_seed(args.seed)

        # --- Resolve HF token ---
        hf_token = None
        if args.hf_token_file and Path(args.hf_token_file).exists():
            hf_token = Path(args.hf_token_file).read_text().strip()

        # --- Load model ---
        lora_dir = args.lora_dir
        if args.use_best_ckpt:
            best = Path(lora_dir) / "best"
            if best.exists():
                lora_dir = str(best)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = VanillaLoRAEngine(
            base_model_id=args.base_model,
            lora_dir=lora_dir,
            qlora=args.qlora,
            online=args.online,
            hf_token=hf_token,
            device=self.device,
        )

        # --- Load user pool ---
        self.profiles, self.user_ids = self._build_profiles(args)
        LOG.info("User pool: %d users (%d rage, %d empath, %d neutral)",
                 len(self.user_ids),
                 sum(1 for p in self.profiles.values() if p.author_type == "rage"),
                 sum(1 for p in self.profiles.values() if p.author_type == "empath"),
                 sum(1 for p in self.profiles.values() if p.author_type == "neutral"))

        # --- Optional: install Arditi residual-stream patch ---
        # This is what makes the M5/M6 factorial cells distinct (Arditi ON / OFF
        # against the vanilla LoRA baseline). When --arditi_patch is empty we
        # skip the install entirely and the patch is a no-op for the run.
        ar_path = str(getattr(args, "arditi_patch", "") or "").strip()
        if ar_path:
            if install_arditi_on_model is None:
                LOG.warning(
                    "[arditi] arditi_install_util.install_arditi_on_model is "
                    "not importable; skipping --arditi_patch=%s", ar_path)
            else:
                try:
                    install_arditi_on_model(
                        self.engine.model,
                        directions_path=ar_path,
                        alpha=float(getattr(args, "arditi_alpha", 1.0)),
                        layer_spec=str(getattr(args, "arditi_layers", "15-23")),
                        mode=str(getattr(args, "arditi_mode", "orthogonal")),
                        label_csv_dir=str(getattr(args, "arditi_label_csv_dir", "/tmp")),
                        log=LOG,
                        profiles_uids=list(self.profiles.keys()),
                        device=self.device,
                    )
                except Exception as e:
                    LOG.error("[arditi] install failed: %s -- continuing without patch", e)
        else:
            LOG.info("[arditi] no --arditi_patch supplied; running without residual-stream patch")

    def _build_profiles(self, args) -> Tuple[Dict[int, UserProfile], List[int]]:
        """Select user pool and build simplified profiles (no gvec)."""
        author_df = pd.read_parquet(args.author_parquet)
        if "target_user_id" not in author_df.columns:
            raise KeyError("author_parquet must include 'target_user_id'")

        # --- Select rage/empath pools ---
        labels: Dict[int, str] = {}
        if args.labels_csv and Path(args.labels_csv).exists():
            ldf = pd.read_csv(args.labels_csv)
            for _, row in ldf.iterrows():
                uid = int(row["target_user_id"])
                lab = str(row.get("label", "neutral")).strip().lower()
                labels[uid] = lab
        else:
            # Fallback: use sentiment column
            col = args.threshold_col
            if col in author_df.columns:
                s = pd.to_numeric(author_df[col], errors="coerce")
                sorted_df = author_df.loc[s.dropna().index].copy()
                sorted_df["_sent"] = s.loc[sorted_df.index]
                sorted_df = sorted_df.sort_values("_sent")
                for _, row in sorted_df.head(args.n_rage).iterrows():
                    labels[int(row["target_user_id"])] = "rage"
                for _, row in sorted_df.tail(args.n_empath).iterrows():
                    labels[int(row["target_user_id"])] = "empath"

        rage_uids = [uid for uid, lab in labels.items() if lab == "rage"]
        empath_uids = [uid for uid, lab in labels.items() if lab == "empath"]

        # Sort by sentiment extremity if possible
        sent_col = args.threshold_col
        if sent_col in author_df.columns:
            uid_to_sent = dict(zip(
                author_df["target_user_id"].astype(int),
                pd.to_numeric(author_df[sent_col], errors="coerce").fillna(0.0)))
            rage_uids.sort(key=lambda u: uid_to_sent.get(u, 0.0))
            empath_uids.sort(key=lambda u: uid_to_sent.get(u, 0.0), reverse=True)

        rage_uids = rage_uids[:args.n_rage]
        empath_uids = empath_uids[:args.n_empath]

        # Remove overlap
        overlap = set(rage_uids) & set(empath_uids)
        empath_uids = [u for u in empath_uids if u not in overlap]

        pool_uids = rage_uids + empath_uids
        author_types: Dict[int, str] = {}
        for u in rage_uids:
            author_types[u] = "rage"
        for u in empath_uids:
            author_types[u] = "empath"

        # Build simplified profiles
        uid_set = set(pool_uids)
        pool_df = author_df[author_df["target_user_id"].isin(uid_set)].copy()

        def _get(row, candidates, default):
            for c in candidates:
                if c in row and pd.notnull(row[c]):
                    try:
                        return float(row[c])
                    except Exception:
                        pass
            return float(default)

        profiles: Dict[int, UserProfile] = {}
        for _, row in pool_df.iterrows():
            uid = int(row["target_user_id"])
            atype = author_types.get(uid, "neutral")

            post_rate = _get(row, ["gstat_post_rate", "gstat_posts_per_day"], 2.0)
            post_rate = max(0.5, min(20.0, post_rate))

            hour_hist = np.ones(24, dtype=np.float32) / 24.0
            for c in row.index:
                if c.startswith("gstat_psage") or c.startswith("gstat_hour"):
                    v = row[c]
                    if isinstance(v, (list, tuple, np.ndarray)):
                        a = np.asarray(v, dtype=np.float32).ravel()
                        if a.size == 24:
                            hour_hist = a
                            break

            hsum = hour_hist.sum()
            if hsum > 0:
                hour_hist = hour_hist / hsum
            else:
                hour_hist = np.ones(24, dtype=np.float32) / 24.0

            profiles[uid] = UserProfile(
                user_id=uid,
                author_type=atype,
                post_rate_per_day=post_rate,
                hour_hist=hour_hist,
                reply_delay_mu=math.log(max(1.0, _get(row, ["gstat_reply_delay_median"], 30.0))),
                reply_delay_sigma=0.7,
                question_ratio=_get(row, ["gstat_question_frac", "gstat_question"], 0.1),
                agreement_ratio=_get(row, ["gstat_agreement_frac"], 0.2),
                disagreement_ratio=_get(row, ["gstat_disagreement_frac"], 0.2),
                secondperson_ratio=_get(row, ["gstat_secondperson_frac", "gstat_secondperson"], 0.1),
                caps_ratio=_get(row, ["gstat_caps_frac", "gstat_caps"], 0.02),
                hedge_ratio=_get(row, ["gstat_hedge_frac", "gstat_hedge"], 0.1),
                topic_affinity=0.0,  # No topic affinity for vanilla baseline
                cooldown_min=max(5.0, min(120.0, 60.0 / max(0.1, post_rate))),
                daily_cap=max(5, min(50, int(post_rate * 3))),
            )

        return profiles, pool_uids

    def simulate_thread(self, gid: int, title: str, topic: str,
                        topic_mode: str) -> ThreadState:
        start_min = 0.0
        horizon = float(self.args.horizon_min)

        thread = ThreadState(gid=int(gid), title=title.strip(), topic=topic.strip(),
                             created_min=start_min)

        caps: Dict[int, int] = {}
        for d, c in enumerate(self.args.fanout):
            caps[d] = int(c)

        decider = Decider(self.profiles, self.rng, topic_mode=topic_mode)
        sched = Scheduler(self.profiles, decider, start_min=start_min,
                          horizon_min=horizon, rng=self.rng)

        generator = BranchGenerator(
            engine=self.engine,
            thread=thread,
            max_len=self.args.max_len,
            gamma_recency=self.args.gamma_recency,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=self.args.do_sample,
            top_p=self.args.top_p,
            temperature=self.args.temperature,
            repetition_penalty=self.args.repetition_penalty,
            no_repeat_ngram_size=int(getattr(self.args, "no_repeat_ngram_size", 0)),
            device=self.device,
        )

        def _gen(uid: int, parent_cid: Optional[int], now_min: float) -> Tuple[str, List[int]]:
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
            if not _is_coherent(text):
                text = ""

            if prefix and text:
                text = prefix + text
            return text, mention_list

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
            rows.append(dict(
                gid=int(th.gid), subject=str(subject), thread_title=str(th.title),
                comment_id=int(nd.cid),
                parent_comment_id=(None if nd.parent_cid is None else int(nd.parent_cid)),
                depth=int(nd.depth), author_user_id=int(nd.author_user_id),
                author_type=str(nd.author_type), created_min=float(nd.created_min),
                mentions=[int(x) for x in nd.mentions], text=str(nd.text),
                path=[int(x) for x in nd.path], popularity=int(len(nd.children)),
            ))
        rows.sort(key=lambda r: (r["created_min"], r["comment_id"]))
        return rows

    def _thread_to_markdown(self, th: ThreadState, subject: str) -> str:
        lines: List[str] = [f"# r/{subject}: {th.title}", ""]

        def walk(cid: int, indent: int) -> None:
            nd = th.nodes[cid]
            if nd.cid != 0:
                lines.append("  " * indent + f"- u{nd.author_user_id} [{nd.author_type}]: {nd.text}")
            for ch in nd.children:
                walk(ch, indent + (0 if nd.cid == 0 else 1))

        walk(0, 0)
        return "\n".join(lines) + "\n"

    def run(self) -> None:
        out_dir = Path(self.args.out_dir)
        _ensure_dir(out_dir)

        persona = str(self.args.sentiment_target or "neutral").strip().lower()
        if persona not in ("rage", "empath", "neutral"):
            persona = "neutral"

        if persona == "rage":
            default_titles = DEFAULT_TITLES_RAGE
        elif persona == "empath":
            default_titles = DEFAULT_TITLES_EMPATH
        else:
            default_titles = DEFAULT_TITLES_NEUTRAL

        titles: List[str] = list(default_titles)
        n = int(self.args.threads_from_default or len(default_titles))
        titles = titles[:n]

        topic = str(self.args.topic or "").strip()
        raw_subject = topic.lower().strip() if topic else persona
        subject = re.sub(r"[^a-z0-9]+", "_", raw_subject).strip("_") or persona

        LOG.info("VANILLA LORA BASELINE — no per-user conditioning")
        LOG.info("Config | persona=%s | subject=%s | n_threads=%d | n_users=%d",
                 persona, subject, len(titles), len(self.user_ids))
        LOG.info("Generation | max_len=%d | max_new_tokens=%d | top_p=%.3f | temp=%.3f",
                 self.args.max_len, self.args.max_new_tokens, self.args.top_p,
                 self.args.temperature)

        all_rows: List[Dict[str, Any]] = []
        md_chunks: List[str] = []

        for i, title in enumerate(titles):
            t0 = _now()
            LOG.info("Thread %d/%d: %s", i + 1, len(titles), title)
            th = self.simulate_thread(gid=i, title=title, topic=topic, topic_mode=persona)
            all_rows.extend(self._thread_to_rows(th, subject=subject))
            md_chunks.append(self._thread_to_markdown(th, subject=subject))
            md_chunks.append("\n---\n\n")
            n_comments = max(0, len(th.nodes) - 1)
            LOG.info("Thread %d done | comments=%d | %.1fs", i + 1, n_comments, _now() - t0)

        df = pd.DataFrame(all_rows)
        if df.empty:
            df = pd.DataFrame(columns=[
                "gid", "subject", "thread_title", "comment_id", "parent_comment_id",
                "depth", "author_user_id", "author_type", "created_min", "mentions",
                "text", "path", "popularity"])

        # --- Save outputs ---
        forum_parquet = out_dir / "forum.parquet"
        df.to_parquet(forum_parquet, index=False)
        LOG.info("Saved %s (%d rows)", forum_parquet, len(df))

        forum_md = out_dir / "forum.md"
        forum_md.write_text("".join(md_chunks), encoding="utf-8")

        forum_jsonl = out_dir / "forum.jsonl"
        with open(forum_jsonl, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, default=str) + "\n")

        # --- Coherence stats ---
        if not df.empty and "text" in df.columns:
            coherent = df["text"].apply(_is_coherent)
            frac = coherent.mean()
            LOG.info("Coherence: %.1f%% (%d/%d)", frac * 100, coherent.sum(), len(coherent))
        else:
            frac = 0.0

        # --- Metadata ---
        type_counts = {}
        if not df.empty and "author_type" in df.columns:
            type_counts = df["author_type"].value_counts().to_dict()

        metadata = {
            "model_type": "vanilla_lora",
            "note": "No per-user conditioning. All users share identical model weights.",
            "persona": persona,
            "subject": subject,
            "topic": topic,
            "n_threads": len(titles),
            "n_comments": len(df),
            "n_users": len(self.user_ids),
            "user_type_counts": {str(k): int(v) for k, v in type_counts.items()},
            "coherent_frac": float(frac),
            "lora_dir": str(self.args.lora_dir),
            "base_model": str(self.args.base_model),
            "max_new_tokens": self.args.max_new_tokens,
            "top_p": self.args.top_p,
            "temperature": self.args.temperature,
            "seed": self.args.seed,
            "args": vars(self.args),
        }
        meta_path = out_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        LOG.info("Saved metadata to %s", meta_path)

        LOG.info("DONE — %d comments across %d threads", len(df), len(titles))


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vanilla LoRA forum baseline generator")

    # --- Paths ---
    p.add_argument("--base_model", type=str, default="EleutherAI/pythia-1.4b")
    p.add_argument("--lora_dir", type=str, required=True,
                   help="Path to vanilla LoRA checkpoint dir (with adapter_config.json)")
    p.add_argument("--use_best_ckpt", action="store_true", default=True,
                   help="Look for lora_dir/best/ subdirectory")
    p.add_argument("--no_best_ckpt", dest="use_best_ckpt", action="store_false")
    p.add_argument("--author_parquet", type=str, required=True,
                   help="Path to author_static_10000.parquet")
    p.add_argument("--labels_csv", type=str, default="",
                   help="Path to labels_sentiment.csv (target_user_id,label)")
    p.add_argument("--threshold_source", type=str, default="",
                   help="Parquet/CSV for sentiment threshold computation (used by scorer)")
    p.add_argument("--threshold_col", type=str, default="gstat_user_sent_mean")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--hf_token_file", type=str, default="")
    p.add_argument("--online", action="store_true", default=False)
    p.add_argument("--qlora", action="store_true", default=True)
    p.add_argument("--no_qlora", dest="qlora", action="store_false")

    # --- User pool ---
    p.add_argument("--n_rage", type=int, default=100)
    p.add_argument("--n_empath", type=int, default=100)

    # --- Forum structure ---
    p.add_argument("--sentiment_target", type=str, default="neutral",
                   choices=["rage", "empath", "neutral"])
    p.add_argument("--topic", type=str, default="")
    p.add_argument("--threads_from_default", type=int, default=12)
    p.add_argument("--fanout", type=int, nargs="+", default=[5, 5, 3, 1])
    p.add_argument("--horizon_min", type=float, default=1440.0,
                   help="Simulation horizon in minutes (default 24h)")
    p.add_argument("--max_posts", type=int, default=0,
                   help="Max posts per thread (0 = auto)")

    # --- Generation ---
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--gamma_recency", type=float, default=1.25)
    p.add_argument("--do_sample", action="store_true", default=True)
    p.add_argument("--no_sample", dest="do_sample", action="store_false")
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument("--temperature", type=float, default=0.70)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)

    # --- Misc ---
    p.add_argument("--seed", type=int, default=142)

    # --- Arditi residual-stream patch (optional; M5/M6 factorial cells) ---
    p.add_argument("--arditi_patch", type=str, default="",
                   help="Path to arditi_directions.safetensors. Empty = skip patch.")
    p.add_argument("--arditi_mode", type=str, default="orthogonal",
                   help="Arditi patch mode: single|signed_sentiment|signed_multi|"
                        "per_cohort_dir|orthogonal|orthogonal_multi")
    p.add_argument("--arditi_alpha", type=float, default=1.0,
                   help="Scalar gain on the Arditi projection subtraction.")
    p.add_argument("--arditi_layers", type=str, default="15-23",
                   help="Layer spec, e.g. '15-23' or 'all' or '0,4,8'.")
    p.add_argument("--arditi_label_csv_dir", type=str, default="/tmp",
                   help="Directory containing the 9 per-dim label CSVs.")

    args = p.parse_args()
    return args


def main() -> None:
    args = parse_args()
    builder = VanillaLoRAForumBuilder(args)
    builder.run()


if __name__ == "__main__":
    main()
