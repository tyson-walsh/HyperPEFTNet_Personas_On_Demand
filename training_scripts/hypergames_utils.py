#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hypergames_utils.py — Shared utilities for Pythia-1.4B HyperPEFT-LoRA (M1, RQ3) game scripts
=================================================================================

Consolidates duplicated code across:
  - run_consensus_game_pythia_v6.py
  - run_moderation_game_pythia_v6.py
  - run_diffusion_game_pythia_v6.py
  - test_robustness_pythia_v6.py
  - score_hyperlora_forum_sentiment.py
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

LOG = logging.getLogger("hypergames_utils")


# =====================================================================
# Patent / boilerplate-leak suppression (C1 + C3 patches, 2026-05-14)
# =====================================================================
#
# Pythia checkpoints occasionally regress into Pile patent-text mode under
# the M1 hyper-PEFT delta injection: generations open with phrases like
# "The present invention relates generally to..." or boilerplate
# "wherein", "embodiments described herein", etc. The fixes are two-fold:
#
#   C3 (root cause): a `bad_words_ids` allowlist passed to HF generate at
#       call-time, blocking the phrase tokens before they can be sampled.
#       See `get_patent_suppression_ids`.
#
#   C1 (post-hoc safety net): a regex check inside `_is_coherent` that
#       drops any reply matching common patent-document templates. This
#       catches phrasing the suppression list misses (e.g. paraphrases,
#       BPE boundaries) before metrics see the row.
#
# Both helpers are imported by build_forum_pythia_v6.py,
# score_hyperlora_forum_sentiment.py, and score_persona_signature.py
# so the canonical definition lives here and there is no drift.
# =====================================================================

# Patent-phrase regex used by the C1 safety-net branch in `_is_coherent`.
# Conservative: must hit a multi-word phrase, not a single word like
# "wherein" in casual prose.
_PATENT_RE = re.compile(
    r"\b(?:the present (?:invention|disclosure|application)"
    r"|relates generally to|in accordance with(?:\s+(?:the present|an?))?"
    r"|wherein|embodiments? described herein|a method for"
    r"|a system for|plurality of)\b",
    re.IGNORECASE,
)

# Phrase list for the C3 `bad_words_ids` suppression branch. Mirrors the
# allowlist originally inlined in build_hyperlora_forum.py around the M1
# legacy path (2026-05-03). Promoted here so all Pythia eval
# pipelines share one source of truth.
_PATENT_SUPPRESSION_PHRASES: Tuple[str, ...] = (
    "this hyperlink",
    "this url",
    "<url>",
    "slash command",
    "slashcmd:",
    # Code-language attractors that pull Pythia into the Pile / Stack
    # Overflow distribution (observed in M1 Pythia 2c output, 2026-05-03).
    "<?php",
    "</a>",
    "</div>",
    "</body>",
    "</html>",
    "</script>",
    "function(",
    "var ",
    "let ",
    "const ",
    "import ",
    "package ",
    "public class",
    "public static",
    "private static",
    "def __init__",
    "self.",
    "$_GET",
    "$_POST",
    "console.log",
    "System.out.println",
    "How to ",
    "How do I",
    "How can I",
    "I have this code",
    "I'm trying to write",
    "A:",
    "Q:",
    "stackoverflow",
    "Stack Overflow",
    # Patent / boilerplate openers.
    "The present invention",
    "The present disclosure",
    "The present application",
    "relates generally to",
    "in accordance with",
    "abstract",
    "claims",
    "wherein",
    "embodiments described herein",
    "embodiment described herein",
    "a method for",
    "a system for",
    "plurality of",
)


def _push_bad_words_ids(engine, override: Optional[List[List[int]]]):
    """Temporarily override ``engine.bad_words_ids`` for one generate call.

    Returns a sentinel that ``_pop_bad_words_ids`` restores afterwards.
    No-op (returns None) when override is None, so the engine's own
    construction-time suppression list stays in force.

    Pushes to both the wrapper engine AND the underlying ``_hl``
    HyperLoRAEngine when present, because ``generate_reply`` reads the
    attribute off ``_hl`` (not the wrapper).
    """
    if override is None:
        return None
    saved = (
        getattr(engine, "bad_words_ids", None),
        getattr(getattr(engine, "_hl", None), "bad_words_ids", None),
    )
    try:
        engine.bad_words_ids = override
    except Exception:
        pass
    hl = getattr(engine, "_hl", None)
    if hl is not None:
        try:
            hl.bad_words_ids = override
        except Exception:
            pass
    return saved


def _pop_bad_words_ids(engine, saved) -> None:
    """Restore the suppression list pushed by ``_push_bad_words_ids``."""
    if saved is None:
        return
    outer, inner = saved
    try:
        engine.bad_words_ids = outer
    except Exception:
        pass
    hl = getattr(engine, "_hl", None)
    if hl is not None:
        try:
            hl.bad_words_ids = inner
        except Exception:
            pass


def get_patent_suppression_ids(tokenizer) -> List[List[int]]:
    """Return tokenized `bad_words_ids` for HF generate phrase suppression.

    Encodes both bare and leading-space variants of each phrase in
    ``_PATENT_SUPPRESSION_PHRASES`` and deduplicates by id-sequence. This
    is the C3 fix: callers pass the result as ``bad_words_ids`` into HF
    ``generate`` so the model cannot sample the leaked patent / Stack
    Overflow boilerplate in the first place.

    Returns an empty list if encoding raises (e.g. tokenizer unavailable).
    """
    seqs: List[List[int]] = []
    try:
        for ph in _PATENT_SUPPRESSION_PHRASES:
            for s in (ph, " " + ph):
                enc = tokenizer(s, add_special_tokens=False, return_tensors=None)
                ids = [int(x) for x in (enc.get("input_ids", []) or [])]
                if ids:
                    seqs.append(ids)
    except Exception:
        return []

    uniq: List[List[int]] = []
    seen: set = set()
    for ids in seqs:
        key = tuple(int(x) for x in ids)
        if key in seen:
            continue
        seen.add(key)
        uniq.append([int(x) for x in ids])
    return uniq


def _is_coherent(
    text: str,
    min_alpha_frac: float = 0.40,
    min_words: int = 3,
    min_ascii_frac: float = 0.70,
) -> bool:
    """Heuristic coherence gate for generated forum replies.

    Returns False for garbled / gibberish text, repetition loops, and
    patent / boilerplate leaks. SST-2 and GoEmotions both score
    incoherent strings near the polarity extremes, which systematically
    mis-labels cohorts at user-level aggregation, so every eval pipeline
    must filter at this gate before any sentiment metric is computed.

    Canonical definition; ``build_forum_pythia_v6.py``,
    ``score_hyperlora_forum_sentiment.py``, and
    ``score_persona_signature.py`` import this rather than
    redefining it to prevent drift.
    """
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
    # C1 safety net: drop patent-document boilerplate the C3 suppression
    # list missed (paraphrases, BPE boundaries).
    if _PATENT_RE.search(t):
        return False
    if has_repetition_loop(t):
        return False
    return True


# =====================================================================
# User pool loading
# =====================================================================

def load_user_pool(
    author_parquet: str,
    labels_csv: str,
    feature_names: Optional[List[str]],
    K: int,
    gvec_from_manifest_fn: Callable,
    *,
    n_rage: int = 0,
    n_empath: int = 0,
    n_neutral: int = 0,
    label_a: str = "rage",
    label_b: str = "empath",
    n_a: Optional[int] = None,
    n_b: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[Dict[str, Any]], Dict[int, np.ndarray], Optional[Dict[int, torch.Tensor]]]:
    """Load user pool from author parquet and labels CSV.

    Supports arbitrary binary labels via label_a/label_b (defaults: rage/empath).
    n_a/n_b override n_rage/n_empath when provided.

    Returns
    -------
    users : list of dicts with keys (user_id, author_type)
    gvecs : dict mapping user_id -> g-vector (np.ndarray)
    gvec_tensors : dict mapping user_id -> g-vector as torch.Tensor on device
                   (None if device is None)
    """
    # Resolve counts: n_a/n_b take priority, fall back to n_rage/n_empath
    count_a = n_a if n_a is not None else n_rage
    count_b = n_b if n_b is not None else n_empath

    author_df = pd.read_parquet(author_parquet)
    if "target_user_id" not in author_df.columns:
        raise KeyError("author_parquet must include 'target_user_id'")

    # Load labels
    labels: Dict[int, str] = {}
    if labels_csv and Path(labels_csv).exists():
        ldf = pd.read_csv(labels_csv)
        for _, row in ldf.iterrows():
            uid = int(row["target_user_id"])
            lab = str(row.get("label", "neutral")).strip().lower()
            labels[uid] = lab

    a_uids = [uid for uid, lab in labels.items() if lab == label_a]
    b_uids = [uid for uid, lab in labels.items() if lab == label_b]

    # Sort by sentiment extremity if available (label_a = negative, label_b = positive)
    sent_col = "gstat_user_sent_mean"
    if sent_col in author_df.columns:
        uid_to_sent = dict(zip(
            author_df["target_user_id"].astype(int),
            pd.to_numeric(author_df[sent_col], errors="coerce").fillna(0.0)))
        a_uids.sort(key=lambda u: uid_to_sent.get(u, 0.0))
        b_uids.sort(key=lambda u: uid_to_sent.get(u, 0.0), reverse=True)

    a_uids = a_uids[:count_a]
    b_uids = [u for u in b_uids if u not in set(a_uids)][:count_b]
    labeled_set = set(a_uids) | set(b_uids)

    # Neutral pool (unlabeled parquet users)
    neutral_uids: List[int] = []
    if n_neutral > 0:
        all_pq = set(author_df["target_user_id"].astype(int).tolist())
        neutral_cands = sorted(all_pq - labeled_set - set(labels.keys()))
        if len(neutral_cands) < n_neutral:
            neutral_cands = sorted(all_pq - labeled_set)
        if rng is not None:
            rng.shuffle(neutral_cands)  # type: ignore[arg-type]
        neutral_uids = list(neutral_cands[:n_neutral])

    # Build type map and pool — author_type uses the actual label name
    atypes: Dict[int, str] = {}
    for u in a_uids:
        atypes[u] = label_a
    for u in b_uids:
        atypes[u] = label_b
    for u in neutral_uids:
        atypes[u] = "neutral"
    pool_uids = a_uids + b_uids + neutral_uids

    uid_set = set(pool_uids)
    pool_df = author_df[author_df["target_user_id"].isin(uid_set)].copy()
    row_by_uid: Dict[int, pd.Series] = {
        int(r["target_user_id"]): r for _, r in pool_df.iterrows()
    }

    users: List[Dict[str, Any]] = []
    gvecs: Dict[int, np.ndarray] = {}
    for uid in pool_uids:
        atype = atypes.get(uid, "neutral")
        row = row_by_uid.get(uid)
        if row is not None and feature_names is not None:
            gvec = gvec_from_manifest_fn(row, feature_names, K)
        else:
            gvec = np.zeros(K, dtype=np.float32)
        users.append({"user_id": uid, "author_type": atype})
        gvecs[uid] = gvec

    # Pre-cache tensors if device provided
    gvec_tensors: Optional[Dict[int, torch.Tensor]] = None
    if device is not None:
        gvec_tensors = {
            uid: torch.tensor([gv], dtype=torch.float32, device=device)
            for uid, gv in gvecs.items()
        }

    return users, gvecs, gvec_tensors


def select_user_uids(
    author_parquet: str,
    labels_csv: str,
    *,
    n_rage: int = 0,
    n_empath: int = 0,
    n_neutral: int = 0,
    label_a: str = "rage",
    label_b: str = "empath",
    n_a: Optional[int] = None,
    n_b: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], List[int], List[int], pd.DataFrame, Dict[int, str]]:
    """Lower-level helper returning UID lists without building g-vectors.

    Supports arbitrary binary labels via label_a/label_b (defaults: rage/empath).
    n_a/n_b override n_rage/n_empath when provided.

    Returns (a_uids, b_uids, neutral_uids, author_df, labels_dict)
    """
    count_a = n_a if n_a is not None else n_rage
    count_b = n_b if n_b is not None else n_empath

    author_df = pd.read_parquet(author_parquet)
    if "target_user_id" not in author_df.columns:
        raise KeyError("author_parquet must include 'target_user_id'")

    labels: Dict[int, str] = {}
    if labels_csv and Path(labels_csv).exists():
        ldf = pd.read_csv(labels_csv)
        for _, row in ldf.iterrows():
            labels[int(row["target_user_id"])] = str(row.get("label", "neutral")).strip().lower()

    a_uids = [uid for uid, lab in labels.items() if lab == label_a]
    b_uids = [uid for uid, lab in labels.items() if lab == label_b]

    sent_col = "gstat_user_sent_mean"
    if sent_col in author_df.columns:
        uid_to_sent = dict(zip(
            author_df["target_user_id"].astype(int),
            pd.to_numeric(author_df[sent_col], errors="coerce").fillna(0.0)))
        a_uids.sort(key=lambda u: uid_to_sent.get(u, 0.0))
        b_uids.sort(key=lambda u: uid_to_sent.get(u, 0.0), reverse=True)

    a_uids = a_uids[:count_a]
    b_uids = [u for u in b_uids if u not in set(a_uids)][:count_b]
    labeled_set = set(a_uids) | set(b_uids)

    neutral_uids: List[int] = []
    if n_neutral > 0:
        all_pq = set(author_df["target_user_id"].astype(int).tolist())
        neutral_cands = sorted(all_pq - labeled_set - set(labels.keys()))
        if len(neutral_cands) < n_neutral:
            neutral_cands = sorted(all_pq - labeled_set)
        if rng is not None:
            rng.shuffle(neutral_cands)  # type: ignore[arg-type]
        neutral_uids = list(neutral_cands[:n_neutral])

    return a_uids, b_uids, neutral_uids, author_df, labels


# =====================================================================
# Context building
# =====================================================================

def build_context(
    tokenizer,
    *,
    topic: str,
    segments: List[str],
    sep_id: int,
    max_context_tokens: int,
    truncation_strategy: str = "drop_oldest",
    max_segments: int = 0,
) -> Tuple[List[int], List[int]]:
    """Build token sequence: topic + segments + "\\n\\nReply: " text anchor.

    The prompt ends on the literal "\\n\\nReply: " marker, NOT on sep_id
    (EOS-aliasing fix; see the marker comment below).

    Parameters
    ----------
    truncation_strategy :
        "drop_oldest" — include as many recent segments as fit, dropping oldest first.
        "per_segment_budget" — allocate equal budget per segment, truncate each.
    max_segments :
        If > 0, keep only the most recent ``max_segments`` segments BEFORE the
        token budget is applied. This is the fixed, fair, size-independent
        context budget (2026-05-20). Without it the consensus task packs as many
        prior replies as fit, which silently drops 96-100% of history at large
        community sizes (a hidden, size-dependent variable) and makes the prompt
        long and slow. With it, every agent sees the same fixed number of recent
        replies at any community size, and the prompt stays short. moderation and
        diffusion already cap context upstream (context_window / neighbor slice),
        so this mainly matters for consensus. See experiments docs 02 and 08.

    Returns (input_ids, attention_mask) as plain lists.
    """
    if max_segments and max_segments > 0 and len(segments) > max_segments:
        segments = segments[-max_segments:]
    topic_tokens = tokenizer.encode(topic, add_special_tokens=False)
    nl_tokens = tokenizer.encode("\n", add_special_tokens=False)

    # Budget: max_context_tokens minus topic overhead and reply-marker room
    overhead = len(topic_tokens) + len(nl_tokens) + 4  # ~4 for "\n\nReply: "
    budget = max(16, max_context_tokens - overhead)

    if truncation_strategy == "drop_oldest":
        # Encode all segments
        encoded_segs: List[List[int]] = []
        for seg in segments:
            encoded_segs.append(tokenizer.encode(seg, add_special_tokens=False))

        # Include as many recent segments as fit, dropping oldest first
        kept: List[List[int]] = []
        used = 0
        for seg_toks in reversed(encoded_segs):
            seg_cost = len(seg_toks) + len(nl_tokens)
            if used + seg_cost > budget:
                break
            kept.append(seg_toks)
            used += seg_cost
        kept.reverse()  # restore chronological order

        n_dropped = len(encoded_segs) - len(kept)
        if n_dropped > 0 and len(encoded_segs) > 3:
            drop_frac = n_dropped / len(encoded_segs)
            # 2026-05-12: bumped from 0.5 -> 0.95 after Paper 3 pre-reg locked
            # max_context_tokens=1536. Under that setting, consensus rounds
            # with N=100-200 fan-in steady-state at 88-95% drop by design;
            # the recent-17 segments carry the load-bearing cascade signal.
            # Real pathology only shows up above 95% (the regime the v1
            # max_context_tokens 512 -> 1536 fix was meant to escape).
            if drop_frac > 0.95:
                LOG.warning(
                    "build_context: dropped %d/%d segments (%.0f%%) -- budget=%d tokens",
                    n_dropped, len(encoded_segs), drop_frac * 100, budget)

        body: List[int] = list(topic_tokens)
        for seg_toks in kept:
            body.extend(nl_tokens)
            body.extend(seg_toks)

    elif truncation_strategy == "per_segment_budget":
        all_segs = [topic] + segments
        per_seg = max(8, budget // max(1, len(all_segs)))
        body = []
        for i, seg in enumerate(all_segs):
            ids = tokenizer.encode(seg, add_special_tokens=False)
            if len(ids) > per_seg:
                ids = ids[-per_seg:]
            body.extend(ids)
            if i + 1 < len(all_segs) and nl_tokens:
                body.extend(nl_tokens)
    elif truncation_strategy == "balanced":
        # Representative sampling for large communities. drop_oldest shows each agent only the
        # most-recent messages, which at large N is a tiny, recency-biased sliver of the
        # conversation (confirmed to flatten H1c at large N: experiments docs 02, 14). "balanced"
        # instead selects segments EVENLY SPACED across the whole conversation that fit the token
        # budget, so the agent sees a representative slice of the deliberation at any community
        # size, within the model's fixed context window. Chronological order is preserved.
        encoded_segs = [tokenizer.encode(seg, add_special_tokens=False) for seg in segments]
        seg_costs = [len(t) + len(nl_tokens) for t in encoded_segs]
        n = len(encoded_segs)
        if n == 0:
            sel_idx: List[int] = []
        else:
            avg_cost = max(1.0, sum(seg_costs) / n)
            n_fit = max(1, int(budget // avg_cost))
            if n <= n_fit:
                sel_idx = list(range(n))
            else:
                sel_idx = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, n_fit)))
        # Budget-fit the evenly-spaced selection (drop oldest of the SELECTION if still over).
        kept_idx: List[int] = []
        used = 0
        for i in reversed(sel_idx):
            c = seg_costs[i]
            if used + c > budget:
                break
            kept_idx.append(i)
            used += c
        kept_idx.sort()
        body = list(topic_tokens)
        for i in kept_idx:
            body.extend(nl_tokens)
            body.extend(encoded_segs[i])

    else:
        raise ValueError(f"Unknown truncation_strategy: {truncation_strategy!r}")

    # Anchor in "Reddit-reply mode" by ending the prompt on the literal text
    # "\n\nReply: " (post-2026-05-03 EOS-aliasing fix).  This now matches
    # build_hyperlora_forum.py _encode_branch exactly: the prompt MUST end on
    # the natural-text marker and MUST NOT end on sep_id (<|reply|>).  The
    # special tokens (<|reply|>, <|eoreply|>) were initialized from the EOS
    # embedding and remain cos~0.9995 to EOS at inference, so a prompt that
    # ends on sep_id reads as "document complete" to Pythia and triggers Pile
    # boilerplate / patent regurgitation.  The text marker tokenizes to ordinary
    # trained tokens that do NOT alias EOS.  (Earlier revisions of this helper
    # appended the reply marker AND then sep_id, reintroducing the bug; the
    # trailing sep_id is now removed.)  See Appendix S of the technical
    # reference and project memory special-token-eos-aliasing.
    reply_marker = tokenizer.encode("\n\nReply: ", add_special_tokens=False)

    # Final trim (safety net) — leave room for the reply marker.
    cap = max_context_tokens - len(reply_marker)
    if cap < 1:
        cap = max_context_tokens - 1
    if len(body) > cap:
        body = body[-cap:]

    if reply_marker:
        body.extend(reply_marker)
    # NOTE: sep_id is intentionally NOT appended (EOS-aliasing fix).  The
    # `sep_id` parameter is retained for call-site compatibility.
    _ = sep_id
    attn = [1] * len(body)
    return body, attn


# =====================================================================
# Generation helpers
# =====================================================================

def generate_one(
    engine,
    input_ids: List[int],
    attention_mask: List[int],
    gvec: np.ndarray,
    postprocess_fn: Callable[[str], str],
    coherence_fn: Callable[[str], bool],
    *,
    gvec_tensor: Optional[torch.Tensor] = None,
    max_new_tokens: int = 64,
    do_sample: bool = True,
    top_p: float = 0.90,
    temperature: float = 0.70,
    user_id: Optional[int] = None,
    bad_words_ids: Optional[List[List[int]]] = None,
) -> str:
    """Generate a single response for one user, returning cleaned text.

    If `user_id` is provided and an Arditi Patch is installed on the engine,
    the per-batch UID state is primed for per-user mode lookups.

    If `bad_words_ids` is provided, it overrides ``engine.bad_words_ids`` for
    the duration of this call (and is restored afterwards). Callers normally
    leave it None so the engine's own suppression list (populated by
    ``get_patent_suppression_ids`` at construction) is used."""
    device = engine.device

    ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask_t = torch.tensor([attention_mask], dtype=torch.long, device=device)
    g_t = gvec_tensor if gvec_tensor is not None else torch.tensor(
        [gvec], dtype=torch.float32, device=device)

    # Per-call override of the C3 suppression list. Push to both the wrapper
    # engine and its underlying _hl (HyperLoRAEngine) since generate_reply
    # reads getattr(self, "bad_words_ids", None) on _hl.
    _bw_saved = _push_bad_words_ids(engine, bad_words_ids)
    try:
        out_ids = engine.generate(
            input_ids=ids_t,
            attention_mask=mask_t,
            global_features=g_t,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            user_id=user_id,
        )
    finally:
        _pop_bad_words_ids(engine, _bw_saved)

    token_ids = out_ids[0] if out_ids else []
    if token_ids and token_ids[-1] == engine.end_id:
        token_ids = token_ids[:-1]

    text = engine.tok.decode(token_ids, skip_special_tokens=True).strip()
    text = postprocess_fn(text)
    if not coherence_fn(text):
        text = ""
    return text


def batch_generate_same_context(
    engine,
    input_ids: List[int],
    attention_mask: List[int],
    gvecs: List[np.ndarray],
    postprocess_fn: Callable[[str], str],
    coherence_fn: Callable[[str], bool],
    *,
    gvec_tensors: Optional[List[torch.Tensor]] = None,
    batch_size: int = 8,
    max_new_tokens: int = 64,
    do_sample: bool = True,
    top_p: float = 0.90,
    temperature: float = 0.70,
    user_ids: Optional[List[int]] = None,
    bad_words_ids: Optional[List[List[int]]] = None,
) -> List[str]:
    """Generate responses for multiple users sharing the same context.

    All users see identical input_ids but have different g-vectors.
    Splits into mini-batches of batch_size for memory efficiency.

    If `user_ids` is provided AND engine has an installed Arditi Patch
    (engine._arditi_state is not None), the per-batch UID tensor is primed
    before each mini-batch so per-user modes (orthogonal, signed_*, etc.)
    can look up the correct direction. Otherwise the patch falls back to
    'main' or no-op behavior.

    If `bad_words_ids` is provided, it overrides ``engine.bad_words_ids``
    for the duration of this call. The default (None) keeps whatever
    suppression list the engine was constructed with via
    ``get_patent_suppression_ids``.
    """
    device = engine.device
    results: List[str] = []
    N = len(gvecs)
    _arditi_state = getattr(engine, "_arditi_state", None)

    # Per-call override of the C3 suppression list, applied once for the
    # whole multi-batch loop (restored in the finally below).
    _bw_saved = _push_bad_words_ids(engine, bad_words_ids)

    try:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start

            ids_t = torch.tensor([input_ids] * B, dtype=torch.long, device=device)
            mask_t = torch.tensor([attention_mask] * B, dtype=torch.long, device=device)

            if gvec_tensors is not None:
                g_t = torch.cat(gvec_tensors[start:end], dim=0)
            else:
                g_batch = np.stack(gvecs[start:end], axis=0)
                g_t = torch.tensor(g_batch, dtype=torch.float32, device=device)

            # Prime per-batch UID state for the Arditi Patch (per-user modes).
            if _arditi_state is not None and user_ids is not None:
                _arditi_state.set_batch_uids(
                    [int(u) for u in user_ids[start:end]], device=device)

            out_ids_batch = engine.generate(
                input_ids=ids_t,
                attention_mask=mask_t,
                global_features=g_t,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
            )

            # Clear the UID state so a stale value doesn't leak to the next call.
            if _arditi_state is not None and user_ids is not None:
                _arditi_state.set_batch_uids(None)

            for b_out in out_ids_batch:
                token_ids = list(b_out) if b_out else []
                if token_ids and token_ids[-1] == engine.end_id:
                    token_ids = token_ids[:-1]
                text = engine.tok.decode(token_ids, skip_special_tokens=True).strip()
                text = postprocess_fn(text)
                if not coherence_fn(text):
                    text = ""
                results.append(text)
    finally:
        _pop_bad_words_ids(engine, _bw_saved)

    return results


def batch_generate_diff_context(
    engine,
    contexts: List[Tuple[List[int], List[int]]],
    gvecs: List[np.ndarray],
    postprocess_fn: Callable[[str], str],
    coherence_fn: Callable[[str], bool],
    *,
    gvec_tensors: Optional[List[torch.Tensor]] = None,
    batch_size: int = 8,
    max_new_tokens: int = 64,
    do_sample: bool = True,
    top_p: float = 0.90,
    temperature: float = 0.70,
    user_ids: Optional[List[int]] = None,
    bad_words_ids: Optional[List[List[int]]] = None,
    pad_id: Optional[int] = None,
) -> List[str]:
    """Generate responses for multiple users with DIFFERENT contexts.

    Mirror of ``batch_generate_same_context`` for the diffusion cascade, where
    every node has its own neighbor context. Each mini-batch is LEFT-padded to
    a common length (so the real tokens sit flush-right and attention masks zero
    out the pad), which is the correct padding side for autoregressive decode.
    Pythia/GPT-NeoX uses rotary position embeddings; left-padding is only valid
    if ``engine.generate`` derives position_ids from the attention mask. The
    equivalence probe (scripts/probe_batched_diff_context.py) verifies this
    against the unbatched ``generate_one`` path under greedy decode BEFORE this
    helper is trusted in production.

    All other behavior (per-sequence g-vectors, Arditi per-UID priming, C3
    bad_words override) matches ``batch_generate_same_context``. The caller is
    responsible for grouping nodes that share the same delta_scale into one call
    (delta_scale is a scalar engine override, not per-sequence)."""
    device = engine.device
    results: List[str] = []
    N = len(contexts)
    _arditi_state = getattr(engine, "_arditi_state", None)
    if pad_id is None:
        pad_id = getattr(engine.tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(engine, "end_id", 0) or 0

    _bw_saved = _push_bad_words_ids(engine, bad_words_ids)
    try:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start
            chunk = contexts[start:end]
            L = max(len(ids) for ids, _ in chunk)

            # LEFT-pad each sequence to L (real tokens flush-right).
            ids_rows, mask_rows = [], []
            for ids, msk in chunk:
                pad = L - len(ids)
                ids_rows.append([pad_id] * pad + list(ids))
                mask_rows.append([0] * pad + list(msk))
            ids_t = torch.tensor(ids_rows, dtype=torch.long, device=device)
            mask_t = torch.tensor(mask_rows, dtype=torch.long, device=device)

            if gvec_tensors is not None:
                g_t = torch.cat(gvec_tensors[start:end], dim=0)
            else:
                g_batch = np.stack(gvecs[start:end], axis=0)
                g_t = torch.tensor(g_batch, dtype=torch.float32, device=device)

            if _arditi_state is not None and user_ids is not None:
                _arditi_state.set_batch_uids(
                    [int(u) for u in user_ids[start:end]], device=device)

            out_ids_batch = engine.generate(
                input_ids=ids_t,
                attention_mask=mask_t,
                global_features=g_t,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
            )

            if _arditi_state is not None and user_ids is not None:
                _arditi_state.set_batch_uids(None)

            for b_out in out_ids_batch:
                token_ids = list(b_out) if b_out else []
                if token_ids and token_ids[-1] == engine.end_id:
                    token_ids = token_ids[:-1]
                text = engine.tok.decode(token_ids, skip_special_tokens=True).strip()
                text = postprocess_fn(text)
                if not coherence_fn(text):
                    text = ""
                results.append(text)
    finally:
        _pop_bad_words_ids(engine, _bw_saved)

    return results


# =====================================================================
# Delta scale control
# =====================================================================

@contextmanager
def delta_scale_override(engine, scale: float):
    """Temporarily override delta scale on the hypernetwork forward pass.

    Saves and restores the original forward method via try/finally.
    """
    if not hasattr(engine, "_orig_hyper_forward"):
        engine._orig_hyper_forward = engine.hyper.forward
    base_forward = engine._orig_hyper_forward
    engine.hyper.forward = lambda g, _of=base_forward, _s=scale: _of(g) * _s
    try:
        yield
    finally:
        engine.hyper.forward = base_forward


def set_engine_delta_scale(engine, scale: float) -> None:
    """Permanently set delta scale on the hypernetwork forward pass.

    Stores the original forward once; safe to call multiple times.
    """
    if not hasattr(engine, "_orig_hyper_forward"):
        engine._orig_hyper_forward = engine.hyper.forward
    base_forward = engine._orig_hyper_forward
    if scale == 0.0:
        engine.hyper.forward = lambda g, _of=base_forward: _of(g) * 0.0
    else:
        engine.hyper.forward = lambda g, _of=base_forward, _s=scale: _of(g) * _s
    LOG.info("Delta scale -> %.4f (permanent)", scale)


# =====================================================================
# Checkpointing
# =====================================================================

def save_round_checkpoint(
    rows: List[Dict[str, Any]],
    checkpoint_path: Path,
    round_idx: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save per-round checkpoint as parquet with metadata sidecar."""
    df = pd.DataFrame(rows)
    df.to_parquet(checkpoint_path / f"checkpoint_round_{round_idx:04d}.parquet", index=False)
    if metadata is not None:
        meta_path = checkpoint_path / f"checkpoint_round_{round_idx:04d}_meta.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def load_checkpoint(
    checkpoint_path: Path,
) -> Tuple[Optional[pd.DataFrame], int]:
    """Load the most recent checkpoint, returning (df, last_round_idx).

    Returns (None, -1) if no checkpoint found.
    """
    ckpts = sorted(checkpoint_path.glob("checkpoint_round_*.parquet"))
    if not ckpts:
        return None, -1

    # Parse round index from filename: checkpoint_round_0003.parquet -> 3
    last_ckpt = ckpts[-1]
    stem = last_ckpt.stem  # "checkpoint_round_0003"
    round_idx = int(stem.split("_")[-1])

    df = pd.read_parquet(last_ckpt)
    LOG.info("Resumed from checkpoint: %s (round %d, %d rows)",
             last_ckpt.name, round_idx, len(df))
    return df, round_idx


# =====================================================================
# Metadata & git
# =====================================================================

def get_git_hash() -> str:
    """Return short git hash of the current repo, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def enrich_metadata(
    meta: Dict[str, Any],
    *,
    start_time: float,
    end_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Add git hash and timing to metadata dict (in-place + returned)."""
    meta["git_hash"] = get_git_hash()
    meta["start_time_unix"] = start_time
    if end_time is not None:
        meta["end_time_unix"] = end_time
        meta["total_elapsed_sec"] = round(end_time - start_time, 2)
    return meta


# =====================================================================
# Bootstrap confidence intervals
# =====================================================================

def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable = np.mean,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 142,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)

    point = float(stat_fn(values))
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_stats = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_stats[b] = stat_fn(sample)

    alpha = 1.0 - ci
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (point, lo, hi)


# =====================================================================
# Text quality helpers
# =====================================================================

def has_repetition_loop(
    text: str,
    max_ngram_frac: float = 0.125,
    ngram_n: int = 3,
) -> bool:
    """Detect degenerate repetition loops in generated text.

    Catches failure modes like:
      "I have to learn how to make this work. I have to learn how to..."

    Two checks:
      1. Most-frequent word n-gram exceeds ``max(2, len(words) * max_ngram_frac)``
      2. Any whole sentence (split on '. ') appears more than 2 times
    """
    t = (text or "").strip()
    words = t.split()
    if len(words) < ngram_n * 3:
        return False

    # --- n-gram repetition ---
    from collections import Counter

    ngrams = [
        tuple(words[i : i + ngram_n]) for i in range(len(words) - ngram_n + 1)
    ]
    if ngrams:
        _, top_count = Counter(ngrams).most_common(1)[0]
        threshold = max(2, int(len(words) * max_ngram_frac))
        if top_count > threshold:
            return True

    # --- whole-sentence repetition ---
    sentences = [s.strip() for s in t.split(". ") if len(s.strip()) > 5]
    if sentences:
        _, top_sent_count = Counter(sentences).most_common(1)[0]
        if top_sent_count > 2:
            return True

    return False


# =====================================================================
# Shared sentiment pipeline
# =====================================================================

_CACHED_SENT_PIPE: Any = None
_CACHED_SENT_KEY: Optional[Tuple[str, int]] = None

# VADER is cached after first init (CPU-only, thread-safe)
_CACHED_VADER_SIA: Any = None

# GoEmotions cached model + tokenizer (loaded once, kept on device)
_CACHED_GOEMO_MODEL: Any = None
_CACHED_GOEMO_TOKENIZER: Any = None
_CACHED_GOEMO_DEVICE: Optional[int] = None

# GoEmotions emotion-to-polarity mappings (Demszky et al., 2020)
GOEMO_POSITIVE = frozenset({
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief",
})
GOEMO_NEGATIVE = frozenset({
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
})
# Sensitivity analysis: rage/empath-specific composites
GOEMO_RAGE = frozenset({"anger", "annoyance", "disapproval", "disgust"})
GOEMO_EMPATH = frozenset({"caring", "approval", "gratitude", "admiration", "joy"})


def _get_vader_sia():
    """Lazily create and cache a VADER SentimentIntensityAnalyzer."""
    global _CACHED_VADER_SIA  # noqa: PLW0603
    if _CACHED_VADER_SIA is not None:
        return _CACHED_VADER_SIA
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError:
        raise ImportError("NLTK is required for VADER. pip install nltk")
    try:
        _CACHED_VADER_SIA = SentimentIntensityAnalyzer()
    except LookupError:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        _CACHED_VADER_SIA = SentimentIntensityAnalyzer()
    return _CACHED_VADER_SIA


def _is_vader_mode(model_id: str) -> bool:
    """Check if the model_id indicates VADER mode (empty, 'vader', or unset)."""
    return not model_id or model_id.lower() == "vader"


def get_sentiment_pipe(
    model_id: str = "",
    device_id: int = -1,
) -> Any:
    """Lazily create and cache a sentiment-analysis pipeline.

    Routing:
      - 'goemo' / 'go_emotions' / path containing 'go_emotions' -> GoEmotions
      - '' or 'vader' -> VADER (legacy)
      - anything else -> SST-2 HF pipeline (legacy)

    The default is now GoEmotions (env SENTIMENT_MODEL, fallback 'goemo').
    """
    global _CACHED_SENT_PIPE, _CACHED_SENT_KEY  # noqa: PLW0603

    import os

    if not model_id:
        model_id = os.environ.get("SENTIMENT_MODEL", "goemo")

    # GoEmotions path (preferred)
    if _is_goemo_mode(model_id):
        LOG.info("Sentiment backend: GoEmotions (Reddit-trained RoBERTa, 28 emotions)")
        return ("goemo", device_id)  # sentinel tuple

    # VADER path (legacy)
    if _is_vader_mode(model_id):
        LOG.info("Sentiment backend: VADER (CPU, no model loading)")
        return "vader"

    # Legacy SST-2 HF pipeline path
    key = (model_id, int(device_id))
    if _CACHED_SENT_PIPE is not None and _CACHED_SENT_KEY == key:
        return _CACHED_SENT_PIPE

    try:
        from transformers import pipeline as hf_pipeline
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()

        # 2026-05-19 fix: framework="pt" prevents Keras 3 / tf-keras compat
        # crash on local Python envs where pipeline auto-falls-back to TF.
        _CACHED_SENT_PIPE = hf_pipeline(
            task="sentiment-analysis",
            model=model_id,
            tokenizer=model_id,
            device=int(device_id),
            top_k=None,
            truncation=True,
            batch_size=64,
            framework="pt",
        )
        _CACHED_SENT_KEY = key
        LOG.info("Sentiment pipe loaded: model=%s device=%d", model_id, device_id)
        return _CACHED_SENT_PIPE
    except Exception as exc:
        LOG.warning("Failed to create sentiment pipe: %s", exc)
        return None


def score_texts_sentiment(
    texts: List[str],
    pipe: Any,
    chunk_size: int = 256,
    goemo_mode: str = "simple",
) -> List[float]:
    """Score a list of texts, returning polarity in approx [-1, +1].

    Routing:
      - GoEmotions sentinel tuple ('goemo', device_id) -> GoEmotions
      - VADER sentinel string 'vader' -> VADER compound
      - HF pipeline object -> legacy SST-2 (POSITIVE - NEGATIVE)

    Parameters
    ----------
    goemo_mode : str
        'simple' or 'rage_empath' (only used when pipe is GoEmotions sentinel).

    Returns list of floats (same length as *texts*); NaN for empty/failed.
    """
    if pipe is None:
        return [float("nan")] * len(texts)

    # GoEmotions path (preferred)
    if isinstance(pipe, tuple) and len(pipe) == 2 and pipe[0] == "goemo":
        device_id = pipe[1]
        return score_texts_goemo(texts, device_id=device_id, mode=goemo_mode)

    # VADER path (legacy)
    if isinstance(pipe, str) and pipe == "vader":
        return score_texts_vader(texts)

    # Legacy SST-2 path
    out: List[float] = []
    for i in range(0, len(texts), chunk_size):
        batch = texts[i : i + chunk_size]
        batch_clean = [t if t.strip() else "neutral" for t in batch]
        try:
            results = pipe(batch_clean)
            for res in results:
                scores = {r["label"]: r["score"] for r in res}
                pos = scores.get("POSITIVE", 0.5)
                neg = scores.get("NEGATIVE", 0.5)
                out.append(float(pos - neg))
        except Exception:
            out.extend([float("nan")] * len(batch))
    return out


def score_texts_vader(texts: List[str]) -> List[float]:
    """Score texts using VADER compound sentiment in [-1, +1].

    VADER is rule-based and designed for social media text (handles caps
    emphasis, exclamation marks, slang, degree modifiers).  No GPU needed.
    """
    sia = _get_vader_sia()
    return [sia.polarity_scores(t or "")["compound"] for t in texts]


# =====================================================================
# GoEmotions (Reddit-trained RoBERTa, 28 emotion categories)
# =====================================================================

def _is_goemo_mode(model_id: str) -> bool:
    """Check if model_id indicates GoEmotions mode."""
    if not model_id:
        return False
    return model_id.lower() in ("goemo", "go_emotions", "goemotions") or "go_emotions" in model_id.lower()


def _get_goemo_model(device_id: int = -1):
    """Lazily load and cache the GoEmotions model + tokenizer."""
    global _CACHED_GOEMO_MODEL, _CACHED_GOEMO_TOKENIZER, _CACHED_GOEMO_DEVICE  # noqa: PLW0603

    if (
        _CACHED_GOEMO_MODEL is not None
        and _CACHED_GOEMO_TOKENIZER is not None
        and _CACHED_GOEMO_DEVICE == device_id
    ):
        return _CACHED_GOEMO_MODEL, _CACHED_GOEMO_TOKENIZER

    import os
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Resolve model path: check local cache first, then env, then default
    model_path = os.environ.get(
        "GOEMO_MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "models" / "roberta-goemo"),
    )
    if not os.path.isdir(model_path):
        model_path = "SamLowe/roberta-base-go_emotions"
        LOG.info("GoEmotions local path not found; using HuggingFace Hub: %s", model_path)

    LOG.info("Loading GoEmotions model from %s (device=%d)", model_path, device_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    if device_id >= 0 and torch.cuda.is_available():
        model = model.to(f"cuda:{device_id}")
    LOG.info("GoEmotions model loaded: %d labels", model.config.num_labels or 28)

    _CACHED_GOEMO_MODEL = model
    _CACHED_GOEMO_TOKENIZER = tokenizer
    _CACHED_GOEMO_DEVICE = device_id
    return model, tokenizer


def _goemo_predict_batch(
    texts: List[str],
    model,
    tokenizer,
    batch_size: int = 64,
) -> List[Dict[str, float]]:
    """Run GoEmotions inference, returning per-text dicts of {emotion: prob}.

    Uses sigmoid (multi-label) since GoEmotions is multi-label classification.
    """
    id2label = model.config.id2label
    device = next(model.parameters()).device
    results: List[Dict[str, float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_clean = [t if (t and t.strip()) else "neutral" for t in batch]
        inputs = tokenizer(
            batch_clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu()  # multi-label: sigmoid

        for row in probs:
            results.append({
                id2label[j]: float(row[j]) for j in range(len(id2label))
            })
    return results


def score_texts_goemo(
    texts: List[str],
    device_id: int = -1,
    batch_size: int = 64,
    mode: str = "simple",
) -> List[float]:
    """Score texts using GoEmotions, returning polarity in approx [-1, +1].

    Parameters
    ----------
    texts : list of str
        Input texts to score.
    device_id : int
        CUDA device (-1 for CPU).
    batch_size : int
        Inference batch size.
    mode : str
        'simple' -- mean(positive_emotions) - mean(negative_emotions)
        'rage_empath' -- mean(empath_emotions) - mean(rage_emotions)

    Returns
    -------
    list of float
        Polarity scores. Range depends on mode but typically in [-1, +1].
    """
    model, tokenizer = _get_goemo_model(device_id)
    emotion_dicts = _goemo_predict_batch(texts, model, tokenizer, batch_size)

    if mode == "rage_empath":
        pos_set, neg_set = GOEMO_EMPATH, GOEMO_RAGE
    else:
        pos_set, neg_set = GOEMO_POSITIVE, GOEMO_NEGATIVE

    polarities: List[float] = []
    for ed in emotion_dicts:
        pos_vals = [ed[k] for k in pos_set if k in ed]
        neg_vals = [ed[k] for k in neg_set if k in ed]
        pos_mean = sum(pos_vals) / len(pos_vals) if pos_vals else 0.0
        neg_mean = sum(neg_vals) / len(neg_vals) if neg_vals else 0.0
        polarities.append(pos_mean - neg_mean)
    return polarities


def score_texts_goemo_full(
    texts: List[str],
    device_id: int = -1,
    batch_size: int = 64,
) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
    """Score texts with GoEmotions, returning both polarity mappings + raw emotions.

    Returns
    -------
    (simple_polarity, rage_empath_polarity, raw_emotion_dicts)
    """
    model, tokenizer = _get_goemo_model(device_id)
    emotion_dicts = _goemo_predict_batch(texts, model, tokenizer, batch_size)

    simple_pol: List[float] = []
    re_pol: List[float] = []
    for ed in emotion_dicts:
        # Simple: all positive vs all negative
        pos_vals = [ed[k] for k in GOEMO_POSITIVE if k in ed]
        neg_vals = [ed[k] for k in GOEMO_NEGATIVE if k in ed]
        pos_mean = sum(pos_vals) / len(pos_vals) if pos_vals else 0.0
        neg_mean = sum(neg_vals) / len(neg_vals) if neg_vals else 0.0
        simple_pol.append(pos_mean - neg_mean)

        # Rage/empath composites
        emp_vals = [ed[k] for k in GOEMO_EMPATH if k in ed]
        rage_vals = [ed[k] for k in GOEMO_RAGE if k in ed]
        emp_mean = sum(emp_vals) / len(emp_vals) if emp_vals else 0.0
        rage_mean = sum(rage_vals) / len(rage_vals) if rage_vals else 0.0
        re_pol.append(emp_mean - rage_mean)

    return simple_pol, re_pol, emotion_dicts


# ── Toxicity scoring (dedicated RoBERTa model) ─────────────────────────
_CACHED_TOX_PIPE: Any = None
_CACHED_TOX_DEVICE: Optional[int] = None


def _get_toxicity_pipe(device_id: int = -1, batch_size: int = 64):
    """Lazily load and cache a toxicity classification pipeline."""
    global _CACHED_TOX_PIPE, _CACHED_TOX_DEVICE  # noqa: PLW0603
    if _CACHED_TOX_PIPE is not None and _CACHED_TOX_DEVICE == device_id:
        return _CACHED_TOX_PIPE
    try:
        from transformers import pipeline as hf_pipeline
        dev = device_id if device_id >= 0 else -1
        # 2026-05-19 fix: framework="pt" prevents Keras 3 / tf-keras compat crash.
        _CACHED_TOX_PIPE = hf_pipeline(
            "text-classification",
            model="s-nlp/roberta_toxicity_classifier",
            device=dev,
            batch_size=batch_size,
            truncation=True,
            max_length=512,
            framework="pt",
        )
        _CACHED_TOX_DEVICE = device_id
    except Exception as exc:
        print(f"[toxicity] WARNING: failed to load toxicity model: {exc}", flush=True)
        _CACHED_TOX_PIPE = None
    return _CACHED_TOX_PIPE


def score_texts_toxicity(
    texts: List[str],
    *,
    device_id: int = -1,
    batch_size: int = 64,
) -> List[float]:
    """Score texts for toxicity using s-nlp/roberta_toxicity_classifier.

    Returns per-text toxicity probability in [0, 1].
    Texts that fail to score return 0.0.
    """
    pipe = _get_toxicity_pipe(device_id=device_id, batch_size=batch_size)
    if pipe is None:
        return [0.0] * len(texts)

    clean = [t.strip() if (t and t.strip()) else "neutral" for t in texts]
    scores: List[float] = []
    for i in range(0, len(clean), batch_size):
        batch = clean[i : i + batch_size]
        try:
            results = pipe(batch)
            for r in results:
                # The model outputs {"label": "toxic"/"neutral", "score": float}
                if r["label"].lower() == "toxic":
                    scores.append(float(r["score"]))
                else:
                    scores.append(1.0 - float(r["score"]))
        except Exception:
            scores.extend([0.0] * len(batch))
    return scores
