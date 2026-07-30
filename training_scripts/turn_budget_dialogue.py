"""
turn_budget_dialogue.py
========================

Per-user multi-turn dialogue generator for the HyperPEFT-LoRA turn-budget
experiment (Paper 2 Appendix H).  For one (condition, cohort) configuration,
a fixed pool of users each writes T sequential turns on a shared neutral
topic, accumulating their own dialogue history.  At each turn we record only
the raw text plus a coherence flag; all derived metrics (style cosine,
signature drift, Hedges' g, AUC, bootstrap CIs, permutation tests) are
deferred to ``analyze_turn_budget.py`` so this script stays
generation-only.

Conditions:
  - hyperpeft   : full HyperPEFT-LoRA delta channel active
  - zero_delta  : same engine, force_zero_delta=True (descriptor channel cut)
  - vanilla     : population vanilla LoRA, no hypernetwork

Output layout (under ``out_dir``)::

    dialogue.parquet         columns = uid, atype, turn, condition, text,
                                       coherent, prompt_len, gen_len
    metadata.json            run config + pool composition
    pool.json                {"rage": [uids...], "empath": [uids...]}
    progress.txt             advances every batch, used by phase resume guard
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Reuse the production engine + helpers; keep the runner light.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_hyperlora_forum import (  # noqa: E402
    HyperPEFTLoRAEngine,
    ProfileBuilder,
    _is_coherent,
    _postprocess_generated_text,
    _set_seed,
)

LOG = logging.getLogger("turn_budget")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


# ---------------------------------------------------------------------
# Pool selection (stratified by cohort, deterministic given seed)
# ---------------------------------------------------------------------

def _load_cohort_map(labels_csv: Path) -> Dict[int, str]:
    df = pd.read_csv(labels_csv)
    if "target_user_id" not in df.columns:
        raise KeyError("labels_csv must contain target_user_id")
    label_col = "label" if "label" in df.columns else "sentiment_label"
    if label_col not in df.columns:
        raise KeyError("labels_csv must contain label or sentiment_label")
    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            uid = int(row["target_user_id"])
        except Exception:
            continue
        lab = str(row[label_col]).strip().lower()
        if lab in {"rage", "empath"}:
            out[uid] = lab
    return out


def _select_pool(
    author_df: pd.DataFrame,
    cohort_map: Dict[int, str],
    n_per_cohort: int,
    seed: int,
) -> Dict[str, List[int]]:
    rng = np.random.RandomState(seed)
    candidates: Dict[str, List[int]] = {"rage": [], "empath": []}
    valid_uids = set(int(u) for u in author_df["target_user_id"].tolist())
    for uid, lab in cohort_map.items():
        if uid in valid_uids:
            candidates[lab].append(uid)

    out: Dict[str, List[int]] = {}
    for cohort, uids in candidates.items():
        if len(uids) < n_per_cohort:
            LOG.warning("cohort=%s only %d candidates < requested %d",
                        cohort, len(uids), n_per_cohort)
            chosen = list(uids)
        else:
            idx = rng.choice(len(uids), size=n_per_cohort, replace=False)
            chosen = [uids[int(i)] for i in idx]
        chosen = sorted(int(u) for u in chosen)
        out[cohort] = chosen
        LOG.info("pool[%s] = %d users", cohort, len(chosen))
    return out


# ---------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------

def _maybe_norm_stats(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _build_engine_hyperpeft(args: argparse.Namespace) -> HyperPEFTLoRAEngine:
    """Treatment + zero-delta both go through the production HyperPEFT engine.

    Zero-delta is selected per-batch in _generate_batch_hyperpeft by zeroing
    the emitted delta parts; we do not rebuild the engine.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_modules = [t for t in str(args.target_modules).split(",") if t]
    engine = HyperPEFTLoRAEngine(
        base_model=args.base_model,
        hyper_dir=args.hyper_dir,
        target_modules=target_modules,
        lora_r=int(args.lora_r),
        lora_alpha=float(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        inject_clamp=float(args.inject_clamp),
        delta_gain=float(args.delta_gain),
        use_best_ckpt=True,
        online=True,
        qlora=False,
        device=device,
        emit_both=True,
    )
    # Batched HF .generate with left-pad requires left-side padding.
    try:
        engine.tok.padding_side = "left"
    except Exception:
        pass
    # Per-turn truncation in _build_user_prompt already keeps the prompt
    # under args.max_len; raise the tokenizer's model_max_length so the
    # encode-time warning ("sequence length is longer than ...") does not
    # fire on the rare path where a single tokenize call sees the full
    # accumulated history before our windowing logic clips it.
    try:
        engine.tok.model_max_length = int(1_000_000_000)
    except Exception:
        pass
    return engine


def _build_engine_vanilla(args: argparse.Namespace) -> Any:
    """Vanilla LoRA path: load PeftModel directly, no hypernetwork.

    Returns a tuple (model, tokenizer, device) since there is no
    HyperPEFTLoRAEngine wrapper for the vanilla case.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.vanilla_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        tok.model_max_length = int(1_000_000_000)
    except Exception:
        pass
    backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    ).to(device)
    backbone.resize_token_embeddings(len(tok))
    # Vanilla LoRA adapter directory — best/ if present, else last/
    candidate = Path(args.vanilla_dir) / "best"
    if not candidate.exists():
        candidate = Path(args.vanilla_dir) / "last"
    if not candidate.exists():
        candidate = Path(args.vanilla_dir)
    model = PeftModel.from_pretrained(backbone, str(candidate))
    model.eval()
    return model, tok, device


# ---------------------------------------------------------------------
# Prompt construction (system + topic + accumulated history)
# ---------------------------------------------------------------------

SYSTEM_LINE = (
    "You are participating in an online forum discussion. Reply in your own "
    "voice, briefly and naturally."
)


def _build_user_prompt(
    *,
    topic: str,
    history: List[str],
    max_len: int,
    tokenizer: Any,
    sep_id: Optional[int],
) -> Tuple[List[int], List[int]]:
    """Build a one-user prompt: system + topic + recency-truncated history.

    Returns (token_ids, attention_mask).  We tokenize each turn separately and
    accumulate from newest backward, dropping oldest turns until the running
    total fits inside the budget.  This avoids handing the HF tokenizer a
    sequence longer than its model_max_length (which triggers a noisy warning
    around turn 21+), while still emitting an identical clipped prefix that
    the model would have received under the previous post-tokenize truncation.
    """
    bos = getattr(tokenizer, "bos_token_id", None)
    prefix: List[int] = []
    if bos is not None and int(bos) >= 0:
        prefix.append(int(bos))

    sep_slots = 1 if (sep_id is not None and int(sep_id) >= 0) else 0
    budget = int(max(64, max_len - 8 - len(prefix) - sep_slots))

    header_str = SYSTEM_LINE + "\n" + f"Topic: {topic.strip()}"
    header_ids = list(tokenizer(header_str, add_special_tokens=False)["input_ids"])
    if len(header_ids) > budget:
        header_ids = header_ids[-budget:]
    remaining = budget - len(header_ids)

    history_ids: List[int] = []
    if history and remaining > 0:
        intro_ids = list(tokenizer("\nPrevious turns:", add_special_tokens=False)["input_ids"])
        if len(intro_ids) < remaining:
            kept_rev: List[List[int]] = []
            used = len(intro_ids)
            # Tokenize newest -> oldest, accumulate while we have room
            for i, h in enumerate(reversed(history), start=1):
                turn_idx = len(history) - i + 1
                t_str = f"\n{turn_idx}. {h.strip()}"
                t_ids = list(tokenizer(t_str, add_special_tokens=False)["input_ids"])
                if used + len(t_ids) > remaining:
                    break
                kept_rev.append(t_ids)
                used += len(t_ids)
            if kept_rev:
                history_ids = list(intro_ids)
                for t_ids in reversed(kept_rev):
                    history_ids.extend(t_ids)

    ids: List[int] = list(prefix) + list(header_ids) + list(history_ids)
    if sep_id is not None and int(sep_id) >= 0:
        ids.append(int(sep_id))

    attn = [1] * len(ids)
    return ids, attn


def _left_pad_batch(
    sequences: List[List[int]],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Left-pad a list of token id lists to a common length.

    Returns (input_ids[B,L], attention_mask[B,L], orig_lens[B]).
    """
    L = max(len(s) for s in sequences)
    B = len(sequences)
    out_ids = np.full((B, L), pad_id, dtype=np.int64)
    out_msk = np.zeros((B, L), dtype=np.int64)
    orig_lens: List[int] = []
    for i, s in enumerate(sequences):
        n = len(s)
        out_ids[i, L - n:] = np.asarray(s, dtype=np.int64)
        out_msk[i, L - n:] = 1
        orig_lens.append(n)
    return (torch.from_numpy(out_ids), torch.from_numpy(out_msk), orig_lens)


# ---------------------------------------------------------------------
# Generation paths
# ---------------------------------------------------------------------

def _generate_batch_hyperpeft(
    engine: HyperPEFTLoRAEngine,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    g_batch: torch.Tensor,
    force_zero: bool,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Batched generate through the HyperPEFT engine.

    The engine's forward hooks are batch-aware: when _g_for_forward has shape
    [B, K] the FlatHypernetwork emits a [B, ...] delta that the einsum hook
    consumes per-batch.  We set the buffers explicitly so generate_reply does
    not collapse them back to single-sample.
    """
    model = engine.model
    eos_ids: List[int] = []
    if engine.end_id is not None:
        eos_ids.append(int(engine.end_id))
    if getattr(engine.tok, "eos_token_id", None) is not None:
        eos_ids.append(int(engine.tok.eos_token_id))
    eos_ids = sorted({int(x) for x in eos_ids if int(x) >= 0})
    eos_arg: Any = None
    if eos_ids:
        eos_arg = eos_ids[0] if len(eos_ids) == 1 else eos_ids

    model._g_for_forward = g_batch
    model._force_zero_flag = bool(force_zero)
    try:
        try:
            emitted = model._emit_delta_parts(g_batch, force_zero=force_zero)
        except TypeError:
            emitted = model._emit_delta_parts(g_batch, force_zero)
        if isinstance(emitted, tuple) and len(emitted) == 3:
            delta_parts, delta_A_parts, _ = emitted
        else:
            delta_parts, _ = emitted
            delta_A_parts = None

        if force_zero:
            delta_parts = [torch.zeros_like(d) for d in delta_parts]
            if delta_A_parts is not None:
                delta_A_parts = [torch.zeros_like(d) for d in delta_A_parts]

        model._delta_for_forward = delta_parts
        model._delta_A_for_forward = delta_A_parts

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=True,
            top_p=float(args.top_p),
            temperature=float(args.temperature),
            pad_token_id=int(engine.tok.pad_token_id) if engine.tok.pad_token_id is not None else int(engine.tok.eos_token_id),
        )
        if eos_arg is not None:
            gen_kwargs["eos_token_id"] = eos_arg
        if args.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = float(args.repetition_penalty)
        if int(args.no_repeat_ngram_size) > 0:
            gen_kwargs["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)

        model.backbone.config.use_cache = True
        with torch.no_grad():
            out = model.backbone.generate(**gen_kwargs)
        return out
    finally:
        model.backbone.config.use_cache = False
        model._g_for_forward = None
        model._delta_for_forward = None
        if hasattr(model, "_delta_A_for_forward"):
            model._delta_A_for_forward = None
        model._force_zero_flag = False


def _generate_batch_vanilla(
    model: Any,
    tok: Any,
    device: torch.device,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else int(tok.eos_token_id)
    eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else pad_id
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=True,
        top_p=float(args.top_p),
        temperature=float(args.temperature),
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    if args.repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = float(args.repetition_penalty)
    if int(args.no_repeat_ngram_size) > 0:
        gen_kwargs["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)
    model.config.use_cache = True
    try:
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        return out
    finally:
        model.config.use_cache = False


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Per-user multi-turn dialogue runner (turn-budget).")
    ap.add_argument("--condition", required=True, choices=["hyperpeft", "zero_delta", "vanilla"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--author_parquet", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--norm_stats_json", default="")
    ap.add_argument("--base_model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--hyper_dir", default="")
    ap.add_argument("--vanilla_dir", default="")
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--lora_r", type=int, default=24)
    ap.add_argument("--lora_alpha", type=float, default=48.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--inject_clamp", type=float, default=0.020)
    ap.add_argument("--delta_gain", type=float, default=8.0)
    ap.add_argument("--feature_clamp", type=float, default=3.0)
    ap.add_argument("--outlier_threshold", type=float, default=4.0)
    ap.add_argument("--filter_outliers", action="store_true")
    ap.add_argument("--n_per_cohort", type=int, default=300)
    ap.add_argument("--n_turns", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--top_p", type=float, default=0.90)
    ap.add_argument("--temperature", type=float, default=0.70)
    ap.add_argument("--repetition_penalty", type=float, default=1.10)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--seed", type=int, default=142)
    ap.add_argument("--save_every_turns", type=int, default=5)
    ap.add_argument("--g_columns", default="")
    ap.add_argument("--force_zero_delta", action="store_true",
                    help="Ablation: zero all hypernetwork deltas (descriptor channel cut).")
    return ap.parse_args()


def _load_g_columns(args: argparse.Namespace) -> List[str]:
    """Resolve g column names from the saved feature_names.json under hyper_dir.

    Falls back to args.g_columns if provided.  For vanilla we still need
    profiles for the profile-based topic prompt, so we still load them but
    they are ignored at generate time.
    """
    if args.g_columns:
        return [c for c in args.g_columns.split(",") if c]
    candidates = []
    if args.hyper_dir:
        candidates.append(Path(args.hyper_dir) / "best" / "feature_names.json")
        candidates.append(Path(args.hyper_dir) / "last" / "feature_names.json")
    for c in candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                cols = obj.get("feature_names") or obj.get("columns") or obj.get("g_columns")
                if cols:
                    return list(cols)
            except Exception:
                continue
    raise FileNotFoundError(
        "could not resolve g_columns (provide --g_columns or hyper_dir with feature_names.json)"
    )


def _g_dim(g_columns: List[str]) -> int:
    return len(g_columns)


def _materialize_g(uid: int, profiles: Dict[int, Any]) -> np.ndarray:
    p = profiles.get(int(uid), None)
    if p is None:
        return np.zeros((1,), dtype=np.float32)
    return np.asarray(p.gvec, dtype=np.float32)


def main() -> int:
    args = parse_args()
    _set_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("turn_budget runner | condition=%s | out=%s", args.condition, str(out_dir))
    LOG.info("topic=%s | n_per_cohort=%d | n_turns=%d | batch_size=%d",
             args.topic, args.n_per_cohort, args.n_turns, args.batch_size)

    # -- Resume guard: if a complete dialogue.parquet exists with all turns,
    #    exit fast.  Used by the YAML phase wrapper for retry safety.
    final_path = out_dir / "dialogue.parquet"
    if final_path.exists():
        try:
            existing = pd.read_parquet(final_path)
            done_marker = out_dir / "dialogue.complete"
            if done_marker.exists() and "turn" in existing.columns:
                if int(existing["turn"].max()) >= int(args.n_turns):
                    LOG.info("resume: found complete dialogue.parquet (max_turn=%d); exiting",
                             int(existing["turn"].max()))
                    return 0
        except Exception:
            pass

    # -- Load author parquet, labels, build pool
    LOG.info("loading author parquet: %s", args.author_parquet)
    author_df = pd.read_parquet(args.author_parquet)
    LOG.info("loading labels csv: %s", args.labels_csv)
    cohort_map = _load_cohort_map(Path(args.labels_csv))
    pool = _select_pool(author_df, cohort_map, int(args.n_per_cohort), int(args.seed))
    with open(out_dir / "pool.json", "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2)
    all_uids = sorted(pool["rage"]) + sorted(pool["empath"])
    uid_to_atype = {uid: "rage" for uid in pool["rage"]}
    uid_to_atype.update({uid: "empath" for uid in pool["empath"]})

    # -- Build user profiles (gvec needed for hyperpeft + zero_delta paths)
    g_columns = _load_g_columns(args)
    g_dim = _g_dim(g_columns)
    pb = ProfileBuilder(
        author_df,
        g_columns=g_columns,
        g_dim=g_dim,
        feature_clamp=float(args.feature_clamp),
        outlier_threshold=float(args.outlier_threshold),
    )
    profiles, filtered = pb.build_profiles(
        all_uids,
        uid_to_atype,
        filter_outliers=bool(args.filter_outliers),
    )
    if filtered:
        LOG.warning("filtered %d outlier users", len(filtered))
        all_uids = [u for u in all_uids if u in profiles]

    # Optional norm stats (currently unused for inference but kept for parity)
    _ = _maybe_norm_stats(args.norm_stats_json)

    # -- Build engine path
    if args.condition in ("hyperpeft", "zero_delta"):
        engine = _build_engine_hyperpeft(args)
        tok = engine.tok
        device = engine.device if hasattr(engine, "device") else torch.device("cuda")
        sep_id = engine.sep_id
        force_zero = (args.condition == "zero_delta") or bool(getattr(args, "force_zero_delta", False))
        vanilla_pack = None
    else:
        model_v, tok, device = _build_engine_vanilla(args)
        engine = None
        sep_id = None
        force_zero = False
        vanilla_pack = (model_v, tok, device)

    pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else int(tok.eos_token_id)

    # -- Per-user state: each user owns a rolling history of generated turns
    histories: Dict[int, List[str]] = {uid: [] for uid in all_uids}

    # -- Assemble batches once.  Order is deterministic per seed.
    rng = np.random.RandomState(int(args.seed))
    order = list(all_uids)
    rng.shuffle(order)
    B = int(args.batch_size)
    batches: List[List[int]] = [order[i:i + B] for i in range(0, len(order), B)]
    LOG.info("batches per turn = %d (B=%d, pool=%d)", len(batches), B, len(order))

    rows: List[Dict[str, Any]] = []
    progress_path = out_dir / "progress.txt"

    t0 = time.time()
    for t in range(1, int(args.n_turns) + 1):
        turn_t0 = time.time()
        for bi, batch in enumerate(batches):
            # --- Build batched prompt
            seqs: List[List[int]] = []
            for uid in batch:
                ids, _ = _build_user_prompt(
                    topic=args.topic,
                    history=histories[uid],
                    max_len=int(args.max_len),
                    tokenizer=tok,
                    sep_id=sep_id,
                )
                seqs.append(ids)
            input_ids, attention_mask, orig_lens = _left_pad_batch(seqs, pad_id=pad_id)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # --- Build batched g
            if engine is not None:
                gs = np.stack([_materialize_g(uid, profiles) for uid in batch], axis=0)
                g_batch = torch.from_numpy(gs).float().to(device)
                out_ids = _generate_batch_hyperpeft(
                    engine,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    g_batch=g_batch,
                    force_zero=force_zero,
                    args=args,
                )
            else:
                model_v, _, _ = vanilla_pack
                out_ids = _generate_batch_vanilla(
                    model_v,
                    tok,
                    device,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    args=args,
                )

            # --- Decode + record
            for i, uid in enumerate(batch):
                full = out_ids[i].tolist()
                pad_count = int(input_ids.size(1)) - int(orig_lens[i])
                gen_only = full[int(input_ids.size(1)):]
                # Strip trailing pads / eos
                eos_set = {int(tok.eos_token_id) if tok.eos_token_id is not None else -1, int(pad_id)}
                while gen_only and int(gen_only[-1]) in eos_set:
                    gen_only.pop()
                text = tok.decode(gen_only, skip_special_tokens=True)
                text = _postprocess_generated_text(text)
                coh = bool(_is_coherent(text))
                histories[uid].append(text)
                rows.append({
                    "uid": int(uid),
                    "atype": uid_to_atype[int(uid)],
                    "turn": int(t),
                    "condition": args.condition,
                    "text": text,
                    "coherent": coh,
                    "prompt_len": int(orig_lens[i]),
                    "gen_len": int(len(gen_only)),
                })

        # Periodic checkpoint of generated text
        turn_dt = time.time() - turn_t0
        LOG.info(
            "turn=%d/%d  batches=%d  coh=%.3f  elapsed=%.1fs  total=%.1fs",
            t, int(args.n_turns), len(batches),
            float(np.mean([r["coherent"] for r in rows[-len(order):]])) if rows else float("nan"),
            turn_dt,
            time.time() - t0,
        )
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(f"turn={t} elapsed_total={time.time() - t0:.1f}\n")

        if int(t) % max(1, int(args.save_every_turns)) == 0 or t == int(args.n_turns):
            df = pd.DataFrame(rows)
            tmp_path = out_dir / "dialogue.parquet.tmp"
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, final_path)

    # Final write + complete marker
    df = pd.DataFrame(rows)
    df.to_parquet(final_path, index=False)
    metadata = {
        "condition": args.condition,
        "topic": args.topic,
        "n_per_cohort": int(args.n_per_cohort),
        "n_turns": int(args.n_turns),
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "max_len": int(args.max_len),
        "top_p": float(args.top_p),
        "temperature": float(args.temperature),
        "repetition_penalty": float(args.repetition_penalty),
        "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
        "lora_r": int(args.lora_r),
        "lora_alpha": float(args.lora_alpha),
        "inject_clamp": float(args.inject_clamp),
        "delta_gain": float(args.delta_gain),
        "seed": int(args.seed),
        "g_columns": list(g_columns),
        "n_pool": int(len(order)),
        "n_rage": int(len(pool["rage"])),
        "n_empath": int(len(pool["empath"])),
        "wallclock_sec": float(time.time() - t0),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    (out_dir / "dialogue.complete").write_text("ok\n", encoding="utf-8")
    LOG.info("turn_budget runner DONE | rows=%d | wallclock=%.1fs",
             len(rows), time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
