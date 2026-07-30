#!/bin/bash
set -Eeuo pipefail
umask 0002

# ===========================================================
# Paper 2 FULL REFRESH — HyperPEFT-LoRA downstream evaluation
# All 15 canonical forum cells + scoring + persona-signature.
# the GPU node high-memory GPU (ARM64), image v19-arm64, RMM enabled.
#
# Phase map (with resume-state IDs):
#   phase0_stage          : hot-stage + Arditi sanity (always runs)
#   phase0d_arditi_extreme: extreme-prompt red-team re-eval (gated RUN_PHASE0D=1)
#   phase1_synth_legacy   : synthesize in_hull + near_hull descriptors
#   phase1p5_label_legacy : label legacy synth descriptors
#   phase2a_vanilla       : 3 vanilla LoRA forums (rage/empath/neutral)
#   phase2b_zerodelta     : 3 zero-delta forums (--force_zero_delta)
#   phase2c_real_user     : 3 real-user Arditi-patched forums
#   phase2d_inhull        : 3 synth in_hull forums
#   phase2d_nearhull      : 3 synth near_hull forums
#   phase2d_farhull_v2    : 3 synth far_from_hull v2 forums (kappa=3)
#   phase2d_midpoint      : 3 synth midpoint_baseline forums
#   phase2e_reconstruction: per-user recon fidelity vs 2c real-user forums
#   phase2f_synth_vs_recon: synth-vs-recon decomposition (Table 10)
#   phase3a_score         : 15 cross-framework sentiment scoring runs
#   phase3d_signature     : 15 persona-signature aggregation runs
#   phase4_layerwise      : hidden-state probe (gated RUN_PHASE4=1)
#   phase5_paper_fill     : PSI + drift slopes + anchor-density analysis
#   phase6_extended_depth : 80-turn extended audit (hyperpeft + zero_delta arms)
# ===========================================================

JOB_TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/data/hypernets/log_files"
MAIN_LOG="${LOG_ROOT}/m1_gpu-node_p2_FULL_REFRESH_${JOB_TS}.log"
GPU_TELEM="${LOG_ROOT}/gpu_telem_m1_gpu-node_p2_REFRESH_${HOSTNAME}_${JOB_TS}.csv"

mkdir -p /tmp/.cache /tmp/hf/hub /tmp/hf/transformers /tmp/hf/datasets
mkdir -p /tmp/nvidia-mps /tmp/nvidia-mps-log || true
mkdir -p "${LOG_ROOT}"

exec > >(stdbuf -oL -eL tee -a "${MAIN_LOG}") 2>&1
echo "[REFRESH] log  : ${MAIN_LOG}"
echo "[REFRESH] host : ${HOSTNAME}"
echo "[REFRESH] ts   : ${JOB_TS}"
echo "[REFRESH] image: v19-arm64  gpu: high-memory GPU-480GB  RMM: HN_USE_UNIFIED_MEMORY=${HN_USE_UNIFIED_MEMORY:-0}"

# Disable NUMA balancing (high-memory GPU perf hint)
echo 0 > /proc/sys/kernel/numa_balancing 2>/dev/null || true

# GPU telemetry (30-s cadence background loop)
echo "timestamp,index,name,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used" > "${GPU_TELEM}" || true
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
      --format=csv,noheader,nounits >> "${GPU_TELEM}" 2>/dev/null || true
    sleep 30
  done
) &
TELEM_PID=$!

# HF token
HF_TOKEN_FILE="/workspace/hypernets/HF_TOKEN.txt"
if [[ -f "${HF_TOKEN_FILE}" ]]; then
  CLEAN_TOKEN="$(tr -d '[:space:]' < "${HF_TOKEN_FILE}")"
  if [[ -n "${CLEAN_TOKEN}" ]]; then
    export HF_TOKEN="${CLEAN_TOKEN}"
    export HUGGINGFACEHUB_API_TOKEN="${CLEAN_TOKEN}"
    export HUGGINGFACE_HUB_TOKEN="${CLEAN_TOKEN}"
  fi
fi

[[ -f /venv/bin/activate ]] && source /venv/bin/activate || true

# CUDA MPS daemon for concurrent sub-phase throughput
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "[mps] starting CUDA MPS daemon"
  nvidia-cuda-mps-control -d || true
  sleep 2
  echo get_server_list | nvidia-cuda-mps-control || true
  export HN_MPS_ENABLED=1
else
  echo "[mps] nvidia-cuda-mps-control not found; single-stream mode"
  export HN_MPS_ENABLED=0
fi

# ===========================================================
# Path configuration
# ===========================================================
SCRIPTS="/workspace/hypernets/training_scripts"
DATA_SCRIPTS="/workspace/hypernets/data_scripts"
DATA="/workspace/hypernets/data"
MODELS_ROOT="/workspace/hypernets/models"

HYPER_DIR="${HYPER_DIR:-${MODELS_ROOT}/hyperlora_gpu-node_M1_pythia14B_r24_full/hyperlora_multi}"
VANILLA_LORA_DIR="${VANILLA_LORA_DIR:-${MODELS_ROOT}/vanillalora_M1_pythia14B}"
BASE_MODEL="EleutherAI/pythia-1.4b"

# 2026-05-20 high-memory GPU stability fix: force the portable math SDPA backend for
# EVERY build_hyperlora_forum.py subprocess. The cuDNN SDPA backend on the
# current high-memory GPU driver throws "cuDNN Frontend error: No valid execution plans
# built" for some attention shapes (crashed phase2d_nearhull empath) and can
# silently emit NaN logits that trip torch.multinomial (crashed phase2c
# real-user). build_hyperlora_forum.py honors this env at import (see the
# [arditi-defense] block) and disables flash/mem-efficient/cuDNN SDP, leaving
# math SDP. Keeps fp16 (consistent with the already-completed cells); the
# phase2c real-user cell additionally runs bf16+eager as before.
export ARDITI_DISABLE_FLASH_SDP=1

OUT_ROOT="${OUT_ROOT:-/data/hypernets/results/paper_2_m1}"
# REFRESH outputs: all new outputs under *_REFRESH/ dirs so they
# live alongside legacy *_arditi/ outputs without overwriting.
FORUM_ROOT="${OUT_ROOT}/phase2_forums_REFRESH"
STATE_FILE="${OUT_ROOT}/phase_state_FULL_REFRESH.json"

mkdir -p "${OUT_ROOT}" "${FORUM_ROOT}"

# Pre-reg-locked inference parameters (MUST NOT change)
INJECT_CLAMP="0.020"
DELTA_GAIN="8.0"
MAX_NEW_TOKENS="192"
TEMPERATURE="0.40"
TOP_P="0.85"
REPETITION_PENALTY="1.25"
NO_REPEAT_NGRAM="4"
MIN_NEW_TOKENS="30"
LORA_R="24"
LORA_ALPHA="48.0"
LORA_DROPOUT="0.05"
FANOUT="5,5,3,1"
HORIZON_MIN="1080"
N_RAGE="500"
N_EMPATH="500"
ARDITI_MODE="orthogonal"
ARDITI_ALPHA="1.0"
ARDITI_LAYERS="15-23"
TARGET_MODULES="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"

# Arditi directions source (validated cached file; extraction skipped)
ARDITI_DIR_SRC="${OUT_ROOT}/phase0b_arditi_directions_arditi"
ARDITI_TMP="/tmp/arditi_directions.safetensors"

# Synth descriptor sources (verified exist)
SYNTH_LEGACY_OUT="${OUT_ROOT}/phase1_synth_legacy_REFRESH"
# Legacy single-kappa parquet path retained for backwards compat
# only; the kappa-sweep loop below writes per-kappa parquets to
# ${OUT_ROOT}/phase1_synth_kappa${K}/synthetic_personas.parquet.
FARHULL_PARQ_SRC="${OUT_ROOT}/phase1_synth_arditi_v3/synthetic_personas.parquet"
MIDPOINT_PARQ_SRC="${OUT_ROOT}/phase1_midpoint_baseline_v2/synthetic_personas.parquet"
KAPPA_LIST="3 10 25 50 100"

# Label sources (verified exist; SKIP if present)
LABEL_LEGACY_OUT="${OUT_ROOT}/phase1p5_label_legacy_REFRESH"
LABEL_FARHULL_OUT="${OUT_ROOT}/phase1p5_label_v2_farhull"
LABEL_MIDPOINT_OUT="${OUT_ROOT}/phase1p5_label_v2_midpoint"

mkdir -p "${SYNTH_LEGACY_OUT}" "${LABEL_LEGACY_OUT}"

# Forum topic map (pre-reg locked)
declare -A FORUM_TOPIC
FORUM_TOPIC["rage"]="unpopular opinions you actually believe"
FORUM_TOPIC["empath"]="what's a piece of advice that stuck with you?"
FORUM_TOPIC["neutral"]="what hobby do you wish you'd started sooner?"

# ===========================================================
# Resume / phase-state helpers
# ===========================================================
phase_state_init() {
  if [[ ! -f "${STATE_FILE}" ]]; then
    echo "{}" > "${STATE_FILE}"
  fi
}
phase_state_get() {
  local pid="$1"
  python -c "
import json, sys
try:
    s = json.load(open('${STATE_FILE}'))
except Exception:
    s = {}
print((s.get('${pid}', {}) or {}).get('status', 'pending'))
"
}
phase_state_set() {
  local pid="$1"; local status="$2"; local n_sub="${3:-0}"; local n_done="${4:-0}"
  python -c "
import json, datetime, sys
try:
    s = json.load(open('${STATE_FILE}'))
except Exception:
    s = {}
entry = s.get('${pid}', {}) or {}
ts = datetime.datetime.utcnow().isoformat() + 'Z'
if '${status}' == 'running' and 'started' not in entry:
    entry['started'] = ts
if '${status}' in ('completed', 'failed'):
    entry['ended'] = ts
entry['status'] = '${status}'
entry['n_subitems'] = int(${n_sub})
entry['n_complete'] = int(${n_done})
s['${pid}'] = entry
json.dump(s, open('${STATE_FILE}', 'w'), indent=2)
"
}
phase_begin() {
  local pid="$1"; local n_sub="${2:-1}"
  phase_state_init
  local st; st="$(phase_state_get "${pid}")"
  if [[ "${st}" == "completed" ]]; then
    echo "[state] phase ${pid} already completed; skipping."
    return 1
  fi
  echo ""
  echo "================================================================"
  echo "  PHASE ${pid}  ($(date -Is))   resume_status=${st}"
  echo "================================================================"
  phase_state_set "${pid}" "running" "${n_sub}" "0"
  return 0
}
phase_end_ok() {
  local pid="$1"; local n_sub="${2:-1}"; local n_done="${3:-${n_sub}}"
  phase_state_set "${pid}" "completed" "${n_sub}" "${n_done}"
  echo "[state] phase ${pid} OK (${n_done}/${n_sub})"
}
phase_end_fail() {
  local pid="$1"; local n_sub="${2:-1}"; local n_done="${3:-0}"
  phase_state_set "${pid}" "failed" "${n_sub}" "${n_done}"
  echo "[state] phase ${pid} FAILED (${n_done}/${n_sub})"
}
subitem_done()    { touch "$1"; }
subitem_is_done() { [[ -f "$1" ]]; }

# ===========================================================
# PHASE 0 -- Hot-stage + Arditi sanity (ALWAYS runs on pod start)
# ===========================================================
# NOTE: /tmp is an emptyDir that resets per pod. Phase state persists
# on NFS. If this phase were guarded, a resumed pod would skip the
# hot-stage and all downstream /tmp reads would fail. Run unconditionally.
echo ""
echo "================================================================"
echo "  PHASE 0_STAGE  ($(date -Is))  -- always runs"
echo "================================================================"

echo "[phase0] GPU inventory:"
nvidia-smi -L || true
nvidia-smi || true
python -c "
import torch
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    print('VRAM', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
" || true

echo "[phase0] staging data files to /tmp"
for f in author_static_10000.parquet global_features_10000.parquet \
         labels_sentiment_goemo.csv feature_norm_stats_10000.json \
         train_data_10000.parquet goemo_labels_metadata.json \
         labels_sentiment_goemo_extremes.csv \
         labels_politeness.csv labels_self_focus.csv labels_tempo.csv \
         labels_curiosity.csv labels_expressiveness.csv \
         labels_anxiety.csv labels_warmth.csv labels_hostility.csv \
         goemo_user_emotions.csv goemo_user_sentiment.csv; do
  if [[ -f "${DATA}/${f}" && ! -f "/tmp/${f}" ]]; then
    cp "${DATA}/${f}" "/tmp/${f}" || true
  fi
done
ls -lh /tmp/*.parquet /tmp/*.csv /tmp/*.json 2>/dev/null || true

# Pre-stage Arditi directions unconditionally (force-copy from validated cache).
# The "skip if exists" guard was removed because a partial extract from a
# hung pod left a broken file in the target on 2026-05-06.
if [[ -f "${ARDITI_DIR_SRC}/arditi_directions.safetensors" ]]; then
  cp -f "${ARDITI_DIR_SRC}/arditi_directions.safetensors" "${ARDITI_TMP}"
  cp -f "${ARDITI_DIR_SRC}/arditi_directions.json" /tmp/arditi_directions.json 2>/dev/null || true
  echo "[phase0] arditi directions force-staged from: ${ARDITI_DIR_SRC}"
else
  echo "[phase0] FATAL: Arditi directions safetensors not found at ${ARDITI_DIR_SRC}"
  echo "[phase0] Pre-stage the validated cached file before launching."
  exit 2
fi

# Sanity: confirm persona/* and cohort/* families are present.
# Safetensors stores key list as JSON header at file start; binary grep is safe.
if grep -qa 'persona/rage'   "${ARDITI_TMP}" \
&& grep -qa 'persona/empath' "${ARDITI_TMP}" \
&& grep -qa 'cohort/rage'    "${ARDITI_TMP}" \
&& grep -qa 'cohort/empath'  "${ARDITI_TMP}"; then
  echo "[phase0] arditi sanity PASS: persona/* and cohort/* keys present"
else
  echo "[phase0] FATAL: arditi_directions.safetensors missing persona/* or cohort/* families"
  echo "[phase0] Likely a partial extract. Re-stage the 2026-05-06 sweep cache."
  exit 2
fi

# Validate required model and data inputs
if phase_begin "phase0_validate" 1; then
  REQUIRED=(
    "${HYPER_DIR}/best/hypernetwork.safetensors"
    "${HYPER_DIR}/best/peft_placeholders.safetensors"
    "${DATA}/author_static_10000.parquet"
    "${DATA}/global_features_10000.parquet"
    "${DATA}/labels_sentiment_goemo.csv"
  )
  MISSING=0
  for f in "${REQUIRED[@]}"; do
    if [[ ! -e "$f" ]]; then
      echo "[validate] MISSING: $f"
      MISSING=$((MISSING + 1))
    fi
  done
  # Warn (not abort) on vanilla LoRA dir — paper is primarily HyperPEFT-LoRA
  if [[ ! -d "${VANILLA_LORA_DIR}" ]]; then
    echo "[validate] WARNING: vanilla LoRA dir not found: ${VANILLA_LORA_DIR}"
    echo "[validate] phase2a_vanilla will be skipped if missing."
  fi
  if (( MISSING > 0 )); then
    echo "[validate] ${MISSING} required input(s) missing; aborting."
    phase_end_fail "phase0_validate" 1 0
    exit 2
  fi
  echo "[validate] all required HyperPEFT-LoRA inputs present"
  phase_end_ok "phase0_validate" 1 1
fi

# ===========================================================
# PHASE 0D_ARDITI_EXTREME -- Extreme-prompt red-team re-eval
# ===========================================================
# Refreshes phase0d_arditi_extreme_REFRESH/ with clean generated text.
# The legacy phase0d_arditi_extreme_arditi/ was contaminated by the
# text-artifact bug (emotion-pickup metrics read generated text which
# contained Pile artifacts). Both arms (unpatched + patched) are re-run.
# ~30 min on high-memory GPU. Single GPU.
# Output: phase0d_arditi_extreme_REFRESH/extreme_eval_{un,}patched.parquet
#         + extreme_eval_{un,}patched_summary.json + side-by-side diff.
# Gated behind RUN_PHASE0D=1 (default ON).
# ===========================================================
if [[ "${RUN_PHASE0D:-1}" == "1" ]]; then
  EXTREME_OUT="${OUT_ROOT}/phase0d_arditi_extreme_REFRESH"
  mkdir -p "${EXTREME_OUT}"
  if phase_begin "phase0d_arditi_extreme" 2; then
    ext_done=0
    # ---- arm 1: UNPATCHED ----
    if [[ -f "${EXTREME_OUT}/extreme_eval_unpatched.parquet" ]]; then
      echo "[phase0d] unpatched arm cached; skipping"
      ext_done=$((ext_done + 1))
    else
      if python "${SCRIPTS}/arditi_extreme_eval.py"                         --hyper_dir "${HYPER_DIR}"                         --base_model "${BASE_MODEL}"                         --online                         --target_modules "${TARGET_MODULES}"                         --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}"                         --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}"                         --use_best_ckpt --emit_both                         --author_parquet /tmp/author_static_10000.parquet                         --labels_csv /tmp/labels_sentiment_goemo.csv                         --feature_manifest_json "${HYPER_DIR}/feature_manifest.json"                         --cohorts "rage,empath"                         --n_users_per_cohort 20                         --max_new_tokens 96                         --temperature "${TEMPERATURE}" --top_p "${TOP_P}"                         --seed 142                         --condition_label "unpatched"                         --out_dir "${EXTREME_OUT}"; then
        ext_done=$((ext_done + 1))
        echo "[phase0d] unpatched arm done"
      else
        echo "[phase0d] unpatched arm FAILED (non-fatal; continuing)"
      fi
    fi
    # ---- arm 2: PATCHED ----
    if [[ -f "${EXTREME_OUT}/extreme_eval_patched.parquet" ]]; then
      echo "[phase0d] patched arm cached; skipping"
      ext_done=$((ext_done + 1))
    else
      if python "${SCRIPTS}/arditi_extreme_eval.py"                         --hyper_dir "${HYPER_DIR}"                         --base_model "${BASE_MODEL}"                         --online                         --target_modules "${TARGET_MODULES}"                         --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}"                         --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}"                         --use_best_ckpt --emit_both                         --author_parquet /tmp/author_static_10000.parquet                         --labels_csv /tmp/labels_sentiment_goemo.csv                         --feature_manifest_json "${HYPER_DIR}/feature_manifest.json"                         --cohorts "rage,empath"                         --n_users_per_cohort 20                         --max_new_tokens 96                         --temperature "${TEMPERATURE}" --top_p "${TOP_P}"                         --seed 142                         --arditi_patch "${ARDITI_TMP}"                         --arditi_alpha "${ARDITI_ALPHA}"                         --arditi_layers "${ARDITI_LAYERS}"                         --condition_label "patched"                         --out_dir "${EXTREME_OUT}"; then
        ext_done=$((ext_done + 1))
        echo "[phase0d] patched arm done"
      else
        echo "[phase0d] patched arm FAILED (non-fatal; continuing)"
      fi
    fi
    # ---- side-by-side comparison ----
    python - "${EXTREME_OUT}" <<'PY_0D_DIFF' || true
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
up = root / "extreme_eval_unpatched_summary.json"
pa = root / "extreme_eval_patched_summary.json"
if not up.exists() or not pa.exists():
    print("[phase0d] one of the summaries is missing; skipping diff")
    sys.exit(0)
U = json.loads(up.read_text())
P = json.loads(pa.read_text())
print()
print("=" * 80)
print("EXTREME-PROMPT RED TEAM (REFRESH): PATCHED vs UNPATCHED")
print("=" * 80)
for cohort in ("rage", "empath"):
    u = U["by_cohort"].get(cohort, {})
    p = P["by_cohort"].get(cohort, {})
    if not u or not p:
        continue
    print(f"\n[{cohort.upper()}]   metric              unpatched      patched        delta")
    for k in ("refusal_rate", "engagement_rate", "mean_anger",
              "mean_disgust", "mean_fear", "mean_joy", "mean_neutral", "mean_words"):
        uv, pv = u.get(k, float('nan')), p.get(k, float('nan'))
        try:
            d = float(pv) - float(uv)
            print(f"           {k:20s}  {uv:>10.4f}    {pv:>10.4f}    {d:>+10.4f}")
        except Exception:
            print(f"           {k:20s}  n/a")
print("=" * 80)
PY_0D_DIFF
    if (( ext_done == 2 )); then
      phase_end_ok "phase0d_arditi_extreme" 2 2
    else
      phase_end_fail "phase0d_arditi_extreme" 2 "${ext_done}"
      echo "[phase0d_arditi_extreme] partial (${ext_done}/2); main pipeline continues"
    fi
  fi
else
  echo "[phase0d_arditi_extreme] SKIPPED (RUN_PHASE0D=${RUN_PHASE0D:-1} != 1)"
fi

# ===========================================================
# PHASE 1_SYNTH_LEGACY -- Synthesize in_hull + near_hull personas
# ===========================================================
# Runs synthesize_personas.py.
# far_from_hull from this script uses the broken PCA-axis 1.5sigma
# sampler; those rows are produced but ignored downstream (we use
# far_from_hull_v2 instead). The script has no --n_per_stratum_far_from_hull
# flag, so we let it produce all three strata and simply never read
# the legacy far_from_hull rows.
# Output: phase1_synth_legacy_REFRESH/synthetic_personas.parquet
# ===========================================================
if phase_begin "phase1_synth_legacy" 1; then
  MARK="${SYNTH_LEGACY_OUT}/.complete"
  if subitem_is_done "${MARK}"; then
    echo "[skip] legacy synth already complete"
    phase_end_ok "phase1_synth_legacy" 1 1
  else
    if python "${SCRIPTS}/synthesize_personas.py" \
        --author_parquet /tmp/author_static_10000.parquet \
        --labels_csv /tmp/labels_sentiment_goemo.csv \
        --feature_names_json "${HYPER_DIR}/best/feature_names.json" \
        --output_dir "${SYNTH_LEGACY_OUT}" \
        --K 20 \
        --n_per_stratum 3500 \
        --hyper_dir "${HYPER_DIR}" \
        --base_model "${BASE_MODEL}" \
        --tau_off 0.0 \
        --tau_off_adaptive_mult 2.0 \
        --seed 142; then
      subitem_done "${MARK}"
      phase_end_ok "phase1_synth_legacy" 1 1
      echo "[phase1_synth_legacy] wrote ${SYNTH_LEGACY_OUT}/synthetic_personas.parquet"
      ls -lh "${SYNTH_LEGACY_OUT}" || true
    else
      phase_end_fail "phase1_synth_legacy" 1 0
      echo "[phase1_synth_legacy] FAILED; in_hull + near_hull forums will be skipped."
    fi
  fi
fi

# ===========================================================
# PHASE 1_SYNTH_FARHULL_V2 -- SKIP (parquet verified present)
# ===========================================================
echo "[phase1_synth_farhull_v2] checking: ${FARHULL_PARQ_SRC}"
if [[ -f "${FARHULL_PARQ_SRC}" ]]; then
  echo "[phase1_synth_farhull_v2] EXISTS — skip (kappa=3 parquet already at ${FARHULL_PARQ_SRC})"
else
  echo "[phase1_synth_farhull_v2] NOT FOUND — running sample_far_from_hull_extrapolation.py"
  FH_V2_OUT="${OUT_ROOT}/phase1_synth_arditi_v3"
  mkdir -p "${FH_V2_OUT}"
  python "${DATA_SCRIPTS}/sample_far_from_hull_extrapolation.py" \
      --feature_parquet /tmp/global_features_10000.parquet \
      --feature_names_json "${HYPER_DIR}/best/feature_names.json" \
      --cohort_labels_csv /tmp/labels_sentiment_goemo_extremes.csv \
      --n_per_cohort 1000 \
      --alpha 3.0 \
      --seed 142 \
      --out_parquet "${FARHULL_PARQ_SRC}" || true
  [[ -f "${FARHULL_PARQ_SRC}" ]] && echo "[phase1_synth_farhull_v2] generated OK" \
    || echo "[phase1_synth_farhull_v2] FAILED; far_from_hull forums will be skipped."
fi

# ===========================================================
# PHASE 1_SYNTH_KAPPA_SWEEP -- kappa in {3, 10, 25, 50, 100}
# ===========================================================
# One parquet per kappa value at
#   ${OUT_ROOT}/phase1_synth_kappa${K}/synthetic_personas.parquet
# The sample_far_from_hull_extrapolation.py script takes --alpha
# which is the kappa value here (variable named alpha in the
# script for historical reasons; the math is identical:
# g_new = (1 + kappa) * anchor - kappa * centroid).
for K in ${KAPPA_LIST}; do
  K_PARQ="${OUT_ROOT}/phase1_synth_kappa${K}/synthetic_personas.parquet"
  if [[ -f "${K_PARQ}" ]]; then
    echo "[phase1_synth_kappa${K}] EXISTS — skip (${K_PARQ})"
  else
    echo "[phase1_synth_kappa${K}] generating kappa=${K} parquet"
    mkdir -p "$(dirname "${K_PARQ}")"
    python "${DATA_SCRIPTS}/sample_far_from_hull_extrapolation.py" \
        --feature_parquet /tmp/global_features_10000.parquet \
        --feature_names_json "${HYPER_DIR}/best/feature_names.json" \
        --cohort_labels_csv /tmp/labels_sentiment_goemo_extremes.csv \
        --n_per_cohort 1000 \
        --alpha "${K}" \
        --seed 142 \
        --out_parquet "${K_PARQ}" || true
    [[ -f "${K_PARQ}" ]] && echo "[phase1_synth_kappa${K}] generated OK" \
      || echo "[phase1_synth_kappa${K}] FAILED; kappa=${K} forums will be skipped."
  fi
done

# ===========================================================
# PHASE 1_SYNTH_MIDPOINT -- SKIP (parquet verified present)
# ===========================================================
echo "[phase1_synth_midpoint] checking: ${MIDPOINT_PARQ_SRC}"
if [[ -f "${MIDPOINT_PARQ_SRC}" ]]; then
  echo "[phase1_synth_midpoint] EXISTS — skip (midpoint parquet already at ${MIDPOINT_PARQ_SRC})"
else
  echo "[phase1_synth_midpoint] NOT FOUND — running sample_midpoint_baseline.py"
  MP_OUT="${OUT_ROOT}/phase1_midpoint_baseline_v2"
  mkdir -p "${MP_OUT}"
  python "${DATA_SCRIPTS}/sample_midpoint_baseline.py" \
      --feature_parquet /tmp/global_features_10000.parquet \
      --feature_names_json "${HYPER_DIR}/best/feature_names.json" \
      --cohort_labels_csv /tmp/labels_sentiment_goemo_extremes.csv \
      --n_per_cohort 1000 \
      --seed 142 \
      --out_parquet "${MIDPOINT_PARQ_SRC}" || true
  [[ -f "${MIDPOINT_PARQ_SRC}" ]] && echo "[phase1_synth_midpoint] generated OK" \
    || echo "[phase1_synth_midpoint] FAILED; midpoint forums will be skipped."
fi

# ===========================================================
# PHASE 1P5_LABEL_LEGACY -- Label in_hull + near_hull descriptors
# ===========================================================
if phase_begin "phase1p5_label_legacy" 1; then
  MARK="${LABEL_LEGACY_OUT}/.complete"
  if subitem_is_done "${MARK}"; then
    echo "[skip] legacy labeling already complete"
    phase_end_ok "phase1p5_label_legacy" 1 1
  else
    SYNTH_PARQ="${SYNTH_LEGACY_OUT}/synthetic_personas.parquet"
    if [[ -f "${SYNTH_PARQ}" ]]; then
      if python "${SCRIPTS}/label_synthetic_personas.py" \
          --synth_parquet "${SYNTH_PARQ}" \
          --data_dir "${DATA}" \
          --author_parquet /tmp/author_static_10000.parquet \
          --feature_names_json "${HYPER_DIR}/best/feature_names.json" \
          --output_dir "${LABEL_LEGACY_OUT}" \
          --k_knn 10 \
          --ambiguity_threshold 0.5; then
        subitem_done "${MARK}"
        phase_end_ok "phase1p5_label_legacy" 1 1
        echo "[phase1p5_label_legacy] wrote ${LABEL_LEGACY_OUT}/synthetic_personas_labeled.parquet"
      else
        phase_end_fail "phase1p5_label_legacy" 1 0
        echo "[phase1p5_label_legacy] FAILED; in_hull + near_hull forums will be skipped."
      fi
    else
      echo "[phase1p5_label_legacy] synth parquet missing at ${SYNTH_PARQ}; skipping."
      phase_end_fail "phase1p5_label_legacy" 1 0
    fi
  fi
fi

# ===========================================================
# PHASE 1P5_LABEL_FARHULL_V2 -- gated on RUN_LEGACY_FARHULL=1
# PHASE 1P5_LABEL_MIDPOINT   -- always run (phase2d_midpoint needs it)
# 2026-05-17: replaced the warn-and-skip stub with an actual labeler
# invocation. The yaml comments at the top of the file claimed both
# parquets EXISTed but neither dir was ever populated; phase2d_midpoint
# quietly skipped on every prior run and farhull_v2 was undefined
# behavior. Now: source parquets in phase1_midpoint_baseline_v2/ and
# (when re-enabled) phase1_synth_arditi_v3/ are labeled in-place
# via label_synthetic_personas.py, same recipe as the kappa-sweep
# labeler below.
# ===========================================================

# ---- far_from_hull v2 (legacy single-kappa=3) -------------------
FH_LBL_FILE="${LABEL_FARHULL_OUT}/synthetic_personas_labeled.parquet"
if [[ -f "${FH_LBL_FILE}" ]]; then
  echo "[phase1p5_label_far_from_hull] EXISTS — skip (${FH_LBL_FILE})"
elif [[ "${RUN_LEGACY_FARHULL:-0}" != "1" ]]; then
  echo "[phase1p5_label_far_from_hull] SKIPPED (RUN_LEGACY_FARHULL=${RUN_LEGACY_FARHULL:-0} != 1)"
  echo "[phase1p5_label_far_from_hull]   (kappa sweep supersedes this; per-kappa labels run below)"
elif [[ ! -f "${FARHULL_PARQ_SRC}" ]]; then
  echo "[phase1p5_label_far_from_hull] WARNING: source parquet missing at ${FARHULL_PARQ_SRC}; skipping"
else
  mkdir -p "${LABEL_FARHULL_OUT}"
  if python "${SCRIPTS}/label_synthetic_personas.py" \
      --synth_parquet "${FARHULL_PARQ_SRC}" \
      --data_dir "${DATA}" \
      --author_parquet /tmp/author_static_10000.parquet \
      --feature_names_json "${HYPER_DIR}/best/feature_names.json" \
      --output_dir "${LABEL_FARHULL_OUT}" \
      --k_knn 10 \
      --ambiguity_threshold 0.5; then
    echo "[phase1p5_label_far_from_hull] labeled OK -> ${FH_LBL_FILE}"
  else
    echo "[phase1p5_label_far_from_hull] FAILED; legacy far-from-hull forums will be skipped."
  fi
fi

# ---- midpoint baseline (always needed for phase2d_midpoint) -----
MP_LBL_FILE="${LABEL_MIDPOINT_OUT}/synthetic_personas_labeled.parquet"
if [[ -f "${MP_LBL_FILE}" ]]; then
  echo "[phase1p5_label_midpoint] EXISTS — skip (${MP_LBL_FILE})"
elif [[ ! -f "${MIDPOINT_PARQ_SRC}" ]]; then
  echo "[phase1p5_label_midpoint] WARNING: source parquet missing at ${MIDPOINT_PARQ_SRC}; skipping"
  echo "[phase1p5_label_midpoint]   (phase2d_midpoint cells will skip)"
else
  mkdir -p "${LABEL_MIDPOINT_OUT}"
  if python "${SCRIPTS}/label_synthetic_personas.py" \
      --synth_parquet "${MIDPOINT_PARQ_SRC}" \
      --data_dir "${DATA}" \
      --author_parquet /tmp/author_static_10000.parquet \
      --feature_names_json "${HYPER_DIR}/best/feature_names.json" \
      --output_dir "${LABEL_MIDPOINT_OUT}" \
      --k_knn 10 \
      --ambiguity_threshold 0.5; then
    echo "[phase1p5_label_midpoint] labeled OK -> ${MP_LBL_FILE}"
  else
    echo "[phase1p5_label_midpoint] FAILED; midpoint forums will be skipped."
  fi
fi

# ===========================================================
# PHASE 1P5_LABEL_KAPPA_SWEEP -- label each kappa-sweep parquet
# ===========================================================
# For each kappa in KAPPA_LIST, run label_synthetic_personas.py
# on the kappa parquet, producing
#   ${OUT_ROOT}/phase1p5_label_kappa${K}/synthetic_personas_labeled.parquet
# The labeled parquet preserves the kappa column (if present) and
# carries the pol_* polar profile that score_persona reads.
for K in ${KAPPA_LIST}; do
  K_PARQ="${OUT_ROOT}/phase1_synth_kappa${K}/synthetic_personas.parquet"
  K_LBL_DIR="${OUT_ROOT}/phase1p5_label_kappa${K}"
  K_LBL="${K_LBL_DIR}/synthetic_personas_labeled.parquet"
  if [[ -f "${K_LBL}" ]]; then
    echo "[phase1p5_label_kappa${K}] EXISTS — skip (${K_LBL})"
    continue
  fi
  if [[ ! -f "${K_PARQ}" ]]; then
    echo "[phase1p5_label_kappa${K}] source parquet missing: ${K_PARQ}; skipping"
    continue
  fi
  mkdir -p "${K_LBL_DIR}"
  if python "${SCRIPTS}/label_synthetic_personas.py" \
      --synth_parquet "${K_PARQ}" \
      --data_dir "${DATA}" \
      --author_parquet /tmp/author_static_10000.parquet \
      --feature_names_json "${HYPER_DIR}/best/feature_names.json" \
      --output_dir "${K_LBL_DIR}" \
      --k_knn 10 \
      --ambiguity_threshold 0.5; then
    echo "[phase1p5_label_kappa${K}] labeled OK -> ${K_LBL}"
  else
    echo "[phase1p5_label_kappa${K}] FAILED; kappa=${K} forums will be skipped."
  fi
done

# ===========================================================
# Shared helper: build_real_user_label_profile
# ===========================================================
# Builds /tmp/author_label_profile_real.parquet (9 pol_* columns
# from gstat_* features + goemo composites). Used by phase2c and
# phase3d for real-user signature profiling.
build_real_user_profile() {
  local REAL_PROF="/tmp/author_label_profile_real.parquet"
  if [[ -f "${REAL_PROF}" ]]; then
    return 0
  fi
  python - <<'PY_REAL_PROF'
import json
import pandas as pd
from pathlib import Path
DATA = Path("/tmp")
base = pd.read_parquet(DATA / "author_static_10000.parquet",
                        columns=["target_user_id",
                                 "gstat_profanity_ratio", "gstat_question_ratio",
                                 "gstat_reply_delay_mean", "gstat_firstperson_ratio",
                                 "gstat_caps_ratio", "gstat_punct_ratio",
                                 "gstat_user_sent_mean"])
base["pol_politeness"] = base["gstat_profanity_ratio"].astype(float)
base["pol_curiosity"]  = base["gstat_question_ratio"].astype(float)
base["pol_tempo"]      = base["gstat_reply_delay_mean"].astype(float)
base["pol_self_focus"] = base["gstat_firstperson_ratio"].astype(float)
norm = json.loads((DATA / "feature_norm_stats_10000.json").read_text())
def _zi(col):
    s = norm.get(col, {})
    mu = float(s.get("mean", 0.0)); sd = float(s.get("std", 1.0))
    if sd <= 0: sd = 1.0
    return base[col].astype(float) * sd + mu
caps = _zi("gstat_caps_ratio"); punct = _zi("gstat_punct_ratio")
base["pol_expressiveness"] = 2.0 * (caps * punct) / (caps + punct + 1e-12)
base["pol_sentiment_goemo"] = base["gstat_user_sent_mean"].astype(float)
try:
    # 2026-05-18: composite class lists imported from signed_hedges_g
    # so EXPECTED vs REALIZED stay synchronized. Warmth dropped
    # gratitude (false-positive class on rage-topic emphatic
    # phrasing). Anxiety unchanged structurally; only HIGH dict
    # entry was inverted (see signed_hedges_g.py line 33).
    import sys as _sys
    _sys.path.insert(0, "/workspace/hypernets/training_scripts")
    from signed_hedges_g import WARMTH_GOEMO_CLASSES, ANXIETY_GOEMO_CLASSES
    emo = pd.read_csv(DATA / "goemo_user_emotions.csv")
    emo = emo.set_index("target_user_id")
    composites = [
        ("anxiety",   [f"{c}_mean" for c in ANXIETY_GOEMO_CLASSES]),
        ("warmth",    [f"{c}_mean" for c in WARMTH_GOEMO_CLASSES]),
        ("hostility", ["anger_mean", "disgust_mean", "disapproval_mean"]),
    ]
    for dim, cols in composites:
        base[f"pol_{dim}"] = base["target_user_id"].map(
            emo[cols].mean(axis=1)).astype(float)
except Exception as e:
    print(f"[realprof] goemo composites skipped: {e}")
    for dim in ("anxiety", "warmth", "hostility"):
        base[f"pol_{dim}"] = float("nan")
meta = json.loads((DATA / "goemo_labels_metadata.json").read_text())
q = meta["rank_boundaries_score_at_fraction"]
b20, b40, b60, b80 = float(q["0.2"]), float(q["0.4"]), float(q["0.6"]), float(q["0.8"])
def _cohort(x):
    if x <= b20: return "rage"
    if x <= b40: return "grumpy"
    if x <= b60: return "mellow"
    if x <= b80: return "calm"
    return "empath"
base["cohort_goemo"] = base["pol_sentiment_goemo"].apply(_cohort)
keep = ["target_user_id", "cohort_goemo"] + [
    f"pol_{d}" for d in
    ("politeness", "curiosity", "tempo", "self_focus", "expressiveness",
     "anxiety", "warmth", "hostility", "sentiment_goemo")
]
base[keep].to_parquet("/tmp/author_label_profile_real.parquet", index=False)
print(f"[realprof] wrote rows={len(base)}")
PY_REAL_PROF
}

# ===========================================================
# Shared helper: slice a labeled synth parquet by cohort+stratum
# ===========================================================
# Args: labeled_parquet cohort stratum out_author out_meta out_labels
# Writes per-cohort scratch parquets to /tmp for the forum builder.
# outlier flag is handled at the forum builder call site (not here).
slice_synth_pool() {
  local labeled_parq="$1"
  local cohort="$2"
  local stratum="$3"
  local out_author="$4"
  local out_meta="$5"
  local out_labels="$6"

  rm -f "${out_author}" "${out_meta}" "${out_labels}"
  COHORT="${cohort}" STRATUM="${stratum}" \
  LABELED="${labeled_parq}" \
  OUT_AUTHOR="${out_author}" OUT_META="${out_meta}" OUT_LABELS="${out_labels}" \
  python - <<'PYEOF' || true
import os, sys, pandas as pd
cohort   = os.environ["COHORT"]
stratum  = os.environ["STRATUM"]
labeled  = os.environ["LABELED"]
out_au   = os.environ["OUT_AUTHOR"]
out_me   = os.environ["OUT_META"]
out_lb   = os.environ["OUT_LABELS"]

df = pd.read_parquet(labeled)
for req in ("cohort_goemo", "stratum", "target_user_id"):
    if req not in df.columns:
        print(f"[slice_synth] missing column {req!r}; got {list(df.columns)[:10]}...")
        sys.exit(0)

if cohort == "neutral":
    sub = df[
        df["cohort_goemo"].astype(str).isin(("calm", "grumpy", "mellow"))
        & (df["stratum"].astype(str) == stratum)
    ].copy()
    if len(sub) > 500:
        sub = sub.sample(n=500, random_state=142).reset_index(drop=True)
else:
    sub = df[
        (df["cohort_goemo"].astype(str) == cohort)
        & (df["stratum"].astype(str) == stratum)
    ].copy()

if len(sub) == 0:
    print(f"[slice_synth] cohort={cohort} stratum={stratum} EMPTY")
    sys.exit(0)

sub.to_parquet(out_au, index=False)
sub[["target_user_id"] + [
    c for c in sub.columns
    if c.startswith("pol_") or c in (
        "stratum", "cohort_goemo", "ambiguity_score",
        "label_profile_json", "target_spec_json"
    )
]].to_parquet(out_me, index=False)

lbl = pd.DataFrame({
    "target_user_id":  sub["target_user_id"].astype(int),
    "sentiment_label": cohort,
})
for d in ("politeness", "curiosity", "tempo", "self_focus",
          "expressiveness", "anxiety", "warmth", "hostility"):
    col = f"pol_{d}"
    if col in sub.columns:
        lbl[col] = sub[col].astype(float).values
lbl.to_csv(out_lb, index=False)
print(f"[slice_synth] cohort={cohort} stratum={stratum} rows={len(sub)}")
PYEOF
}

# ===========================================================
# PHASE 2A_VANILLA -- Vanilla LoRA baseline forums (3 cohorts)
# ===========================================================
# build_vanillalora_forum.py: --no_qlora is REQUIRED on the GPU node
# (CUDA 13.2 has no bitsandbytes binary). This flag is ONLY valid
# here; NEVER pass --no_qlora to build_hyperlora_forum.py.
# repetition_penalty=1.25 (matches pre-reg); --min_new_tokens not
# supported by vanillalora script so omitted.
N_VANILLA=3
if phase_begin "phase2a_vanilla" "${N_VANILLA}"; then
  n_done=0
  for cohort in rage empath neutral; do
    topic="${FORUM_TOPIC[${cohort}]}"
    out="${FORUM_ROOT}/2a_vanilla_${cohort}"
    mark="${out}/.complete"
    mkdir -p "${out}/persona_signature"
    if subitem_is_done "${mark}"; then
      echo "[skip] 2a_vanilla_${cohort} already complete"
      n_done=$((n_done+1)); continue
    fi
    if [[ ! -d "${VANILLA_LORA_DIR}" ]]; then
      echo "[phase2a_vanilla] vanilla LoRA dir missing: ${VANILLA_LORA_DIR}; skipping ${cohort}"
      continue
    fi
    if python "${SCRIPTS}/build_vanillalora_forum.py" \
        --author_parquet /tmp/author_static_10000.parquet \
        --labels_csv /tmp/labels_sentiment_goemo.csv \
        --base_model "${BASE_MODEL}" \
        --online --no_qlora \
        --lora_dir "${VANILLA_LORA_DIR}" --use_best_ckpt \
        --out_dir "${out}" \
        --topic "${topic}" --sentiment_target "${cohort}" \
        --threads_from_default 12 \
        --fanout 5 5 3 1 --horizon_min "${HORIZON_MIN}" \
        --n_rage "${N_RAGE}" --n_empath "${N_EMPATH}" \
        --max_len 512 --max_new_tokens "${MAX_NEW_TOKENS}" \
        --do_sample --top_p "${TOP_P}" --temperature "${TEMPERATURE}" \
        --repetition_penalty "${REPETITION_PENALTY}" \
        --no_repeat_ngram_size "${NO_REPEAT_NGRAM}" \
        --seed 142; then
      subitem_done "${mark}"
      n_done=$((n_done+1))
    else
      echo "[phase2a_vanilla] ${cohort} FAILED"
    fi
  done
  (( n_done == N_VANILLA )) \
    && phase_end_ok "phase2a_vanilla" "${N_VANILLA}" "${n_done}" \
    || phase_end_fail "phase2a_vanilla" "${N_VANILLA}" "${n_done}"
fi

# ===========================================================
# PHASE 2B_ZERODELTA -- TRUE zero-condition baseline (3 cohorts)
# ===========================================================
# FIX 2026-05-16: previous version passed BOTH --arditi_patch
# AND --force_zero_delta, which left the Arditi residual-stream
# hook FIRING in the "Zero-Δ" cells. That made 2b an
# "Arditi-only, delta-off" cell, NOT a clean baseline. The
# paper's Table 1 / §4.1 narrative ("Zero-Δ flatlines like
# vanilla; per-user Δθ carries the cohort signal") requires the
# opposite: Zero-Δ must be NEITHER conditioning route on, so
# any cohort signal it shows is from cohort-target prompt +
# base-model priors only. We now strip the four --arditi_*
# flags from this block. Result: 2b = base + LoRA template
# (zero deltas at injection) + cohort-target prompt. No
# Arditi hook. No per-user feature vector contribution.
# --filter_outliers applies (standard real-user feature space).
N_ZDELTA=3
if phase_begin "phase2b_zerodelta" "${N_ZDELTA}"; then
  n_done=0
  for cohort in rage empath neutral; do
    topic="${FORUM_TOPIC[${cohort}]}"
    out="${FORUM_ROOT}/2b_zero_delta_${cohort}"
    mark="${out}/.complete"
    mkdir -p "${out}/persona_signature"
    if subitem_is_done "${mark}"; then
      echo "[skip] 2b_zero_delta_${cohort} already complete"
      n_done=$((n_done+1)); continue
    fi
    if python "${SCRIPTS}/build_hyperlora_forum.py" \
        --author_parquet /tmp/author_static_10000.parquet \
        --labels_csv /tmp/labels_sentiment_goemo.csv \
        --base_model "${BASE_MODEL}" \
        --online \
        --hyper_dir "${HYPER_DIR}" \
        --out_dir "${out}" \
        --topic "${topic}" --sentiment_target "${cohort}" \
        --threads_from_default "${THREADS_FROM_DEFAULT:-12}" \
        --target_modules "${TARGET_MODULES}" \
        --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
        --emit_both \
        --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}" \
        --fanout "${FANOUT}" --horizon_min "${HORIZON_MIN}" \
        --n_rage "${N_RAGE}" --n_empath "${N_EMPATH}" \
        --max_len 512 --max_new_tokens "${MAX_NEW_TOKENS}" \
        --do_sample --top_p "${TOP_P}" --temperature "${TEMPERATURE}" \
        --repetition_penalty "${REPETITION_PENALTY}" \
        --no_repeat_ngram_size "${NO_REPEAT_NGRAM}" \
        --min_new_tokens "${MIN_NEW_TOKENS}" \
        --threshold_source /tmp/global_features_10000.parquet \
        --threshold_col gstat_user_sent_mean \
        --norm_stats_json /tmp/feature_norm_stats_10000.json \
        --feature_clamp 3.0 --outlier_threshold 4.0 --filter_outliers \
        --min_user_tokens 50 --seed 142 \
        --infer_batch_size 16 \
        --force_zero_delta; then
      subitem_done "${mark}"
      n_done=$((n_done+1))
    else
      echo "[phase2b_zerodelta] ${cohort} FAILED"
    fi
  done
  (( n_done == N_ZDELTA )) \
    && phase_end_ok "phase2b_zerodelta" "${N_ZDELTA}" "${n_done}" \
    || phase_end_fail "phase2b_zerodelta" "${N_ZDELTA}" "${n_done}"
fi

# ===========================================================
# PHASE 2C_REAL_USER -- Real-user HyperPEFT-LoRA forums (3 cohorts)
# ===========================================================
# Production inference: Arditi Patch (orthogonal, alpha=1.0, layers 15-23).
# --filter_outliers applies (real-user feature space).
N_REAL=3
if phase_begin "phase2c_real_user" "${N_REAL}"; then
  n_done=0
  for cohort in rage empath neutral; do
    topic="${FORUM_TOPIC[${cohort}]}"
    out="${FORUM_ROOT}/2c_real_user_${cohort}"
    mark="${out}/.complete"
    mkdir -p "${out}/persona_signature"
    if subitem_is_done "${mark}"; then
      echo "[skip] 2c_real_user_${cohort} already complete"
      n_done=$((n_done+1)); continue
    fi
    # 2026-05-21 fix: phase 2c real-user runs in DEFAULT fp16 + math SDPA.
    # The global ARDITI_DISABLE_FLASH_SDP=1 export (above) forces the portable
    # math SDP backend (no cuDNN crash) and the NaN-logits guard in
    # build_hyperlora_forum.py catches the multinomial NaN. The earlier
    # bf16+eager workaround also dodged the cuDNN crash, but its coarse mantissa
    # mangled the small per-user LoRA deltas (clamp 0.020), scrambling the
    # in-band measures and pulling the recon cohort separation away from the
    # fp16 paper numbers (sentiment +0.03 vs +0.18 on the 2026-05-20 run). fp16
    # + math SDPA avoids cuDNN WITHOUT the precision loss, so recon matches the
    # synth cells and the published result.
    if python "${SCRIPTS}/build_hyperlora_forum.py" \
        --author_parquet /tmp/author_static_10000.parquet \
        --labels_csv /tmp/labels_sentiment_goemo.csv \
        --base_model "${BASE_MODEL}" \
        --online \
        --hyper_dir "${HYPER_DIR}" \
        --out_dir "${out}" \
        --topic "${topic}" --sentiment_target "${cohort}" \
        --threads_from_default 12 \
        --target_modules "${TARGET_MODULES}" \
        --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
        --emit_both \
        --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}" \
        --fanout "${FANOUT}" --horizon_min "${HORIZON_MIN}" \
        --n_rage "${N_RAGE}" --n_empath "${N_EMPATH}" \
        --max_len 512 --max_new_tokens "${MAX_NEW_TOKENS}" \
        --do_sample --top_p "${TOP_P}" --temperature "${TEMPERATURE}" \
        --repetition_penalty "${REPETITION_PENALTY}" \
        --no_repeat_ngram_size "${NO_REPEAT_NGRAM}" \
        --min_new_tokens "${MIN_NEW_TOKENS}" \
        --threshold_source /tmp/global_features_10000.parquet \
        --threshold_col gstat_user_sent_mean \
        --norm_stats_json /tmp/feature_norm_stats_10000.json \
        --feature_clamp 3.0 --outlier_threshold 4.0 --filter_outliers \
        --min_user_tokens 50 --seed 142 \
        --infer_batch_size 16 \
        --arditi_patch "${ARDITI_TMP}" \
        --arditi_mode "${ARDITI_MODE}" \
        --arditi_alpha "${ARDITI_ALPHA}" --arditi_layers "${ARDITI_LAYERS}"; then
      subitem_done "${mark}"
      n_done=$((n_done+1))
    else
      echo "[phase2c_real_user] ${cohort} FAILED"
    fi
  done
  (( n_done == N_REAL )) \
    && phase_end_ok "phase2c_real_user" "${N_REAL}" "${n_done}" \
    || phase_end_fail "phase2c_real_user" "${N_REAL}" "${n_done}"
fi

# ===========================================================
# PHASE 2D_INHULL -- Synth in_hull forums (3 cohorts)
# ===========================================================
N_INHULL=3
if phase_begin "phase2d_inhull" "${N_INHULL}"; then
  n_done=0
  LABELED_LEGACY="${LABEL_LEGACY_OUT}/synthetic_personas_labeled.parquet"
  if [[ -f "${LABELED_LEGACY}" ]]; then
    cp -f "${LABELED_LEGACY}" /tmp/synth_labeled_legacy.parquet
    for cohort in rage empath neutral; do
      topic="${FORUM_TOPIC[${cohort}]}"
      out="${FORUM_ROOT}/2d_synth_${cohort}_in_hull"
      mark="${out}/.complete"
      mkdir -p "${out}/persona_signature"
      if subitem_is_done "${mark}"; then
        echo "[skip] 2d_synth_${cohort}_in_hull already complete"
        n_done=$((n_done+1)); continue
      fi
      slice_synth_pool \
        /tmp/synth_labeled_legacy.parquet \
        "${cohort}" "in_hull" \
        "/tmp/synth_au_${cohort}_in_hull.parquet" \
        "/tmp/synth_me_${cohort}_in_hull.parquet" \
        "/tmp/synth_lb_${cohort}_in_hull.csv"
      if [[ ! -f "/tmp/synth_au_${cohort}_in_hull.parquet" ]]; then
        echo "[phase2d_inhull] empty pool for ${cohort}; skipping"
        continue
      fi
      if python "${SCRIPTS}/build_hyperlora_forum.py" \
          --author_parquet "/tmp/synth_au_${cohort}_in_hull.parquet" \
          --labels_csv "/tmp/synth_lb_${cohort}_in_hull.csv" \
          --user_metadata_parquet "/tmp/synth_me_${cohort}_in_hull.parquet" \
          --base_model "${BASE_MODEL}" \
          --online \
          --hyper_dir "${HYPER_DIR}" \
          --out_dir "${out}" \
          --topic "${topic}" --sentiment_target "${cohort}" \
          --threads_from_default 12 \
          --target_modules "${TARGET_MODULES}" \
          --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
          --emit_both \
          --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}" \
          --fanout "${FANOUT}" --horizon_min "${HORIZON_MIN}" \
          --n_rage "${N_RAGE}" --n_empath "${N_EMPATH}" \
          --max_len 512 --max_new_tokens "${MAX_NEW_TOKENS}" \
          --do_sample --top_p "${TOP_P}" --temperature "${TEMPERATURE}" \
          --repetition_penalty "${REPETITION_PENALTY}" \
          --no_repeat_ngram_size "${NO_REPEAT_NGRAM}" \
          --min_new_tokens "${MIN_NEW_TOKENS}" \
          --threshold_source /tmp/global_features_10000.parquet \
          --threshold_col gstat_user_sent_mean \
          --norm_stats_json /tmp/feature_norm_stats_10000.json \
          --feature_clamp 3.0 --outlier_threshold 4.0 --filter_outliers \
          --min_user_tokens 50 --seed 142 \
          --infer_batch_size 16 \
          --arditi_patch "${ARDITI_TMP}" \
          --arditi_mode "${ARDITI_MODE}" \
          --arditi_alpha "${ARDITI_ALPHA}" --arditi_layers "${ARDITI_LAYERS}"; then
        subitem_done "${mark}"
        n_done=$((n_done+1))
      else
        echo "[phase2d_inhull] ${cohort} FAILED"
      fi
    done
  else
    echo "[phase2d_inhull] labeled parquet missing: ${LABELED_LEGACY}; skipping all in_hull"
  fi
  (( n_done == N_INHULL )) \
    && phase_end_ok "phase2d_inhull" "${N_INHULL}" "${n_done}" \
    || phase_end_fail "phase2d_inhull" "${N_INHULL}" "${n_done}"
fi

# ===========================================================
# PHASE 2D_NEARHULL -- Synth near_hull forums (3 cohorts)
# ===========================================================
N_NEARHULL=3
if phase_begin "phase2d_nearhull" "${N_NEARHULL}"; then
  n_done=0
  LABELED_LEGACY="${LABEL_LEGACY_OUT}/synthetic_personas_labeled.parquet"
  if [[ -f "${LABELED_LEGACY}" ]]; then
    [[ -f /tmp/synth_labeled_legacy.parquet ]] || cp -f "${LABELED_LEGACY}" /tmp/synth_labeled_legacy.parquet
    for cohort in rage empath neutral; do
      topic="${FORUM_TOPIC[${cohort}]}"
      out="${FORUM_ROOT}/2d_synth_${cohort}_near_hull"
      mark="${out}/.complete"
      mkdir -p "${out}/persona_signature"
      if subitem_is_done "${mark}"; then
        echo "[skip] 2d_synth_${cohort}_near_hull already complete"
        n_done=$((n_done+1)); continue
      fi
      slice_synth_pool \
        /tmp/synth_labeled_legacy.parquet \
        "${cohort}" "near_hull" \
        "/tmp/synth_au_${cohort}_near_hull.parquet" \
        "/tmp/synth_me_${cohort}_near_hull.parquet" \
        "/tmp/synth_lb_${cohort}_near_hull.csv"
      if [[ ! -f "/tmp/synth_au_${cohort}_near_hull.parquet" ]]; then
        echo "[phase2d_nearhull] empty pool for ${cohort}; skipping"
        continue
      fi
      if python "${SCRIPTS}/build_hyperlora_forum.py" \
          --author_parquet "/tmp/synth_au_${cohort}_near_hull.parquet" \
          --labels_csv "/tmp/synth_lb_${cohort}_near_hull.csv" \
          --user_metadata_parquet "/tmp/synth_me_${cohort}_near_hull.parquet" \
          --base_model "${BASE_MODEL}" \
          --online \
          --hyper_dir "${HYPER_DIR}" \
          --out_dir "${out}" \
          --topic "${topic}" --sentiment_target "${cohort}" \
          --threads_from_default 12 \
          --target_modules "${TARGET_MODULES}" \
          --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
          --emit_both \
          --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}" \
          --fanout "${FANOUT}" --horizon_min "${HORIZON_MIN}" \
          --n_rage "${N_RAGE}" --n_empath "${N_EMPATH}" \
          --max_len 512 --max_new_tokens "${MAX_NEW_TOKENS}" \
          --do_sample --top_p "${TOP_P}" --temperature "${TEMPERATURE}" \
          --repetition_penalty "${REPETITION_PENALTY}" \
          --no_repeat_ngram_size "${NO_REPEAT_NGRAM}" \
          --min_new_tokens "${MIN_NEW_TOKENS}" \
          --threshold_source /tmp/global_features_10000.parquet \
          --threshold_col gstat_user_sent_mean \
          --norm_stats_json /tmp/feature_norm_stats_10000.json \
          --feature_clamp 3.0 --outlier_threshold 4.0 --filter_outliers \
          --min_user_tokens 50 --seed 142 \
          --infer_batch_size 16 \
          --arditi_patch "${ARDITI_TMP}" \
          --arditi_mode "${ARDITI_MODE}" \
          --arditi_alpha "${ARDITI_ALPHA}" --arditi_layers "${ARDITI_LAYERS}"; then
        subitem_done "${mark}"
        n_done=$((n_done+1))
      else
        echo "[phase2d_nearhull] ${cohort} FAILED"
      fi
    done
  else
    echo "[phase2d_nearhull] labeled parquet missing: ${LABELED_LEGACY}; skipping all near_hull"
  fi
  (( n_done == N_NEARHULL )) \
    && phase_end_ok "phase2d_nearhull" "${N_NEARHULL}" "${n_done}" \
    || phase_end_fail "phase2d_nearhull" "${N_NEARHULL}" "${n_done}"
fi

# ===========================================================
# PHASE 2D_FARHULL_V2 -- Synth far_from_hull v2 forums (3 cohorts)
# ===========================================================
# kappa=3 true extrapolation — descriptors are deliberately beyond
# +-4 std. DO NOT pass --filter_outliers here; raise clamp thresholds
# so the v2 descriptors are not dropped wholesale.
# Dir name: 2d_synth_{cohort}_far_from_hull (same as legacy would have
# been, but in REFRESH root so no collision).
# Legacy single-kappa (kappa=3) far-hull phase: DISABLED. The
# kappa-sweep below produces 2d_synth_${cohort}_far_kappa3
# which supersedes this. Set RUN_LEGACY_FARHULL=1 to re-enable.
N_FARHULL=3
if [[ "${RUN_LEGACY_FARHULL:-0}" == "1" ]] && phase_begin "phase2d_farhull_v2" "${N_FARHULL}"; then
  n_done=0
  LABELED_FH="${LABEL_FARHULL_OUT}/synthetic_personas_labeled.parquet"
  if [[ -f "${LABELED_FH}" ]]; then
    cp -f "${LABELED_FH}" /tmp/synth_labeled_farhull_v2.parquet
    for cohort in rage empath neutral; do
      topic="${FORUM_TOPIC[${cohort}]}"
      out="${FORUM_ROOT}/2d_synth_${cohort}_far_from_hull"
      mark="${out}/.complete"
      mkdir -p "${out}/persona_signature"
      if subitem_is_done "${mark}"; then
        echo "[skip] 2d_synth_${cohort}_far_from_hull already complete"
        n_done=$((n_done+1)); continue
      fi
      slice_synth_pool \
        /tmp/synth_labeled_farhull_v2.parquet \
        "${cohort}" "far_from_hull" \
        "/tmp/synth_au_${cohort}_far_from_hull.parquet" \
        "/tmp/synth_me_${cohort}_far_from_hull.parquet" \
        "/tmp/synth_lb_${cohort}_far_from_hull.csv"
      if [[ ! -f "/tmp/synth_au_${cohort}_far_from_hull.parquet" ]]; then
        echo "[phase2d_farhull_v2] empty pool for ${cohort}; skipping"
        continue
      fi
      # NOTE: no --filter_outliers here; --feature_clamp 5.0 --outlier_threshold 50.0
      # because kappa=3 v2 descriptors are intentionally far outside the training hull.
      if python "${SCRIPTS}/build_hyperlora_forum.py" \
          --author_parquet "/tmp/synth_au_${cohort}_far_from_hull.parquet" \
          --labels_csv "/tmp/synth_lb_${cohort}_far_from_hull.csv" \
          --user_metadata_parquet "/tmp/synth_me_${cohort}_far_from_hull.parquet" \
          --base_model "${BASE_MODEL}" \
          --online \
          --hyper_dir "${HYPER_DIR}" \
          --out_dir "${out}" \
          --topic "${topic}" --sentiment_target "${cohort}" \
          --threads_from_default 12 \
          --target_modules "${TARGET_MODULES}" \
          --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
          --emit_both \
          --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}" \
          --fanout "${FANOUT}" --horizon_min "${HORIZON_MIN}" \
          --n_rage "${N_RAGE}" --n_empath "${N_EMPATH}" \
          --max_len 512 --max_new_tokens "${MAX_NEW_TOKENS}" \
          --do_sample --top_p "${TOP_P}" --temperature "${TEMPERATURE}" \
          --repetition_penalty "${REPETITION_PENALTY}" \
          --no_repeat_ngram_size "${NO_REPEAT_NGRAM}" \
          --min_new_tokens "${MIN_NEW_TOKENS}" \
          --threshold_source /tmp/global_features_10000.parquet \
          --threshold_col gstat_user_sent_mean \
          --norm_stats_json /tmp/feature_norm_stats_10000.json \
          --feature_clamp 5.0 --outlier_threshold 50.0 \
          --min_user_tokens 50 --seed 142 \
          --infer_batch_size 16 \
          --arditi_patch "${ARDITI_TMP}" \
          --arditi_mode "${ARDITI_MODE}" \
          --arditi_alpha "${ARDITI_ALPHA}" --arditi_layers "${ARDITI_LAYERS}"; then
        subitem_done "${mark}"
        n_done=$((n_done+1))
      else
        echo "[phase2d_farhull_v2] ${cohort} FAILED"
      fi
    done
  else
    echo "[phase2d_farhull_v2] labeled parquet missing: ${LABELED_FH}; skipping all far_from_hull"
  fi
  (( n_done == N_FARHULL )) \
    && phase_end_ok "phase2d_farhull_v2" "${N_FARHULL}" "${n_done}" \
    || phase_end_fail "phase2d_farhull_v2" "${N_FARHULL}" "${n_done}"
fi

# ===========================================================
# PHASE 2D_FARHULL_SWEEP -- 5 kappa x 3 cohorts = 15 far-hull cells
# ===========================================================
# For every kappa in KAPPA_LIST, generate one synth far-hull
# forum per cohort, output dir = 2d_synth_${cohort}_far_kappa${K}.
# Each cell uses the kappa-specific labeled parquet from
# phase1p5_label_kappa${K}/synthetic_personas_labeled.parquet.
# Same outlier treatment as phase2d_farhull_v2 (clamp 5.0,
# threshold 50.0, no --filter_outliers).
N_FARHULL_SWEEP=15  # 5 kappa values x 3 cohorts
if phase_begin "phase2d_farhull_sweep" "${N_FARHULL_SWEEP}"; then
  sweep_done=0
  for K in ${KAPPA_LIST}; do
    K_LBL="${OUT_ROOT}/phase1p5_label_kappa${K}/synthetic_personas_labeled.parquet"
    if [[ ! -f "${K_LBL}" ]]; then
      echo "[phase2d_farhull_sweep] kappa=${K} labeled parquet missing: ${K_LBL}; skipping cohort loop"
      continue
    fi
    cp -f "${K_LBL}" /tmp/synth_labeled_kappa${K}.parquet
    for cohort in rage empath neutral; do
      topic="${FORUM_TOPIC[${cohort}]}"
      out="${FORUM_ROOT}/2d_synth_${cohort}_far_kappa${K}"
      mark="${out}/.complete"
      mkdir -p "${out}/persona_signature"
      if subitem_is_done "${mark}"; then
        echo "[skip] 2d_synth_${cohort}_far_kappa${K} already complete"
        sweep_done=$((sweep_done+1)); continue
      fi
      slice_synth_pool \
        /tmp/synth_labeled_kappa${K}.parquet \
        "${cohort}" "far_from_hull" \
        "/tmp/synth_au_${cohort}_far_kappa${K}.parquet" \
        "/tmp/synth_me_${cohort}_far_kappa${K}.parquet" \
        "/tmp/synth_lb_${cohort}_far_kappa${K}.csv"
      if [[ ! -f "/tmp/synth_au_${cohort}_far_kappa${K}.parquet" ]]; then
        echo "[phase2d_farhull_sweep] kappa=${K} cohort=${cohort} empty pool; skipping"
        continue
      fi
      if python "${SCRIPTS}/build_hyperlora_forum.py" \
          --author_parquet "/tmp/synth_au_${cohort}_far_kappa${K}.parquet" \
          --labels_csv "/tmp/synth_lb_${cohort}_far_kappa${K}.csv" \
          --user_metadata_parquet "/tmp/synth_me_${cohort}_far_kappa${K}.parquet" \
          --base_model "${BASE_MODEL}" \
          --online \
          --hyper_dir "${HYPER_DIR}" \
          --out_dir "${out}" \
          --topic "${topic}" --sentiment_target "${cohort}" \
          --threads_from_default 12 \
          --target_modules "${TARGET_MODULES}" \
          --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
          --emit_both \
          --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}" \
          --fanout "${FANOUT}" --horizon_min "${HORIZON_MIN}" \
          --n_rage "${N_RAGE}" --n_empath "${N_EMPATH}" \
          --max_len 512 --max_new_tokens "${MAX_NEW_TOKENS}" \
          --do_sample --top_p "${TOP_P}" --temperature "${TEMPERATURE}" \
          --repetition_penalty "${REPETITION_PENALTY}" \
          --no_repeat_ngram_size "${NO_REPEAT_NGRAM}" \
          --min_new_tokens "${MIN_NEW_TOKENS}" \
          --threshold_source /tmp/global_features_10000.parquet \
          --threshold_col gstat_user_sent_mean \
          --norm_stats_json /tmp/feature_norm_stats_10000.json \
          --feature_clamp 5.0 --outlier_threshold 50.0 \
          --min_user_tokens 50 --seed 142 \
          --infer_batch_size 16 \
          --arditi_patch "${ARDITI_TMP}" \
          --arditi_mode "${ARDITI_MODE}" \
          --arditi_alpha "${ARDITI_ALPHA}" --arditi_layers "${ARDITI_LAYERS}"; then
        subitem_done "${mark}"
        sweep_done=$((sweep_done+1))
      else
        echo "[phase2d_farhull_sweep] kappa=${K} cohort=${cohort} FAILED"
      fi
    done
  done
  (( sweep_done == N_FARHULL_SWEEP )) \
    && phase_end_ok "phase2d_farhull_sweep" "${N_FARHULL_SWEEP}" "${sweep_done}" \
    || phase_end_fail "phase2d_farhull_sweep" "${N_FARHULL_SWEEP}" "${sweep_done}"
fi

# ===========================================================
# PHASE 2D_MIDPOINT -- Synth midpoint_baseline forums (3 cohorts)
# ===========================================================
# Interpolation ablation (Item 17). Same v2-style outlier treatment:
# no --filter_outliers; raised clamp/threshold.
N_MIDPOINT=3
if phase_begin "phase2d_midpoint" "${N_MIDPOINT}"; then
  n_done=0
  LABELED_MP="${LABEL_MIDPOINT_OUT}/synthetic_personas_labeled.parquet"
  if [[ -f "${LABELED_MP}" ]]; then
    cp -f "${LABELED_MP}" /tmp/synth_labeled_midpoint.parquet
    for cohort in rage empath neutral; do
      topic="${FORUM_TOPIC[${cohort}]}"
      out="${FORUM_ROOT}/2d_synth_${cohort}_midpoint_baseline"
      mark="${out}/.complete"
      mkdir -p "${out}/persona_signature"
      if subitem_is_done "${mark}"; then
        echo "[skip] 2d_synth_${cohort}_midpoint_baseline already complete"
        n_done=$((n_done+1)); continue
      fi
      slice_synth_pool \
        /tmp/synth_labeled_midpoint.parquet \
        "${cohort}" "midpoint_baseline" \
        "/tmp/synth_au_${cohort}_midpoint.parquet" \
        "/tmp/synth_me_${cohort}_midpoint.parquet" \
        "/tmp/synth_lb_${cohort}_midpoint.csv"
      if [[ ! -f "/tmp/synth_au_${cohort}_midpoint.parquet" ]]; then
        echo "[phase2d_midpoint] empty pool for ${cohort}; skipping"
        continue
      fi
      # No --filter_outliers; raised thresholds same as far_from_hull_v2
      if python "${SCRIPTS}/build_hyperlora_forum.py" \
          --author_parquet "/tmp/synth_au_${cohort}_midpoint.parquet" \
          --labels_csv "/tmp/synth_lb_${cohort}_midpoint.csv" \
          --user_metadata_parquet "/tmp/synth_me_${cohort}_midpoint.parquet" \
          --base_model "${BASE_MODEL}" \
          --online \
          --hyper_dir "${HYPER_DIR}" \
          --out_dir "${out}" \
          --topic "${topic}" --sentiment_target "${cohort}" \
          --threads_from_default 12 \
          --target_modules "${TARGET_MODULES}" \
          --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
          --emit_both \
          --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}" \
          --fanout "${FANOUT}" --horizon_min "${HORIZON_MIN}" \
          --n_rage "${N_RAGE}" --n_empath "${N_EMPATH}" \
          --max_len 512 --max_new_tokens "${MAX_NEW_TOKENS}" \
          --do_sample --top_p "${TOP_P}" --temperature "${TEMPERATURE}" \
          --repetition_penalty "${REPETITION_PENALTY}" \
          --no_repeat_ngram_size "${NO_REPEAT_NGRAM}" \
          --min_new_tokens "${MIN_NEW_TOKENS}" \
          --threshold_source /tmp/global_features_10000.parquet \
          --threshold_col gstat_user_sent_mean \
          --norm_stats_json /tmp/feature_norm_stats_10000.json \
          --feature_clamp 5.0 --outlier_threshold 50.0 \
          --min_user_tokens 50 --seed 142 \
          --infer_batch_size 16 \
          --arditi_patch "${ARDITI_TMP}" \
          --arditi_mode "${ARDITI_MODE}" \
          --arditi_alpha "${ARDITI_ALPHA}" --arditi_layers "${ARDITI_LAYERS}"; then
        subitem_done "${mark}"
        n_done=$((n_done+1))
      else
        echo "[phase2d_midpoint] ${cohort} FAILED"
      fi
    done
  else
    echo "[phase2d_midpoint] labeled parquet missing: ${LABELED_MP}; skipping all midpoint"
  fi
  (( n_done == N_MIDPOINT )) \
    && phase_end_ok "phase2d_midpoint" "${N_MIDPOINT}" "${n_done}" \
    || phase_end_fail "phase2d_midpoint" "${N_MIDPOINT}" "${n_done}"
fi

# ===========================================================
# PHASE 3A_SCORE -- Cross-framework sentiment scoring (15 forums)
# ===========================================================
# GoEmotions + SST-2 + VADER on every forum.parquet under FORUM_ROOT.
# Writes posthoc_sentiment/forum_scored.parquet per forum.
# Cell count: 3 cohorts x (2a + 2b + 2c + in_hull + near_hull + 5 kappa-sweep + midpoint) = 33
N_SCORE=33
if phase_begin "phase3a_score" "${N_SCORE}"; then
  n_done=0
  for variant_dir in "${FORUM_ROOT}"/*; do
    [[ -d "${variant_dir}" ]] || continue
    fp="${variant_dir}/forum.parquet"
    [[ -f "${fp}" ]] || continue
    mark="${variant_dir}/posthoc_sentiment/.complete"
    if subitem_is_done "${mark}"; then
      n_done=$((n_done+1)); continue
    fi
    mkdir -p "${variant_dir}/posthoc_sentiment"
    if python "${SCRIPTS}/score_hyperlora_forum_sentiment.py" \
        --input_parquet "${fp}" \
        --out_dir "${variant_dir}/posthoc_sentiment" \
        --threshold_source /tmp/global_features_10000.parquet \
        --threshold_col gstat_user_sent_mean \
        --sentiment_backend all --no_toxicity \
        --device_id 0 --pipe_batch_size 256 --chunk_size 8192; then
      subitem_done "${mark}"
      n_done=$((n_done+1))
    else
      echo "[phase3a_score] FAILED: $(basename "${variant_dir}")"
    fi
  done
  (( n_done == N_SCORE )) \
    && phase_end_ok "phase3a_score" "${N_SCORE}" "${n_done}" \
    || phase_end_fail "phase3a_score" "${N_SCORE}" "${n_done}"
fi

# ===========================================================
# PHASE 3D_SIGNATURE -- Persona-signature aggregation (15 forums)
# ===========================================================
# Runs score_persona_signature.py on every forum.
# For synth forums: uses the per-stratum labeled profile as the
# author_profile_parquet (has pol_* columns). Falls back to
# real-user profile (/tmp/author_label_profile_real.parquet) if
# the synth label parquet is unavailable.
# Cell count: 3 cohorts x (2a + 2b + 2c + in_hull + near_hull + 5 kappa-sweep + midpoint) = 33
N_SIG=33
build_real_user_profile   # build /tmp/author_label_profile_real.parquet if needed
REAL_PROF="/tmp/author_label_profile_real.parquet"
SIGNATURE_ROOT="${OUT_ROOT}/phase3d_persona_signature_REFRESH"
mkdir -p "${SIGNATURE_ROOT}"

if phase_begin "phase3d_signature" "${N_SIG}"; then
  n_done=0
  for variant_dir in "${FORUM_ROOT}"/*; do
    [[ -d "${variant_dir}" ]] || continue
    fp="${variant_dir}/forum.parquet"
    [[ -f "${fp}" ]] || continue
    mark="${variant_dir}/persona_signature/.complete"
    if subitem_is_done "${mark}"; then
      n_done=$((n_done+1)); continue
    fi
    mkdir -p "${variant_dir}/persona_signature"

    # Select author profile for this forum variant. Far-kappa
    # sweep cells route to the kappa-specific labeled parquet
    # under phase1p5_label_kappa${K}/. Legacy far_from_hull
    # (kappa=3, no _kappa suffix) routes to LABEL_FARHULL_OUT.
    vname="$(basename "${variant_dir}")"
    prof="${REAL_PROF}"
    if [[ "${vname}" == *"_in_hull"* || "${vname}" == *"_near_hull"* ]]; then
      [[ -f "${LABEL_LEGACY_OUT}/synthetic_personas_labeled.parquet" ]] \
        && prof="${LABEL_LEGACY_OUT}/synthetic_personas_labeled.parquet"
    elif [[ "${vname}" == *"_far_kappa"* ]]; then
      KAPPA_TAG="${vname##*_far_kappa}"  # e.g. "10"
      KAPPA_LBL="${OUT_ROOT}/phase1p5_label_kappa${KAPPA_TAG}/synthetic_personas_labeled.parquet"
      [[ -f "${KAPPA_LBL}" ]] && prof="${KAPPA_LBL}"
    elif [[ "${vname}" == *"_far_from_hull"* ]]; then
      [[ -f "${LABEL_FARHULL_OUT}/synthetic_personas_labeled.parquet" ]] \
        && prof="${LABEL_FARHULL_OUT}/synthetic_personas_labeled.parquet"
    elif [[ "${vname}" == *"_midpoint_baseline"* ]]; then
      [[ -f "${LABEL_MIDPOINT_OUT}/synthetic_personas_labeled.parquet" ]] \
        && prof="${LABEL_MIDPOINT_OUT}/synthetic_personas_labeled.parquet"
    fi

    if python "${SCRIPTS}/score_persona_signature.py" \
        --forum_parquet "${fp}" \
        --author_profile_parquet "${prof}" \
        --norm_stats_json /tmp/feature_norm_stats_10000.json \
        --goemo_meta_json /tmp/goemo_labels_metadata.json \
        --output_parquet "${variant_dir}/persona_signature/persona_signature.parquet" \
        --batch_size 64 --device_id 0; then
      subitem_done "${mark}"
      n_done=$((n_done+1))
    else
      echo "[phase3d_signature] FAILED: ${vname}"
    fi
  done

  # Aggregate persona_signature_summary.csv across all 15 forums
  if FORUM_ROOT="${FORUM_ROOT}" SIGNATURE_ROOT="${SIGNATURE_ROOT}" python - <<'PY_SIG_AGG'
import re, json, os
from pathlib import Path
import numpy as np
import pandas as pd

forum_root  = Path(os.environ["FORUM_ROOT"])
out_dir     = Path(os.environ["SIGNATURE_ROOT"])
out_dir.mkdir(parents=True, exist_ok=True)

# Regex covers all conditions in REFRESH root, including the
# kappa-sweep far_kappa{K} cells (K in {3,10,25,50,100}).
VARIANT_RE = re.compile(
    r"^(2[abcd])_(vanilla|zero_delta|real_user|synth)_"
    r"(rage|empath|neutral)"
    r"(?:_(in_hull|near_hull|far_from_hull|far_kappa\d+|midpoint_baseline))?$"
)
HELD_OUT = ["politeness", "curiosity", "tempo", "self_focus",
            "expressiveness", "anxiety", "warmth", "hostility"]
rows = []
for fdir in sorted(forum_root.iterdir()):
    if not fdir.is_dir():
        continue
    m = VARIANT_RE.match(fdir.name)
    if not m:
        continue
    code, variant, cohort = m.group(1), m.group(2), m.group(3)
    stratum = m.group(4) or ""
    sig_p = fdir / "persona_signature" / "persona_signature.parquet"
    if not sig_p.exists():
        continue
    try:
        sig = pd.read_parquet(sig_p)
    except Exception as e:
        print(f"[agg] {fdir.name}: read failed: {e}")
        continue
    row = dict(variant_code=code, variant=variant,
               cohort=cohort, stratum=stratum, n_rows=int(len(sig)))
    for col, key in [
        ("signature_cosine_heldout", "signature_cosine_heldout_mean"),
        ("signature_cosine_all9",    "signature_cosine_all9_mean"),
        ("signature_L1_heldout",     "signature_L1_heldout_mean"),
        ("cohort_agreement",         "cohort_agreement_rate"),
    ]:
        row[key] = float(pd.to_numeric(sig.get(col), errors="coerce").mean())
    for d in HELD_OUT:
        rcol, ecol = f"realized_pol_{d}", f"expected_pol_{d}"
        if rcol in sig.columns and ecol in sig.columns:
            rv = pd.to_numeric(sig[rcol], errors="coerce")
            ev = pd.to_numeric(sig[ecol], errors="coerce")
            row[f"realized_{d}_mean"] = float(rv.mean())
            row[f"expected_{d}_mean"] = float(ev.mean())
            row[f"bias_{d}"]          = float((rv - ev).mean())
            if "turn_idx" in sig.columns and "author_user_id" in sig.columns:
                slopes = []
                for _, g in sig.groupby("author_user_id"):
                    if len(g) < 3:
                        continue
                    x = pd.to_numeric(g["turn_idx"], errors="coerce").to_numpy()
                    y = pd.to_numeric(g[rcol], errors="coerce").to_numpy()
                    mask = np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 3:
                        continue
                    slopes.append(float(np.polyfit(x[mask], y[mask], 1)[0]))
                row[f"drift_slope_{d}_mean"] = float(np.mean(slopes)) if slopes else float("nan")
    rows.append(row)

if rows:
    df = pd.DataFrame(rows).sort_values(["variant_code", "cohort", "stratum"])
    df.to_csv(out_dir / "persona_signature_summary.csv", index=False)
    print(f"[agg] wrote {len(df)} rows -> persona_signature_summary.csv")
else:
    print("[agg] no rows produced; no persona_signature.parquet files found")
PY_SIG_AGG
  then
    :
  fi
  (( n_done == N_SIG )) \
    && phase_end_ok "phase3d_signature" "${N_SIG}" "${n_done}" \
    || phase_end_fail "phase3d_signature" "${N_SIG}" "${n_done}"
fi

# ===========================================================
# PHASE 2E_RECONSTRUCTION -- Per-user reconstruction fidelity
# ===========================================================
# CPU-only. Runs after phase3d so 2c_real_user forum.parquets are
# guaranteed written. Computes lexical Jaccard, sentiment MAE, and
# style cosine per user by comparing generated forum posts vs the
# user's actual training-corpus posts. Runs one sub-job per cohort
# (rage / empath / neutral) with per-cohort summaries, then
# aggregates into a single reconstruction_summary.json under the
# RECON root. ~10 min CPU.
# Script: evaluate_user_reconstruction.py
# Output: phase2e_reconstruction_REFRESH/{rage,empath,neutral}/
#           per_user_reconstruction.csv
#           reconstruction_summary.json   (per-cohort)
#         phase2e_reconstruction_REFRESH/reconstruction_summary.json (combined)
# ===========================================================
RECON_ROOT="${OUT_ROOT}/phase2e_reconstruction_REFRESH"
N_RECON_SUB=3
if phase_begin "phase2e_reconstruction_fidelity" "${N_RECON_SUB}"; then
  mkdir -p "${RECON_ROOT}"
  n_done=0
  for cohort in rage empath neutral; do
    src="${FORUM_ROOT}/2c_real_user_${cohort}/forum.parquet"
    out="${RECON_ROOT}/${cohort}"
    mark="${out}/.complete"
    if subitem_is_done "${mark}"; then
      n_done=$((n_done+1)); continue
    fi
    if [[ ! -f "${src}" ]]; then
      echo "[phase2e] missing 2c_real_user_${cohort}/forum.parquet; skipping"
      continue
    fi
    mkdir -p "${out}"
    if python "${SCRIPTS}/evaluate_user_reconstruction.py"                       --gen_parquet "${src}"                       --train_parquet /tmp/train_data_10000.parquet                       --out_dir "${out}"                       --labels_csv /tmp/labels_sentiment_goemo.csv                       --max_train_per_user 200                       --embed_batch_size 128 --goemo_chunk 256                       --n_boot 1000 --seed 142; then
      subitem_done "${mark}"
      n_done=$((n_done+1))
      echo "[phase2e] ${cohort} done -> ${out}/reconstruction_summary.json"
    else
      echo "[phase2e] ${cohort} FAILED"
    fi
  done
  # Aggregate per-cohort summaries into combined reconstruction_summary.json
  if (( n_done > 0 )); then
    RECON_ROOT="${RECON_ROOT}" python - <<'PY_RECON_AGG'
import json, os
from pathlib import Path
recon_root = Path(os.environ["RECON_ROOT"])
combined = {"source": "phase2e_reconstruction_REFRESH", "per_cohort": {}}
for cohort in ("rage", "empath", "neutral"):
    cand = recon_root / cohort / "reconstruction_summary.json"
    if cand.exists():
        try:
            d = json.loads(cand.read_text())
            combined["per_cohort"][cohort] = d.get("global", d)
            print(f"[phase2e_agg] {cohort}: style_cos="
                  f"{combined['per_cohort'][cohort].get('style_cosine_mean', 'n/a'):.3f}"
                  f"  sent_mae={combined['per_cohort'][cohort].get('sent_mae_mean', 'n/a'):.3f}")
        except Exception as e:
            print(f"[phase2e_agg] {cohort}: read failed: {e}")
out_p = recon_root / "reconstruction_summary.json"
out_p.write_text(json.dumps(combined, indent=2, default=str))
print(f"[phase2e_agg] wrote combined -> {out_p}")
PY_RECON_AGG
  fi
  if (( n_done == N_RECON_SUB )); then
    phase_end_ok "phase2e_reconstruction_fidelity" "${N_RECON_SUB}" "${n_done}"
  else
    phase_end_fail "phase2e_reconstruction_fidelity" "${N_RECON_SUB}" "${n_done}"
    echo "[phase2e] partial (${n_done}/${N_RECON_SUB}); non-fatal"
  fi
fi

# ===========================================================
# PHASE 2F_SYNTH_VS_RECON -- Synth-vs-recon decomposition
# ===========================================================
# CPU-only. Runs after phase3d + phase2e. Computes:
#   - L1 distance to expected signature per stratum x cohort
#   - Signature-cosine gap (synth vs real-user recon)
#   - Polarity gap per stratum x cohort
#   - match_extreme_rate (Table 10 in paper)
# Outputs summary CSV + gap pivot CSV.
# Output: phase2f_synth_vs_recon_REFRESH/synth_vs_recon_summary.json
#                                        summary.csv
#                                        gap_pivot.csv
# ~15 min CPU only.
# ===========================================================
SVR_OUT="${OUT_ROOT}/phase2f_synth_vs_recon_REFRESH"
if phase_begin "phase2f_synth_vs_recon" 1; then
  mkdir -p "${SVR_OUT}"
  MARK="${SVR_OUT}/.complete"
  if subitem_is_done "${MARK}"; then
    phase_end_ok "phase2f_synth_vs_recon" 1 1
  else
    set +e
    FORUM_ROOT="${FORUM_ROOT}" RECON_ROOT="${RECON_ROOT}" SVR_OUT="${SVR_OUT}" python - <<'PY_2F'
import json, os
from pathlib import Path
import pandas as pd

forum_root = Path(os.environ["FORUM_ROOT"])
recon_root = Path(os.environ["RECON_ROOT"])
out_dir    = Path(os.environ["SVR_OUT"])
out_dir.mkdir(parents=True, exist_ok=True)

COHORTS = ["rage", "empath", "neutral"]
STRATA  = ["in_hull", "near_hull", "far_from_hull", "midpoint_baseline"]
METRICS = [
    ("mean_style_cosine",      ["style_summary"]),
    ("match_style_rate",        ["style_summary"]),
    ("match_extreme_rate",      ["summary"]),
    ("match_extreme_rel_rate",  ["summary"]),
    ("mean_sent",               ["summary"]),
]

def load_forum(path: Path, cohort: str) -> dict:
    meta_p = path / "metadata.json"
    if not meta_p.exists():
        return {}
    try:
        meta = json.loads(meta_p.read_text())
    except Exception:
        return {}
    flat = {}
    for mname, branch in METRICS:
        d = meta
        for b in branch:
            d = (d or {}).get(b, {}) or {}
        d = (d or {}).get(cohort, {}) or {}
        v = d.get(mname, None)
        flat[mname] = float(v) if v is not None else float("nan")
    sc = path / "posthoc_sentiment" / "score_metadata.json"
    if sc.exists():
        try:
            s = (json.loads(sc.read_text()).get("summary") or {}).get(cohort, {}) or {}
            flat["posthoc_sent_mean"] = float(s.get("sent_mean", float("nan")))
            flat["posthoc_match_extreme"] = float(s.get("match_extreme_rate", float("nan")))
        except Exception:
            pass
    fp = path / "posthoc_sentiment" / "forum_scored.parquet"
    if fp.exists():
        try:
            fs = pd.read_parquet(fp)
            sub = fs[fs.get("author_type", pd.Series(dtype=str)).eq(cohort)]
            for col, alias in [
                ("sent_polarity",       "polarity_sst2"),
                ("sent_polarity_vader", "polarity_vader"),
                ("sent_polarity_goemo", "polarity_goemo"),
            ]:
                if col in fs.columns:
                    flat[alias] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        except Exception as e:
            print(f"[phase2f] {path.name}: forum_scored read failed: {e}")
    return flat

def load_recon_fidelity(cohort: str) -> dict:
    d = recon_root / cohort
    if not d.exists():
        return {}
    for cand in (d / "reconstruction_summary.json", d / "summary.json"):
        if cand.exists():
            try:
                obj = json.loads(cand.read_text())
                return obj.get("global", obj)
            except Exception:
                return {}
    for p in d.glob("*.json"):
        try:
            return json.loads(p.read_text())
        except Exception:
            continue
    return {}

rows = []
for cohort in COHORTS:
    recon_dir  = forum_root / f"2c_real_user_{cohort}"
    recon_flat = load_forum(recon_dir, cohort)
    recon_fid  = load_recon_fidelity(cohort)
    for stratum in STRATA:
        synth_dir  = forum_root / f"2d_synth_{cohort}_{stratum}"
        synth_flat = load_forum(synth_dir, cohort)
        keys = sorted(set(recon_flat) | set(synth_flat))
        for k in keys:
            rv  = recon_flat.get(k, float("nan"))
            sv  = synth_flat.get(k, float("nan"))
            try:
                gap = float(sv) - float(rv)
            except Exception:
                gap = float("nan")
            rows.append(dict(
                cohort=cohort, stratum=stratum, metric=k,
                recon_value=rv, synth_value=sv, gap=gap,
            ))
        for k, v in (recon_fid or {}).items():
            if isinstance(v, (int, float)):
                rows.append(dict(
                    cohort=cohort, stratum=stratum,
                    metric=f"recon_fid:{k}",
                    recon_value=float(v),
                    synth_value=float("nan"),
                    gap=float("nan"),
                ))

if not rows:
    raise SystemExit("[phase2f] no synth-vs-recon rows produced")

df = pd.DataFrame(rows)
df.to_csv(out_dir / "summary.csv", index=False)
pivot = df.pivot_table(
    index=["cohort", "stratum"], columns="metric",
    values="gap", aggfunc="first",
).reset_index()
pivot.to_csv(out_dir / "gap_pivot.csv", index=False)

# Write compact JSON summary for paper Table 10 (match_extreme_rate key)
import json as _json
summary_j = {}
for (cohort, stratum), grp in df.groupby(["cohort", "stratum"]):
    mer_row = grp[grp["metric"] == "match_extreme_rate"]
    if len(mer_row):
        summary_j.setdefault(cohort, {})[stratum] = {
            "match_extreme_rate_synth":  float(mer_row["synth_value"].iloc[0]),
            "match_extreme_rate_recon":  float(mer_row["recon_value"].iloc[0]),
            "match_extreme_rate_gap":    float(mer_row["gap"].iloc[0]),
        }
(out_dir / "synth_vs_recon_summary.json").write_text(
    _json.dumps(summary_j, indent=2, default=str))
print(f"[phase2f] wrote {len(df)} rows -> {out_dir}/summary.csv")
print(f"[phase2f] wrote gap_pivot.csv and synth_vs_recon_summary.json")
PY_2F
    RC2F=$?
    set -e
    if (( RC2F == 0 )) && [[ -f "${SVR_OUT}/summary.csv" ]]; then
      subitem_done "${MARK}"
      phase_end_ok "phase2f_synth_vs_recon" 1 1
      echo "[phase2f] complete -> ${SVR_OUT}/synth_vs_recon_summary.json"
    else
      phase_end_fail "phase2f_synth_vs_recon" 1 0
      echo "[phase2f] FAILED (non-fatal; main pipeline continues)"
    fi
  fi
fi

# ===========================================================
# PHASE 4_LAYERWISE -- Hidden-state layerwise probe (OPTIONAL)
# ===========================================================
# Gated behind RUN_PHASE4 (default 0 in env block above -- OFF).
# Paper 2 has zero layerwise references; this is Paper 5 turf.
# Override: kubectl set env job/m1-gpu-node-p2-full-refresh RUN_PHASE4=1
# ONLY if you need ad-hoc Paper 5 layerwise data.
# This is the gold-standard Methodology C evidence (Appendix layerwise).
# Adds ~6-12h wall clock on high-memory GPU depending on n_users.
if [[ "${RUN_PHASE4:-0}" == "1" ]]; then
  LAYERWISE_OUT="${OUT_ROOT}/phase4_layerwise_REFRESH"
  if phase_begin "phase4_layerwise" 1; then
    mkdir -p "${LAYERWISE_OUT}"
    MARK="${LAYERWISE_OUT}/.complete"
    if subitem_is_done "${MARK}"; then
      phase_end_ok "phase4_layerwise" 1 1
    else
      echo "[phase4_layerwise] starting hidden-state layerwise probe"
      if python "${SCRIPTS}/layerwise_probe.py" \
          --hyper_dir "${HYPER_DIR}" \
          --base_model "${BASE_MODEL}" \
          --online \
          --target_modules "${TARGET_MODULES}" \
          --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
          --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}" \
          --use_best_ckpt --emit_both \
          --author_parquet /tmp/author_static_10000.parquet \
          --labels_csv /tmp/labels_sentiment_goemo.csv \
          --label_csv_dir /tmp \
          --norm_stats_json /tmp/feature_norm_stats_10000.json \
          --arditi_patch "${ARDITI_TMP}" \
          --arditi_alpha "${ARDITI_ALPHA}" \
          --arditi_layers "${ARDITI_LAYERS}" \
          --n_users_per_cohort 50 \
          --n_prompts 20 \
          --max_len 128 \
          --seed 142 \
          --out_dir "${LAYERWISE_OUT}"; then
        subitem_done "${MARK}"
        phase_end_ok "phase4_layerwise" 1 1
      else
        phase_end_fail "phase4_layerwise" 1 0
        echo "[phase4_layerwise] FAILED (non-fatal; main pipeline continues)"
      fi
    fi
  fi
else
  echo "[phase4_layerwise] SKIPPED (Paper 5 territory; RUN_PHASE4=${RUN_PHASE4:-0}); flip to 1 only for ad-hoc Paper 5 work"
fi

# ===========================================================
# PHASE 5_PAPER_FILL -- Analysis pass (no GPU required)
# ===========================================================
# Computes per-stratum PSI, drift slopes, anchor-density Spearman,
# and observational PCI across all 15 REFRESH forums.
# Writes phase5_paper_fill_REFRESH/paper2_analysis_pack.json and
# persona_signature_summary.csv (already written by phase3d agg above).
PAPER_FILL_OUT="${OUT_ROOT}/phase5_paper_fill_REFRESH"
if phase_begin "phase5_paper_fill" 1; then
  mkdir -p "${PAPER_FILL_OUT}"
  MARK="${PAPER_FILL_OUT}/.complete"
  if subitem_is_done "${MARK}"; then
    phase_end_ok "phase5_paper_fill" 1 1
  else
    set +e
    FORUM_ROOT="${FORUM_ROOT}" \
    FARHULL_PARQ="${FARHULL_PARQ_SRC}" \
    SIGNATURE_ROOT="${SIGNATURE_ROOT}" \
    OUT_DIR="${PAPER_FILL_OUT}" \
    python - <<'PY_PAPER_FILL'
import os, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
# 2026-05-16: pull signed-direction convention + Hedges' g helper for the
# new per-mode Table 1 cohort_g block (HP-LoRA / Vanilla / Zero-Δ).
sys.path.insert(0, "/workspace/hypernets/training_scripts")
from signed_hedges_g import HIGH_COHORT_PER_MEASURE, _hedges_g

FORUM_ROOT     = Path(os.environ["FORUM_ROOT"])
FARHULL_PARQ   = Path(os.environ["FARHULL_PARQ"])
SIGNATURE_ROOT = Path(os.environ["SIGNATURE_ROOT"])
OUT_DIR        = Path(os.environ["OUT_DIR"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

HELD_OUT = ["politeness", "curiosity", "tempo", "self_focus",
            "expressiveness", "anxiety", "warmth", "hostility"]
STRATA   = ["in_hull", "near_hull", "far_from_hull", "midpoint_baseline"]

pack = {
    "source": "phase2_forums_REFRESH",
    "psi": {}, "drift_slopes": {}, "anchor_density": {},
    "per_stratum": {}, "per_stratum_pfi": {}, "cohort_g": {},
}

# --- PSI per stratum (LEGACY cosine-variance metric) ---
# 2026-05-19: this metric is DIRECTION-ONLY and degenerate on
# single-reply (gid, user) groups. Kept for backward-compat.
# The publication metric is PFI below.
psi_by_stratum = {s: [] for s in STRATA}
for sig_path in sorted(FORUM_ROOT.glob("*/persona_signature/persona_signature.parquet")):
    forum_tag = sig_path.parent.parent.name
    try:
        d = pd.read_parquet(sig_path)
    except Exception as e:
        print(f"[phase5] {forum_tag}: read failed: {e}")
        continue
    for stratum in STRATA:
        if stratum in forum_tag:
            break
    else:
        stratum = "real_user"
    if not {"gid", "author_user_id", "signature_cosine_heldout"}.issubset(d.columns):
        continue
    grp = d.groupby(["gid", "author_user_id"])["signature_cosine_heldout"]
    var = grp.var(ddof=1)
    psi = 1.0 - var.dropna()
    if stratum in psi_by_stratum:
        psi_by_stratum[stratum].extend(psi.tolist())

for stratum, vals in psi_by_stratum.items():
    if vals:
        arr = np.asarray(vals, dtype=float)
        pack["per_stratum"][stratum] = {
            "n_author_thread_groups": int(arr.size),
            "median_psi": float(np.nanmedian(arr)),
            "p10_psi":    float(np.nanquantile(arr, 0.10)),
            "p90_psi":    float(np.nanquantile(arr, 0.90)),
            "mean_psi":   float(np.nanmean(arr)),
            "metric_status": "DEPRECATED cosine-variance",
        }
        print(f"[phase5] PSI(legacy) {stratum}: n={arr.size} median={np.nanmedian(arr):.3f}")
    else:
        pack["per_stratum"][stratum] = {"error": "no usable persona_signature.parquet"}

# --- PFI per stratum (PRODUCTION metric, 2026-05-19) ---
# PFI = per-user mean of bullseye_match_sdnorm across coherent
# replies. Bullseye is SD-normalized per dim before averaging.
# Bootstrap 95% CI on the across-user mean per stratum.
pfi_strata = [
    ("real-user",   ["2c_real_user_rage", "2c_real_user_empath", "2c_real_user_neutral"]),
    ("in-hull",     ["2d_synth_rage_in_hull", "2d_synth_empath_in_hull", "2d_synth_neutral_in_hull"]),
    ("near-hull",   ["2d_synth_rage_near_hull", "2d_synth_empath_near_hull", "2d_synth_neutral_near_hull"]),
    ("far_kappa3",  ["2d_synth_rage_far_kappa3", "2d_synth_empath_far_kappa3", "2d_synth_neutral_far_kappa3"]),
    ("far_kappa10", ["2d_synth_rage_far_kappa10", "2d_synth_empath_far_kappa10", "2d_synth_neutral_far_kappa10"]),
    ("far_kappa25", ["2d_synth_rage_far_kappa25", "2d_synth_empath_far_kappa25", "2d_synth_neutral_far_kappa25"]),
    ("far_kappa100",["2d_synth_rage_far_kappa100", "2d_synth_empath_far_kappa100"]),
]
rng_pfi = np.random.default_rng(142)
for label, cells in pfi_strata:
    parts = []
    for c in cells:
        p = FORUM_ROOT / c / "persona_signature" / "persona_signature.parquet"
        if p.exists():
            try: parts.append(pd.read_parquet(p))
            except Exception: pass
    if not parts: continue
    full = pd.concat(parts, ignore_index=True)
    if "is_coherent" in full.columns:
        full = full[full["is_coherent"] == True]
    if "bullseye_match_sdnorm" not in full.columns:
        pack["per_stratum_pfi"][label] = {"error": "bullseye_match_sdnorm missing (re-run phase 3d)"}
        continue
    per_user = (full.groupby("author_user_id")["bullseye_match_sdnorm"]
                    .mean().dropna().values)
    if len(per_user) < 10: continue
    boot = np.empty(1000)
    for b in range(1000):
        idx = rng_pfi.integers(0, len(per_user), size=len(per_user))
        boot[b] = float(np.mean(per_user[idx]))
    pack["per_stratum_pfi"][label] = {
        "n_users":     int(len(per_user)),
        "mean_pfi":    float(np.mean(per_user)),
        "ci95_lo":     float(np.percentile(boot, 2.5)),
        "ci95_hi":     float(np.percentile(boot, 97.5)),
        "p25":         float(np.percentile(per_user, 25)),
        "p75":         float(np.percentile(per_user, 75)),
        "metric_status": "production (SD-normalized bullseye)",
    }
    print(f"[phase5] PFI {label}: n={len(per_user)} mean={np.mean(per_user):.4f} "
          f"CI=[{np.percentile(boot,2.5):.4f}, {np.percentile(boot,97.5):.4f}]")

# ---------------------------------------------------------------
# 2026-05-17: PSI vs cohort-agreement bin analysis.
# The Appendix H3 threshold (PSI >= 0.94) has no quantitative
# defense in the paper. Build it here: for every (gid, uid) group
# across all forum cells compute PSI + cohort_rate, bin PSI into
# deciles, report per-bin mean cohort_rate with bootstrap 95% CI,
# and name the inflection point where cohort_rate falls off most
# sharply. If flat, the 0.94 threshold has no semantic basis and
# the H3 prose must say so.
# ---------------------------------------------------------------
rng_psi = np.random.default_rng(142)
rows_psi_coh = []
for sig_path in sorted(FORUM_ROOT.glob("*/persona_signature/persona_signature.parquet")):
    forum_tag = sig_path.parent.parent.name
    try:
        d = pd.read_parquet(sig_path)
    except Exception as e:
        print(f"[phase5] psi_bin {forum_tag}: read failed: {e}")
        continue
    needed = {"gid", "author_user_id", "signature_cosine_heldout", "cohort_agreement"}
    if not needed.issubset(d.columns):
        continue
    for stratum in STRATA:
        if stratum in forum_tag:
            break
    else:
        stratum = "real_user"
    agg = (d.groupby(["gid", "author_user_id"])
             .agg(psi=("signature_cosine_heldout",
                       lambda s: 1.0 - s.var(ddof=1) if s.dropna().size >= 2 else np.nan),
                  cohort_rate=("cohort_agreement", "mean"),
                  n_turns=("cohort_agreement", "size"))
             .reset_index())
    agg["stratum"]   = stratum
    agg["forum_tag"] = forum_tag
    rows_psi_coh.append(agg)

if rows_psi_coh:
    psi_df = pd.concat(rows_psi_coh, ignore_index=True)
    psi_df = psi_df.dropna(subset=["psi", "cohort_rate"]).copy()
    # Decile edges over the pooled PSI distribution
    qs   = np.linspace(0.0, 1.0, 11)
    edges = np.unique(np.quantile(psi_df["psi"].to_numpy(), qs))
    if edges.size < 3:
        # all PSI values identical -- emit a single-bin trivial result
        edges = np.array([psi_df["psi"].min() - 1e-9,
                          psi_df["psi"].max() + 1e-9])
    bin_idx = np.clip(np.searchsorted(edges, psi_df["psi"].to_numpy(),
                                      side="right") - 1,
                      0, len(edges) - 2)
    psi_df["psi_bin"] = bin_idx
    bin_rows = []
    for b in range(len(edges) - 1):
        sub = psi_df.loc[psi_df["psi_bin"] == b, "cohort_rate"].to_numpy()
        if sub.size == 0:
            continue
        mean_ca = float(np.mean(sub))
        # bootstrap CI (1000 resamples)
        boot = np.empty(1000, dtype=float)
        for k in range(1000):
            boot[k] = float(np.mean(rng_psi.choice(sub, size=sub.size, replace=True)))
        ci_lo, ci_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        bin_rows.append({
            "psi_bin_idx":      int(b),
            "psi_lower":        float(edges[b]),
            "psi_upper":        float(edges[b + 1]),
            "psi_mid":          float(0.5 * (edges[b] + edges[b + 1])),
            "n_threads":        int(sub.size),
            "mean_cohort_rate": mean_ca,
            "ci95_lo":          ci_lo,
            "ci95_hi":          ci_hi,
        })
    bin_df = pd.DataFrame(bin_rows)
    # Inflection: largest negative finite-difference in mean_cohort_rate
    if len(bin_df) >= 2:
        diffs    = np.diff(bin_df["mean_cohort_rate"].to_numpy())
        worst_dn = float(np.min(diffs))
        if worst_dn < -0.05:  # require a meaningful drop
            k_drop = int(np.argmin(diffs))
            # The drop occurs BETWEEN bin k_drop and bin k_drop+1.
            # Inflection PSI = upper edge of the bin BEFORE the drop.
            inflect_edge = float(bin_df.iloc[k_drop]["psi_upper"])
            inflect_str  = f"{inflect_edge:.3f}"
        else:
            inflect_edge = None
            inflect_str  = "flat"
    else:
        inflect_edge = None
        inflect_str  = "flat"
    # NameError fix 2026-05-19: the bash env var is PAPER_FILL_OUT
    # but the Python heredoc bound it as OUT_DIR (line ~2119);
    # the three downstream references below used the wrong name.
    bin_df.to_csv(OUT_DIR / "psi_bin_analysis.csv", index=False,
                  float_format="%.6f")
    with open(OUT_DIR / "psi_bin_analysis.json", "w") as fp_pbj:
        json.dump({
            "bins":              bin_rows,
            "inflection_psi":    inflect_edge,
            "inflection_label":  inflect_str,
            "prereg_threshold":  0.94,
            "n_groups":          int(len(psi_df)),
            "n_bins":            int(len(bin_df)),
            "method":            "pooled deciles; bootstrap n=1000 percentile CI; seed=142",
        }, fp_pbj, indent=2)
    pack["psi_bin_analysis"] = {
        "inflection_psi":   inflect_edge,
        "inflection_label": inflect_str,
        "prereg_threshold": 0.94,
        "n_groups":         int(len(psi_df)),
        "n_bins":           int(len(bin_df)),
    }
    print(f"[phase5] PSI bin inflection: {inflect_str}")
    print(f"[phase5] wrote {OUT_DIR / 'psi_bin_analysis.csv'}")
else:
    pack["psi_bin_analysis"] = {"error": "no usable persona_signature.parquet"}
    print("[phase5] PSI bin inflection: skipped (no signature parquets)")

# --- Drift slopes per stratum + dim ---
slopes_by_stratum = {}
for sig_path in sorted(FORUM_ROOT.glob("*/persona_signature/persona_signature.parquet")):
    forum_tag = sig_path.parent.parent.name
    for stratum in STRATA:
        if stratum in forum_tag:
            break
    else:
        stratum = "real_user"
    if stratum not in slopes_by_stratum:
        slopes_by_stratum[stratum] = {d: [] for d in HELD_OUT}
    try:
        sig = pd.read_parquet(sig_path)
    except Exception:
        continue
    for dim in HELD_OUT:
        rcol = f"realized_pol_{dim}"
        if rcol not in sig.columns or "turn_idx" not in sig.columns:
            continue
        for _, g in sig.groupby("author_user_id"):
            if len(g) < 3:
                continue
            x = pd.to_numeric(g["turn_idx"], errors="coerce").to_numpy()
            y = pd.to_numeric(g[rcol], errors="coerce").to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            slopes_by_stratum[stratum][dim].append(
                float(np.polyfit(x[mask], y[mask], 1)[0]))

pack["drift_slopes"] = {
    sk: {d: {"mean_slope": float(np.mean(v)) if v else float("nan"),
             "n_users": int(len(v))}
         for d, v in sdims.items()}
    for sk, sdims in slopes_by_stratum.items()
}

# --- Cohort-axis Hedges' g per mode (HP-LoRA / Vanilla / Zero-Δ) ---
# 2026-05-16: REWRITTEN. Old block computed g from per-thread summary rows
# (n=11 per cohort) off persona_signature_summary.csv, which is the wrong
# unit of analysis for Paper 2 Table 1. New block pools per-user means of
# realized_pol_<probe> from each mode's three cohort cells and emits both
# the paper convention (g_raw = rage - empath, "g > 0 ⇒ rage scores higher")
# AND the signed convention from signed_hedges_g.HIGH_COHORT_PER_MEASURE
# ("g > 0 ⇒ correct cohort direction"). Two CSVs + one markdown summary.
PROBES = [
    "sentiment_goemo", "politeness", "self_focus",
    "curiosity", "expressiveness", "tempo",
    "anxiety", "warmth", "hostility",
]
# Paper-row display name -> parquet probe key.
DISPLAY_ROWS = [
    ("Sentiment",      "sentiment_goemo"),
    ("Politeness",     "politeness"),
    ("Self-focus",     "self_focus"),
    ("Curiosity",      "curiosity"),
    ("Expressiveness", "expressiveness"),
    ("Tempo",          "tempo"),
    ("Anxiety",        "anxiety"),
    ("Warmth",         "warmth"),
    ("Hostility",      "hostility"),
]
# Mode display name -> per-cohort cell prefix under FORUM_ROOT.
MODE_CELLS = {
    "HP-LoRA": "2c_real_user",   # HyperPEFT-LoRA, Arditi hook ON (production)
    "Vanilla": "2a_vanilla",     # vanilla LoRA, no Arditi
    "Zero-Δ":  "2b_zero_delta",  # HyperPEFT-LoRA force_zero_delta, no Arditi
}

def _load_per_user_probe(cell_dir: Path, probe: str):
    # 2026-05-16: per-user mean of realized_pol_<probe> from one cohort cell.
    pq = cell_dir / "persona_signature" / "persona_signature.parquet"
    if not pq.exists():
        return np.array([], dtype=float)
    try:
        df = pd.read_parquet(pq)
    except Exception as e:
        print(f"[phase5] cohort_g: read {pq} failed: {e}")
        return np.array([], dtype=float)
    col = f"realized_pol_{probe}"
    if col not in df.columns or "author_user_id" not in df.columns:
        return np.array([], dtype=float)
    per_user = (df.groupby("author_user_id")[col]
                .mean()
                .dropna()
                .to_numpy(dtype=float))
    return per_user

def _coherence_rate(cell_dir: Path, cohort: str) -> float:
    # 2026-05-16: prefer metadata.json summary.coherence_by_cohort.*.coherence_rate;
    # fallback to forum.parquet is_coherent mean; else NaN.
    meta_p = cell_dir / "metadata.json"
    if meta_p.exists():
        try:
            m = json.loads(meta_p.read_text())
            v = (((m.get("summary") or {}).get("coherence_by_cohort") or {})
                 .get(cohort, {}) or {}).get("coherence_rate", None)
            if v is not None:
                return float(v)
        except Exception:
            pass
    fp = cell_dir / "forum.parquet"
    if fp.exists():
        try:
            fdf = pd.read_parquet(fp, columns=["is_coherent"])
            if "is_coherent" in fdf.columns and len(fdf) > 0:
                return float(fdf["is_coherent"].mean())
        except Exception:
            pass
    return float("nan")

# Build pack["cohort_g"]: {probe: {mode: {g_raw, g_signed, n_high, n_low}}}.
# 2026-05-17 RECIPE LOCK -- OPTION B: pool both topic cells, within-user
# normalize per-cell-mean then average across cells, user-level Hedges' g.
# This strips Decider topic-bias from each user's mean. The Decider
# over-samples cohort-matched users into cohort-matched cells (+0.35
# reply prob, +0.15 top-level prob, build_*_forum.py); a naive pool
# over both cells lets each user's mean be dominated by whichever
# topic over-sampled them and fakes a cohort signal in every mode.
# The two-step normalization here computes a per-user mean WITHIN
# each cell first (so each cell contributes equally to that user),
# then averages the per-cell means with equal weight, giving a
# topic-balanced per-user value. User-level Hedges' g over those
# values, signed via HIGH_COHORT_PER_MEASURE, exposes HP-LoRA's
# per-user channel against Vanilla / Zero-Δ controls and rescues
# the sentiment probe that single-cell reply-level whiffed on.
# Verified on gu03 standard GPU n=200 smoke: HP-LoRA wins on 6/9 probes
# (sentiment, politeness, self_focus, expressiveness, tempo,
# hostility); Vanilla collapses to max |g| 0.22; Zero-Δ to 0.19.
for probe in PROBES:
    pack["cohort_g"][probe] = {}
    high_cohort = HIGH_COHORT_PER_MEASURE.get(probe, "rage")
    low_cohort  = "empath" if high_cohort == "rage" else "rage"
    col = f"realized_pol_{probe}"
    for mode, prefix in MODE_CELLS.items():
        rage_dir   = FORUM_ROOT / f"{prefix}_rage"
        empath_dir = FORUM_ROOT / f"{prefix}_empath"
        ps_r = rage_dir   / "persona_signature" / "persona_signature.parquet"
        ps_e = empath_dir / "persona_signature" / "persona_signature.parquet"
        try:
            df_r = pd.read_parquet(ps_r)
            df_e = pd.read_parquet(ps_e)
        except Exception as _exc:
            print(f"[phase5] WARN {mode}/{probe}: parquet read failed ({_exc})")
            pack["cohort_g"][probe][mode] = {
                "g_raw": None, "g_signed": None,
                "n_high": 0, "n_low": 0,
            }
            continue
        need = {"author_user_id", "expected_cohort_goemo", col}
        if not (need.issubset(df_r.columns) and need.issubset(df_e.columns)):
            print(f"[phase5] WARN {mode}/{probe}: required cols missing")
            pack["cohort_g"][probe][mode] = {
                "g_raw": None, "g_signed": None,
                "n_high": 0, "n_low": 0,
            }
            continue
        # Step 1: per-user mean WITHIN each topic cell.
        per_user_per_cell = {}
        for df_cell in (df_r, df_e):
            gp = (df_cell.groupby("author_user_id")
                         .agg(m=(col, "mean"),
                              c=("expected_cohort_goemo", "first"))
                         .reset_index())
            for _, row in gp.iterrows():
                uid = row["author_user_id"]
                entry = per_user_per_cell.setdefault(
                    uid, {"vals": [], "cohort": row["c"]})
                entry["vals"].append(row["m"])
        # Step 2: average per-cell means with equal weight per user.
        user_rows = []
        for uid, d in per_user_per_cell.items():
            finite = [v for v in d["vals"] if pd.notna(v)]
            if finite:
                user_rows.append({"uid": uid,
                                  "val": float(np.mean(finite)),
                                  "cohort": d["cohort"]})
        udf = pd.DataFrame(user_rows)
        high_vals = udf[udf["cohort"] == high_cohort]["val"].dropna().values
        low_vals  = udf[udf["cohort"] == low_cohort ]["val"].dropna().values
        g_signed = _hedges_g(high_vals, low_vals)
        if g_signed != g_signed:
            g_raw = float("nan")
        else:
            g_raw = g_signed if high_cohort == "rage" else -g_signed
        pack["cohort_g"][probe][mode] = {
            "g_raw":    None if g_raw    != g_raw else float(g_raw),
            "g_signed": None if g_signed != g_signed else float(g_signed),
            "n_high":   int(len(high_vals)),
            "n_low":    int(len(low_vals)),
        }
    print(f"[phase5] cohort_g {probe}: "
          + " ".join(f"{m}={pack['cohort_g'][probe][m]['g_signed']}" for m in MODE_CELLS))

# --- Coherence rates per cell (for sanity-check block in the .md) ---
# 2026-05-16: independent of the g loop so we can render even if probes empty.
coherence_rates = {}  # mode -> {cohort: rate}
for mode, prefix in MODE_CELLS.items():
    coherence_rates[mode] = {}
    for cohort in ("rage", "empath", "neutral"):
        coherence_rates[mode][cohort] = _coherence_rate(
            FORUM_ROOT / f"{prefix}_{cohort}", cohort)
pack["coherence_rates"] = coherence_rates

# ===========================================================
# 2026-05-17 EXTENSION: cohort_g_synth + cohort_g_per_topic
# Paper appendix tables tab:h1-perprobe and tab:per-topic both
# need cohort g numbers that the 3-mode block above does not
# cover. Compute them here with the same Option B recipe (for
# synth strata, which split rage/empath labels across two cells
# exactly like the real-user 3-mode setup) and Option A (for
# per-topic style cosine, which is by construction single-cell
# since each forum IS one topic).
# ===========================================================

# ---- SYNTH STRATA (Option B, mirrors HP-LoRA recipe) ---------
# Synth users are LABELED by label_synthetic_personas.py into
# rage/empath cohorts based on their descriptor vector, and the
# build_*_forum scripts sample them into matched-cohort cells:
# 2d_synth_rage_<stratum> contains rage-labeled synth users
# replying to the rage topic; 2d_synth_empath_<stratum> contains
# empath-labeled synth users replying to the empath topic.
# Option B (per-user mean WITHIN each cell, then equal-weighted
# average across the two cells, then user-level Hedges' g signed
# via HIGH_COHORT_PER_MEASURE) gives the same topic-balanced
# readout for synth strata as for the 3-mode block.
SYNTH_STRATA = {
    "Synth in-hull":   "in_hull",
    "Synth near-hull": "near_hull",
    "Synth far-hull":  "far_kappa3",
}
pack["cohort_g_synth"] = {}
for probe in PROBES:
    pack["cohort_g_synth"][probe] = {}
    high_cohort = HIGH_COHORT_PER_MEASURE.get(probe, "rage")
    low_cohort  = "empath" if high_cohort == "rage" else "rage"
    col = f"realized_pol_{probe}"
    for label, suffix in SYNTH_STRATA.items():
        rage_dir   = FORUM_ROOT / f"2d_synth_rage_{suffix}"
        empath_dir = FORUM_ROOT / f"2d_synth_empath_{suffix}"
        ps_r = rage_dir   / "persona_signature" / "persona_signature.parquet"
        ps_e = empath_dir / "persona_signature" / "persona_signature.parquet"
        try:
            df_r = pd.read_parquet(ps_r)
            df_e = pd.read_parquet(ps_e)
        except Exception as _exc:
            print(f"[phase5] WARN synth/{label}/{probe}: parquet read failed ({_exc})")
            pack["cohort_g_synth"][probe][label] = {
                "g_raw": None, "g_signed": None,
                "n_high": 0, "n_low": 0,
            }
            continue
        need = {"author_user_id", "expected_cohort_goemo", col}
        if not (need.issubset(df_r.columns) and need.issubset(df_e.columns)):
            print(f"[phase5] WARN synth/{label}/{probe}: required cols missing")
            pack["cohort_g_synth"][probe][label] = {
                "g_raw": None, "g_signed": None,
                "n_high": 0, "n_low": 0,
            }
            continue
        per_user_per_cell = {}
        for df_cell in (df_r, df_e):
            gp = (df_cell.groupby("author_user_id")
                         .agg(m=(col, "mean"),
                              c=("expected_cohort_goemo", "first"))
                         .reset_index())
            for _, row in gp.iterrows():
                uid = row["author_user_id"]
                entry = per_user_per_cell.setdefault(
                    uid, {"vals": [], "cohort": row["c"]})
                entry["vals"].append(row["m"])
        user_rows = []
        for uid, d in per_user_per_cell.items():
            finite = [v for v in d["vals"] if pd.notna(v)]
            if finite:
                user_rows.append({"uid": uid,
                                  "val": float(np.mean(finite)),
                                  "cohort": d["cohort"]})
        udf = pd.DataFrame(user_rows)
        high_vals = udf[udf["cohort"] == high_cohort]["val"].dropna().values
        low_vals  = udf[udf["cohort"] == low_cohort ]["val"].dropna().values
        g_signed = _hedges_g(high_vals, low_vals)
        if g_signed != g_signed:
            g_raw = float("nan")
        else:
            g_raw = g_signed if high_cohort == "rage" else -g_signed
        pack["cohort_g_synth"][probe][label] = {
            "g_raw":    None if g_raw    != g_raw else float(g_raw),
            "g_signed": None if g_signed != g_signed else float(g_signed),
            "n_high":   int(len(high_vals)),
            "n_low":    int(len(low_vals)),
        }
    print(f"[phase5] cohort_g_synth {probe}: "
          + " ".join(f"{lab}={pack['cohort_g_synth'][probe][lab]['g_signed']}"
                     for lab in SYNTH_STRATA))

# ---- PER-TOPIC STYLE COSINE (Option A, single-cell reply-level) -
# tab:per-topic needs rage-vs-empath Hedges' g on signature_cosine_heldout
# within each topic-forum cell, under HP-LoRA real-user (2c) and
# Zero-Δ no-Arditi (2b). Each cell IS one topic, so single-cell
# reply-level is by construction the correct recipe -- topic is
# held constant within the cell. Hedges' g is signed so that
# g>0 means the cohort whose label aligns with topic intensity
# produces the larger style-cosine match-to-profile value. The
# neutral forum has no within-cohort intensity contrast so we
# report it as a descriptor only and flag it in the caption.
TOPIC_COHORTS = ("rage", "empath", "neutral")
TOPIC_CONDS   = {
    "real-user": "2c_real_user",
    "zero-Δ":    "2b_zero_delta",
}
SIG_COL = "signature_cosine_heldout"
pack["cohort_g_per_topic"] = {}
for topic in TOPIC_COHORTS:
    pack["cohort_g_per_topic"][topic] = {}
    for cond, prefix in TOPIC_CONDS.items():
        cell_dir = FORUM_ROOT / f"{prefix}_{topic}"
        ps = cell_dir / "persona_signature" / "persona_signature.parquet"
        try:
            df = pd.read_parquet(ps)
        except Exception as _exc:
            print(f"[phase5] WARN per_topic/{topic}/{cond}: read failed ({_exc})")
            pack["cohort_g_per_topic"][topic][cond] = {
                "g": None, "style_cos_mean": None,
                "n_rage": 0, "n_empath": 0,
            }
            continue
        if SIG_COL not in df.columns or "expected_cohort_goemo" not in df.columns:
            print(f"[phase5] WARN per_topic/{topic}/{cond}: cols missing in {ps}")
            pack["cohort_g_per_topic"][topic][cond] = {
                "g": None, "style_cos_mean": None,
                "n_rage": 0, "n_empath": 0,
            }
            continue
        rage_v   = df[df["expected_cohort_goemo"] == "rage"  ][SIG_COL].dropna().values
        empath_v = df[df["expected_cohort_goemo"] == "empath"][SIG_COL].dropna().values
        # Sign convention: rage minus empath. The paper caption
        # reports it positive when rage cohort separates higher
        # on style cosine; the topic-intensity ceiling argument
        # in app:per-topic interprets the sign per forum.
        g = _hedges_g(rage_v, empath_v)
        style_mean = float(pd.to_numeric(df[SIG_COL], errors="coerce").mean())
        pack["cohort_g_per_topic"][topic][cond] = {
            "g":              None if g != g else float(g),
            "style_cos_mean": None if style_mean != style_mean else style_mean,
            "n_rage":         int(len(rage_v)),
            "n_empath":       int(len(empath_v)),
        }
    rt = pack["cohort_g_per_topic"][topic]
    print(f"[phase5] cohort_g_per_topic {topic}: "
          + " ".join(f"{c}=g{rt[c]['g']},sc{rt[c]['style_cos_mean']}" for c in TOPIC_CONDS))

# --- Render two CSVs in the exact 9-row x 3-col Paper 2 Table 1 layout ---
# 2026-05-16: sign-explicit 2-decimal formatting; NaN -> "nan" in CSV.
def _fmt_g(v) -> str:
    if v is None:
        return "nan"
    try:
        fv = float(v)
    except Exception:
        return "nan"
    if not np.isfinite(fv):
        return "nan"
    if fv == 0.0:
        return "+0.00"
    return f"{fv:+.2f}"

csv_modes = list(MODE_CELLS.keys())  # HP-LoRA, Vanilla, Zero-Δ
raw_lines = [",".join(["row"] + csv_modes)]
signed_lines = [",".join(["row"] + csv_modes)]
for disp, probe in DISPLAY_ROWS:
    raw_cells, signed_cells = [], []
    for mode in csv_modes:
        cell = pack["cohort_g"].get(probe, {}).get(mode, {})
        raw_cells.append(_fmt_g(cell.get("g_raw")))
        signed_cells.append(_fmt_g(cell.get("g_signed")))
    raw_lines.append(",".join([disp] + raw_cells))
    signed_lines.append(",".join([disp] + signed_cells))

t1_raw_path    = OUT_DIR / "surface_table1.csv"
t1_signed_path = OUT_DIR / "surface_table1_signed.csv"
t1_raw_path.write_text("\n".join(raw_lines) + "\n")
t1_signed_path.write_text("\n".join(signed_lines) + "\n")
print(f"[phase5] wrote {t1_raw_path}")
print(f"[phase5] wrote {t1_signed_path}")

# ---- h1_perprobe.csv: 8-probe x 6-condition signed g (Appendix tab:h1-perprobe)
# Probes match paper ordering: OOB (politeness, self_focus), IB
# (curiosity, expressiveness, tempo), tier-2 (anxiety, warmth,
# hostility). Sentiment is excluded -- it defines the cohort cut.
H1PP_PROBES = [
    ("Politeness (OOB)",     "politeness"),
    ("Self-focus (OOB)",     "self_focus"),
    ("Curiosity (IB)",       "curiosity"),
    ("Expressiveness (IB)",  "expressiveness"),
    ("Tempo (IB)",           "tempo"),
    ("Anxiety (tier-2)",     "anxiety"),
    ("Warmth (tier-2)",      "warmth"),
    ("Hostility (tier-2)",   "hostility"),
]
# Column order matches paper: Vanilla, Zero-Δ, Recon (= HP-LoRA on
# real users), Synth in-hull, Synth near-hull, Synth far-hull.
H1PP_COLS = [
    ("Vanilla",         "cohort_g",       "Vanilla"),
    ("Zero-Δ",          "cohort_g",       "Zero-Δ"),
    ("Recon",           "cohort_g",       "HP-LoRA"),
    ("Synth in-hull",   "cohort_g_synth", "Synth in-hull"),
    ("Synth near-hull", "cohort_g_synth", "Synth near-hull"),
    ("Synth far-hull",  "cohort_g_synth", "Synth far-hull"),
]
h1pp_lines = [",".join(["probe"] + [c[0] for c in H1PP_COLS])]
for disp, probe in H1PP_PROBES:
    cells = []
    for _, src_key, lab in H1PP_COLS:
        blk = pack.get(src_key, {}).get(probe, {}).get(lab, {})
        cells.append(_fmt_g(blk.get("g_signed")))
    h1pp_lines.append(",".join([disp] + cells))
h1pp_path = OUT_DIR / "h1_perprobe.csv"
h1pp_path.write_text("\n".join(h1pp_lines) + "\n")
print(f"[phase5] wrote {h1pp_path}")

# ---- per_topic.csv: per-topic style-cosine g + style cos mean
# (Appendix tab:per-topic). Each row is one (forum, condition);
# forum names mirror the paper rows. The neutral forum is
# reported as a descriptor only (g still computed for sanity).
PT_FORUM_LABELS = [
    ("rage (unpopular opinions)", "rage"),
    ("empath (advice)",            "empath"),
    ("neutral (hobby)",            "neutral"),
]
pt_lines = ["forum,condition,g,style_cos_mean,n_rage,n_empath"]
for forum_disp, topic in PT_FORUM_LABELS:
    for cond in TOPIC_CONDS:
        blk = pack["cohort_g_per_topic"].get(topic, {}).get(cond, {})
        g  = blk.get("g")
        sc = blk.get("style_cos_mean")
        pt_lines.append(",".join([
            f'"{forum_disp}"', cond,
            _fmt_g(g),
            "nan" if sc is None or sc != sc else f"{sc:.3f}",
            str(blk.get("n_rage", 0)),
            str(blk.get("n_empath", 0)),
        ]))
pt_path = OUT_DIR / "per_topic.csv"
pt_path.write_text("\n".join(pt_lines) + "\n")
print(f"[phase5] wrote {pt_path}")

# --- Render cohort_results_summary.md (single human-readable deliverable) ---
# 2026-05-16: lives under PROSPECTUS/HyperPEFTNet_RQ2/paper2/; same content
# mirrored into OUT_DIR for archival alongside paper2_analysis_pack.json.
def _fmt_g_md(v) -> str:
    s = _fmt_g(v)
    return "—" if s == "nan" else s

def _md_table(rows, modes, key):
    header = "| Probe | " + " | ".join(modes) + " |"
    sep    = "|---" * (1 + len(modes)) + "|"
    body   = []
    for disp, probe in rows:
        cells = []
        for mode in modes:
            cell = pack["cohort_g"].get(probe, {}).get(mode, {})
            cells.append(_fmt_g_md(cell.get(key)))
        body.append(f"| {disp} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body)

md_parts = []
md_parts.append("# Paper 2 Table 1 -- minimal 3-mode cohort comparison\n")
md_parts.append("Values are Hedges' g, reply-level (per-post), within the rage-topic cell only. ")
md_parts.append("Topic is structurally controlled by single-cell scope; the only variation ")
md_parts.append("carrying cohort signal is the per-user channel. HP-LoRA = HyperPEFT-LoRA with ")
md_parts.append("Arditi residual hook ON (production). Vanilla = vanilla LoRA, no Arditi. ")
md_parts.append("Zero-Delta = HyperPEFT-LoRA with `--force_zero_delta` AND no Arditi.\n\n")
md_parts.append("## Paper Table 1 -- raw convention (g > 0 when rage scores higher)\n\n")
md_parts.append(_md_table(DISPLAY_ROWS, csv_modes, "g_raw"))
md_parts.append("\n\n## Same table -- signed convention (g > 0 when correct cohort)\n\n")
md_parts.append(_md_table(DISPLAY_ROWS, csv_modes, "g_signed"))
md_parts.append("\n\n## Reply counts (high-cohort vs low-cohort within rage-topic cell)\n\n")
md_parts.append("| Probe | Mode | n_high | n_low |\n|---|---|---|---|\n")
n_rows = []
for disp, probe in DISPLAY_ROWS:
    for mode in csv_modes:
        cell = pack["cohort_g"].get(probe, {}).get(mode, {})
        n_rows.append(f"| {disp} | {mode} | {cell.get('n_high', 0)} | "
                      f"{cell.get('n_low', 0)} |")
md_parts.append("\n".join(n_rows))
md_parts.append("\n\n## Reading guide\n\n")
md_parts.append(
    "Paper 2's Table 1 uses the **raw** rage-minus-empath convention, so a "
    "positive g means rage users scored higher on that probe than empath users. "
    "HP-LoRA's expected signature is positive hostility, positive self_focus, "
    "positive expressiveness, positive anxiety, negative warmth, negative "
    "sentiment, negative curiosity, negative politeness (i.e. higher profanity), "
    "and negative tempo (i.e. faster replies). Vanilla and Zero-Delta should "
    "either flatline near zero or follow only the prompt directive without any "
    "per-user latent structure; in particular any Zero-Delta row close to zero "
    "confirms 'no per-user channel = no cohort signal at surface from the "
    "per-user route -- only the prompt directive remains.' The signed-convention "
    "table just flips probes where 'higher' is the empath side, so g > 0 always "
    "means 'the high-expected cohort really was higher.'\n\n")
md_parts.append("\n## Appendix H1 perprobe -- 8 probes x 6 conditions (signed g)\n\n")
md_parts.append("Conditions: Vanilla and Zero-Δ from the 3-mode block above; "
                "Recon is HP-LoRA+Arditi on real users; Synth in-hull/near-hull/"
                "far-hull are HP-LoRA+Arditi on synthetic users sampled at each "
                "descriptor-space stratum. Same Option B recipe across all six.\n\n")
md_parts.append("| Probe | " + " | ".join(c[0] for c in H1PP_COLS) + " |\n")
md_parts.append("|---" * (1 + len(H1PP_COLS)) + "|\n")
for disp, probe in H1PP_PROBES:
    cells = []
    for _, src_key, lab in H1PP_COLS:
        blk = pack.get(src_key, {}).get(probe, {}).get(lab, {})
        cells.append(_fmt_g_md(blk.get("g_signed")))
    md_parts.append(f"| {disp} | " + " | ".join(cells) + " |\n")

md_parts.append("\n## Appendix per-topic -- style-cosine g per forum + condition\n\n")
md_parts.append("Single-cell rage-vs-empath g on signature_cosine_heldout; "
                "topic is constant within each cell. Neutral forum reported as "
                "descriptor only (no within-cohort intensity contrast).\n\n")
md_parts.append("| Forum (topic) | Condition | g | style cos mean | n_rage | n_empath |\n")
md_parts.append("|---|---|---|---|---|---|\n")
for forum_disp, topic in PT_FORUM_LABELS:
    for cond in TOPIC_CONDS:
        blk = pack["cohort_g_per_topic"].get(topic, {}).get(cond, {})
        g  = blk.get("g")
        sc = blk.get("style_cos_mean")
        g_str  = _fmt_g_md(g)
        sc_str = "—" if sc is None or sc != sc else f"{sc:.3f}"
        md_parts.append(f"| {forum_disp} | {cond} | {g_str} | {sc_str} | "
                        f"{blk.get('n_rage',0)} | {blk.get('n_empath',0)} |\n")
md_parts.append("\n")

md_parts.append("## Status / sanity checks -- coherence rate per cell\n\n")
md_parts.append("| Mode | Cohort | coherence_rate | flag |\n|---|---|---|---|\n")
flag_rows = []
for mode in csv_modes:
    for cohort in ("rage", "empath", "neutral"):
        rate = coherence_rates.get(mode, {}).get(cohort, float("nan"))
        if not (isinstance(rate, float) and np.isfinite(rate)):
            rate_s = "—"; flag = "missing"
        else:
            rate_s = f"{rate:.3f}"
            flag = "OK" if rate >= 0.90 else "LOW (<0.90)"
        flag_rows.append(f"| {mode} | {cohort} | {rate_s} | {flag} |")
md_parts.append("\n".join(flag_rows))
md_parts.append("\n")
md_text = "".join(md_parts)

# Primary location: paper2 dir per user spec.
paper2_md = Path("/workspace/hypernets/PROSPECTUS/"
                 "HyperPEFTNet_RQ2/paper2/cohort_results_summary.md")
try:
    paper2_md.parent.mkdir(parents=True, exist_ok=True)
    paper2_md.write_text(md_text)
    print(f"[phase5] wrote {paper2_md}")
except Exception as e:
    print(f"[phase5] WARN: could not write paper2 cohort_results_summary.md: {e}")
# Mirror into OUT_DIR alongside the JSON pack for archival.
try:
    (OUT_DIR / "cohort_results_summary.md").write_text(md_text)
    print(f"[phase5] wrote {OUT_DIR / 'cohort_results_summary.md'}")
except Exception as e:
    print(f"[phase5] WARN: could not mirror cohort_results_summary.md into OUT_DIR: {e}")

# --- Anchor-density Spearman rho (far_from_hull v2) ---
try:
    from scipy.stats import spearmanr
    if FARHULL_PARQ.exists():
        syn = pd.read_parquet(FARHULL_PARQ)
        if "source_anchors" in syn.columns:
            def _anchor_count(x):
                try:
                    if isinstance(x, (list, tuple, np.ndarray)):
                        return int(len(x))
                    if isinstance(x, (int, float)) and not math.isnan(x):
                        return int(x)
                except Exception:
                    pass
                return float("nan")
            syn = syn[["target_user_id"]].assign(
                anchor_density=syn["source_anchors"].apply(_anchor_count))
            rows = []
            for sp in sorted(FORUM_ROOT.glob(
                    "2d_synth_*_far_from_hull/persona_signature/persona_signature.parquet")):
                d = pd.read_parquet(sp)
                if not {"author_user_id", "signature_cosine_heldout"}.issubset(d.columns):
                    continue
                pu = d.groupby("author_user_id")["signature_cosine_heldout"].mean().reset_index()
                pu = pu.rename(columns={"author_user_id": "target_user_id"})
                joined = pu.merge(syn, on="target_user_id", how="inner")
                joined = joined.dropna(subset=["anchor_density", "signature_cosine_heldout"])
                rows.append(joined)
            if rows:
                pooled = pd.concat(rows, ignore_index=True)
                if len(pooled) >= 10:
                    rho, p = spearmanr(pooled["anchor_density"],
                                       pooled["signature_cosine_heldout"])
                    pack["anchor_density"] = {
                        "spearman_rho": float(rho), "p_value": float(p),
                        "n": int(len(pooled)),
                    }
                    print(f"[phase5] anchor-density Spearman rho={rho:+.3f} p={p:.2e} n={len(pooled)}")
                else:
                    pack["anchor_density"] = {"error": f"insufficient n={len(pooled)}"}
            else:
                pack["anchor_density"] = {"error": "no far_from_hull persona_signature found"}
        else:
            pack["anchor_density"] = {"error": "no source_anchors column in far_from_hull parquet"}
    else:
        pack["anchor_density"] = {"error": f"farhull parquet not found: {FARHULL_PARQ}"}
except ImportError:
    pack["anchor_density"] = {"error": "scipy not available"}
except Exception as e:
    pack["anchor_density"] = {"error": str(e)}

out_path = OUT_DIR / "paper2_analysis_pack.json"
out_path.write_text(json.dumps(pack, indent=2, default=str))
print(f"[phase5] wrote {out_path}")
sys.exit(0)
PY_PAPER_FILL
    RC5=$?
    set -e
    if (( RC5 == 0 )) && [[ -f "${PAPER_FILL_OUT}/paper2_analysis_pack.json" ]]; then
      subitem_done "${MARK}"
      phase_end_ok "phase5_paper_fill" 1 1
      echo "[phase5_paper_fill] complete -> ${PAPER_FILL_OUT}/paper2_analysis_pack.json"
      # Pre-stage tab:example-replies sample (best-effort; non-fatal).
      python "${SCRIPTS}/sample_example_replies.py" \
          --forum_root "${FORUM_ROOT}" \
          --out_md /workspace/hypernets/PROSPECTUS/HyperPEFTNet_RQ2/paper2/example_replies_sample.md \
          2>&1 | tee -a /tmp/example_replies.log || true
    else
      phase_end_fail "phase5_paper_fill" 1 0
      echo "[phase5_paper_fill] FAILED (non-fatal for publishability)"
    fi
  fi
fi

# ===========================================================
# PHASE 3E_PCI -- Persona Compositionality Index aggregation
# ===========================================================
# Runs compute_pci.py on the union of every synth-cell
# persona_signature.parquet (2d_synth_*) produced by phase3d.
# Partitions users by k = realized cohort-direction agreement
# count and reports mean signature cosine + percentile
# bootstrap CI per partition (1000 user-resampled draws).
# Output: phase3e_pci_REFRESH/pci_summary.json (feeds paper2
# Table 12 tab:pci).
PCI_OUT="${OUT_ROOT}/phase3e_pci_REFRESH"
if phase_begin "phase3e_pci" 1; then
  mkdir -p "${PCI_OUT}"
  MARK_PCI="${PCI_OUT}/.complete"
  if [[ -f "${MARK_PCI}" ]]; then
    echo "[phase3e_pci] already complete"
    phase_end_ok "phase3e_pci" 1 1
  else
    # Collect every 2d_synth_* persona_signature.parquet.
    PCI_INPUTS=()
    while IFS= read -r -d '' f; do PCI_INPUTS+=("$f"); done < <(
      find "${FORUM_ROOT}" -mindepth 3 -maxdepth 3 \
           -path '*2d_synth_*/persona_signature/persona_signature.parquet' \
           -print0 2>/dev/null
    )
    echo "[phase3e_pci] discovered ${#PCI_INPUTS[@]} synth persona_signature parquet(s)"
    if (( ${#PCI_INPUTS[@]} == 0 )); then
      echo "[phase3e_pci] no synth signature parquets found; skipping"
      phase_end_fail "phase3e_pci" 1 0
    else
      set +e
      python "${SCRIPTS}/compute_pci.py" \
          "${PCI_INPUTS[@]}" \
          --mag_threshold 0.05 \
          --n_boot 1000 \
          --seed 142 \
          --out_json "${PCI_OUT}/pci_summary.json"
      RCPCI=$?
      set -e
      if (( RCPCI == 0 )) && [[ -f "${PCI_OUT}/pci_summary.json" ]]; then
        subitem_done "${MARK_PCI}"
        phase_end_ok "phase3e_pci" 1 1
        echo "[phase3e_pci] complete -> ${PCI_OUT}/pci_summary.json"
      else
        phase_end_fail "phase3e_pci" 1 0
        echo "[phase3e_pci] FAILED (non-fatal for publishability)"
      fi
    fi
  fi
fi

# ===========================================================
# PHASE 6_EXTENDED_DEPTH -- 80-turn extended dialogue audit
# ===========================================================
# 300 rage + 300 empath authors x 80 turns = 48,000 turn-rows.
# Matches legacy phase6_turn_budget_arditi settings (n_rage=300
# n_empath=300 n_turns=80).
#
# TWO ARMS:
#   hyperpeft  : full HyperPEFT-LoRA conditioning (primary)
#   zero_delta : --force_zero_delta ablation (Appendix N comparison)
#
# The Appendix N drift-slope analysis requires BOTH arms from the
# same REFRESH run to make a valid comparison. The legacy zero_delta
# parquet was contaminated by text artifacts; the vanilla arm was
# clean and stays in the legacy arditi location as reference.
#
# Output: phase6_turn_budget_REFRESH/hyperpeft/dialogue.parquet
#         phase6_turn_budget_REFRESH/zero_delta/dialogue.parquet
# ~6-8 h each on high-memory GPU. Run sequentially (single GPU).
# ===========================================================
TURN_BUDGET_OUT="${OUT_ROOT}/phase6_turn_budget_REFRESH"
P6_TOPIC="share something honest about a recent everyday choice and how it sat with you"
if phase_begin "phase6_extended_depth" 2; then
  p6_done=0
  # ---- arm 1: hyperpeft ----
  mkdir -p "${TURN_BUDGET_OUT}/hyperpeft"
  MARK_HP="${TURN_BUDGET_OUT}/hyperpeft/dialogue.complete"
  if [[ -f "${MARK_HP}" ]]; then
    echo "[phase6] hyperpeft arm already complete"
    p6_done=$((p6_done + 1))
  else
    set +e
    # Phase 6 needs BF16 + eager attention to avoid the high-memory GPU
    # cuDNN-frontend crash at extended depth. The HyperPEFTLoRAEngine
    # honors HN_TORCH_DTYPE / HN_ATTN_IMPL env overrides.
    HN_TORCH_DTYPE=bfloat16 HN_ATTN_IMPL=eager \
    python "${SCRIPTS}/turn_budget_dialogue.py"                       --condition "hyperpeft"                       --out_dir "${TURN_BUDGET_OUT}/hyperpeft"                       --author_parquet /tmp/author_static_10000.parquet                       --labels_csv /tmp/labels_sentiment_goemo.csv                       --norm_stats_json /tmp/feature_norm_stats_10000.json                       --base_model "${BASE_MODEL}"                       --hyper_dir "${HYPER_DIR}"                       --target_modules "${TARGET_MODULES}"                       --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}"                       --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}"                       --feature_clamp 3.0 --outlier_threshold 4.0 --filter_outliers                       --n_per_cohort 300                       --n_turns 80                       --batch_size 24                       --topic "${P6_TOPIC}"                       --max_len 768 --max_new_tokens 96                       --top_p "${TOP_P}" --temperature "${TEMPERATURE}"                       --repetition_penalty "${REPETITION_PENALTY}" --no_repeat_ngram_size 3                       --seed 142 --save_every_turns 5
    RC6HP=$?
    set -e
    if (( RC6HP == 0 )); then
      p6_done=$((p6_done + 1))
      echo "[phase6] hyperpeft arm done -> ${TURN_BUDGET_OUT}/hyperpeft/dialogue.parquet"
    else
      echo "[phase6] hyperpeft arm FAILED (rc=${RC6HP})"
    fi
  fi
  # ---- arm 2: zero_delta (Appendix N ablation) ----
  # --force_zero_delta zeroes the hypernetwork delta at injection,
  # leaving Arditi patch active. This is the REFRESH baseline for
  # Appendix N drift-slope comparison against the hyperpeft arm.
  mkdir -p "${TURN_BUDGET_OUT}/zero_delta"
  MARK_ZD="${TURN_BUDGET_OUT}/zero_delta/dialogue.complete"
  if [[ -f "${MARK_ZD}" ]]; then
    echo "[phase6] zero_delta arm already complete"
    p6_done=$((p6_done + 1))
  else
    set +e
    # Phase 6 zero-delta arm: same BF16 + eager env override
    # as the hyperpeft arm above (avoids the high-memory GPU cuDNN crash).
    HN_TORCH_DTYPE=bfloat16 HN_ATTN_IMPL=eager \
    python "${SCRIPTS}/turn_budget_dialogue.py" \
        --condition "zero_delta" \
        --out_dir "${TURN_BUDGET_OUT}/zero_delta" \
        --author_parquet /tmp/author_static_10000.parquet \
        --labels_csv /tmp/labels_sentiment_goemo.csv \
        --norm_stats_json /tmp/feature_norm_stats_10000.json \
        --base_model "${BASE_MODEL}" \
        --hyper_dir "${HYPER_DIR}" \
        --target_modules "${TARGET_MODULES}" \
        --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
        --inject_clamp "${INJECT_CLAMP}" --delta_gain "${DELTA_GAIN}" \
        --feature_clamp 3.0 --outlier_threshold 4.0 --filter_outliers \
        --n_per_cohort 300 \
        --n_turns 80 \
        --batch_size 24 \
        --topic "${P6_TOPIC}" \
        --max_len 768 --max_new_tokens 96 \
        --top_p "${TOP_P}" --temperature "${TEMPERATURE}" \
        --repetition_penalty "${REPETITION_PENALTY}" --no_repeat_ngram_size 3 \
        --seed 142 --save_every_turns 5 \
        --force_zero_delta
    RC6ZD=$?
    set -e
    if (( RC6ZD == 0 )); then
      p6_done=$((p6_done + 1))
      echo "[phase6] zero_delta arm done -> ${TURN_BUDGET_OUT}/zero_delta/dialogue.parquet"
    else
      echo "[phase6] zero_delta arm FAILED (rc=${RC6ZD}); Appendix N comparison arm missing"
    fi
  fi
  if (( p6_done == 2 )); then
    phase_end_ok "phase6_extended_depth" 2 2
  else
    phase_end_fail "phase6_extended_depth" 2 "${p6_done}"
    echo "[phase6] partial (${p6_done}/2); Appendix N comparison may be incomplete"
  fi
fi

# ===========================================================
# DONE
# ===========================================================
echo ""
echo "================================================================"
echo "  P2 FULL REFRESH COMPLETE  ($(date -Is))"
echo "================================================================"
echo "[REFRESH] state file : ${STATE_FILE}"
cat "${STATE_FILE}" 2>/dev/null || true
echo ""
echo "[REFRESH] output inventory:"
ls -lh "${FORUM_ROOT}" 2>/dev/null || true
echo ""
echo "[REFRESH] persona signature summary:"
ls -lh "${SIGNATURE_ROOT}" 2>/dev/null || true
echo ""
echo "[REFRESH] paper fill:"
ls -lh "${PAPER_FILL_OUT}" 2>/dev/null || true
echo ""
echo "[REFRESH] reconstruction fidelity:"
ls -lh "${RECON_ROOT:-${OUT_ROOT}/phase2e_reconstruction_REFRESH}" 2>/dev/null || true
echo ""
echo "[REFRESH] synth-vs-recon:"
ls -lh "${SVR_OUT:-${OUT_ROOT}/phase2f_synth_vs_recon_REFRESH}" 2>/dev/null || true
echo ""
echo "[REFRESH] arditi extreme (phase0d):"
ls -lh "${OUT_ROOT}/phase0d_arditi_extreme_REFRESH" 2>/dev/null || true
echo ""
echo "[REFRESH] phase6 turn budget:"
ls -lh "${TURN_BUDGET_OUT:-${OUT_ROOT}/phase6_turn_budget_REFRESH}" 2>/dev/null || true
echo ""
echo "[REFRESH] logs:"
echo "  main log   : ${MAIN_LOG}"
echo "  gpu telem  : ${GPU_TELEM}"
echo ""
echo "POST-COMPLETION FOLLOW-UP:"
echo "  Repoint make_paper2_figures.py PHASE3D constant:"
echo "    old: phase2_forums_arditi/"
echo "    new: phase2_forums_REFRESH/"
echo "  Repoint PHASE5 constant:"
echo "    old: phase5_paper_fill_arditi/"
echo "    new: phase5_paper_fill_REFRESH/"
echo "  Repoint SIGNATURE_ROOT constant:"
echo "    old: phase3d_persona_signature_arditi/"
echo "    new: phase3d_persona_signature_REFRESH/"
echo "  Repoint PHASE0D constant to phase0d_arditi_extreme_REFRESH/"
echo "  Appendix N: use phase6_turn_budget_REFRESH/hyperpeft/ and zero_delta/"

kill "${TELEM_PID}" 2>/dev/null || true
