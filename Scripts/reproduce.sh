#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${PY:-python}"
PHASES="${PHASES:-0 1 2 3 4 5 6}"
SEEDS="${SEEDS:-1 2 3}"

FORCE_FLAG=()
SKIP_FLAG=(--skip_if_current)
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_FLAG=(--force)
  SKIP_FLAG=()
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  export SMOKE=1
  export ARTIFACT_ROOT="${ARTIFACT_ROOT:-Smoke}"
  export DATA="${DATA:-Smoke/Data/dataset/smoke.parquet}"
else
  export DATA="${DATA:-Data/dataset/PJ_dataset_qcd_flat_15to7000.parquet}"
fi

# Config overrides; neither num_workers nor wandb_* changes a run's identity hash.
OVERRIDES=()
if [[ -n "${NUM_WORKERS:-}" ]]; then
  OVERRIDES+=("num_workers=${NUM_WORKERS}")
fi
EVAL_OVERRIDE=()
if (( ${#OVERRIDES[@]} )); then
  EVAL_OVERRIDE=(--override "${OVERRIDES[@]}")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  OVERRIDES+=("wandb_project=${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    OVERRIDES+=("wandb_entity=${WANDB_ENTITY}")
  fi
fi
TRAIN_OVERRIDE=()
if (( ${#OVERRIDES[@]} )); then
  TRAIN_OVERRIDE=(--override "${OVERRIDES[@]}")
fi

have() { [[ " ${PHASES} " == *" $1 "* ]]; }
say()  { printf '\n[reproduce] %s\n' "$*"; }
run()  { printf '[reproduce] $ %s\n' "$*"; "$@"; }

if have 0; then
  say "phase 0: preflight"
  if [[ "${SMOKE:-0}" == "1" && ! -f "$DATA" ]]; then
    run "$PY" Scripts/make_smoke_data.py --out "$DATA"
  fi
  if [[ ! -f "$DATA" ]]; then
    echo "[reproduce] data not found: $DATA" >&2
    exit 1
  fi
  CUDA_FLAG=()
  for p in 1 2 3; do
    if have "$p"; then CUDA_FLAG=(--require_cuda); fi
  done
  run "$PY" Scripts/record_environment.py ${CUDA_FLAG[@]+"${CUDA_FLAG[@]}"}
fi

if have 1; then
  say "phase 1: train"
  for cfg in $("$PY" Experiments/paper_runs.py list); do
    run "$PY" Experiments/run_experiment.py "$cfg" --seeds $SEEDS \
      ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} ${TRAIN_OVERRIDE[@]+"${TRAIN_OVERRIDE[@]}"}
  done
fi

if have 2; then
  say "phase 2: common-event-set evaluation"
  for cfg in $("$PY" Experiments/paper_runs.py list --group parton2reco); do
    run "$PY" Experiments/run_experiment.py "$cfg" --mode eval_common --seeds $SEEDS \
      ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} ${EVAL_OVERRIDE[@]+"${EVAL_OVERRIDE[@]}"}
  done
fi

if have 3; then
  say "phase 3: dumps"
  for cfg in $("$PY" Experiments/paper_runs.py list --group parton2reco); do
    run "$PY" Experiments/run_experiment.py "$cfg" --mode dump --seeds $SEEDS \
      ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} ${EVAL_OVERRIDE[@]+"${EVAL_OVERRIDE[@]}"}
  done
fi

if have 4; then
  say "phase 4: analysis"
  for s in $SEEDS; do
    for script in compare_models dummy_sampler_baseline training_curve stage_ceilings cone_narrowing_control; do
      run "$PY" "Experiments/${script}.py" --seed "$s" ${SKIP_FLAG[@]+"${SKIP_FLAG[@]}"}
    done
  done
fi

if have 5; then
  say "phase 5: numbers"
  run "$PY" Paper/build_numbers.py
fi

if have 6; then
  say "phase 6: figures"
  run "$PY" Paper/build_figures.py
fi

say "done. Next: update the tex from ${ARTIFACT_ROOT:+${ARTIFACT_ROOT}/}Paper/numbers_report.md, then run"
say "  $PY Paper/build_numbers.py --check-tex && (cd Paper && latexmk -pdf neurips_2026_v2.tex)"
