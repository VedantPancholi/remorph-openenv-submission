#!/usr/bin/env bash
# Two-stage GRPO only: 500-row dataset → v1, 1500-row dataset → v2 (no 3000 / no stage 3).
# Defaults tuned for a fast wall-clock run: MAX_STEPS_V1=4, MAX_STEPS_V2=8 (override via env).
#
# Always uploads the full artifacts/submission tree to HF_DATASET_REPO / UPLOAD_PATH_IN_REPO
# so models, plots, datasets, and JSON land on the Hub (public dataset = free browser access).
#
# Optional HF Storage Bucket backup (same pattern as hf_run_track_a_fast_persist.sh):
#   PERSIST_ROOT=/mnt/trl_backup
#   hf jobs run ... -v hf://buckets/<user>/<bucket>:/mnt/trl_backup
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/VedantPancholi/remorph-openenv-submission.git}"
REPO_DIR="${REPO_DIR:-remorph-openenv-submission}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HF_DATASET_REPO="${HF_DATASET_REPO:-Jenish31/remorph-training-artifacts}"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-staged2_500_1500_${RUN_TS}}"
UPLOAD_PATH_IN_REPO="${UPLOAD_PATH_IN_REPO:-${RUN_ID}}"

PERSIST_ROOT="${PERSIST_ROOT:-}"

MAX_STEPS_V1="${MAX_STEPS_V1:-4}"
MAX_STEPS_V2="${MAX_STEPS_V2:-8}"
# Short runs: keep eval frequent (must be <= max-steps for meaningful mid-run eval).
EVAL_STEPS="${EVAL_STEPS:-2}"

export RUN_ID HF_DATASET_REPO UPLOAD_PATH_IN_REPO MAX_STEPS_V1 MAX_STEPS_V2
export SKIP_UPLOAD=0

sync_persist() {
  local label="$1"
  if [[ -z "$PERSIST_ROOT" ]]; then
    return 0
  fi
  if [[ ! -d "$PERSIST_ROOT" ]]; then
    echo "ERROR: PERSIST_ROOT=$PERSIST_ROOT is not a directory." >&2
    exit 1
  fi
  local dest="$PERSIST_ROOT/$RUN_ID"
  mkdir -p "$dest"
  cp -a artifacts/submission/. "$dest/"
  echo "{\"persist_sync\":\"${label}\",\"run_id\":\"${RUN_ID}\",\"local_mirror\":\"${dest}\"}" > "$dest/_persist_sync_${label}.json"
  echo "Persist sync (${label}) -> ${dest}"
}

if [[ -f scripts/hf_run_staged_grpo_two_stage_persist.sh ]]; then
  REPO_DIR="."
fi
if [[ "$REPO_DIR" != "." && ! -d "$REPO_DIR" ]]; then
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "RUN_ID=${RUN_ID}"
echo "UPLOAD_PATH_IN_REPO=${UPLOAD_PATH_IN_REPO}"
echo "HF_DATASET_REPO=${HF_DATASET_REPO}"
echo "MAX_STEPS_V1=${MAX_STEPS_V1} MAX_STEPS_V2=${MAX_STEPS_V2} EVAL_STEPS=${EVAL_STEPS}"
echo "PERSIST_ROOT=${PERSIST_ROOT:-<none>}"

pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-training.txt -c pip_constraints.txt
pip install --upgrade --force-reinstall --no-cache-dir "transformers>=5.2.0,<6" -c pip_constraints.txt
pip install "jmespath>=1.0.0,<2"
python3 -c "import transformers as t; v=t.__version__.split('+')[0]; p=tuple(int(x) for x in v.split('.')[:3]); assert p>=(5,2,0), f'transformers {t.__version__} < 5.2'"
python3 -c "import jmespath; print('jmespath ok')"
python scripts/smoke_grpo_trainer_deps.py --model "$MODEL"

python scripts/generate_grpo_dataset.py \
  --output-dir artifacts/submission/grpo_dataset_500 \
  --train-seed-start 1000 --train-seed-count 50 \
  --eval-seed-start 2000 --eval-seed-count 10 \
  --execution-mode simulated

python scripts/generate_grpo_dataset.py \
  --output-dir artifacts/submission/grpo_dataset_1500 \
  --train-seed-start 3000 --train-seed-count 150 \
  --eval-seed-start 5000 --eval-seed-count 20 \
  --execution-mode simulated

python scripts/train_trl_grpo.py --train \
  --output-dir artifacts/submission/trl_grpo_run_v1 \
  --train-path artifacts/submission/grpo_dataset_500/train_prompts.jsonl \
  --eval-path artifacts/submission/grpo_dataset_500/eval_prompts.jsonl \
  --model "$MODEL" \
  --max-steps "$MAX_STEPS_V1" \
  --learning-rate 1e-6 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-generations 4 \
  --max-prompt-length 2048 \
  --max-completion-length 768 \
  --eval-steps "$EVAL_STEPS"

sync_persist "after_stage1_train"

python scripts/plot_trl_grpo.py \
  --metrics-path artifacts/submission/trl_grpo_run_v1/trl_grpo_metrics.json \
  --output-dir artifacts/submission/plots/stage1

python scripts/train_trl_grpo.py --train \
  --output-dir artifacts/submission/trl_grpo_run_v2 \
  --train-path artifacts/submission/grpo_dataset_1500/train_prompts.jsonl \
  --eval-path artifacts/submission/grpo_dataset_1500/eval_prompts.jsonl \
  --model "$MODEL" \
  --max-steps "$MAX_STEPS_V2" \
  --learning-rate 1e-6 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-generations 4 \
  --max-prompt-length 2048 \
  --max-completion-length 768 \
  --eval-steps "$EVAL_STEPS"

sync_persist "after_stage2_train"

python scripts/plot_trl_grpo.py \
  --metrics-path artifacts/submission/trl_grpo_run_v2/trl_grpo_metrics.json \
  --output-dir artifacts/submission/plots/stage2

python scripts/plot_trl_master_compare.py \
  --summary-1 artifacts/submission/trl_grpo_run_v1/trl_grpo_training_summary.json \
  --summary-2 artifacts/submission/trl_grpo_run_v2/trl_grpo_training_summary.json \
  --label-1 "stage1_500" \
  --label-2 "stage2_1500" \
  --output-dir artifacts/submission/plots/master

python scripts/space_submission_checks.py
python scripts/sqlite_artifact_bridge.py
python scripts/export_frontend_payload.py
python scripts/promote_long_run_outputs.py || true

sync_persist "before_hub_upload"

python - <<PY
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

repo_id = os.environ["HF_DATASET_REPO"]
path_in_repo = os.environ["UPLOAD_PATH_IN_REPO"]
run_id = os.environ.get("RUN_ID", "")
manifest = {
    "run_id": run_id,
    "pipeline": "staged_grpo_two_stage_500_1500",
    "max_steps_v1": os.environ.get("MAX_STEPS_V1", ""),
    "max_steps_v2": os.environ.get("MAX_STEPS_V2", ""),
    "uploaded_repo": repo_id,
    "path_in_repo": path_in_repo,
    "dataset_url": f"https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}",
    "best_model_hint": "artifacts/submission/trl_grpo_run_v2/model (check best_run_promotion.json)",
}
Path("artifacts/submission/hub_upload_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
api.upload_folder(
    folder_path="artifacts/submission",
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo=path_in_repo,
)
print(json.dumps({"status": "uploaded", **manifest}, indent=2))
PY

sync_persist "after_hub_upload"

echo "STAGED_TWO_STAGE_PERSIST_OK run_id=${RUN_ID}"
