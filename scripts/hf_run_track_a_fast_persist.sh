#!/usr/bin/env bash
# Track A — fast two-stage GRPO (small data, low steps), same family as the "Sachin fast" / race jobs.
# - Always syncs artifacts to an optional HF Storage Bucket mount (backup during the run).
# - Always uploads artifacts/submission to HF_DATASET_REPO (no SKIP_UPLOAD path).
#
# HF Jobs: mount a writable bucket, e.g.
#   -v hf://buckets/<your_user>/<your_bucket>:/mnt/trl_backup
# and set:
#   PERSIST_ROOT=/mnt/trl_backup
#
# Hub Dataset copy (canonical public handoff) lives under:
#   https://huggingface.co/datasets/$HF_DATASET_REPO/tree/main/$UPLOAD_PATH_IN_REPO
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/VedantPancholi/remorph-openenv-submission.git}"
REPO_DIR="${REPO_DIR:-remorph-openenv-submission}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HF_DATASET_REPO="${HF_DATASET_REPO:-Jenish31/remorph-training-artifacts}"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-tracka_fast_${RUN_TS}}"
UPLOAD_PATH_IN_REPO="${UPLOAD_PATH_IN_REPO:-${RUN_ID}}"

# Writable mount from `hf jobs run -v hf://buckets/org/name:/mnt/trl_backup`
PERSIST_ROOT="${PERSIST_ROOT:-}"

MAX_STEPS_V1="${MAX_STEPS_V1:-8}"
MAX_STEPS_V2="${MAX_STEPS_V2:-12}"

export RUN_ID HF_DATASET_REPO UPLOAD_PATH_IN_REPO
# This script always uploads to the Hub dataset repo (never skip).
export SKIP_UPLOAD=0

sync_persist() {
  local label="$1"
  if [[ -z "$PERSIST_ROOT" ]]; then
    return 0
  fi
  if [[ ! -d "$PERSIST_ROOT" ]]; then
    echo "ERROR: PERSIST_ROOT is set to $PERSIST_ROOT but that path is not a directory." >&2
    echo "Mount a bucket with: hf jobs run -v hf://buckets/<user>/<bucket>:$PERSIST_ROOT ..." >&2
    exit 1
  fi
  local dest="$PERSIST_ROOT/$RUN_ID"
  mkdir -p "$dest"
  cp -a artifacts/submission/. "$dest/"
  echo "{\"persist_sync\":\"${label}\",\"run_id\":\"${RUN_ID}\",\"local_mirror\":\"${dest}\"}" > "$dest/_persist_sync_${label}.json"
  echo "Persist sync (${label}) -> ${dest}"
}

if [[ -f scripts/hf_run_track_a_fast_persist.sh ]]; then
  REPO_DIR="."
fi
if [[ "$REPO_DIR" != "." && ! -d "$REPO_DIR" ]]; then
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "RUN_ID=${RUN_ID}"
echo "UPLOAD_PATH_IN_REPO=${UPLOAD_PATH_IN_REPO}"
echo "HF_DATASET_REPO=${HF_DATASET_REPO}"
echo "MAX_STEPS_V1=${MAX_STEPS_V1} MAX_STEPS_V2=${MAX_STEPS_V2}"
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
  --output-dir artifacts/submission/grpo_dataset_tracka_120 \
  --train-seed-start 21000 --train-seed-count 12 \
  --eval-seed-start 22000 --eval-seed-count 4 \
  --execution-mode simulated

python scripts/generate_grpo_dataset.py \
  --output-dir artifacts/submission/grpo_dataset_tracka_240 \
  --train-seed-start 23000 --train-seed-count 24 \
  --eval-seed-start 24000 --eval-seed-count 6 \
  --execution-mode simulated

python scripts/train_trl_grpo.py --train \
  --output-dir artifacts/submission/trl_grpo_run_tracka_v1 \
  --train-path artifacts/submission/grpo_dataset_tracka_120/train_prompts.jsonl \
  --eval-path artifacts/submission/grpo_dataset_tracka_120/eval_prompts.jsonl \
  --model "$MODEL" \
  --max-steps "$MAX_STEPS_V1" \
  --learning-rate 1e-6 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-generations 4 \
  --max-prompt-length 1536 \
  --max-completion-length 512 \
  --eval-steps 8

sync_persist "after_stage1_train"

python scripts/plot_trl_grpo.py \
  --metrics-path artifacts/submission/trl_grpo_run_tracka_v1/trl_grpo_metrics.json \
  --output-dir artifacts/submission/plots/tracka_stage1

python scripts/train_trl_grpo.py --train \
  --output-dir artifacts/submission/trl_grpo_run_tracka_v2 \
  --train-path artifacts/submission/grpo_dataset_tracka_240/train_prompts.jsonl \
  --eval-path artifacts/submission/grpo_dataset_tracka_240/eval_prompts.jsonl \
  --model "$MODEL" \
  --max-steps "$MAX_STEPS_V2" \
  --learning-rate 1e-6 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-generations 4 \
  --max-prompt-length 1536 \
  --max-completion-length 512 \
  --eval-steps 8

sync_persist "after_stage2_train"

python scripts/plot_trl_grpo.py \
  --metrics-path artifacts/submission/trl_grpo_run_tracka_v2/trl_grpo_metrics.json \
  --output-dir artifacts/submission/plots/tracka_stage2

python scripts/plot_trl_master_compare.py \
  --summary-1 artifacts/submission/trl_grpo_run_tracka_v1/trl_grpo_training_summary.json \
  --summary-2 artifacts/submission/trl_grpo_run_tracka_v2/trl_grpo_training_summary.json \
  --label-1 tracka_stage1 \
  --label-2 tracka_stage2 \
  --output-dir artifacts/submission/plots/tracka_master

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
    "uploaded_repo": repo_id,
    "path_in_repo": path_in_repo,
    "dataset_url": f"https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}",
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

echo "TRACK_A_FAST_PERSIST_OK run_id=${RUN_ID}"
