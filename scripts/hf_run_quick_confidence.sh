#!/usr/bin/env bash
# Fast confidence profile: complete a small end-to-end GRPO run quickly.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/VedantPancholi/remorph-openenv-submission.git}"
REPO_DIR="${REPO_DIR:-remorph-openenv-submission}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HF_DATASET_REPO="${HF_DATASET_REPO:-Jenish31/remorph-training-artifacts}"
UPLOAD_PATH_IN_REPO="${UPLOAD_PATH_IN_REPO:-quick_confidence_run}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
MAX_STEPS_V1="${MAX_STEPS_V1:-30}"
MAX_STEPS_V2="${MAX_STEPS_V2:-45}"

export HF_DATASET_REPO UPLOAD_PATH_IN_REPO

if [[ -f scripts/hf_run_quick_confidence.sh ]]; then
  REPO_DIR="."
fi
if [[ "$REPO_DIR" != "." && ! -d "$REPO_DIR" ]]; then
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-training.txt -c pip_constraints.txt
pip install --upgrade --force-reinstall --no-cache-dir "transformers>=5.2.0,<6"
pip install "jmespath>=1.0.0,<2"
python3 -c "import transformers as t; v=t.__version__.split('+')[0]; p=tuple(int(x) for x in v.split('.')[:3]); assert p>=(5,2,0), f'transformers {t.__version__} < 5.2'"
python3 -c "import jmespath; print('jmespath ok')"
python scripts/smoke_grpo_trainer_deps.py --model "$MODEL"

python scripts/generate_grpo_dataset.py \
  --output-dir artifacts/submission/grpo_dataset_quick_500 \
  --train-seed-start 11000 --train-seed-count 25 \
  --eval-seed-start 12000 --eval-seed-count 8 \
  --execution-mode simulated

python scripts/generate_grpo_dataset.py \
  --output-dir artifacts/submission/grpo_dataset_quick_1500 \
  --train-seed-start 13000 --train-seed-count 60 \
  --eval-seed-start 14000 --eval-seed-count 12 \
  --execution-mode simulated

python scripts/train_trl_grpo.py --train \
  --output-dir artifacts/submission/trl_grpo_run_quick_v1 \
  --train-path artifacts/submission/grpo_dataset_quick_500/train_prompts.jsonl \
  --eval-path artifacts/submission/grpo_dataset_quick_500/eval_prompts.jsonl \
  --model "$MODEL" \
  --max-steps "$MAX_STEPS_V1" --learning-rate 1e-6 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 2 --num-generations 4 \
  --max-prompt-length 1536 --max-completion-length 512 --eval-steps 15

python scripts/plot_trl_grpo.py \
  --metrics-path artifacts/submission/trl_grpo_run_quick_v1/trl_grpo_metrics.json \
  --output-dir artifacts/submission/plots/quick_stage1

python scripts/train_trl_grpo.py --train \
  --output-dir artifacts/submission/trl_grpo_run_quick_v2 \
  --train-path artifacts/submission/grpo_dataset_quick_1500/train_prompts.jsonl \
  --eval-path artifacts/submission/grpo_dataset_quick_1500/eval_prompts.jsonl \
  --model "$MODEL" \
  --max-steps "$MAX_STEPS_V2" --learning-rate 1e-6 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 2 --num-generations 4 \
  --max-prompt-length 1536 --max-completion-length 512 --eval-steps 15

python scripts/plot_trl_grpo.py \
  --metrics-path artifacts/submission/trl_grpo_run_quick_v2/trl_grpo_metrics.json \
  --output-dir artifacts/submission/plots/quick_stage2

python scripts/plot_trl_master_compare.py \
  --summary-1 artifacts/submission/trl_grpo_run_quick_v1/trl_grpo_training_summary.json \
  --summary-2 artifacts/submission/trl_grpo_run_quick_v2/trl_grpo_training_summary.json \
  --label-1 "quick_stage1" --label-2 "quick_stage2" \
  --output-dir artifacts/submission/plots/quick_master

if [[ "$SKIP_UPLOAD" != "1" ]]; then
  python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ.get("HF_DATASET_REPO", "Jenish31/remorph-training-artifacts")
path_in_repo = os.environ.get("UPLOAD_PATH_IN_REPO", "quick_confidence_run")
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
api.upload_folder(
    folder_path="artifacts/submission",
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo=path_in_repo,
)
print({"uploaded_repo": repo_id, "path": path_in_repo})
PY
fi

echo "HF quick confidence pipeline OK"
