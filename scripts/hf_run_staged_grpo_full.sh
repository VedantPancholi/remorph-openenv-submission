#!/usr/bin/env bash
# Full hackathon path: GRPO datasets 500 → 1500 → 3000, staged TRL runs, master charts, Hub upload.
# Run on HF Jobs / Colab / any GPU box with: bash scripts/hf_run_staged_grpo_full.sh
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/VedantPancholi/remorph-openenv-submission.git}"
REPO_DIR="${REPO_DIR:-remorph-openenv-submission}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HF_DATASET_REPO="${HF_DATASET_REPO:-Jenish31/remorph-training-artifacts}"
UPLOAD_PATH_IN_REPO="${UPLOAD_PATH_IN_REPO:-full_staged_grpo_run}"
MAX_STEPS_V1="${MAX_STEPS_V1:-100}"
MAX_STEPS_V2="${MAX_STEPS_V2:-200}"
MAX_STEPS_V3="${MAX_STEPS_V3:-300}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
export HF_DATASET_REPO UPLOAD_PATH_IN_REPO

# If already inside the repo (e.g. Colab after manual clone), stay here — do not nest-clone.
if [[ -f scripts/hf_run_staged_grpo_full.sh ]]; then
  REPO_DIR="."
fi
if [[ "$REPO_DIR" != "." && ! -d "$REPO_DIR" ]]; then
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-training.txt -c pip_constraints.txt
# Colab/HF: dependency solver sometimes leaves transformers 5.0.x; force before TRL runs.
pip install --upgrade --force-reinstall --no-cache-dir "transformers>=5.2.0,<6"
pip install "jmespath>=1.0.0,<2"
python3 -c "import transformers as t; v=t.__version__.split('+')[0]; p=tuple(int(x) for x in v.split('.')[:3]); assert p>=(5,2,0), f'transformers {t.__version__} < 5.2 — fix pip install'"
python3 -c "import jmespath; print('jmespath ok')"

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

python scripts/generate_grpo_dataset.py \
  --output-dir artifacts/submission/grpo_dataset_3000 \
  --train-seed-start 7000 --train-seed-count 300 \
  --eval-seed-start 9000 --eval-seed-count 30 \
  --execution-mode simulated

python scripts/train_trl_grpo.py --train \
  --output-dir artifacts/submission/trl_grpo_run_v1 \
  --train-path artifacts/submission/grpo_dataset_500/train_prompts.jsonl \
  --eval-path artifacts/submission/grpo_dataset_500/eval_prompts.jsonl \
  --model "$MODEL" \
  --max-steps "$MAX_STEPS_V1" --learning-rate 1e-6 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 --num-generations 4 \
  --max-prompt-length 2048 --max-completion-length 768 --eval-steps 25

python scripts/plot_trl_grpo.py \
  --metrics-path artifacts/submission/trl_grpo_run_v1/trl_grpo_metrics.json \
  --output-dir artifacts/submission/plots/stage1

python scripts/train_trl_grpo.py --train \
  --output-dir artifacts/submission/trl_grpo_run_v2 \
  --train-path artifacts/submission/grpo_dataset_1500/train_prompts.jsonl \
  --eval-path artifacts/submission/grpo_dataset_1500/eval_prompts.jsonl \
  --model "$MODEL" \
  --max-steps "$MAX_STEPS_V2" --learning-rate 1e-6 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 --num-generations 4 \
  --max-prompt-length 2048 --max-completion-length 768 --eval-steps 25

python scripts/plot_trl_grpo.py \
  --metrics-path artifacts/submission/trl_grpo_run_v2/trl_grpo_metrics.json \
  --output-dir artifacts/submission/plots/stage2

python scripts/train_trl_grpo.py --train \
  --output-dir artifacts/submission/trl_grpo_run_v3 \
  --train-path artifacts/submission/grpo_dataset_3000/train_prompts.jsonl \
  --eval-path artifacts/submission/grpo_dataset_3000/eval_prompts.jsonl \
  --model "$MODEL" \
  --max-steps "$MAX_STEPS_V3" --learning-rate 1e-6 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 --num-generations 4 \
  --max-prompt-length 2048 --max-completion-length 768 --eval-steps 25

python scripts/plot_trl_grpo.py \
  --metrics-path artifacts/submission/trl_grpo_run_v3/trl_grpo_metrics.json \
  --output-dir artifacts/submission/plots/stage3

python scripts/plot_trl_master_compare.py \
  --summary-1 artifacts/submission/trl_grpo_run_v1/trl_grpo_training_summary.json \
  --summary-2 artifacts/submission/trl_grpo_run_v2/trl_grpo_training_summary.json \
  --summary-3 artifacts/submission/trl_grpo_run_v3/trl_grpo_training_summary.json \
  --label-1 "stage1_500" --label-2 "stage2_1500" --label-3 "stage3_3000" \
  --output-dir artifacts/submission/plots/master

if [[ "$SKIP_UPLOAD" != "1" ]]; then
  python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ.get("HF_DATASET_REPO", "Jenish31/remorph-training-artifacts")
path_in_repo = os.environ.get("UPLOAD_PATH_IN_REPO", "full_staged_grpo_run")
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

echo "HF full staged GRPO pipeline OK"
