#!/usr/bin/env bash
# Run on Hugging Face Jobs: short entrypoint to avoid "File name too long" on giant one-liners.
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/VedantPancholi/remorph-openenv-submission.git}"
REPO_DIR="${REPO_DIR:-remorph-openenv-submission}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

if [[ -f scripts/hf_run_stage1_stage2.sh ]]; then
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
  --max-steps 100 --learning-rate 1e-6 \
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
  --max-steps 200 --learning-rate 1e-6 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 --num-generations 4 \
  --max-prompt-length 2048 --max-completion-length 768 --eval-steps 25

python scripts/plot_trl_grpo.py \
  --metrics-path artifacts/submission/trl_grpo_run_v2/trl_grpo_metrics.json \
  --output-dir artifacts/submission/plots/stage2

python scripts/plot_trl_master_compare.py

python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
repo_id = "Jenish31/remorph-training-artifacts"
api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
api.upload_folder(
    folder_path="artifacts/submission",
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="stage1_stage2_run",
)
print({"uploaded_repo": repo_id, "path": "stage1_stage2_run"})
PY

echo "HF pipeline OK"
