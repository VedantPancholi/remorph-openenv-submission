# OpenEnv Validation Notes

Use these commands from the repository root:

```bash
pip install -r requirements.txt
python -c "import yaml; yaml.safe_load(open('openenv.yaml')); print('openenv.yaml OK')"
python scripts/smoke_test_openenv.py
python scripts/inference.py
python scripts/generate_splits.py --seed 42
python scripts/live_local_demo.py
python scripts/train_submission.py --output-dir artifacts/submission/training_run --train-manifest artifacts/submission/splits/train_manifest.json --eval-manifest artifacts/submission/splits/eval_manifest.json
python scripts/evaluate_submission.py --policy baseline --split eval --train-manifest artifacts/submission/splits/train_manifest.json --eval-manifest artifacts/submission/splits/eval_manifest.json
python scripts/evaluate_submission.py --policy supervised --split eval --train-manifest artifacts/submission/splits/train_manifest.json --eval-manifest artifacts/submission/splits/eval_manifest.json
python scripts/evaluate_submission.py --policy adaptive_reference --split eval --train-manifest artifacts/submission/splits/train_manifest.json --eval-manifest artifacts/submission/splits/eval_manifest.json
python scripts/generate_submission_plots.py
openenv validate
```

If `openenv validate` fails locally, capture the exact validator output and fix only the reported structure issue rather than regenerating the repo with `openenv init`.

Phase 1 benchmark hardening adds:

- `20` total scenarios
- `4` frozen validator scenarios
- `16` benchmark scenarios
- `10` train split scenarios
- `6` eval split scenarios

The split manifests live in `artifacts/submission/splits/`.

Phase 2 adds a live-local FastAPI-backed execution mode:

- `8` live-local scenarios
- `3` services: CRM, billing, identity
- runnable demo: `python scripts/live_local_demo.py`

Phase 3 adds:

- split-aware supervised training on `train` and held-out `eval`
- telemetry output at `artifacts/submission/telemetry/rollouts.jsonl`
- benchmark reports at `artifacts/submission/benchmark_report.md` and `.json`
- manifest-driven training flags: `--train-manifest` and `--eval-manifest`
- TRL-ready dataset export at `artifacts/submission/training_run/trl_dataset.jsonl`

Next phase target:

- build the Hugging Face Space on top of the now-stable benchmark core
- expose scenario selection, state view, policy comparison, action JSON, reward breakdown, step playback, simulated vs live-local comparison, and benchmark plots

Prompt-driven notebook flow:

- notebook path: `notebooks/remorph_openenv_colab.ipynb`
- includes validation, training, baseline/supervised/adaptive evaluation, plot generation, and report inspection

Manual verification flow before starting the Space:

1. `python -c "import yaml; yaml.safe_load(open('openenv.yaml')); print('openenv.yaml OK')"`
2. `python scripts/smoke_test_openenv.py`
3. `python scripts/inference.py`
4. `python scripts/generate_splits.py --seed 42`
5. `python scripts/live_local_demo.py`
6. `python scripts/train_submission.py --output-dir artifacts/submission/training_run --seed 42 --train-manifest artifacts/submission/splits/train_manifest.json --eval-manifest artifacts/submission/splits/eval_manifest.json`
7. `python scripts/generate_submission_plots.py`
8. `openenv validate`
