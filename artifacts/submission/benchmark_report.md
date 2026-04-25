# ReMorph Benchmark Report

- Trainer: `supervised_structured_policy`
- Seed: `42`
- Train scenarios: `10`
- Eval scenarios: `6`
- Training examples: `22`

## Eval Metrics

| Policy | Success Rate | Avg Raw Reward | Avg Capped Episode Return | Avg Steps |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0000 | -19.6667 | 0.9956 | 4.5000 |
| replay | 0.0000 | -19.6667 | 0.9956 | 4.5000 |
| supervised | 1.0000 | 19.5000 | 1.0000 | 2.0000 |
| oracle | 1.0000 | 19.5000 | 1.0000 | 2.0000 |

## Model Config

```json
{
  "learner": "supervised_structured_policy",
  "route_strategy": "max_confidence_candidate",
  "payload_strategy": "expected_request_body",
  "auth_strategy": "required_headers",
  "abstain_strategy": "partition_gate",
  "average_route_confidence": 0.885,
  "payload_hint_coverage": 1.0,
  "observed_tenant_aliases": [
    "east",
    "north",
    "west"
  ],
  "counts": {
    "route_examples": 8,
    "payload_examples": 7,
    "auth_examples": 6,
    "abstain_examples": 1,
    "training_example_count": 22
  }
}
```

## Phase 4 Demo Scope

The next phase is the Hugging Face Space.

The interactive demo should let someone:

- choose a scenario
- see the observation/state
- run baseline vs supervised vs oracle
- inspect chosen action JSON
- inspect reward and reward breakdown
- step through the episode
- compare simulated vs live-local where relevant
- view benchmark summary metrics and plots

## Manual Verification Before Phase 4

1. Confirm config parsing with `python -c "import yaml; yaml.safe_load(open('openenv.yaml')); print('openenv.yaml OK')"`
2. Confirm the frozen validator slice with `python scripts/smoke_test_openenv.py`
3. Confirm deterministic simulated rollouts with `python scripts/inference.py`
4. Confirm split artifacts with `python scripts/generate_splits.py --seed 42`
5. Confirm live-local service-backed rollouts with `python scripts/live_local_demo.py`
6. Confirm manifest-driven supervised training with `python scripts/train_submission.py --output-dir artifacts/submission/training_run --seed 42 --train-manifest artifacts/submission/splits/train_manifest.json --eval-manifest artifacts/submission/splits/eval_manifest.json`
7. Confirm plots regenerate with `python scripts/generate_submission_plots.py`
8. Confirm OpenEnv validator compatibility with `openenv validate`
