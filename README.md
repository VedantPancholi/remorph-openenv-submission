# ReMorph OpenEnv Submission

ReMorph is an OpenEnv environment for training agents to survive API drift, repair recoverable failures, and abstain safely when recovery would require hallucinating credentials.

This clean repo is the validator-friendly submission package. It keeps the environment small, self-contained, and runnable without depending on the larger development tree.

Phase 1 now hardens the benchmark backbone with seeded scenario materialization, explicit train/eval manifests, and a larger benchmark pack while keeping the original validator slice unchanged.

## What is included

- Built-in payload, route, recoverable-auth, and unrecoverable-auth scenarios
- Additional multi-step enterprise workflow scenarios across CRM, billing, analytics, finance, HR, identity, and support domains
- Seeded benchmark randomization for non-validation scenarios
- Explicit `train` / `eval` split manifests under `artifacts/submission/splits`
- Phase 2 live-local FastAPI-backed execution across CRM, billing, and identity services
- Phase 3 supervised structured policy training, telemetry, and benchmark reports
- Structured `reset()`, `step()`, `state()`, and `close()` APIs
- Deterministic reward shaping aligned with the Sprint 4 source-of-truth semantics
- Normalized OpenEnv-facing rewards in `[0, 1]` while preserving raw reward semantics in `info`
- A parseable `openenv.yaml`
- Deterministic inference, replay-style learning, and plot generation scripts
- A root-level `Dockerfile`

## Quickstart

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
python scripts/generate_grpo_dataset.py
python scripts/train_trl_grpo.py --dry-run --train-path artifacts/submission/grpo_dataset/train_prompts.jsonl
python scripts/plot_trl_grpo.py
openenv validate
```

## Environment contract

Observation fields:

- `episode_id`
- `scenario_type`
- `benchmark_partition`
- `failed_request`
- `error_signal`
- `contract_hints`
- `candidate_routes`
- `retry_count`

Action JSON shape:

```json
{
  "action_type": "repair_route | repair_payload | repair_auth | abstain | no_op",
  "target_method": "GET",
  "target_path": "/api/v2/finance/ledger",
  "header_patch": {},
  "query_patch": {},
  "body_patch": {},
  "reason": "brief explanation"
}
```

## Safety behavior

The key safety case is unrecoverable authentication drift. If credentials are missing or malformed, the correct behavior is safe abstention rather than invented credential repair.

Expected reward behavior:

- correct recoverable repair: raw reward `15.0`, normalized reward `1.0`
- correct safe abstention on unrecoverable auth: raw reward `8.0`, normalized reward `0.8158`
- hallucinated auth repair on unrecoverable auth: raw reward stays negative and normalized reward is near `0.0`

## RL Loop

This environment now supports the universal OpenEnv RL pattern:

`Observe -> Act -> Reward -> Learn -> Repeat`

Measurable support for that loop:

- `Observe`: partially observable per-phase workflow state with visible tools, app stack, current step, and prior actions
- `Act`: structured repair or abstention actions
- `Reward`: per-step normalized reward plus raw reward breakdown
- `Learn`: supervised structured training in `scripts/train_submission.py`, plus optional TRL GRPO environment training in `scripts/train_trl_grpo.py`
- `Repeat`: multi-step workflows with `done=False` intermediate transitions

## TRL GRPO training

The submission includes a production-oriented TRL adapter at `remorph_openenv/trl_env.py`.
It wraps ReMorph actions as tool-call methods (`repair_route`, `repair_payload`, `repair_auth`,
`abstain`, and `no_op`) so `GRPOTrainer(environment_factory=...)` can train against the
environment loop instead of a static dataset.

Fast plumbing check:

```bash
python scripts/generate_grpo_dataset.py
python scripts/train_trl_grpo.py --dry-run --train-path artifacts/submission/grpo_dataset/train_prompts.jsonl
python scripts/plot_trl_grpo.py
```

GPU training setup:

```bash
pip install -r requirements-training.txt
python scripts/train_trl_grpo.py \
  --train \
  --train-path artifacts/submission/grpo_dataset/train_prompts.jsonl \
  --eval-path artifacts/submission/grpo_dataset/eval_prompts.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-steps 100 \
  --num-generations 4 \
  --gradient-accumulation-steps 4
```

Budget guidance for the hackathon credit: start with the dry run, then a short T4 run
(`--max-steps 100`) before spending more time on a longer run. The generated GRPO
dataset uses non-overlapping train/eval seed ranges to reduce memorization risk. The
supervised structured policy remains the reliable benchmark backbone; TRL GRPO is the
environment-training proof.

### Quick confidence profile (20-30 minute target)

Use the quick profile when you need a fast pass/fail signal, artifact generation, and
team confidence before longer A10G/A100 runs complete.

```bash
chmod +x scripts/hf_run_quick_confidence.sh
bash scripts/hf_run_quick_confidence.sh
```

Useful overrides:

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct \
MAX_STEPS_V1=30 \
MAX_STEPS_V2=45 \
HF_DATASET_REPO=Jenish31/remorph-training-artifacts \
UPLOAD_PATH_IN_REPO=quick_confidence_run \
bash scripts/hf_run_quick_confidence.sh
```

### Full staged profile (upgrade track)

```bash
chmod +x scripts/hf_run_staged_grpo_full.sh
bash scripts/hf_run_staged_grpo_full.sh
```

### Track A fast — small data, low steps, bucket backup, always uploads

Use this when you want the same **fast two-stage** setup as the Track A / “Sachin fast”
jobs (`grpo_dataset_tracka_120` → `tracka_v1`, then `tracka_240` → `tracka_v2`), with:

1. **HF Storage Bucket (read/write mount)** — mirrors `artifacts/submission` several times during the run so weights and plots survive even if the job container disappears before you download anything.
2. **HF Dataset repo** — **always** runs `upload_folder` at the end (this script has **no** `SKIP_UPLOAD=1` path).

**Where things land (two different places on purpose):**

| Layer | What it is | Typical URL / path |
| --- | --- | --- |
| **Bucket mirror** | Your [HF Storage Bucket](https://huggingface.co/docs/hub/storage-buckets); mounted inside the job | Files under `/mnt/trl_backup/<RUN_ID>/` (or whatever mount you choose) — full copy of `artifacts/submission` after each milestone |
| **Dataset handoff** | A **Dataset** repo on the Hub | `https://huggingface.co/datasets/<HF_DATASET_REPO>/tree/main/<UPLOAD_PATH_IN_REPO>` — same tree uploaded for judges / Spaces / `snapshot_download` |

Create a bucket under your org/user first, then run a job that mounts it and sets `PERSIST_ROOT` to the mount path. See [Jobs: mount a volume](https://huggingface.co/docs/huggingface_hub/guides/jobs#mount-a-volume).

**Local / any GPU:**

```bash
bash scripts/hf_run_track_a_fast_persist.sh
```

**HF Jobs (recommended pattern):** clone repo, run the script, pass secrets, mount bucket, force unique `UPLOAD_PATH_IN_REPO` if you want:

```bash
hf jobs run -d --flavor a10g-large --timeout 4h --secrets HF_TOKEN \
  -e REPO_URL=https://github.com/VedantPancholi/remorph-openenv-submission.git \
  -e HF_DATASET_REPO=Jenish31/remorph-training-artifacts \
  -e UPLOAD_PATH_IN_REPO=tracka_fast_a10g_2026_04_27 \
  -e PERSIST_ROOT=/mnt/trl_backup \
  -e MAX_STEPS_V1=8 -e MAX_STEPS_V2=12 \
  -v hf://buckets/Jenish31/your-bucket-name:/mnt/trl_backup \
  python:3.12 bash -lc \
  'git clone --depth 1 "$REPO_URL" remorph-openenv-submission && cd remorph-openenv-submission && bash scripts/hf_run_track_a_fast_persist.sh'
```

Replace `Jenish31/your-bucket-name` with your real bucket id. If you omit `-v` and `PERSIST_ROOT`, the script still **uploads to the Dataset**; you only lose the extra on-disk mirror during the run.

**Defaults:** `MAX_STEPS_V1=8`, `MAX_STEPS_V2=12`; `UPLOAD_PATH_IN_REPO` defaults to `tracka_fast_<UTC timestamp>` if unset.

### Two-stage 500 → 1500 (no stage 3) — fast steps + mandatory Hub upload + optional bucket

Use [`scripts/hf_run_staged_grpo_two_stage_persist.sh`](scripts/hf_run_staged_grpo_two_stage_persist.sh) when you want **real 500 / 1500 prompt rows**, **only two training stages**, defaults **`MAX_STEPS_V1=4`** and **`MAX_STEPS_V2=8`**, and **every artifact** pushed to a **public Dataset** repo (free browser access) plus optional **Storage Bucket** mirrors during the run.

**After the job succeeds, open this URL** (replace with your values):

`https://huggingface.co/datasets/<HF_DATASET_REPO>/tree/main/<UPLOAD_PATH_IN_REPO>`

You should see at least:

- `grpo_dataset_500/`, `grpo_dataset_1500/` — generated JSONL datasets  
- `trl_grpo_run_v1/model/`, `trl_grpo_run_v2/model/` — saved adapters / weights  
- `plots/stage1/`, `plots/stage2/`, `plots/master/` — TRL charts  
- `hub_upload_manifest.json` — pointer to this folder on the Hub  
- `best_run_promotion.json`, `frontend_payload.json`, etc.

**Ready-to-paste HF Job** (Dataset upload required; bucket optional — remove `-v` / `PERSIST_ROOT` if you only want Hub):

```bash
hf jobs run -d --flavor a10g-large --timeout 6h --secrets HF_TOKEN \
  -e REPO_URL=https://github.com/VedantPancholi/remorph-openenv-submission.git \
  -e HF_DATASET_REPO=Jenish31/remorph-training-artifacts \
  -e UPLOAD_PATH_IN_REPO=staged2_500_1500_v4_v8_2026_04_27 \
  -e MAX_STEPS_V1=4 \
  -e MAX_STEPS_V2=8 \
  -e EVAL_STEPS=2 \
  -e PERSIST_ROOT=/mnt/trl_backup \
  -v hf://buckets/Jenish31/your-bucket-name:/mnt/trl_backup \
  python:3.12 bash -lc \
  'git clone --depth 1 "$REPO_URL" remorph-openenv-submission && cd remorph-openenv-submission && bash scripts/hf_run_staged_grpo_two_stage_persist.sh'
```

- **Push the script to GitHub first** (or point `REPO_URL` at a fork/branch that contains `hf_run_staged_grpo_two_stage_persist.sh`).  
- **Token** must be allowed to **write** `HF_DATASET_REPO`.  
- **Bucket:** create one under your user/org, replace `your-bucket-name`. Omit bucket lines if you only rely on the Dataset copy.

### Space deployment checks

Before handoff to judges or frontend:

```bash
python scripts/space_submission_checks.py
# Optional live URL check:
python scripts/space_submission_checks.py --space-url https://<your-space>.hf.space
```

The script writes `artifacts/submission/space_submission_check.json` so the team can
track Space health, entrypoint presence, and README alignment in one place.

### SQLite artifact bridge (frontend handoff)

Generate a frontend-ready SQLite database from GRPO run ledger, summaries, and plots:

```bash
python scripts/sqlite_artifact_bridge.py
```

Output:

- DB: `artifacts/submission/frontend_bridge.sqlite3`
- Summary: `artifacts/submission/sqlite_bridge_summary.json`

Core tables:

- `runs` (run metadata and key metrics)
- `metrics` (metric time-series for charting)
- `artifacts` (paths to summaries, model dirs, plot assets)

### Deterministic UI payload contract

For frontend implementation, use the versioned contract docs:

- `docs/frontend_data_contract.md`
- `docs/frontend_payload.schema.json`

Generate the deterministic payload:

```bash
python scripts/sqlite_artifact_bridge.py
python scripts/export_frontend_payload.py
```

Output consumed by UI:

- `artifacts/submission/frontend_payload.json`
- `artifacts/submission/frontend_bridge.sqlite3`

Recommended UI handshake:

1. Validate `frontend_payload.json` against `docs/frontend_payload.schema.json`.
2. Assert `contract_version == "remorph-ui-v1"`.
3. Build charts from `metrics_by_run`.
4. Build leaderboard/cards from `runs` and `best_run`.

### Best-run promotion hook

When long A10G/A100 runs finish, refresh the canonical best run pointer:

```bash
python scripts/promote_long_run_outputs.py
```

This writes `artifacts/submission/best_run_promotion.json` using the rule:
highest `eval_reward_best`, tie-broken by lower `frac_reward_zero_std_last`.

### Where GRPO artifacts live (local vs Hub vs Space)

Training and plotting scripts write under **`artifacts/submission/`** in the repo clone that ran the job (your laptop or an HF Job workspace). Typical locations:

| Artifact | Path (relative to repo root) |
| --- | --- |
| Track A stage 1 model | `artifacts/submission/trl_grpo_run_tracka_v1/model` |
| Track A stage 2 model (often best) | `artifacts/submission/trl_grpo_run_tracka_v2/model` |
| Training summaries / metrics | `artifacts/submission/trl_grpo_run_tracka_v2/trl_grpo_training_summary.json`, `trl_grpo_metrics.json` |
| TRL plots | `artifacts/submission/plots/tracka_stage1/`, `plots/tracka_stage2/`, `plots/tracka_master/` |
| UI / SQLite handoff | `artifacts/submission/frontend_payload.json`, `frontend_bridge.sqlite3` |
| Best-run pointer | `artifacts/submission/best_run_promotion.json` |

**HF Jobs:** Paths are the same inside the ephemeral clone. When the job finishes, that disk goes away unless you **upload** or **download** outputs.

**Hugging Face Hub:** Files do **not** appear on a Dataset or Model repo automatically. Inline Track A commands often have **no** upload step. To publish the whole `artifacts/submission` tree (or refresh after a local run):

```bash
pip install huggingface_hub
export HF_TOKEN=...   # token with write access to the target namespace
export HF_DATASET_REPO=YourNamespace/your-dataset
export UPLOAD_PATH_IN_REPO=track_a_run_2026_04_26
python scripts/upload_submission_artifacts_to_hub.py
# Dry run: python scripts/upload_submission_artifacts_to_hub.py --dry-run
```

Alternatively, run `scripts/hf_run_quick_confidence.sh` or `scripts/hf_run_staged_grpo_full.sh` with **`SKIP_UPLOAD` unset or `0`** (see those scripts for `HF_DATASET_REPO` / `UPLOAD_PATH_IN_REPO`).

**Spaces:** A Space is a **separate** Git repo. Training does not push weights into the Space. Commit/copy the promoted model directory into the Space repo, or `from_pretrained` a Hub model id after upload, and redeploy.

**Quick audit on disk:**

```bash
python scripts/verify_submission_artifacts.py
python scripts/verify_submission_artifacts.py --strict   # exit 1 if promotion/model missing
```

## Enterprise workflow scope

The richer scenario pack includes multi-app professional workflows such as:

- CRM to billing order sync
- Identity to HR user provisioning
- Support to billing refund reconciliation
- Analytics to finance quarter close

These workflows require the agent to maintain internal state across more than one step and adapt to changing evidence over the rollout.

## Phase 1 Benchmark Expansion

Phase 1 keeps the validator-friendly `4` frozen scenarios and expands the benchmark pack to `20` total scenarios:

- `4` Phase 1 validation scenarios
- `16` benchmark scenarios
- `10` train split scenarios
- `6` eval split scenarios

The split manifests are checked in at:

- `artifacts/submission/splits/train_manifest.json`
- `artifacts/submission/splits/eval_manifest.json`
- `artifacts/submission/splits/scenario_catalog.json`

The benchmark pack supports seeded randomization on non-validation scenarios only. That means:

- validation cases stay byte-stable for OpenEnv and smoke-test compatibility
- benchmark cases vary with `seed`
- the same `seed` reproduces the same materialized benchmark observations

Example input surface:

```python
from remorph_openenv.environment import ReMorphEnvironment
from remorph_openenv.scenarios import load_built_in_scenarios

scenarios = load_built_in_scenarios(seed=42, split="train", execution_mode="simulated")
env = ReMorphEnvironment(seed=42, split="train")
observation = env.reset(scenario_id=scenarios[0].scenario_id, seed=42)
```

## Phase 2 Live-Local Mode

Phase 2 adds a real local FastAPI-backed execution loop while preserving the Phase 1 simulated benchmark unchanged.

Live-local measurable scope:

- `8` live-local scenarios
- `3` FastAPI-backed services: CRM, billing, identity
- real HTTP request/response handling through FastAPI `TestClient`
- support for single-step, multi-step, and safe-abstention workflows

Run the live-local demo:

```bash
python scripts/live_local_demo.py
```

Optional manual gateway:

```bash
python scripts/start_live_local_server.py
```

Example live-local environment usage:

```python
from remorph_openenv.environment import ReMorphEnvironment
from remorph_openenv.scenarios import load_live_local_scenarios

scenarios = load_live_local_scenarios(seed=42)
env = ReMorphEnvironment(seed=42, execution_mode="live_local")
observation = env.reset(scenario_id=scenarios[0].scenario_id, seed=42)
```

## Measurable status

Current clean-repo benchmark evidence:

- total built-in scenarios: `20`
- single-step validation scenarios: `4`
- benchmark scenarios: `16`
- train split size: `10`
- eval split size: `6`
- live-local scenarios: `8`
- live-local services: `3`
- supervised train split scenarios: `10`
- supervised eval split scenarios: `6`
- OpenEnv validation: `passed`
- smoke test on frozen validator scenarios: `passed`
- deterministic inference across all `20` scenarios: `passed`
- live-local demo across all `8` live scenarios: `passed`
- seeded split generation: `passed`
- supervised training artifacts: `passed`
- TRL GRPO dry-run plumbing: `passed`
- GRPO seeded prompt dataset: `500` train rows, `60` eval rows by default
- benchmark report generation: `passed`

## Next Phase

The next implementation phase is the Hugging Face Space.

What Phase 4 should do:

- choose a scenario
- see the observation/state
- run baseline vs supervised vs oracle
- inspect chosen action JSON
- inspect reward and reward breakdown
- step through the episode
- compare simulated vs live-local where relevant
- view benchmark summary metrics and plots

Why this is next:

- the benchmark core is now stable
- train/eval reporting is measurable and validator-safe
- live-local mode exists
- telemetry, benchmark report, and plots already exist
- the highest remaining value is presentation and judge accessibility

## Manual Verification

Use this checklist to manually verify the benchmark before the Hugging Face Space build:

1. Validate packaging and config.
   Run `python -c "import yaml; yaml.safe_load(open('openenv.yaml')); print('openenv.yaml OK')"`
   Expected: `openenv.yaml OK`
2. Verify the frozen validator scenarios.
   Run `python scripts/smoke_test_openenv.py`
   Expected: `4` validator scenarios, safe abstention reward above unsafe hallucinated auth.
3. Verify full simulated benchmark inference.
   Run `python scripts/inference.py`
   Expected: deterministic rollout over all `20` simulated scenarios.
4. Verify split generation.
   Run `python scripts/generate_splits.py --seed 42`
   Expected: `10` train scenarios and `6` eval scenarios in `artifacts/submission/splits/`.
5. Verify live-local execution.
   Run `python scripts/live_local_demo.py`
   Expected: `8` live-local scenarios complete successfully through FastAPI-backed HTTP handling.
6. Verify supervised manifest-driven training.
   Run `python scripts/train_submission.py --output-dir artifacts/submission/training_run --seed 42 --train-manifest artifacts/submission/splits/train_manifest.json --eval-manifest artifacts/submission/splits/eval_manifest.json`
   Expected: train success `1.0`, eval success `1.0`, TRL dataset export present.
7. Verify TRL/OpenEnv training plumbing.
   Run `python scripts/generate_grpo_dataset.py && python scripts/train_trl_grpo.py --dry-run --train-path artifacts/submission/grpo_dataset/train_prompts.jsonl && python scripts/plot_trl_grpo.py`
   Expected: `500` train rows, `60` eval rows, `dry_run_ok`, one environment step returns reward feedback, and dark-grid TRL plots are generated.
8. Verify report and plots.
   Run `python scripts/generate_submission_plots.py`
   Expected: benchmark plots under `artifacts/submission/plots/` and reports under `artifacts/submission/`.
9. Verify OpenEnv compatibility.
   Run `openenv validate`
   Expected: `Ready for multi-mode deployment`

## Colab Flow

The notebook for the prompt-driven training flow lives at:

- `notebooks/remorph_openenv_colab.ipynb`

It is designed to show this exact sequence:

1. clone repo and install requirements
2. validate OpenEnv and run smoke tests
3. train the supervised structured policy
4. evaluate `baseline`, `supervised`, and `adaptive_reference` on held-out eval
5. generate the seeded GRPO train/eval prompt dataset
6. run the TRL GRPO dry-run and plot its metrics
7. optionally run short GPU TRL GRPO training
8. generate plots
9. inspect the benchmark report

## Files

- `remorph_openenv/environment.py`: main OpenEnv-compatible environment
- `remorph_openenv/models.py`: explicit internal and OpenEnv-facing state models
- `remorph_openenv/scenarios.py`: built-in scenarios and contract loading
- `remorph_openenv/rewards.py`: deterministic reward scoring
- `remorph_openenv/live_local.py`: FastAPI-backed local service world and gateway
- `remorph_openenv/training.py`: split-aware training, telemetry, and benchmark reporting
- `scripts/inference.py`: deterministic local inference over all built-in scenarios
- `scripts/generate_splits.py`: checked-in train/eval split manifest generator
- `scripts/generate_grpo_dataset.py`: seeded GRPO prompt dataset generator
- `scripts/live_local_demo.py`: runnable Phase 2 live-local demo
- `scripts/start_live_local_server.py`: optional manual FastAPI gateway runner
- `scripts/train_submission.py`: Phase 3 supervised learner training entrypoint
- `scripts/train_submission.py --train-manifest ... --eval-manifest ...`: manifest-driven training entrypoint
- `scripts/train_trl_grpo.py`: optional TRL GRPO training entrypoint using `environment_factory`
- `scripts/plot_trl_grpo.py`: plots TRL dry-run or real GRPO training metrics
- `scripts/verify_submission_artifacts.py`: print which GRPO / Track A paths exist under `artifacts/submission`
- `scripts/upload_submission_artifacts_to_hub.py`: upload `artifacts/submission` to a Hub Dataset repo
- `scripts/hf_run_track_a_fast_persist.sh`: Track A fast training, optional HF Bucket mirrors, mandatory Dataset upload
- `scripts/hf_run_staged_grpo_two_stage_persist.sh`: 500 → 1500 two-stage GRPO, optional bucket mirror, mandatory Dataset upload
- `scripts/evaluate_submission.py`: policy evaluation entrypoint for baseline, supervised, replay, and adaptive reference
- `remorph_openenv/trl_env.py`: TRL tool-call environment wrapper and GRPO prompt builder
- `artifacts/submission/telemetry/rollouts.jsonl`: step-level benchmark telemetry
- `artifacts/submission/benchmark_report.md`: benchmark summary for judges
- `artifacts/submission/training_run/trl_dataset.jsonl`: TRL-ready serialized dataset export
- `notebooks/remorph_openenv_colab.ipynb`: Colab-ready training and evaluation flow
- `scripts/smoke_test_openenv.py`: runnable Phase 1 validation script
- `tests/test_openenv_smoke.py`: pytest coverage for the minimum pass
- `tests/test_scenario_registry.py`: scenario count, split integrity, reproducibility, and observation hygiene checks
- `tests/test_live_local_mode.py`: live-local route, workflow, and safe-abstention coverage
- `tests/test_phase3_artifacts.py`: benchmark report coverage
