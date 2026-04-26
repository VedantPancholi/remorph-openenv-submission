# Frontend Data Contract (Deterministic)

This document defines the **stable payload contract** that frontend should use.
The payload is generated from training artifacts through:

1. `python scripts/sqlite_artifact_bridge.py`
2. `python scripts/export_frontend_payload.py`

## Contract version

- `contract_version`: `remorph-ui-v1`
- Backward-compatibility rule:
  - `v1.x`: additive-only changes (safe for UI)
  - `v2.0+`: breaking changes (UI migration required)

## Artifact outputs

- SQLite DB: `artifacts/submission/frontend_bridge.sqlite3`
- JSON payload: `artifacts/submission/frontend_payload.json`
- JSON schema: `docs/frontend_payload.schema.json`

## SQLite schema (source of truth)

### `runs`

- `run_id` `TEXT PRIMARY KEY` (deterministic ID)
- `timestamp_utc` `TEXT | null`
- `model_name` `TEXT | null`
- `stage_label` `TEXT` (`stage1|stage2|stage3|unknown`)
- `status` `TEXT | null`
- `prompt_row_count` `INTEGER | null`
- `eval_prompt_row_count` `INTEGER | null`
- `max_steps` `INTEGER | null`
- `reward_best` `REAL | null`
- `reward_last` `REAL | null`
- `eval_reward_best` `REAL | null`
- `eval_reward_last` `REAL | null`
- `frac_reward_zero_std_last` `REAL | null`
- `model_output_dir` `TEXT | null`
- `source_type` `TEXT` (`ledger|summary`)

### `metrics`

- `id` `INTEGER PRIMARY KEY AUTOINCREMENT`
- `run_id` `TEXT`
- `step` `INTEGER`
- `metric_name` `TEXT`
- `metric_value` `REAL`
- `source_type` `TEXT`

### `artifacts`

- `id` `INTEGER PRIMARY KEY AUTOINCREMENT`
- `run_id` `TEXT`
- `artifact_type` `TEXT`
- `artifact_path` `TEXT`

## Frontend JSON payload contract

Top-level payload object:

- `contract_version`: string (currently `remorph-ui-v1`)
- `generated_at_utc`: ISO timestamp
- `db_path`: project-relative path string
- `summary`: object
- `runs`: array of run objects (sorted by `eval_reward_best DESC`, then `run_id ASC`)
- `metrics_by_run`: object keyed by `run_id`
- `artifacts_by_run`: object keyed by `run_id`
- `best_run`: run object or `null`

### `summary`

- `total_runs`: integer
- `total_metric_points`: integer
- `total_artifacts`: integer
- `metric_names`: sorted string array

### Run object

- Same keys as `runs` table columns (except DB internal IDs)

### `metrics_by_run[run_id]`

Array of metric rows sorted by:

1. `step ASC`
2. `metric_name ASC`

Each row:

- `step`: integer
- `metric_name`: string
- `metric_value`: number
- `source_type`: string

### `artifacts_by_run[run_id]`

Array sorted by:

1. `artifact_type ASC`
2. `artifact_path ASC`

Each row:

- `artifact_type`: string
- `artifact_path`: string

## Determinism guarantees

The exporter enforces deterministic ordering:

- fixed SQL `ORDER BY`
- stable object/array construction
- explicit sorted metric name list

For identical DB inputs, JSON output is byte-stable except `generated_at_utc`.

## Frontend usage guidance

- Use `runs` for leaderboard/table cards.
- Use `metrics_by_run[run_id]` to build line charts.
- Use `artifacts_by_run[run_id]` for drill-down links.
- Highlight `best_run` in dashboard header.

## Recommended frontend checks

- Assert `contract_version === "remorph-ui-v1"`.
- Validate payload against `docs/frontend_payload.schema.json`.
- Gracefully handle empty arrays and `null` numeric fields.
