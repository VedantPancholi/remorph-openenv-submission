from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.training import load_scenarios_from_manifest, train_supervised_policy


def test_train_eval_manifests_are_disjoint_and_loadable() -> None:
    train_manifest = REPO_ROOT / "artifacts" / "submission" / "splits" / "train_manifest.json"
    eval_manifest = REPO_ROOT / "artifacts" / "submission" / "splits" / "eval_manifest.json"

    train_payload = json.loads(train_manifest.read_text(encoding="utf-8"))
    eval_payload = json.loads(eval_manifest.read_text(encoding="utf-8"))
    train_ids = {row["scenario_id"] for row in train_payload["scenarios"]}
    eval_ids = {row["scenario_id"] for row in eval_payload["scenarios"]}

    assert train_ids
    assert eval_ids
    assert train_ids.isdisjoint(eval_ids)
    assert len(load_scenarios_from_manifest(train_manifest, seed=42)) == len(train_ids)
    assert len(load_scenarios_from_manifest(eval_manifest, seed=42)) == len(eval_ids)


def test_telemetry_schema_contains_stable_signature_fields() -> None:
    result = train_supervised_policy(seed=42)
    telemetry_rows = result["telemetry"]["supervised_eval"]

    assert telemetry_rows
    first = telemetry_rows[0]
    required_keys = {
        "policy_name",
        "scenario_id",
        "workflow_id",
        "step_index",
        "observation_signature",
        "belief",
        "confidence",
        "belief_changed",
        "action",
        "normalized_reward",
        "raw_reward",
        "done",
        "success",
        "reward_breakdown",
    }
    assert required_keys.issubset(first.keys())
    assert isinstance(first["observation_signature"], str)
    assert isinstance(first["reward_breakdown"], dict)
