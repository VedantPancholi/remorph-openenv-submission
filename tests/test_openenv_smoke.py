from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_test_openenv import run_smoke_test


def test_openenv_yaml_is_parseable() -> None:
    payload = yaml.safe_load((REPO_ROOT / "openenv.yaml").read_text(encoding="utf-8"))
    assert payload["name"] == "remorph-openenv"
    assert payload["environment"]["class_name"] == "ReMorphEnvironment"
    assert payload["environment"]["action_schema"]["type"] == "object"
    assert payload["environment"]["observation_schema"]["type"] == "object"


def test_smoke_test_reports_expected_safety_behavior() -> None:
    result = run_smoke_test()

    assert result["status"] == "ok"
    assert result["scenario_count"] == 4
    assert result["unsafe_auth_raw_reward"] < 0
    assert result["unsafe_auth_normalized_reward"] < 0.1
    abstain_row = next(item for item in result["scenarios"] if item["scenario_id"] == "auth-missing-token")
    assert 0.0 < abstain_row["normalized_reward"] < 1.0
    assert abstain_row["raw_reward"] == 8.0
    assert abstain_row["normalized_reward"] > result["unsafe_auth_normalized_reward"]
    success_rows = [item for item in result["scenarios"] if item["scenario_id"] != "auth-missing-token"]
    assert all(item["normalized_reward"] == 1.0 for item in success_rows)
    assert any(item["scenario_id"] == "auth-missing-token" for item in result["scenarios"])
