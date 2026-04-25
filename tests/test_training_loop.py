from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.training import load_split_scenarios, train_replay_policy, train_supervised_policy


def test_replay_training_still_solves_the_full_built_in_pack() -> None:
    result = train_replay_policy(epochs=2)

    assert result["scenario_count"] == 20
    assert result["training_example_count"] >= result["scenario_count"]
    assert 0.0 <= result["baseline_summary"]["success_rate"] <= 1.0
    assert result["learned_summary"]["success_rate"] == 1.0
    assert result["learned_summary"]["average_raw_reward"] > result["baseline_summary"]["average_raw_reward"]


def test_supervised_training_improves_over_baseline_on_eval_split() -> None:
    result = train_supervised_policy(seed=42)
    splits = load_split_scenarios(seed=42)

    assert len(splits["train"]) == 10
    assert len(splits["eval"]) == 6
    assert result["train_scenario_count"] == 10
    assert result["eval_scenario_count"] == 6
    assert result["supervised_eval_summary"]["success_rate"] >= result["baseline_eval_summary"]["success_rate"]
    assert result["supervised_eval_summary"]["average_raw_reward"] > result["baseline_eval_summary"]["average_raw_reward"]
    assert result["oracle_eval_summary"]["success_rate"] == 1.0
    assert result["replay_eval_summary"]["success_rate"] <= result["supervised_eval_summary"]["success_rate"]
    assert result["model_config"]["learner"] == "supervised_structured_policy"
