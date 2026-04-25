from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.environment import ReMorphEnvironment
from remorph_openenv.scenarios import load_world_model_scenarios


def test_multi_step_world_model_scenario_requires_more_than_one_step() -> None:
    scenario = next(item for item in load_world_model_scenarios() if item.scenario_id == "crm-billing-order-sync")
    env = ReMorphEnvironment(scenarios=[scenario], seed=0)

    observation = env.reset(scenario_id=scenario.scenario_id)
    reward_observation, reward, done, info = env.step(scenario.phases[0].expected_action)

    assert done is False
    assert reward > 0.0
    assert info["done_reason"] == "progressed"
    assert reward_observation["step_index"] == 1
