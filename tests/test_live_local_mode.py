from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.environment import ReMorphEnvironment
from remorph_openenv.models import PolicyAction
from remorph_openenv.scenarios import load_live_local_scenarios


def test_live_local_pack_has_expected_size() -> None:
    scenarios = load_live_local_scenarios(seed=42)

    assert len(scenarios) == 8
    assert all(item.execution_mode == "live_local" for item in scenarios)


def test_live_local_single_step_route_repair_succeeds() -> None:
    env = ReMorphEnvironment(seed=42, execution_mode="live_local")
    scenario = next(item for item in load_live_local_scenarios(seed=42) if item.scenario_id == "live-crm-route-repair")

    env.reset(scenario_id=scenario.scenario_id, seed=42)
    _, reward, done, info = env.step(scenario.reference_action)

    assert done is True
    assert reward == 1.0
    assert info["status_code"] == 200
    assert info["success"] is True


def test_live_local_multi_step_workflow_progresses() -> None:
    env = ReMorphEnvironment(seed=42, execution_mode="live_local")
    scenario = next(item for item in load_live_local_scenarios(seed=42) if item.scenario_id == "live-crm-billing-order-sync")

    observation = env.reset(scenario_id=scenario.scenario_id, seed=42)
    observation, reward, done, info = env.step(scenario.phases[0].expected_action)

    assert done is False
    assert reward > 0.0
    assert info["done_reason"] == "progressed"
    assert observation["step_index"] == 1


def test_live_local_safe_abstention_beats_hallucinated_auth() -> None:
    env = ReMorphEnvironment(seed=42, execution_mode="live_local")
    scenario = next(item for item in load_live_local_scenarios(seed=42) if item.scenario_id == "live-identity-safe-abstain")

    env.reset(scenario_id=scenario.scenario_id, seed=42)
    _, abstain_reward, abstain_done, abstain_info = env.step(scenario.reference_action)

    env.reset(scenario_id=scenario.scenario_id, seed=42)
    _, unsafe_reward, unsafe_done, unsafe_info = env.step(
        PolicyAction(
            action_type="repair_auth",
            target_method="POST",
            target_path="/identity/v2/secure/export",
            header_patch={"Authorization": "Bearer invented"},
            reason="Unsafe fabricated credential repair.",
        )
    )

    assert abstain_done is True
    assert unsafe_done is True
    assert abstain_info["success"] is True
    assert abstain_reward > unsafe_reward
    assert unsafe_info["raw_reward"] < 0
