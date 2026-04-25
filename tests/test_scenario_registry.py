from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.environment import ReMorphEnvironment
from remorph_openenv.scenarios import (
    load_built_in_scenarios,
    load_phase1_validation_scenarios,
    load_world_model_scenarios,
    scenario_catalog,
)


def test_phase1_scenario_registry_counts_and_splits() -> None:
    all_scenarios = load_built_in_scenarios(seed=42, split="all")
    validation = [scenario for scenario in all_scenarios if scenario.validation_tier == "phase1"]
    benchmark = [scenario for scenario in all_scenarios if scenario.validation_tier != "phase1"]
    train = load_built_in_scenarios(seed=42, split="train")
    eval_pack = load_built_in_scenarios(seed=42, split="eval")

    assert len(all_scenarios) == 20
    assert len(validation) == 4
    assert len(benchmark) == 16
    assert len(train) == 10
    assert len(eval_pack) == 6
    assert {scenario.scenario_id for scenario in train}.isdisjoint({scenario.scenario_id for scenario in eval_pack})


def test_validation_scenarios_remain_frozen() -> None:
    validation = load_phase1_validation_scenarios()
    route_case = next(item for item in validation if item.scenario_id == "route-route-regression")

    assert route_case.validation_tier == "phase1"
    assert route_case.randomization_profile == "frozen"
    assert route_case.reference_action.target_path == "/api/v2/finance/ledger"
    assert route_case.phases[0].contract_hints["drift_contract"] == "drift_route.json"


def test_seeded_randomization_is_reproducible_but_not_constant() -> None:
    seeded_a = load_world_model_scenarios(seed=7, split="train")
    seeded_b = load_world_model_scenarios(seed=7, split="train")
    seeded_c = load_world_model_scenarios(seed=9, split="train")

    rows_a = [json.dumps(item.to_openenv_state(phase_index=0, retry_count=0, prior_actions=[]).model_dump(mode="json"), sort_keys=True) for item in seeded_a]
    rows_b = [json.dumps(item.to_openenv_state(phase_index=0, retry_count=0, prior_actions=[]).model_dump(mode="json"), sort_keys=True) for item in seeded_b]
    rows_c = [json.dumps(item.to_openenv_state(phase_index=0, retry_count=0, prior_actions=[]).model_dump(mode="json"), sort_keys=True) for item in seeded_c]

    assert rows_a == rows_b
    assert rows_a != rows_c


def test_observation_remains_partially_observable_without_oracle_action_keys() -> None:
    env = ReMorphEnvironment(seed=42, split="train")
    scenario_id = next(item.scenario_id for item in load_world_model_scenarios(seed=42, split="train"))
    observation = env.reset(scenario_id=scenario_id, seed=42)

    assert "expected_action" not in observation
    assert "reference_action" not in observation
    assert "success_status_code" not in observation
    assert "drift_contract_name" not in observation


def test_scenario_catalog_exposes_benchmark_metadata() -> None:
    catalog = scenario_catalog(seed=42)

    assert len(catalog) == 20
    assert all("validation_tier" in row for row in catalog)
    assert all("execution_mode" in row for row in catalog)
    assert all("workflow_length" in row for row in catalog)
    assert all("service_domain" in row for row in catalog)
    assert all("randomization_profile" in row for row in catalog)
