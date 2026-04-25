from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.scenarios import load_built_in_scenarios
from remorph_openenv.trl_env import (
    ReMorphToolEnv,
    build_grpo_prompt_rows,
    environment_reward,
    make_environment_factory,
)


def test_grpo_prompt_rows_include_scenario_routing() -> None:
    rows = build_grpo_prompt_rows(seed=42, split="train", repeats=2)

    assert rows
    assert len(rows) == 20
    assert {
        "prompt",
        "scenario_id",
        "workflow_id",
        "split",
        "seed",
        "scenario_type",
        "benchmark_partition",
        "workflow_length",
    } <= set(rows[0])
    assert rows[0]["split"] == "train"
    assert rows[0]["seed"] == 42
    assert rows[0]["prompt"][0]["role"] == "system"


def test_tool_env_executes_reference_action_and_sets_reward() -> None:
    scenario = load_built_in_scenarios(seed=42, split="train")[0]
    action = scenario.phases[0].expected_action
    env = ReMorphToolEnv(seed=42, split="train")

    observation_text = env.reset(scenario_id=scenario.scenario_id)
    assert scenario.scenario_id in observation_text

    if action.action_type == "repair_route":
        feedback = env.repair_route(action.target_method or "GET", action.target_path or "/", "test route")
    elif action.action_type == "repair_payload":
        feedback = env.repair_payload(
            action.target_method or "POST",
            action.target_path or "/",
            json.dumps(action.body_patch or {}),
            "test payload",
        )
    elif action.action_type == "repair_auth":
        feedback = env.repair_auth(
            action.target_method or "GET",
            action.target_path or "/",
            json.dumps(action.header_patch or {}),
            "test auth",
        )
    elif action.action_type == "abstain":
        feedback = env.abstain("test abstain")
    else:
        feedback = env.no_op("test no-op")

    payload = json.loads(feedback)
    assert payload["reward"] == env.reward
    assert env.reward > 0.0
    assert environment_reward([env]) == [env.reward]


def test_tool_env_rejects_bad_json_tool_arguments() -> None:
    env = ReMorphToolEnv(seed=42, split="train")
    scenario = load_built_in_scenarios(seed=42, split="train")[0]
    env.reset(scenario_id=scenario.scenario_id)

    with pytest.raises(ValueError, match="body_patch_json"):
        env.repair_payload("POST", "/example", "not-json", "bad json")


def test_make_environment_factory_returns_no_arg_class() -> None:
    factory = make_environment_factory(seed=42, split="train")
    env = factory()

    assert isinstance(env, ReMorphToolEnv)
    assert env.split == "train"


def test_tool_env_seed_routing_is_deterministic_and_variable() -> None:
    scenario_id = load_built_in_scenarios(seed=1000, split="train")[0].scenario_id
    env = ReMorphToolEnv(seed=42, split="train")

    first = env.reset(seed=1000, split="train", scenario_id=scenario_id)
    second = env.reset(seed=1000, split="train", scenario_id=scenario_id)
    third = env.reset(seed=1001, split="train", scenario_id=scenario_id)

    assert first == second
    assert first != third
