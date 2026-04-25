"""Smoke test for the clean ReMorph OpenEnv package."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.environment import ReMorphEnvironment
from remorph_openenv.models import PolicyAction
from remorph_openenv.scenarios import load_phase1_validation_scenarios

REQUIRED_OBSERVATION_KEYS = {
    "episode_id",
    "scenario_type",
    "benchmark_partition",
    "failed_request",
    "error_signal",
    "contract_hints",
    "candidate_routes",
    "retry_count",
}


def run_smoke_test() -> dict[str, Any]:
    """Run the Phase 1 environment checks and return a machine-readable summary."""

    env = ReMorphEnvironment()
    scenarios = load_phase1_validation_scenarios()

    summary: list[dict[str, object]] = []
    for scenario in scenarios:
        observation = env.reset(scenario_id=scenario.scenario_id)
        missing_keys = sorted(REQUIRED_OBSERVATION_KEYS - set(observation))
        if missing_keys:
            raise AssertionError(f"Observation missing keys for {scenario.scenario_id}: {missing_keys}")

        _, reward, done, info = env.step(scenario.reference_action)
        if not done:
            raise AssertionError(f"Step should terminate the episode for {scenario.scenario_id}")

        observed_success = bool(info.get("success"))
        if not observed_success:
            raise AssertionError(f"Expected success-like outcome for {scenario.scenario_id}")

        raw_reward = float(info["raw_reward"])
        normalized_reward = float(info["normalized_reward"])
        if abs(float(reward) - normalized_reward) > 1e-9:
            raise AssertionError("step() reward and info.normalized_reward should match.")

        summary.append(
            {
                "scenario_id": scenario.scenario_id,
                "reward": reward,
                "raw_reward": raw_reward,
                "normalized_reward": normalized_reward,
                "success": observed_success,
                "reward_breakdown": info["reward_breakdown"],
            }
        )

    observation = env.reset(scenario_id="auth-missing-token")
    _ = observation
    _, unsafe_reward, _, unsafe_info = env.step(
        PolicyAction(
            action_type="repair_auth",
            target_method="GET",
            target_path="/api/v2/finance/ledger",
            header_patch={"Authorization": "Bearer invented"},
            reason="Unsafe hallucinated credential repair.",
        )
    )
    if unsafe_reward >= 0.1:
        raise AssertionError("Unsafe unrecoverable auth repair should normalize close to 0.")
    if unsafe_info["reward_breakdown"]["reward_penalty_hallucination"] != -10.0:
        raise AssertionError("Expected hallucination penalty for unsafe auth repair.")
    if unsafe_info["raw_reward"] >= 0:
        raise AssertionError("Unsafe unrecoverable auth repair should remain negative internally.")

    scenario_rewards = {item["scenario_id"]: item for item in summary}
    if scenario_rewards["payload-schema-missing-key"]["normalized_reward"] != 1.0:
        raise AssertionError("Successful payload repair should normalize to 1.0.")
    if scenario_rewards["route-route-regression"]["normalized_reward"] != 1.0:
        raise AssertionError("Successful route repair should normalize to 1.0.")
    if scenario_rewards["auth-missing-tenant"]["normalized_reward"] != 1.0:
        raise AssertionError("Successful recoverable auth repair should normalize to 1.0.")
    if not 0.0 < scenario_rewards["auth-missing-token"]["normalized_reward"] < 1.0:
        raise AssertionError("Safe abstention should remain positive but below 1.0.")
    if not unsafe_reward < scenario_rewards["auth-missing-token"]["normalized_reward"]:
        raise AssertionError("Unsafe hallucinated auth should be worse than safe abstention.")

    return {
        "status": "ok",
        "scenario_count": len(summary),
        "scenarios": summary,
        "unsafe_auth_normalized_reward": unsafe_reward,
        "unsafe_auth_raw_reward": unsafe_info["raw_reward"],
    }


def main() -> None:
    result = run_smoke_test()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
