"""Minimal server entrypoint for OpenEnv validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.environment import ReMorphEnvironment
from remorph_openenv.models import PolicyAction
from remorph_openenv.scenarios import load_live_local_scenarios


def run_demo_episode() -> dict[str, Any]:
    """Execute one deterministic simulated demo episode."""

    env = ReMorphEnvironment(seed=0)
    observation = env.reset(scenario_id="route-route-regression")
    _ = observation
    _, reward, done, info = env.step(
        PolicyAction(
            action_type="repair_route",
            target_method="GET",
            target_path="/api/v2/finance/ledger",
            reason="Validation demo route repair.",
        )
    )
    return {
        "status": "ok",
        "done": done,
        "reward": reward,
        "raw_reward": info["raw_reward"],
        "normalized_reward": info["normalized_reward"],
        "success": info["success"],
    }


def run_live_local_demo_episode() -> dict[str, Any]:
    """Execute one deterministic live-local demo episode."""

    env = ReMorphEnvironment(seed=42, execution_mode="live_local")
    scenario = load_live_local_scenarios(seed=42)[0]
    observation = env.reset(scenario_id=scenario.scenario_id, seed=42)
    _ = observation
    _, reward, done, info = env.step(
        PolicyAction(
            action_type="repair_route",
            target_method="GET",
            target_path="/crm/v2/accounts/export",
            reason="Live-local validation demo route repair.",
        )
    )
    return {
        "status": "ok",
        "mode": "live_local",
        "scenario_id": scenario.scenario_id,
        "done": done,
        "reward": reward,
        "raw_reward": info["raw_reward"],
        "normalized_reward": info["normalized_reward"],
        "success": info["success"],
        "status_code": info["status_code"],
    }


def main() -> None:
    print(json.dumps({"simulated": run_demo_episode(), "live_local": run_live_local_demo_episode()}, indent=2))


if __name__ == "__main__":
    main()
