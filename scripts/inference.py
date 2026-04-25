"""Deterministic local inference runner for the built-in ReMorph scenarios."""

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
from remorph_openenv.scenarios import load_built_in_scenarios


def choose_action(observation: dict[str, Any]) -> PolicyAction:
    """Pick a deterministic action from the built-in scenario evidence."""

    scenario_type = str(observation.get("scenario_type") or "unknown")
    benchmark_partition = str(observation.get("benchmark_partition") or "other")
    contract_hints = dict(observation.get("contract_hints") or {})
    candidate_routes = list(observation.get("candidate_routes") or [])
    failed_request = dict(observation.get("failed_request") or {})

    if benchmark_partition == "unrecoverable":
        return PolicyAction(
            action_type="abstain",
            reason="Deterministic policy abstains on unrecoverable credential failures.",
        )

    if scenario_type == "route_drift" and candidate_routes:
        route = max(
            (dict(candidate) for candidate in candidate_routes),
            key=lambda row: float(row.get("confidence") or 0.0),
        )
        return PolicyAction(
            action_type="repair_route",
            target_method=str(route.get("method") or failed_request.get("method") or "GET").upper(),
            target_path=str(route.get("path") or ""),
            reason="Deterministic route repair from top contract candidate.",
        )

    if scenario_type == "payload_drift":
        expected_body = dict(contract_hints.get("expected_request_body") or {})
        if not expected_body:
            failed_body = dict(failed_request.get("body") or {})
            expected_body = {
                "user": {
                    "f_name": str(failed_body.get("first_name", "")),
                    "l_name": str(failed_body.get("last_name", "")),
                }
            }
        return PolicyAction(
            action_type="repair_payload",
            target_method=str(failed_request.get("method") or "POST").upper(),
            target_path=str(failed_request.get("path") or "/"),
            body_patch=expected_body,
            reason="Deterministic payload repair from expected request body hint.",
        )

    if scenario_type == "auth_drift":
        required_headers = list(contract_hints.get("required_headers") or [])
        tenant_alias = str(contract_hints.get("tenant_alias") or "").strip("-")
        tenant_key = f"demo-tenant-key-{tenant_alias}" if tenant_alias else "demo-tenant-key"
        header_patch = {
            header_name: tenant_key
            for header_name in required_headers
            if header_name == "x-api-key"
        }
        return PolicyAction(
            action_type="repair_auth",
            target_method=str(failed_request.get("method") or "GET").upper(),
            target_path=str(failed_request.get("path") or "/"),
            header_patch=header_patch,
            reason="Deterministic auth repair only for recoverable tenant/header drift.",
        )

    return PolicyAction(action_type="no_op", reason="No deterministic action available.")


def main() -> None:
    env = ReMorphEnvironment(seed=0)
    results: list[dict[str, Any]] = []

    for scenario in load_built_in_scenarios():
        scenario_id = scenario.scenario_id
        observation = env.reset(scenario_id=scenario_id)
        done = False
        step_results: list[dict[str, Any]] = []
        while not done:
            action = choose_action(observation)
            observation, reward, done, info = env.step(action)
            step_results.append(
                {
                    "action": action.model_dump(mode="json"),
                    "done": done,
                    "success": info["success"],
                    "raw_reward": info["raw_reward"],
                    "normalized_reward": reward,
                    "reward_breakdown": info["reward_breakdown"],
                    "phase_index": info.get("phase_index"),
                }
            )
        results.append(
            {
                "scenario_id": scenario_id,
                "workflow_id": scenario.workflow_id,
                "app_stack": scenario.app_stack,
                "step_count": len(step_results),
                "final_success": step_results[-1]["success"],
                "steps": step_results,
            }
        )

    print(json.dumps({"status": "ok", "results": results}, indent=2))


if __name__ == "__main__":
    main()
