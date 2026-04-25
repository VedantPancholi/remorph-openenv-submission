"""Run the Phase 2 live-local FastAPI-backed environment pack."""

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


def choose_action(observation: dict[str, Any]) -> PolicyAction:
    scenario_type = str(observation.get("scenario_type") or "unknown")
    benchmark_partition = str(observation.get("benchmark_partition") or "other")
    contract_hints = dict(observation.get("contract_hints") or {})
    candidate_routes = list(observation.get("candidate_routes") or [])
    failed_request = dict(observation.get("failed_request") or {})

    if benchmark_partition == "unrecoverable":
        return PolicyAction(action_type="abstain", reason="Live-local demo abstains on unrecoverable auth.")

    if scenario_type == "route_drift" and candidate_routes:
        route = max((dict(candidate) for candidate in candidate_routes), key=lambda row: float(row.get("confidence") or 0.0))
        return PolicyAction(
            action_type="repair_route",
            target_method=str(route.get("method") or failed_request.get("method") or "GET").upper(),
            target_path=str(route.get("path") or "/"),
            reason="Live-local demo chooses the highest-confidence route candidate.",
        )

    if scenario_type == "payload_drift":
        expected_body = dict(contract_hints.get("expected_request_body") or {})
        return PolicyAction(
            action_type="repair_payload",
            target_method=str(failed_request.get("method") or "POST").upper(),
            target_path=str(failed_request.get("path") or "/"),
            body_patch=expected_body,
            reason="Live-local demo replays the expected payload shape from hints.",
        )

    if scenario_type == "auth_drift":
        tenant_alias = str(contract_hints.get("tenant_alias") or "").strip("-")
        tenant_key = f"demo-tenant-key-{tenant_alias}" if tenant_alias else "demo-tenant-key"
        return PolicyAction(
            action_type="repair_auth",
            target_method=str(failed_request.get("method") or "POST").upper(),
            target_path=str(failed_request.get("path") or "/"),
            header_patch={"x-api-key": tenant_key},
            reason="Live-local demo adds the tenant routing header.",
        )

    return PolicyAction(action_type="no_op", reason="No deterministic live-local action available.")


def run_live_local_demo() -> dict[str, Any]:
    env = ReMorphEnvironment(seed=42, execution_mode="live_local")
    results: list[dict[str, Any]] = []
    for scenario in load_live_local_scenarios(seed=42):
        observation = env.reset(scenario_id=scenario.scenario_id, seed=42)
        done = False
        steps: list[dict[str, Any]] = []
        while not done:
            action = choose_action(observation)
            observation, reward, done, info = env.step(action)
            steps.append(
                {
                    "action": action.model_dump(mode="json"),
                    "done": done,
                    "status_code": info["status_code"],
                    "success": info["success"],
                    "raw_reward": info["raw_reward"],
                    "normalized_reward": reward,
                }
            )
        results.append(
            {
                "scenario_id": scenario.scenario_id,
                "workflow_id": scenario.workflow_id,
                "step_count": len(steps),
                "final_success": steps[-1]["success"],
                "steps": steps,
            }
        )
    return {"status": "ok", "scenario_count": len(results), "results": results}


def main() -> None:
    print(json.dumps(run_live_local_demo(), indent=2))


if __name__ == "__main__":
    main()
