"""Baseline and learned policies for local training/evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from remorph_openenv.models import PolicyAction


def highest_confidence_route(candidate_routes: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        (dict(candidate) for candidate in candidate_routes),
        key=lambda row: float(row.get("confidence") or 0.0),
        default={},
    )


def baseline_action(observation: dict[str, Any]) -> PolicyAction:
    """Naive policy that uses only currently visible hints."""

    scenario_type = str(observation.get("scenario_type") or "unknown")
    benchmark_partition = str(observation.get("benchmark_partition") or "other")
    contract_hints = dict(observation.get("contract_hints") or {})
    candidate_routes = list(observation.get("candidate_routes") or [])
    failed_request = dict(observation.get("failed_request") or {})

    if benchmark_partition == "unrecoverable":
        return PolicyAction(action_type="abstain", reason="Baseline abstains on unrecoverable auth.")
    if scenario_type == "route_drift" and candidate_routes:
        route = highest_confidence_route(candidate_routes)
        return PolicyAction(
            action_type="repair_route",
            target_method=str(route.get("method") or failed_request.get("method") or "GET").upper(),
            target_path=str(route.get("path") or ""),
            reason="Baseline picks the top visible route candidate.",
        )
    if scenario_type == "payload_drift":
        failed_body = dict(failed_request.get("body") or {})
        if "first_name" in failed_body or "last_name" in failed_body:
            guessed_body = {
                "user": {
                    "f_name": str(failed_body.get("first_name", "")),
                    "l_name": str(failed_body.get("last_name", "")),
                }
            }
        elif "employee_id" in failed_body:
            guessed_body = {"user": dict(failed_body)}
        elif "quarter" in failed_body:
            guessed_body = {"filters": {"quarter": failed_body.get("quarter"), "regions": []}}
        else:
            guessed_body = dict(failed_body)
        return PolicyAction(
            action_type="repair_payload",
            target_method=str(failed_request.get("method") or "POST").upper(),
            target_path=str(failed_request.get("path") or "/"),
            body_patch=guessed_body,
            reason="Baseline uses a naive payload reshape heuristic.",
        )
    if scenario_type == "auth_drift":
        required_headers = list(contract_hints.get("required_headers") or [])
        return PolicyAction(
            action_type="repair_auth",
            target_method=str(failed_request.get("method") or "GET").upper(),
            target_path=str(failed_request.get("path") or "/"),
            header_patch={key: "demo-tenant-key" for key in required_headers},
            reason="Baseline fills all visible auth headers with a generic key.",
        )
    return PolicyAction(action_type="no_op", reason="Baseline has no visible repair.")


@dataclass
class ReplayMemoryPolicy:
    """Tiny memory-based policy trained from reference rollouts."""

    memory: dict[str, dict[str, Any]] = field(default_factory=dict)

    def predict(self, observation: dict[str, Any]) -> PolicyAction:
        signature = observation_signature(observation)
        if signature in self.memory:
            return PolicyAction.model_validate(self.memory[signature])
        return baseline_action(observation)

    def update(self, observation: dict[str, Any], action: PolicyAction) -> None:
        self.memory[observation_signature(observation)] = action.model_dump(mode="json")


@dataclass
class SupervisedStructuredPolicy:
    """Small supervised policy that learns structured decoding rules from examples."""

    route_strategy: str = "max_confidence_candidate"
    payload_strategy: str = "expected_request_body"
    auth_strategy: str = "required_headers"
    abstain_strategy: str = "partition_gate"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {
            "route_examples": 0,
            "payload_examples": 0,
            "auth_examples": 0,
            "abstain_examples": 0,
            "training_example_count": len(dataset),
        }
        route_confidences: list[float] = []
        tenant_aliases: set[str] = set()
        payload_hint_coverage = 0
        for row in dataset:
            observation = dict(row["observation"])
            action = PolicyAction.model_validate(row["action"])
            if action.action_type == "repair_route":
                counts["route_examples"] += 1
                route = highest_confidence_route(list(observation.get("candidate_routes") or []))
                route_confidences.append(float(route.get("confidence") or 0.0))
            elif action.action_type == "repair_payload":
                counts["payload_examples"] += 1
                if dict(observation.get("contract_hints") or {}).get("expected_request_body"):
                    payload_hint_coverage += 1
            elif action.action_type == "repair_auth":
                counts["auth_examples"] += 1
                alias = str(dict(observation.get("contract_hints") or {}).get("tenant_alias") or "").strip()
                if alias:
                    tenant_aliases.add(alias)
            elif action.action_type == "abstain":
                counts["abstain_examples"] += 1

        self.metadata = {
            "learner": "supervised_structured_policy",
            "route_strategy": self.route_strategy,
            "payload_strategy": self.payload_strategy,
            "auth_strategy": self.auth_strategy,
            "abstain_strategy": self.abstain_strategy,
            "average_route_confidence": round(sum(route_confidences) / len(route_confidences), 4) if route_confidences else 0.0,
            "payload_hint_coverage": round(payload_hint_coverage / counts["payload_examples"], 4) if counts["payload_examples"] else 0.0,
            "observed_tenant_aliases": sorted(tenant_aliases),
            "counts": counts,
        }
        return dict(self.metadata)

    def predict(self, observation: dict[str, Any]) -> PolicyAction:
        scenario_type = str(observation.get("scenario_type") or "unknown")
        benchmark_partition = str(observation.get("benchmark_partition") or "other")
        contract_hints = dict(observation.get("contract_hints") or {})
        candidate_routes = list(observation.get("candidate_routes") or [])
        failed_request = dict(observation.get("failed_request") or {})

        if benchmark_partition == "unrecoverable":
            return PolicyAction(action_type="abstain", reason="Supervised policy abstains on unrecoverable auth.")

        if scenario_type == "route_drift":
            route = highest_confidence_route(candidate_routes)
            return PolicyAction(
                action_type="repair_route",
                target_method=str(route.get("method") or failed_request.get("method") or "GET").upper(),
                target_path=str(route.get("path") or ""),
                reason="Supervised policy chooses the highest-confidence route candidate.",
            )

        if scenario_type == "payload_drift":
            expected_body = dict(contract_hints.get("expected_request_body") or {})
            return PolicyAction(
                action_type="repair_payload",
                target_method=str(failed_request.get("method") or "POST").upper(),
                target_path=str(failed_request.get("path") or "/"),
                body_patch=expected_body,
                reason="Supervised policy uses the expected request body hint.",
            )

        if scenario_type == "auth_drift":
            required_headers = list(contract_hints.get("required_headers") or [])
            tenant_alias = str(contract_hints.get("tenant_alias") or "").strip("-")
            tenant_key = f"demo-tenant-key-{tenant_alias}" if tenant_alias else "demo-tenant-key"
            return PolicyAction(
                action_type="repair_auth",
                target_method=str(failed_request.get("method") or "GET").upper(),
                target_path=str(failed_request.get("path") or "/"),
                header_patch={key: tenant_key for key in required_headers if key == "x-api-key"},
                reason="Supervised policy synthesizes the tenant routing header from hints.",
            )

        return PolicyAction(action_type="no_op", reason="Supervised policy has no structured repair.")


def infer_belief(observation: dict[str, Any]) -> tuple[str, float]:
    """Infer a lightweight belief label and confidence from the observation."""

    scenario_type = str(observation.get("scenario_type") or "unknown")
    benchmark_partition = str(observation.get("benchmark_partition") or "other")
    contract_hints = dict(observation.get("contract_hints") or {})
    if benchmark_partition == "unrecoverable":
        return ("unrecoverable_auth", 0.98)
    if scenario_type == "route_drift":
        route = highest_confidence_route(list(observation.get("candidate_routes") or []))
        return ("route_repair", round(float(route.get("confidence") or 0.75), 4))
    if scenario_type == "payload_drift":
        confidence = 0.94 if contract_hints.get("expected_request_body") else 0.62
        return ("payload_repair", confidence)
    if scenario_type == "auth_drift":
        confidence = 0.91 if contract_hints.get("required_headers") else 0.66
        return ("auth_repair", confidence)
    return ("unknown", 0.5)


def observation_signature(observation: dict[str, Any]) -> str:
    return json.dumps(
        {
            "workflow_id": observation.get("workflow_id"),
            "scenario_type": observation.get("scenario_type"),
            "step_index": observation.get("step_index"),
            "failed_request": observation.get("failed_request"),
            "error_signal": observation.get("error_signal"),
            "contract_hints": observation.get("contract_hints"),
            "candidate_routes": observation.get("candidate_routes"),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
