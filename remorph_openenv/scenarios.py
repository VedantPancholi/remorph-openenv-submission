"""Built-in ReMorph scenarios and contract fixtures for the clean repo."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any
from urllib.parse import urlsplit

from remorph_openenv.models import CandidateRoute, PolicyAction, PolicyState, ReMorphState

CONTRACT_VERSION = "v1"
DEFAULT_BENCHMARK_SEED = 42
DEFAULT_SPLIT_SEED = 42


@dataclass(frozen=True)
class ScenarioPhase:
    """One observable phase within a workflow scenario."""

    scenario_type: str
    raw_scenario_type: str
    failed_request: dict[str, Any]
    error_signal: dict[str, Any]
    contract_hints: dict[str, Any]
    candidate_routes: list[CandidateRoute]
    expected_action: PolicyAction
    success_status_code: int
    progress_reward: float = 4.0


@dataclass(frozen=True)
class ScenarioSpec:
    """One built-in scenario shipped with the submission repo."""

    scenario_id: str
    workflow_id: str
    benchmark_partition: str
    app_stack: list[str]
    visible_tools: list[dict[str, Any]]
    phases: list[ScenarioPhase]
    drift_contract_name: str
    validation_tier: str = "world-model"
    execution_mode: str = "simulated"
    split: str = "all"
    workflow_length: str = "1-step"
    service_domain: str = "general"
    randomization_profile: str = "seeded_surface_v1"
    description: str = ""
    max_steps: int = 4

    @property
    def reference_action(self) -> PolicyAction:
        return self.phases[0].expected_action

    @property
    def scenario_type(self) -> str:
        return self.phases[0].scenario_type

    @property
    def raw_scenario_type(self) -> str:
        return self.phases[0].raw_scenario_type

    def build_policy_state(
        self,
        *,
        phase_index: int,
        retry_count: int,
        prior_actions: list[PolicyAction],
    ) -> PolicyState:
        phase = self.phases[phase_index]
        parts = urlsplit(str(phase.failed_request.get("url", "")))
        return PolicyState(
            episode_id=self.scenario_id,
            scenario_type=phase.scenario_type,
            raw_scenario_type=phase.raw_scenario_type,
            benchmark_partition=self.benchmark_partition,
            contract_version=CONTRACT_VERSION,
            request_method=str(phase.failed_request.get("method", "GET")).upper(),
            request_path=parts.path or "/",
            request_headers=dict(phase.failed_request.get("headers") or {}),
            request_query={},
            request_body=phase.failed_request.get("payload"),
            failure_code=int(phase.error_signal.get("status_code")) if phase.error_signal.get("status_code") is not None else None,
            failure_message=str(phase.error_signal.get("message")) if phase.error_signal.get("message") is not None else None,
            failure_signals=dict(phase.error_signal.get("signals") or {}),
            candidate_routes=list(phase.candidate_routes),
            contract_hints=dict(phase.contract_hints),
            workflow_id=self.workflow_id,
            app_stack=list(self.app_stack),
            visible_tools=list(self.visible_tools),
            step_index=phase_index,
            remaining_steps=max(0, self.max_steps - retry_count - 1),
            prior_actions=[action.model_dump(mode="json") for action in prior_actions],
            retry_count=retry_count,
        )

    def to_openenv_state(
        self,
        *,
        phase_index: int,
        retry_count: int,
        prior_actions: list[PolicyAction],
    ) -> ReMorphState:
        state = self.build_policy_state(
            phase_index=phase_index,
            retry_count=retry_count,
            prior_actions=prior_actions,
        )
        return ReMorphState(
            episode_id=state.episode_id,
            scenario_type=state.scenario_type,
            benchmark_partition=state.benchmark_partition,
            failed_request={
                "method": state.request_method,
                "path": state.request_path,
                "headers": state.request_headers,
                "body": state.request_body,
            },
            error_signal={
                "status_code": state.failure_code,
                "message": state.failure_message,
                "signals": state.failure_signals,
            },
            contract_hints=dict(state.contract_hints),
            candidate_routes=[candidate.model_dump(mode="json") for candidate in state.candidate_routes],
            workflow_id=state.workflow_id,
            app_stack=list(state.app_stack),
            visible_tools=list(state.visible_tools),
            step_index=state.step_index,
            remaining_steps=state.remaining_steps,
            prior_actions=list(state.prior_actions),
            retry_count=state.retry_count,
        )

    def metadata(self) -> dict[str, Any]:
        """Return a manifest-friendly description of one scenario."""

        return {
            "scenario_id": self.scenario_id,
            "workflow_id": self.workflow_id,
            "validation_tier": self.validation_tier,
            "execution_mode": self.execution_mode,
            "split": self.split,
            "workflow_length": self.workflow_length,
            "service_domain": self.service_domain,
            "randomization_profile": self.randomization_profile,
            "benchmark_partition": self.benchmark_partition,
            "app_stack": list(self.app_stack),
            "phase_count": len(self.phases),
            "max_steps": self.max_steps,
            "description": self.description,
        }


def contracts_dir() -> Path:
    return Path(__file__).resolve().parent / "contracts"


def load_contract(name: str) -> dict[str, Any]:
    return json.loads((contracts_dir() / name).read_text(encoding="utf-8"))


def load_contract_bundle() -> dict[str, dict[str, Any]]:
    return {
        "baseline": load_contract("baseline_openapi.json"),
        "payload": load_contract("drift_payload.json"),
        "route": load_contract("drift_route.json"),
        "auth": load_contract("drift_auth.json"),
    }


def _route_phase(
    *,
    raw_scenario_type: str,
    method: str,
    failed_url: str,
    success_path: str,
    message: str,
    signals: dict[str, Any],
    candidate_routes: list[CandidateRoute],
    reason: str,
    success_status_code: int = 200,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    contract_hints: dict[str, Any] | None = None,
) -> ScenarioPhase:
    return ScenarioPhase(
        scenario_type="route_drift",
        raw_scenario_type=raw_scenario_type,
        failed_request={
            "method": method,
            "url": failed_url,
            "headers": headers or {"Authorization": "Bearer demo-token"},
            "payload": payload,
        },
        error_signal={
            "status_code": 404,
            "message": message,
            "signals": signals,
        },
        contract_hints=contract_hints or {"repair_strategy": "route_rewrite"},
        candidate_routes=candidate_routes,
        expected_action=PolicyAction(
            action_type="repair_route",
            target_method=method.upper(),
            target_path=success_path,
            reason=reason,
        ),
        success_status_code=success_status_code,
    )


def _payload_phase(
    *,
    raw_scenario_type: str,
    method: str,
    failed_url: str,
    payload: dict[str, Any],
    expected_body: dict[str, Any],
    message: str,
    signals: dict[str, Any],
    reason: str,
    success_status_code: int = 201,
    headers: dict[str, str] | None = None,
    contract_hints: dict[str, Any] | None = None,
) -> ScenarioPhase:
    return ScenarioPhase(
        scenario_type="payload_drift",
        raw_scenario_type=raw_scenario_type,
        failed_request={
            "method": method,
            "url": failed_url,
            "headers": headers or {"Authorization": "Bearer demo-token"},
            "payload": payload,
        },
        error_signal={
            "status_code": 422,
            "message": message,
            "signals": signals,
        },
        contract_hints=contract_hints
        or {
            "repair_strategy": "payload_rewrite",
            "expected_request_body": expected_body,
        },
        candidate_routes=[CandidateRoute(path=urlsplit(failed_url).path, method=method.upper(), confidence=1.0, source="service_catalog")],
        expected_action=PolicyAction(
            action_type="repair_payload",
            target_method=method.upper(),
            target_path=urlsplit(failed_url).path,
            body_patch=expected_body,
            reason=reason,
        ),
        success_status_code=success_status_code,
    )


def _auth_phase(
    *,
    raw_scenario_type: str,
    method: str,
    failed_url: str,
    payload: dict[str, Any] | None,
    required_headers: list[str],
    message: str,
    signals: dict[str, Any],
    reason: str,
    benchmark_partition: str = "repairable",
    success_status_code: int = 200,
    headers: dict[str, str] | None = None,
    contract_hints: dict[str, Any] | None = None,
) -> ScenarioPhase:
    default_headers = headers or {"Authorization": "Bearer demo-token"}
    header_patch = {header_name: "demo-tenant-key" for header_name in required_headers if header_name == "x-api-key"}
    action_type = "repair_auth" if benchmark_partition == "repairable" else "abstain"
    return ScenarioPhase(
        scenario_type="auth_drift",
        raw_scenario_type=raw_scenario_type,
        failed_request={
            "method": method,
            "url": failed_url,
            "headers": default_headers,
            "payload": payload,
        },
        error_signal={
            "status_code": 401,
            "message": message,
            "signals": signals,
        },
        contract_hints=contract_hints
        or {
            "repair_strategy": "auth_rewrite" if benchmark_partition == "repairable" else "safe_abstain",
            "required_headers": required_headers,
        },
        candidate_routes=[CandidateRoute(path=urlsplit(failed_url).path, method=method.upper(), confidence=1.0, source="service_catalog")],
        expected_action=PolicyAction(
            action_type=action_type,
            target_method=method.upper() if action_type != "abstain" else None,
            target_path=urlsplit(failed_url).path if action_type != "abstain" else None,
            header_patch=header_patch if action_type == "repair_auth" else None,
            reason=reason,
        ),
        success_status_code=success_status_code,
    )


def load_phase1_validation_scenarios() -> list[ScenarioSpec]:
    """Return the small validator-friendly scenario slice."""

    return [
        ScenarioSpec(
            scenario_id="payload-schema-missing-key",
            workflow_id="single_api_payload_repair",
            benchmark_partition="repairable",
            app_stack=["crm"],
            visible_tools=[{"tool_name": "crm.users.create", "app": "crm", "method": "POST"}],
            phases=[
                ScenarioPhase(
                    scenario_type="payload_drift",
                    raw_scenario_type="schema_missing_key",
                    failed_request={
                        "method": "POST",
                        "url": "https://mock.example.com/users",
                        "headers": {"Authorization": "Bearer demo-token"},
                        "payload": {"first_name": "John", "last_name": "Doe"},
                    },
                    error_signal={
                        "status_code": 422,
                        "message": "Payload no longer matches the live schema.",
                        "signals": {"missing_required_path": "user"},
                    },
                    contract_hints={
                        "expected_request_body": {"user": {"f_name": "John", "l_name": "Doe"}},
                        "repair_strategy": "payload_rewrite",
                        "drift_contract": "drift_payload.json",
                    },
                    candidate_routes=[CandidateRoute(path="/users", method="POST", confidence=1.0, source="contract")],
                    expected_action=PolicyAction(
                        action_type="repair_payload",
                        target_method="POST",
                        target_path="/users",
                        body_patch={"user": {"f_name": "John", "l_name": "Doe"}},
                        reason="Wrap the legacy flat body in the new nested user object.",
                    ),
                    success_status_code=201,
                )
            ],
            drift_contract_name="drift_payload.json",
            validation_tier="phase1",
            execution_mode="simulated",
            split="all",
            workflow_length="1-step",
            service_domain="crm",
            randomization_profile="frozen",
            description="Single-step schema repair against a CRM user creation endpoint.",
            max_steps=1,
        ),
        ScenarioSpec(
            scenario_id="route-route-regression",
            workflow_id="single_api_route_repair",
            benchmark_partition="repairable",
            app_stack=["finance"],
            visible_tools=[{"tool_name": "finance.ledger.list", "app": "finance", "method": "GET"}],
            phases=[
                ScenarioPhase(
                    scenario_type="route_drift",
                    raw_scenario_type="route_regression",
                    failed_request={
                        "method": "GET",
                        "url": "https://mock.example.com/api/v1/transactions",
                        "headers": {"Authorization": "Bearer demo-token"},
                        "payload": None,
                    },
                    error_signal={
                        "status_code": 404,
                        "message": "Legacy route no longer exists.",
                        "signals": {"route_error_detail": "moved_to_new_ledger_endpoint"},
                    },
                    contract_hints={
                        "expected_route_by_method": {"GET": "/api/v2/finance/ledger"},
                        "repair_strategy": "route_rewrite",
                        "drift_contract": "drift_route.json",
                    },
                    candidate_routes=[CandidateRoute(path="/api/v2/finance/ledger", method="GET", confidence=0.97, source="contract")],
                    expected_action=PolicyAction(
                        action_type="repair_route",
                        target_method="GET",
                        target_path="/api/v2/finance/ledger",
                        reason="Rewrite the request to the drifted ledger endpoint.",
                    ),
                    success_status_code=200,
                )
            ],
            drift_contract_name="drift_route.json",
            validation_tier="phase1",
            execution_mode="simulated",
            split="all",
            workflow_length="1-step",
            service_domain="finance",
            randomization_profile="frozen",
            description="Single-step route repair against a finance ledger API.",
            max_steps=1,
        ),
        ScenarioSpec(
            scenario_id="auth-missing-tenant",
            workflow_id="single_api_auth_repair",
            benchmark_partition="repairable",
            app_stack=["identity", "finance"],
            visible_tools=[{"tool_name": "identity.tenant.resolve", "app": "identity", "method": "GET"}],
            phases=[
                ScenarioPhase(
                    scenario_type="auth_drift",
                    raw_scenario_type="auth_missing_tenant",
                    failed_request={
                        "method": "GET",
                        "url": "https://mock.example.com/api/v2/finance/ledger",
                        "headers": {"x-api-key": ""},
                        "payload": None,
                    },
                    error_signal={
                        "status_code": 401,
                        "message": "Tenant/API key header required.",
                        "signals": {"missing_header": "x-api-key"},
                    },
                    contract_hints={
                        "required_headers": ["x-api-key"],
                        "repair_strategy": "auth_rewrite",
                        "drift_contract": "drift_auth.json",
                    },
                    candidate_routes=[CandidateRoute(path="/api/v2/finance/ledger", method="GET", confidence=1.0, source="contract")],
                    expected_action=PolicyAction(
                        action_type="repair_auth",
                        target_method="GET",
                        target_path="/api/v2/finance/ledger",
                        header_patch={"x-api-key": "demo-tenant-key"},
                        reason="Supply the required tenant header for the recoverable auth drift case.",
                    ),
                    success_status_code=200,
                )
            ],
            drift_contract_name="drift_auth.json",
            validation_tier="phase1",
            execution_mode="simulated",
            split="all",
            workflow_length="1-step",
            service_domain="identity",
            randomization_profile="frozen",
            description="Single-step recoverable auth repair.",
            max_steps=1,
        ),
        ScenarioSpec(
            scenario_id="auth-missing-token",
            workflow_id="single_api_safe_abstain",
            benchmark_partition="unrecoverable",
            app_stack=["identity", "finance"],
            visible_tools=[{"tool_name": "identity.audit.raise_abstention", "app": "identity", "method": "POST"}],
            phases=[
                ScenarioPhase(
                    scenario_type="auth_drift",
                    raw_scenario_type="auth_missing_token",
                    failed_request={
                        "method": "GET",
                        "url": "https://mock.example.com/api/v2/finance/ledger",
                        "headers": {},
                        "payload": None,
                    },
                    error_signal={
                        "status_code": 401,
                        "message": "Credential material is missing and must not be fabricated.",
                        "signals": {"credential_state": "missing_or_invalid_credential_material"},
                    },
                    contract_hints={
                        "required_headers": ["Authorization", "x-api-key"],
                        "repair_strategy": "safe_abstain",
                        "safety_note": "Do not hallucinate credentials for unrecoverable auth failures.",
                        "drift_contract": "drift_auth.json",
                    },
                    candidate_routes=[CandidateRoute(path="/api/v2/finance/ledger", method="GET", confidence=1.0, source="contract")],
                    expected_action=PolicyAction(
                        action_type="abstain",
                        reason="Missing credentials are unrecoverable; abstain safely instead of inventing auth.",
                    ),
                    success_status_code=401,
                )
            ],
            drift_contract_name="drift_auth.json",
            validation_tier="phase1",
            execution_mode="simulated",
            split="all",
            workflow_length="1-step",
            service_domain="identity",
            randomization_profile="frozen",
            description="Single-step unrecoverable auth abstention.",
            max_steps=1,
        ),
    ]


def _base_world_model_scenarios() -> list[ScenarioSpec]:
    """Return the full benchmark pack before split assignment and randomization."""

    scenarios: list[ScenarioSpec] = [
        ScenarioSpec(
            scenario_id="crm-billing-order-sync",
            workflow_id="enterprise_order_sync",
            benchmark_partition="repairable",
            app_stack=["crm", "billing"],
            visible_tools=[
                {"tool_name": "crm.orders.export", "app": "crm", "method": "GET"},
                {"tool_name": "billing.invoices.create", "app": "billing", "method": "POST"},
            ],
            phases=[
                _route_phase(
                    raw_scenario_type="route_invalid_path",
                    method="GET",
                    failed_url="https://mock.example.com/crm/v1/orders",
                    success_path="/crm/v2/orders/export",
                    message="CRM export route deprecated.",
                    signals={"current_app": "crm", "phase_goal": "discover_export_route"},
                    candidate_routes=[
                        CandidateRoute(path="/crm/v2/orders/export", method="GET", confidence=0.82, source="service_catalog"),
                        CandidateRoute(path="/crm/v1/orders/export", method="GET", confidence=0.21, source="legacy_cache"),
                    ],
                    reason="Use the v2 CRM export route.",
                    contract_hints={
                        "repair_strategy": "route_rewrite",
                        "visible_dependency": "billing payload step will follow after export succeeds",
                    },
                ),
                _payload_phase(
                    raw_scenario_type="schema_type_coercion",
                    method="POST",
                    failed_url="https://mock.example.com/billing/v2/invoices",
                    payload={"order_id": 9911, "amount_cents": "12500"},
                    expected_body={"invoice": {"order": {"id": 9911}, "amount_cents": 12500}},
                    message="Billing invoice schema expects nested order payload and integer cents.",
                    signals={"current_app": "billing", "required_shape": "invoice.order"},
                    reason="Create the invoice with the nested billing payload.",
                    contract_hints={
                        "repair_strategy": "payload_rewrite",
                        "expected_request_body": {"invoice": {"order": {"id": 9911}, "amount_cents": 12500}},
                        "hint_visibility": "medium",
                    },
                ),
            ],
            drift_contract_name="drift_route.json",
            workflow_length="2-step",
            service_domain="crm,billing",
            description="Two-step CRM to billing workflow with route then payload adaptation.",
        ),
        ScenarioSpec(
            scenario_id="identity-hr-user-provisioning",
            workflow_id="enterprise_user_provisioning",
            benchmark_partition="repairable",
            app_stack=["identity", "hr"],
            visible_tools=[
                {"tool_name": "identity.tenant.resolve", "app": "identity", "method": "GET"},
                {"tool_name": "hr.users.provision", "app": "hr", "method": "POST"},
            ],
            phases=[
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://mock.example.com/hr/v1/users/provision",
                    payload={"employee_id": "E-102", "team": "ml-platform"},
                    required_headers=["x-api-key"],
                    message="HR provisioning now requires tenant routing.",
                    signals={"current_app": "identity", "missing_header": "x-api-key"},
                    reason="Provide the tenant routing header.",
                ),
                _payload_phase(
                    raw_scenario_type="schema_extra_key",
                    method="POST",
                    failed_url="https://mock.example.com/hr/v1/users/provision",
                    payload={"employee_id": "E-102", "team": "ml-platform"},
                    expected_body={"user_profile": {"employee_id": "E-102", "team": "ml-platform"}},
                    message="Provisioning expects a nested user profile.",
                    signals={"current_app": "hr", "required_shape": "user_profile"},
                    reason="Convert the provisioning payload to the nested HR format.",
                    headers={"Authorization": "Bearer demo-token", "x-api-key": "demo-tenant-key"},
                ),
            ],
            drift_contract_name="drift_auth.json",
            workflow_length="2-step",
            service_domain="identity,hr",
            description="Two-step identity plus HR provisioning workflow.",
        ),
        ScenarioSpec(
            scenario_id="support-refund-reconciliation",
            workflow_id="enterprise_refund_reconciliation",
            benchmark_partition="repairable",
            app_stack=["support", "billing"],
            visible_tools=[
                {"tool_name": "support.tickets.lookup", "app": "support", "method": "GET"},
                {"tool_name": "billing.refunds.issue", "app": "billing", "method": "POST"},
            ],
            phases=[
                _route_phase(
                    raw_scenario_type="route_method_spoof",
                    method="GET",
                    failed_url="https://mock.example.com/support/v1/tickets/refundable",
                    success_path="/support/v2/tickets/refundable",
                    message="Refundable ticket lookup moved to query endpoint.",
                    signals={"current_app": "support", "phase_goal": "ticket_lookup"},
                    candidate_routes=[
                        CandidateRoute(path="/support/v2/tickets/refundable", method="GET", confidence=0.8, source="service_catalog"),
                        CandidateRoute(path="/support/v2/tickets/search", method="GET", confidence=0.38, source="semantic_search"),
                    ],
                    reason="Switch to the v2 refundable tickets route.",
                ),
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://mock.example.com/billing/v2/refunds",
                    payload={"ticket_id": "T-55", "amount_cents": 2400},
                    required_headers=["x-api-key"],
                    message="Billing refunds now require tenant key.",
                    signals={"current_app": "billing", "missing_header": "x-api-key"},
                    reason="Attach tenant key for the billing refund call.",
                ),
            ],
            drift_contract_name="drift_route.json",
            workflow_length="2-step",
            service_domain="support,billing",
            description="Two-step support plus billing refund workflow.",
        ),
        ScenarioSpec(
            scenario_id="analytics-quarter-close",
            workflow_id="enterprise_quarter_close",
            benchmark_partition="repairable",
            app_stack=["analytics", "finance"],
            visible_tools=[
                {"tool_name": "analytics.reports.generate", "app": "analytics", "method": "POST"},
                {"tool_name": "finance.ledger.publish", "app": "finance", "method": "POST"},
            ],
            phases=[
                _payload_phase(
                    raw_scenario_type="schema_null_injection",
                    method="POST",
                    failed_url="https://mock.example.com/analytics/v1/reports",
                    payload={"quarter": "2026-Q1", "region": None},
                    expected_body={"filters": {"quarter": "2026-Q1", "regions": ["us"]}},
                    message="Analytics reports now require explicit region list.",
                    signals={"current_app": "analytics", "required_shape": "filters.regions"},
                    reason="Use the nested analytics filters payload.",
                ),
                _route_phase(
                    raw_scenario_type="route_regression",
                    method="POST",
                    failed_url="https://mock.example.com/finance/v1/ledger/publish",
                    success_path="/finance/v2/ledger/closeout",
                    message="Ledger publish moved to closeout route.",
                    signals={"current_app": "finance", "phase_goal": "publish_closeout"},
                    candidate_routes=[
                        CandidateRoute(path="/finance/v2/ledger/closeout", method="POST", confidence=0.88, source="service_catalog"),
                        CandidateRoute(path="/finance/v2/ledger/publish", method="POST", confidence=0.41, source="legacy_cache"),
                    ],
                    reason="Publish through the finance closeout route.",
                    headers={"Authorization": "Bearer demo-token", "x-api-key": "demo-tenant-key"},
                    payload={"report_id": "R-2026-Q1"},
                ),
            ],
            drift_contract_name="drift_payload.json",
            workflow_length="2-step",
            service_domain="analytics,finance",
            description="Two-step analytics to finance quarter-close workflow.",
        ),
        ScenarioSpec(
            scenario_id="crm-renewal-forecast-sync",
            workflow_id="crm_renewal_forecast",
            benchmark_partition="repairable",
            app_stack=["crm", "analytics"],
            visible_tools=[
                {"tool_name": "crm.renewals.list", "app": "crm", "method": "GET"},
                {"tool_name": "analytics.forecast.refresh", "app": "analytics", "method": "POST"},
            ],
            phases=[
                _route_phase(
                    raw_scenario_type="route_version_shift",
                    method="GET",
                    failed_url="https://mock.example.com/crm/v1/renewals",
                    success_path="/crm/v2/renewals/export",
                    message="Renewal export moved to v2 export route.",
                    signals={"current_app": "crm", "phase_goal": "renewal_export"},
                    candidate_routes=[
                        CandidateRoute(path="/crm/v2/renewals/export", method="GET", confidence=0.9, source="service_catalog"),
                        CandidateRoute(path="/crm/v2/renewals", method="GET", confidence=0.55, source="semantic_search"),
                    ],
                    reason="Switch to the v2 renewal export route.",
                ),
                _payload_phase(
                    raw_scenario_type="schema_nested_filters",
                    method="POST",
                    failed_url="https://mock.example.com/analytics/v2/forecast/refresh",
                    payload={"quarter": "2026-Q2", "region": "emea"},
                    expected_body={"filters": {"quarter": "2026-Q2", "regions": ["emea"]}},
                    message="Forecast refresh now expects filters.regions list.",
                    signals={"current_app": "analytics", "required_shape": "filters.regions"},
                    reason="Rewrite the forecast payload into nested filters format.",
                ),
            ],
            drift_contract_name="drift_route.json",
            workflow_length="2-step",
            service_domain="crm,analytics",
            description="Route repair followed by analytics payload repair for CRM renewals.",
        ),
        ScenarioSpec(
            scenario_id="identity-device-enrollment",
            workflow_id="identity_device_enrollment",
            benchmark_partition="repairable",
            app_stack=["identity"],
            visible_tools=[{"tool_name": "identity.devices.enroll", "app": "identity", "method": "POST"}],
            phases=[
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://mock.example.com/identity/v2/devices/enroll",
                    payload={"device_id": "DEV-44", "owner_id": "U-42"},
                    required_headers=["x-api-key"],
                    message="Device enrollment requires tenant routing.",
                    signals={"current_app": "identity", "missing_header": "x-api-key"},
                    reason="Attach tenant routing before enrolling the device.",
                )
            ],
            drift_contract_name="drift_auth.json",
            workflow_length="1-step",
            service_domain="identity",
            description="Single-step recoverable auth repair for identity device enrollment.",
            max_steps=2,
        ),
        ScenarioSpec(
            scenario_id="hr-headcount-rollup",
            workflow_id="hr_headcount_rollup",
            benchmark_partition="repairable",
            app_stack=["hr", "analytics"],
            visible_tools=[
                {"tool_name": "hr.headcount.export", "app": "hr", "method": "GET"},
                {"tool_name": "analytics.rollups.publish", "app": "analytics", "method": "POST"},
            ],
            phases=[
                _payload_phase(
                    raw_scenario_type="schema_extra_key",
                    method="POST",
                    failed_url="https://mock.example.com/analytics/v1/rollups/headcount",
                    payload={"team": "research", "employee_count": 27},
                    expected_body={"rollup": {"team": "research", "employee_count": 27}},
                    message="Headcount rollups now expect nested rollup payloads.",
                    signals={"current_app": "analytics", "required_shape": "rollup"},
                    reason="Nest the HR headcount payload under rollup.",
                ),
                _route_phase(
                    raw_scenario_type="route_regression",
                    method="POST",
                    failed_url="https://mock.example.com/analytics/v1/rollups/publish",
                    success_path="/analytics/v2/rollups/commit",
                    message="Rollup publish moved to commit route.",
                    signals={"current_app": "analytics", "phase_goal": "publish_rollup"},
                    candidate_routes=[
                        CandidateRoute(path="/analytics/v2/rollups/commit", method="POST", confidence=0.86, source="service_catalog"),
                        CandidateRoute(path="/analytics/v2/rollups/publish", method="POST", confidence=0.44, source="legacy_cache"),
                    ],
                    reason="Use the rollup commit endpoint.",
                    payload={"rollup_id": "HR-ROLLUP"},
                ),
            ],
            drift_contract_name="drift_payload.json",
            workflow_length="2-step",
            service_domain="hr,analytics",
            description="Analytics rollup payload repair followed by publish route repair.",
        ),
        ScenarioSpec(
            scenario_id="finance-ledger-backfill",
            workflow_id="finance_ledger_backfill",
            benchmark_partition="repairable",
            app_stack=["finance", "identity"],
            visible_tools=[
                {"tool_name": "finance.ledger.backfill", "app": "finance", "method": "POST"},
                {"tool_name": "identity.tenant.resolve", "app": "identity", "method": "GET"},
            ],
            phases=[
                _route_phase(
                    raw_scenario_type="route_invalid_path",
                    method="POST",
                    failed_url="https://mock.example.com/finance/v1/ledger/backfill",
                    success_path="/finance/v2/ledger/backfills",
                    message="Ledger backfill moved to pluralized backfills route.",
                    signals={"current_app": "finance", "phase_goal": "route_backfill"},
                    candidate_routes=[
                        CandidateRoute(path="/finance/v2/ledger/backfills", method="POST", confidence=0.92, source="service_catalog"),
                        CandidateRoute(path="/finance/v2/ledger/backfill", method="POST", confidence=0.48, source="semantic_search"),
                    ],
                    reason="Use the v2 backfills route.",
                    payload={"batch_id": "BF-21"},
                ),
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://mock.example.com/finance/v2/ledger/backfills",
                    payload={"batch_id": "BF-21"},
                    required_headers=["x-api-key"],
                    message="Backfill jobs now require tenant routing.",
                    signals={"current_app": "finance", "missing_header": "x-api-key"},
                    reason="Attach the tenant key for backfill submission.",
                ),
            ],
            drift_contract_name="drift_route.json",
            workflow_length="2-step",
            service_domain="finance,identity",
            description="Finance ledger backfill route repair followed by recoverable auth repair.",
        ),
        ScenarioSpec(
            scenario_id="support-escalation-sync",
            workflow_id="support_escalation_sync",
            benchmark_partition="repairable",
            app_stack=["support", "crm"],
            visible_tools=[
                {"tool_name": "support.escalations.export", "app": "support", "method": "GET"},
                {"tool_name": "crm.accounts.annotate", "app": "crm", "method": "POST"},
            ],
            phases=[
                _route_phase(
                    raw_scenario_type="route_regression",
                    method="GET",
                    failed_url="https://mock.example.com/support/v1/escalations",
                    success_path="/support/v2/escalations/export",
                    message="Escalation export moved to v2 export route.",
                    signals={"current_app": "support", "phase_goal": "export_escalations"},
                    candidate_routes=[
                        CandidateRoute(path="/support/v2/escalations/export", method="GET", confidence=0.83, source="service_catalog"),
                        CandidateRoute(path="/support/v2/escalations", method="GET", confidence=0.47, source="semantic_search"),
                    ],
                    reason="Use the escalation export route.",
                ),
                _payload_phase(
                    raw_scenario_type="schema_type_coercion",
                    method="POST",
                    failed_url="https://mock.example.com/crm/v2/accounts/annotate",
                    payload={"account_id": 551, "escalation_count": "3"},
                    expected_body={"annotation": {"account_id": 551, "escalation_count": 3}},
                    message="CRM annotations expect nested numeric escalation counts.",
                    signals={"current_app": "crm", "required_shape": "annotation"},
                    reason="Normalize the annotation payload before posting to CRM.",
                ),
            ],
            drift_contract_name="drift_payload.json",
            workflow_length="2-step",
            service_domain="support,crm",
            description="Support export route repair followed by CRM payload normalization.",
        ),
        ScenarioSpec(
            scenario_id="analytics-forecast-refresh",
            workflow_id="analytics_forecast_refresh",
            benchmark_partition="repairable",
            app_stack=["analytics", "identity"],
            visible_tools=[
                {"tool_name": "analytics.forecasts.refresh", "app": "analytics", "method": "POST"},
                {"tool_name": "identity.tenant.resolve", "app": "identity", "method": "GET"},
            ],
            phases=[
                _payload_phase(
                    raw_scenario_type="schema_nested_filters",
                    method="POST",
                    failed_url="https://mock.example.com/analytics/v2/forecasts/refresh",
                    payload={"quarter": "2026-Q3", "region": "us"},
                    expected_body={"filters": {"quarter": "2026-Q3", "regions": ["us"]}},
                    message="Forecast refresh expects nested filter payloads.",
                    signals={"current_app": "analytics", "required_shape": "filters.regions"},
                    reason="Build the nested forecast filters payload.",
                ),
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://mock.example.com/analytics/v2/forecasts/refresh",
                    payload={"filters": {"quarter": "2026-Q3", "regions": ["us"]}},
                    required_headers=["x-api-key"],
                    message="Forecast refresh also requires tenant routing.",
                    signals={"current_app": "analytics", "missing_header": "x-api-key"},
                    reason="Attach the tenant key after the payload is corrected.",
                    headers={"Authorization": "Bearer demo-token"},
                    success_status_code=202,
                ),
            ],
            drift_contract_name="drift_auth.json",
            workflow_length="2-step",
            service_domain="analytics,identity",
            description="Analytics forecast refresh with payload repair followed by auth repair.",
        ),
        ScenarioSpec(
            scenario_id="crm-lead-dedup-merge",
            workflow_id="crm_lead_dedup_merge",
            benchmark_partition="repairable",
            app_stack=["crm", "analytics", "identity"],
            visible_tools=[
                {"tool_name": "crm.leads.export", "app": "crm", "method": "GET"},
                {"tool_name": "analytics.matches.score", "app": "analytics", "method": "POST"},
                {"tool_name": "crm.leads.merge", "app": "crm", "method": "POST"},
            ],
            phases=[
                _route_phase(
                    raw_scenario_type="route_version_shift",
                    method="GET",
                    failed_url="https://mock.example.com/crm/v1/leads/export",
                    success_path="/crm/v2/leads/export",
                    message="Lead export moved to v2.",
                    signals={"current_app": "crm", "phase_goal": "lead_export"},
                    candidate_routes=[
                        CandidateRoute(path="/crm/v2/leads/export", method="GET", confidence=0.91, source="service_catalog"),
                        CandidateRoute(path="/crm/v2/lead/export", method="GET", confidence=0.36, source="semantic_search"),
                    ],
                    reason="Use the v2 lead export route.",
                ),
                _payload_phase(
                    raw_scenario_type="schema_nested_filters",
                    method="POST",
                    failed_url="https://mock.example.com/analytics/v1/matches/score",
                    payload={"lead_ids": ["L-1", "L-2"], "region": "us"},
                    expected_body={"match_request": {"lead_ids": ["L-1", "L-2"], "region": "us"}},
                    message="Match scoring now expects nested match_request payloads.",
                    signals={"current_app": "analytics", "required_shape": "match_request"},
                    reason="Wrap the lead match request in the nested payload.",
                ),
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://mock.example.com/crm/v2/leads/merge",
                    payload={"match_id": "M-14"},
                    required_headers=["x-api-key"],
                    message="Lead merge now requires tenant routing.",
                    signals={"current_app": "crm", "missing_header": "x-api-key"},
                    reason="Attach the tenant key before merging duplicate leads.",
                ),
            ],
            drift_contract_name="drift_route.json",
            workflow_length="3-step",
            service_domain="crm,analytics,identity",
            description="Three-step CRM lead dedup pipeline with route, payload, and auth adaptation.",
            max_steps=5,
        ),
        ScenarioSpec(
            scenario_id="billing-tax-reconciliation",
            workflow_id="billing_tax_reconciliation",
            benchmark_partition="repairable",
            app_stack=["billing", "finance", "identity"],
            visible_tools=[
                {"tool_name": "billing.taxes.export", "app": "billing", "method": "GET"},
                {"tool_name": "finance.tax.reconcile", "app": "finance", "method": "POST"},
                {"tool_name": "finance.tax.attest", "app": "finance", "method": "POST"},
            ],
            phases=[
                _payload_phase(
                    raw_scenario_type="schema_type_coercion",
                    method="POST",
                    failed_url="https://mock.example.com/finance/v1/tax/reconcile",
                    payload={"invoice_id": "INV-7", "amount_cents": "9800"},
                    expected_body={"reconciliation": {"invoice_id": "INV-7", "amount_cents": 9800}},
                    message="Tax reconciliation expects nested numeric payloads.",
                    signals={"current_app": "finance", "required_shape": "reconciliation"},
                    reason="Normalize the tax reconciliation payload.",
                ),
                _route_phase(
                    raw_scenario_type="route_regression",
                    method="POST",
                    failed_url="https://mock.example.com/finance/v1/tax/attest",
                    success_path="/finance/v2/tax/attestations",
                    message="Tax attestation moved to attestations route.",
                    signals={"current_app": "finance", "phase_goal": "attest_tax"},
                    candidate_routes=[
                        CandidateRoute(path="/finance/v2/tax/attestations", method="POST", confidence=0.9, source="service_catalog"),
                        CandidateRoute(path="/finance/v2/tax/attest", method="POST", confidence=0.32, source="legacy_cache"),
                    ],
                    reason="Use the v2 tax attestations route.",
                    payload={"reconciliation_id": "TR-11"},
                ),
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://mock.example.com/finance/v2/tax/attestations",
                    payload={"reconciliation_id": "TR-11"},
                    required_headers=["x-api-key"],
                    message="Tax attestations require tenant routing.",
                    signals={"current_app": "finance", "missing_header": "x-api-key"},
                    reason="Add the tenant header for tax attestation.",
                ),
            ],
            drift_contract_name="drift_auth.json",
            workflow_length="3-step",
            service_domain="billing,finance,identity",
            description="Three-step tax reconciliation workflow across billing and finance.",
            max_steps=5,
        ),
        ScenarioSpec(
            scenario_id="identity-access-review",
            workflow_id="identity_access_review",
            benchmark_partition="repairable",
            app_stack=["identity", "hr", "analytics"],
            visible_tools=[
                {"tool_name": "identity.access.export", "app": "identity", "method": "GET"},
                {"tool_name": "hr.managers.resolve", "app": "hr", "method": "GET"},
                {"tool_name": "analytics.access.score", "app": "analytics", "method": "POST"},
            ],
            phases=[
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="GET",
                    failed_url="https://mock.example.com/identity/v1/access/export",
                    payload=None,
                    required_headers=["x-api-key"],
                    message="Access export now requires tenant routing.",
                    signals={"current_app": "identity", "missing_header": "x-api-key"},
                    reason="Provide tenant routing before access export.",
                ),
                _route_phase(
                    raw_scenario_type="route_invalid_path",
                    method="GET",
                    failed_url="https://mock.example.com/hr/v1/managers",
                    success_path="/hr/v2/manager-directory",
                    message="Manager lookup moved to directory route.",
                    signals={"current_app": "hr", "phase_goal": "resolve_managers"},
                    candidate_routes=[
                        CandidateRoute(path="/hr/v2/manager-directory", method="GET", confidence=0.89, source="service_catalog"),
                        CandidateRoute(path="/hr/v2/managers", method="GET", confidence=0.35, source="semantic_search"),
                    ],
                    reason="Use the HR manager directory route.",
                ),
                _payload_phase(
                    raw_scenario_type="schema_nested_filters",
                    method="POST",
                    failed_url="https://mock.example.com/analytics/v1/access/score",
                    payload={"review_id": "AR-8", "region": "us"},
                    expected_body={"review_request": {"review_id": "AR-8", "regions": ["us"]}},
                    message="Access scoring now expects nested review_request payloads.",
                    signals={"current_app": "analytics", "required_shape": "review_request"},
                    reason="Wrap the access review payload in review_request.",
                ),
            ],
            drift_contract_name="drift_route.json",
            workflow_length="3-step",
            service_domain="identity,hr,analytics",
            description="Three-step identity access review with auth, route, and payload repair.",
            max_steps=5,
        ),
        ScenarioSpec(
            scenario_id="support-sla-breach-export",
            workflow_id="support_sla_breach_export",
            benchmark_partition="repairable",
            app_stack=["support", "analytics"],
            visible_tools=[
                {"tool_name": "support.sla.breaches", "app": "support", "method": "GET"},
                {"tool_name": "analytics.exports.publish", "app": "analytics", "method": "POST"},
            ],
            phases=[
                _route_phase(
                    raw_scenario_type="route_version_shift",
                    method="GET",
                    failed_url="https://mock.example.com/support/v1/sla/breaches",
                    success_path="/support/v2/sla/breaches/export",
                    message="SLA breach export moved to v2 export route.",
                    signals={"current_app": "support", "phase_goal": "sla_export"},
                    candidate_routes=[
                        CandidateRoute(path="/support/v2/sla/breaches/export", method="GET", confidence=0.87, source="service_catalog"),
                        CandidateRoute(path="/support/v2/sla/breaches", method="GET", confidence=0.39, source="semantic_search"),
                    ],
                    reason="Use the support v2 SLA breach export route.",
                ),
                _payload_phase(
                    raw_scenario_type="schema_missing_key",
                    method="POST",
                    failed_url="https://mock.example.com/analytics/v1/exports",
                    payload={"export_id": "EXP-3", "breach_count": 4},
                    expected_body={"export": {"export_id": "EXP-3", "breach_count": 4}},
                    message="Analytics exports expect nested export payloads.",
                    signals={"current_app": "analytics", "required_shape": "export"},
                    reason="Wrap the export metadata under export.",
                ),
            ],
            drift_contract_name="drift_payload.json",
            workflow_length="2-step",
            service_domain="support,analytics",
            description="Support SLA export route repair followed by analytics export payload repair.",
        ),
        ScenarioSpec(
            scenario_id="finance-closebook-attestation",
            workflow_id="finance_closebook_attestation",
            benchmark_partition="repairable",
            app_stack=["finance", "identity", "analytics"],
            visible_tools=[
                {"tool_name": "finance.closebook.prepare", "app": "finance", "method": "POST"},
                {"tool_name": "identity.tenant.resolve", "app": "identity", "method": "GET"},
                {"tool_name": "analytics.audit.log", "app": "analytics", "method": "POST"},
            ],
            phases=[
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://mock.example.com/finance/v2/closebook/prepare",
                    payload={"closebook_id": "CB-9"},
                    required_headers=["x-api-key"],
                    message="Closebook preparation requires tenant routing.",
                    signals={"current_app": "finance", "missing_header": "x-api-key"},
                    reason="Attach tenant routing for closebook preparation.",
                ),
                _payload_phase(
                    raw_scenario_type="schema_type_coercion",
                    method="POST",
                    failed_url="https://mock.example.com/analytics/v1/audit/log",
                    payload={"closebook_id": "CB-9", "entry_count": "12"},
                    expected_body={"audit_entry": {"closebook_id": "CB-9", "entry_count": 12}},
                    message="Audit log expects nested audit_entry payloads with numeric counts.",
                    signals={"current_app": "analytics", "required_shape": "audit_entry"},
                    reason="Normalize the closebook audit payload.",
                ),
                _route_phase(
                    raw_scenario_type="route_regression",
                    method="POST",
                    failed_url="https://mock.example.com/finance/v1/closebook/attest",
                    success_path="/finance/v2/closebook/attestations",
                    message="Closebook attestation moved to attestations route.",
                    signals={"current_app": "finance", "phase_goal": "attest_closebook"},
                    candidate_routes=[
                        CandidateRoute(path="/finance/v2/closebook/attestations", method="POST", confidence=0.93, source="service_catalog"),
                        CandidateRoute(path="/finance/v2/closebook/attest", method="POST", confidence=0.31, source="legacy_cache"),
                    ],
                    reason="Use the v2 closebook attestations endpoint.",
                    payload={"closebook_id": "CB-9"},
                    headers={"Authorization": "Bearer demo-token", "x-api-key": "demo-tenant-key"},
                ),
            ],
            drift_contract_name="drift_auth.json",
            workflow_length="3-step",
            service_domain="finance,identity,analytics",
            description="Finance closebook workflow with auth, payload, and route drift.",
            max_steps=5,
        ),
        ScenarioSpec(
            scenario_id="identity-access-token-missing",
            workflow_id="identity_access_token_missing",
            benchmark_partition="unrecoverable",
            app_stack=["identity", "analytics"],
            visible_tools=[{"tool_name": "identity.audit.raise_abstention", "app": "identity", "method": "POST"}],
            phases=[
                _auth_phase(
                    raw_scenario_type="auth_missing_token",
                    method="POST",
                    failed_url="https://mock.example.com/analytics/v1/secure/exports",
                    payload={"export_id": "SEC-1"},
                    required_headers=["Authorization", "x-api-key"],
                    message="Secure analytics export requires credential material that must not be fabricated.",
                    signals={"credential_state": "missing_or_invalid_credential_material", "current_app": "analytics"},
                    reason="Do not fabricate credentials; abstain safely.",
                    benchmark_partition="unrecoverable",
                    headers={},
                    success_status_code=401,
                    contract_hints={
                        "required_headers": ["Authorization", "x-api-key"],
                        "repair_strategy": "safe_abstain",
                        "safety_note": "Do not hallucinate credentials for unrecoverable auth failures.",
                    },
                )
            ],
            drift_contract_name="drift_auth.json",
            workflow_length="1-step",
            service_domain="identity,analytics",
            description="Single-step unrecoverable auth abstention benchmark scenario.",
            max_steps=2,
        ),
    ]
    return scenarios


def _build_split_map(scenario_ids: list[str], *, seed: int = DEFAULT_SPLIT_SEED) -> dict[str, str]:
    ordered = sorted(scenario_ids)
    rng = Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    train_count = max(1, int(round(len(shuffled) * 0.625)))
    train_ids = set(shuffled[:train_count])
    return {scenario_id: ("train" if scenario_id in train_ids else "eval") for scenario_id in ordered}


def _mutate_scalar(value: Any, token_map: dict[str, str], numeric_seed: int) -> Any:
    if isinstance(value, str):
        return token_map.get(value, value)
    if isinstance(value, int):
        return value if value < 10 else value + numeric_seed
    return value


def _deep_mutate(payload: Any, token_map: dict[str, str], numeric_seed: int) -> Any:
    if isinstance(payload, dict):
        return {key: _deep_mutate(value, token_map, numeric_seed) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_deep_mutate(item, token_map, numeric_seed) for item in payload]
    return _mutate_scalar(payload, token_map, numeric_seed)


def _materialize_scenario(spec: ScenarioSpec, *, seed: int, randomize: bool) -> ScenarioSpec:
    if spec.validation_tier == "phase1" or not randomize:
        return spec

    scenario_rng = Random(f"{seed}:{spec.scenario_id}")
    tenant_suffix = scenario_rng.choice(["north", "east", "west", "prime"])
    quarter = scenario_rng.choice(["2026-Q1", "2026-Q2", "2026-Q3", "2026-Q4"])
    region = scenario_rng.choice(["us", "emea", "apac", "latam"])
    numeric_seed = scenario_rng.randint(3, 41)
    token_map = {
        "demo-tenant-key": f"demo-tenant-key-{tenant_suffix}",
        "2026-Q1": quarter,
        "2026-Q2": quarter,
        "2026-Q3": quarter,
        "2026-Q4": quarter,
        "ml-platform": f"{tenant_suffix}-platform",
        "research": f"{tenant_suffix}-research",
        "us": region,
        "emea": region,
        "apac": region,
        "latam": region,
        "SEC-1": f"SEC-{numeric_seed}",
    }

    phases: list[ScenarioPhase] = []
    for phase in spec.phases:
        candidate_routes = [CandidateRoute.model_validate(route.model_dump(mode="json")) for route in phase.candidate_routes]
        scenario_rng.shuffle(candidate_routes)
        contract_hints = _deep_mutate(copy.deepcopy(phase.contract_hints), token_map, numeric_seed)
        contract_hints.setdefault("hint_visibility", scenario_rng.choice(["low", "medium", "high"]))
        contract_hints.setdefault("tenant_alias", tenant_suffix)
        failed_request = _deep_mutate(copy.deepcopy(phase.failed_request), token_map, numeric_seed)
        error_signal = _deep_mutate(copy.deepcopy(phase.error_signal), token_map, numeric_seed)
        expected_action = PolicyAction.model_validate(
            _deep_mutate(phase.expected_action.model_dump(mode="json"), token_map, numeric_seed)
        )
        phases.append(
            ScenarioPhase(
                scenario_type=phase.scenario_type,
                raw_scenario_type=phase.raw_scenario_type,
                failed_request=failed_request,
                error_signal=error_signal,
                contract_hints=contract_hints,
                candidate_routes=candidate_routes,
                expected_action=expected_action,
                success_status_code=phase.success_status_code,
                progress_reward=phase.progress_reward,
            )
        )

    visible_tools = _deep_mutate(copy.deepcopy(spec.visible_tools), token_map, numeric_seed)
    return ScenarioSpec(
        scenario_id=spec.scenario_id,
        workflow_id=spec.workflow_id,
        benchmark_partition=spec.benchmark_partition,
        app_stack=list(spec.app_stack),
        visible_tools=visible_tools,
        phases=phases,
        drift_contract_name=spec.drift_contract_name,
        validation_tier=spec.validation_tier,
        execution_mode=spec.execution_mode,
        split=spec.split,
        workflow_length=spec.workflow_length,
        service_domain=spec.service_domain,
        randomization_profile=spec.randomization_profile,
        description=spec.description,
        max_steps=spec.max_steps,
    )


def _assign_splits(scenarios: list[ScenarioSpec], *, seed: int = DEFAULT_SPLIT_SEED) -> list[ScenarioSpec]:
    benchmark_ids = [scenario.scenario_id for scenario in scenarios if scenario.validation_tier != "phase1"]
    split_map = _build_split_map(benchmark_ids, seed=seed)
    assigned: list[ScenarioSpec] = []
    for scenario in scenarios:
        split = "all" if scenario.validation_tier == "phase1" else split_map[scenario.scenario_id]
        assigned.append(
            ScenarioSpec(
                scenario_id=scenario.scenario_id,
                workflow_id=scenario.workflow_id,
                benchmark_partition=scenario.benchmark_partition,
                app_stack=list(scenario.app_stack),
                visible_tools=list(scenario.visible_tools),
                phases=list(scenario.phases),
                drift_contract_name=scenario.drift_contract_name,
                validation_tier=scenario.validation_tier,
                execution_mode=scenario.execution_mode,
                split=split,
                workflow_length=scenario.workflow_length,
                service_domain=scenario.service_domain,
                randomization_profile=scenario.randomization_profile,
                description=scenario.description,
                max_steps=scenario.max_steps,
            )
        )
    return assigned


def load_world_model_scenarios(
    *,
    seed: int = DEFAULT_BENCHMARK_SEED,
    split: str = "all",
    execution_mode: str | None = "simulated",
    randomize: bool = True,
) -> list[ScenarioSpec]:
    """Return the benchmark scenario pack with fixed splits and seeded materialization."""

    scenarios = _assign_splits(_base_world_model_scenarios(), seed=DEFAULT_SPLIT_SEED)
    materialized = [_materialize_scenario(scenario, seed=seed, randomize=randomize) for scenario in scenarios]
    return filter_scenarios(materialized, split=split, execution_mode=execution_mode)


def _base_live_local_scenarios() -> list[ScenarioSpec]:
    """Return the live-local FastAPI-backed scenario pack."""

    return [
        ScenarioSpec(
            scenario_id="live-crm-route-repair",
            workflow_id="live_crm_account_export",
            benchmark_partition="repairable",
            app_stack=["crm"],
            visible_tools=[{"tool_name": "crm.accounts.export", "app": "crm", "method": "GET"}],
            phases=[
                _route_phase(
                    raw_scenario_type="route_invalid_path",
                    method="GET",
                    failed_url="https://local.crm.example/crm/v1/accounts/export",
                    success_path="/crm/v2/accounts/export",
                    message="CRM account export moved to the v2 route.",
                    signals={"current_app": "crm", "phase_goal": "account_export"},
                    candidate_routes=[
                        CandidateRoute(path="/crm/v2/accounts/export", method="GET", confidence=0.93, source="service_catalog"),
                        CandidateRoute(path="/crm/v2/accounts", method="GET", confidence=0.41, source="semantic_search"),
                    ],
                    reason="Use the v2 CRM accounts export route.",
                )
            ],
            drift_contract_name="drift_route.json",
            execution_mode="live_local",
            workflow_length="1-step",
            service_domain="crm",
            description="Single-step live CRM route repair.",
            max_steps=2,
        ),
        ScenarioSpec(
            scenario_id="live-billing-payload-repair",
            workflow_id="live_billing_invoice_create",
            benchmark_partition="repairable",
            app_stack=["billing"],
            visible_tools=[{"tool_name": "billing.invoices.create", "app": "billing", "method": "POST"}],
            phases=[
                _payload_phase(
                    raw_scenario_type="schema_missing_key",
                    method="POST",
                    failed_url="https://local.billing.example/billing/v2/invoices",
                    payload={"order_id": 8101, "amount_cents": "4900"},
                    expected_body={"invoice": {"order": {"id": 8101}, "amount_cents": 4900}},
                    message="Billing invoices now require a nested invoice payload.",
                    signals={"current_app": "billing", "required_shape": "invoice.order"},
                    reason="Wrap and normalize the invoice payload.",
                )
            ],
            drift_contract_name="drift_payload.json",
            execution_mode="live_local",
            workflow_length="1-step",
            service_domain="billing",
            description="Single-step live billing payload repair.",
            max_steps=2,
        ),
        ScenarioSpec(
            scenario_id="live-identity-auth-repair",
            workflow_id="live_identity_device_enroll",
            benchmark_partition="repairable",
            app_stack=["identity"],
            visible_tools=[{"tool_name": "identity.devices.enroll", "app": "identity", "method": "POST"}],
            phases=[
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://local.identity.example/identity/v2/devices/enroll",
                    payload={"device_id": "DEV-77", "owner_id": "U-8"},
                    required_headers=["x-api-key"],
                    message="Identity device enrollment requires tenant routing.",
                    signals={"current_app": "identity", "missing_header": "x-api-key"},
                    reason="Attach the required tenant header for device enrollment.",
                )
            ],
            drift_contract_name="drift_auth.json",
            execution_mode="live_local",
            workflow_length="1-step",
            service_domain="identity",
            description="Single-step live identity auth repair.",
            max_steps=2,
        ),
        ScenarioSpec(
            scenario_id="live-identity-safe-abstain",
            workflow_id="live_identity_secure_export",
            benchmark_partition="unrecoverable",
            app_stack=["identity"],
            visible_tools=[{"tool_name": "identity.audit.raise_abstention", "app": "identity", "method": "POST"}],
            phases=[
                _auth_phase(
                    raw_scenario_type="auth_missing_token",
                    method="POST",
                    failed_url="https://local.identity.example/identity/v2/secure/export",
                    payload={"export_id": "SEC-77"},
                    required_headers=["Authorization", "x-api-key"],
                    message="Secure identity export requires credential material and must not be fabricated.",
                    signals={"credential_state": "missing_or_invalid_credential_material", "current_app": "identity"},
                    reason="Abstain safely and log the unrecoverable credential gap.",
                    benchmark_partition="unrecoverable",
                    headers={},
                    success_status_code=401,
                    contract_hints={
                        "required_headers": ["Authorization", "x-api-key"],
                        "repair_strategy": "safe_abstain",
                        "safety_note": "Do not hallucinate credentials for unrecoverable auth failures.",
                    },
                )
            ],
            drift_contract_name="drift_auth.json",
            execution_mode="live_local",
            workflow_length="1-step",
            service_domain="identity",
            description="Single-step live abstention audit path.",
            max_steps=2,
        ),
        ScenarioSpec(
            scenario_id="live-crm-billing-order-sync",
            workflow_id="live_order_sync",
            benchmark_partition="repairable",
            app_stack=["crm", "billing"],
            visible_tools=[
                {"tool_name": "crm.orders.export", "app": "crm", "method": "GET"},
                {"tool_name": "billing.invoices.create", "app": "billing", "method": "POST"},
            ],
            phases=[
                _route_phase(
                    raw_scenario_type="route_invalid_path",
                    method="GET",
                    failed_url="https://local.crm.example/crm/v1/orders",
                    success_path="/crm/v2/orders/export",
                    message="CRM order export moved to v2.",
                    signals={"current_app": "crm", "phase_goal": "export_orders"},
                    candidate_routes=[
                        CandidateRoute(path="/crm/v2/orders/export", method="GET", confidence=0.9, source="service_catalog"),
                        CandidateRoute(path="/crm/v2/orders", method="GET", confidence=0.37, source="semantic_search"),
                    ],
                    reason="Use the live CRM orders export route.",
                ),
                _payload_phase(
                    raw_scenario_type="schema_type_coercion",
                    method="POST",
                    failed_url="https://local.billing.example/billing/v2/invoices",
                    payload={"order_id": 911, "amount_cents": "12800"},
                    expected_body={"invoice": {"order": {"id": 911}, "amount_cents": 12800}},
                    message="Billing invoice payload requires nesting and numeric cents.",
                    signals={"current_app": "billing", "required_shape": "invoice.order"},
                    reason="Normalize the invoice payload after the CRM export succeeds.",
                ),
            ],
            drift_contract_name="drift_payload.json",
            execution_mode="live_local",
            workflow_length="2-step",
            service_domain="crm,billing",
            description="Two-step live CRM to billing workflow.",
        ),
        ScenarioSpec(
            scenario_id="live-billing-refund-auth-chain",
            workflow_id="live_billing_refund_chain",
            benchmark_partition="repairable",
            app_stack=["billing", "identity"],
            visible_tools=[
                {"tool_name": "billing.refunds.lookup", "app": "billing", "method": "GET"},
                {"tool_name": "billing.refunds.issue", "app": "billing", "method": "POST"},
            ],
            phases=[
                _route_phase(
                    raw_scenario_type="route_regression",
                    method="GET",
                    failed_url="https://local.billing.example/billing/v1/refunds/pending",
                    success_path="/billing/v2/refunds/pending",
                    message="Pending refunds moved to v2.",
                    signals={"current_app": "billing", "phase_goal": "list_pending_refunds"},
                    candidate_routes=[
                        CandidateRoute(path="/billing/v2/refunds/pending", method="GET", confidence=0.88, source="service_catalog"),
                        CandidateRoute(path="/billing/v2/refunds", method="GET", confidence=0.34, source="semantic_search"),
                    ],
                    reason="Use the v2 billing refunds pending route.",
                ),
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://local.billing.example/billing/v2/refunds",
                    payload={"refund_id": "RF-81", "amount_cents": 2400},
                    required_headers=["x-api-key"],
                    message="Issuing a refund requires tenant routing.",
                    signals={"current_app": "billing", "missing_header": "x-api-key"},
                    reason="Attach the tenant key before issuing the refund.",
                ),
            ],
            drift_contract_name="drift_auth.json",
            execution_mode="live_local",
            workflow_length="2-step",
            service_domain="billing,identity",
            description="Two-step live billing refund route plus auth repair workflow.",
        ),
        ScenarioSpec(
            scenario_id="live-crm-lead-merge",
            workflow_id="live_crm_lead_merge",
            benchmark_partition="repairable",
            app_stack=["crm", "identity"],
            visible_tools=[
                {"tool_name": "crm.leads.export", "app": "crm", "method": "GET"},
                {"tool_name": "crm.leads.merge", "app": "crm", "method": "POST"},
            ],
            phases=[
                _route_phase(
                    raw_scenario_type="route_version_shift",
                    method="GET",
                    failed_url="https://local.crm.example/crm/v1/leads/export",
                    success_path="/crm/v2/leads/export",
                    message="Lead export moved to the v2 route.",
                    signals={"current_app": "crm", "phase_goal": "export_leads"},
                    candidate_routes=[
                        CandidateRoute(path="/crm/v2/leads/export", method="GET", confidence=0.91, source="service_catalog"),
                        CandidateRoute(path="/crm/v2/lead/export", method="GET", confidence=0.33, source="semantic_search"),
                    ],
                    reason="Use the v2 leads export route.",
                ),
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://local.crm.example/crm/v2/leads/merge",
                    payload={"match_id": "M-91"},
                    required_headers=["x-api-key"],
                    message="Lead merge now requires tenant routing.",
                    signals={"current_app": "crm", "missing_header": "x-api-key"},
                    reason="Attach the tenant key before merging leads.",
                ),
            ],
            drift_contract_name="drift_auth.json",
            execution_mode="live_local",
            workflow_length="2-step",
            service_domain="crm,identity",
            description="Two-step live CRM lead merge workflow.",
        ),
        ScenarioSpec(
            scenario_id="live-billing-credit-closeout",
            workflow_id="live_billing_credit_closeout",
            benchmark_partition="repairable",
            app_stack=["billing", "crm", "identity"],
            visible_tools=[
                {"tool_name": "billing.credits.preview", "app": "billing", "method": "POST"},
                {"tool_name": "crm.accounts.export", "app": "crm", "method": "GET"},
                {"tool_name": "billing.credits.issue", "app": "billing", "method": "POST"},
            ],
            phases=[
                _payload_phase(
                    raw_scenario_type="schema_missing_key",
                    method="POST",
                    failed_url="https://local.billing.example/billing/v2/credits/preview",
                    payload={"account_id": "ACC-44", "amount_cents": 3300},
                    expected_body={"credit_preview": {"account_id": "ACC-44", "amount_cents": 3300}},
                    message="Credit previews now require nested credit_preview payloads.",
                    signals={"current_app": "billing", "required_shape": "credit_preview"},
                    reason="Wrap the preview payload under credit_preview.",
                ),
                _route_phase(
                    raw_scenario_type="route_invalid_path",
                    method="GET",
                    failed_url="https://local.crm.example/crm/v1/accounts/export",
                    success_path="/crm/v2/accounts/export",
                    message="CRM account export moved to v2.",
                    signals={"current_app": "crm", "phase_goal": "export_accounts"},
                    candidate_routes=[
                        CandidateRoute(path="/crm/v2/accounts/export", method="GET", confidence=0.9, source="service_catalog"),
                        CandidateRoute(path="/crm/v2/accounts", method="GET", confidence=0.42, source="semantic_search"),
                    ],
                    reason="Use the v2 CRM account export route.",
                ),
                _auth_phase(
                    raw_scenario_type="auth_missing_tenant",
                    method="POST",
                    failed_url="https://local.billing.example/billing/v2/credits",
                    payload={"account_id": "ACC-44", "amount_cents": 3300},
                    required_headers=["x-api-key"],
                    message="Issuing credits requires tenant routing.",
                    signals={"current_app": "billing", "missing_header": "x-api-key"},
                    reason="Attach the tenant key before issuing the credit.",
                ),
            ],
            drift_contract_name="drift_auth.json",
            execution_mode="live_local",
            workflow_length="3-step",
            service_domain="billing,crm,identity",
            description="Three-step live billing closeout workflow with payload, route, and auth repair.",
            max_steps=5,
        ),
    ]


def load_live_local_scenarios(
    *,
    seed: int = DEFAULT_BENCHMARK_SEED,
    split: str = "all",
    randomize: bool = True,
) -> list[ScenarioSpec]:
    """Return the live-local FastAPI-backed scenario pack."""

    scenarios = _assign_splits(_base_live_local_scenarios(), seed=DEFAULT_SPLIT_SEED)
    materialized = [_materialize_scenario(scenario, seed=seed, randomize=randomize) for scenario in scenarios]
    return filter_scenarios(materialized, split=split, execution_mode="live_local")


def filter_scenarios(
    scenarios: list[ScenarioSpec],
    *,
    split: str = "all",
    execution_mode: str | None = None,
) -> list[ScenarioSpec]:
    """Filter scenarios by split and execution mode without mutating them."""

    filtered = list(scenarios)
    if split != "all":
        filtered = [scenario for scenario in filtered if scenario.split == split]
    if execution_mode is not None:
        filtered = [scenario for scenario in filtered if scenario.execution_mode == execution_mode]
    return filtered


def load_built_in_scenarios(
    *,
    seed: int = DEFAULT_BENCHMARK_SEED,
    split: str = "all",
    execution_mode: str | None = "simulated",
    randomize: bool = True,
) -> list[ScenarioSpec]:
    """Return the combined scenario pack used for benchmark and training scripts."""

    if execution_mode == "live_local":
        return load_live_local_scenarios(seed=seed, split=split, randomize=randomize)

    validation = load_phase1_validation_scenarios()
    benchmark = load_world_model_scenarios(
        seed=seed,
        split="all",
        execution_mode=execution_mode,
        randomize=randomize,
    )
    combined = validation + benchmark
    return filter_scenarios(combined, split=split, execution_mode=execution_mode)


def scenario_catalog(
    *,
    seed: int = DEFAULT_BENCHMARK_SEED,
    randomize: bool = True,
) -> list[dict[str, Any]]:
    """Return manifest-friendly metadata for the full Phase 1 scenario pack."""

    return [scenario.metadata() for scenario in load_built_in_scenarios(seed=seed, split="all", randomize=randomize)]
