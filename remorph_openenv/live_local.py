"""FastAPI-backed local service world for Phase 2 live execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from remorph_openenv.models import PolicyAction
from remorph_openenv.scenarios import ScenarioSpec


@dataclass
class _WorldState:
    scenario: ScenarioSpec | None = None
    phase_index: int = 0
    abstention_events: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.abstention_events is None:
            self.abstention_events = []

    def current_phase(self) -> Any:
        if self.scenario is None:
            return None
        return self.scenario.phases[self.phase_index]

    def expected_path(self) -> str:
        phase = self.current_phase()
        if phase is None:
            return "/"
        return str(phase.expected_action.target_path or urlsplit(str(phase.failed_request.get("url", ""))).path or "/")

    def expected_method(self) -> str:
        phase = self.current_phase()
        if phase is None:
            return "GET"
        return str(phase.expected_action.target_method or phase.failed_request.get("method") or "GET").upper()


def _service_name_from_path(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    return parts[0] if parts else "unknown"


def _normalize_headers(headers: dict[str, Any]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _build_service_app(service_name: str, world: _WorldState) -> FastAPI:
    app = FastAPI(title=f"ReMorph {service_name} service")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": service_name}

    @app.get("/state")
    def state() -> dict[str, Any]:
        scenario = world.scenario
        return {
            "service": service_name,
            "scenario_id": scenario.scenario_id if scenario else None,
            "phase_index": world.phase_index,
            "expected_path": world.expected_path() if scenario else None,
        }

    @app.post("/identity/v1/audit/abstain")
    async def audit_abstain(request: Request) -> JSONResponse:
        payload = await request.json()
        scenario = world.scenario
        if scenario is None or scenario.benchmark_partition != "unrecoverable":
            return JSONResponse(status_code=400, content={"status": "invalid", "correct_abstention": False})
        world.abstention_events.append(dict(payload))
        return JSONResponse(
            status_code=202,
            content={
                "status": "abstained",
                "correct_abstention": True,
                "phase_completed": True,
                "route_match": False,
                "payload_valid": False,
            },
        )

    @app.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def dispatch(full_path: str, request: Request) -> JSONResponse:
        scenario = world.scenario
        if scenario is None:
            return JSONResponse(status_code=503, content={"error": "world_not_initialized"})

        method = request.method.upper()
        path = "/" + full_path
        payload = None
        if method in {"POST", "PUT", "PATCH"}:
            try:
                payload = await request.json()
            except Exception:
                payload = None

        phase = world.current_phase()
        expected_action = phase.expected_action
        expected_path = world.expected_path()
        expected_method = world.expected_method()
        headers = _normalize_headers(dict(request.headers))
        required_headers = _normalize_headers(dict(expected_action.header_patch or {}))

        route_match = method == expected_method and path == expected_path
        payload_valid = expected_action.action_type != "repair_payload" or dict(payload or {}) == dict(expected_action.body_patch or {})
        auth_valid = True
        if expected_action.action_type == "repair_auth":
            auth_valid = all(headers.get(key) == value for key, value in required_headers.items())

        if _service_name_from_path(path) != service_name:
            return JSONResponse(status_code=404, content={"route_match": False, "payload_valid": False, "auth_valid": False})

        if not route_match:
            return JSONResponse(
                status_code=404,
                content={
                    "route_match": False,
                    "payload_valid": False,
                    "auth_valid": auth_valid,
                    "phase_completed": False,
                },
            )

        if expected_action.action_type == "repair_payload" and not payload_valid:
            return JSONResponse(
                status_code=422,
                content={
                    "route_match": True,
                    "payload_valid": False,
                    "auth_valid": True,
                    "phase_completed": False,
                },
            )

        if expected_action.action_type == "repair_auth" and not auth_valid:
            return JSONResponse(
                status_code=401,
                content={
                    "route_match": True,
                    "payload_valid": True,
                    "auth_valid": False,
                    "phase_completed": False,
                },
            )

        return JSONResponse(
            status_code=phase.success_status_code,
            content={
                "route_match": route_match,
                "payload_valid": payload_valid,
                "auth_valid": auth_valid,
                "phase_completed": True,
                "service": service_name,
            },
        )

    return app


class LiveLocalServiceHarness:
    """In-process FastAPI service world exercised through real HTTP calls."""

    def __init__(self) -> None:
        self._world = _WorldState()
        self.crm_app = _build_service_app("crm", self._world)
        self.billing_app = _build_service_app("billing", self._world)
        self.identity_app = _build_service_app("identity", self._world)
        self._clients = {
            "crm": TestClient(self.crm_app),
            "billing": TestClient(self.billing_app),
            "identity": TestClient(self.identity_app),
        }

    def reset(self, scenario: ScenarioSpec, *, phase_index: int = 0) -> None:
        self._world.scenario = scenario
        self._world.phase_index = phase_index
        self._world.abstention_events = []

    def set_phase(self, phase_index: int) -> None:
        self._world.phase_index = phase_index

    def close(self) -> None:
        for client in self._clients.values():
            client.close()

    def _client_for_path(self, path: str) -> TestClient:
        service = _service_name_from_path(path)
        if service not in self._clients:
            raise ValueError(f"Unsupported live_local service path: {path}")
        return self._clients[service]

    def execute_action(self, action: PolicyAction, scenario: ScenarioSpec, phase_index: int) -> dict[str, Any]:
        self.set_phase(phase_index)
        phase = scenario.phases[phase_index]

        if action.action_type == "abstain":
            response = self._clients["identity"].post(
                "/identity/v1/audit/abstain",
                json={"scenario_id": scenario.scenario_id, "phase_index": phase_index, "reason": action.reason},
            )
            payload = response.json()
            return {
                "status_code": response.status_code,
                "phase_completed": bool(payload.get("phase_completed")),
                "correct_abstention": bool(payload.get("correct_abstention")),
                "route_match": False,
                "payload_valid": False,
                "used_hallucinated_auth": False,
            }

        request_path = str(action.target_path or urlsplit(str(phase.failed_request.get("url", ""))).path or "/")
        request_method = str(action.target_method or phase.failed_request.get("method") or "GET").upper()
        headers = dict(phase.failed_request.get("headers") or {})
        headers.update(dict(action.header_patch or {}))
        payload = action.body_patch if action.action_type == "repair_payload" else phase.failed_request.get("payload")
        client = self._client_for_path(request_path)
        response = client.request(request_method, request_path, headers=headers, json=payload)
        content = response.json()

        expected_headers = dict(phase.expected_action.header_patch or {})
        used_hallucinated_auth = (
            scenario.benchmark_partition == "unrecoverable"
            and action.action_type == "repair_auth"
            and bool(action.header_patch)
            and dict(action.header_patch or {}) != expected_headers
        )
        return {
            "status_code": response.status_code,
            "phase_completed": bool(content.get("phase_completed")),
            "correct_abstention": False,
            "route_match": bool(content.get("route_match")),
            "payload_valid": bool(content.get("payload_valid")),
            "used_hallucinated_auth": used_hallucinated_auth,
        }


def create_live_local_gateway_app() -> FastAPI:
    """Create a combined gateway app for manual local inspection."""

    harness = LiveLocalServiceHarness()
    app = FastAPI(title="ReMorph live-local gateway")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/services")
    def services() -> dict[str, list[str]]:
        return {"services": ["crm", "billing", "identity"]}

    @app.get("/crm/health")
    def crm_health() -> dict[str, str]:
        return {"status": "ok", "service": "crm"}

    @app.get("/billing/health")
    def billing_health() -> dict[str, str]:
        return {"status": "ok", "service": "billing"}

    @app.get("/identity/health")
    def identity_health() -> dict[str, str]:
        return {"status": "ok", "service": "identity"}

    @app.on_event("shutdown")
    def shutdown() -> None:
        harness.close()

    return app
