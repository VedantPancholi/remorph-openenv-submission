"""Typed contracts for the clean ReMorph OpenEnv package."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ActionType = Literal[
    "repair_route",
    "repair_payload",
    "repair_auth",
    "abstain",
    "no_op",
]


class CandidateRoute(BaseModel):
    """One route candidate visible to the agent."""

    path: str
    method: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str = "scenario"


class PolicyState(BaseModel):
    """Normalized observation shown to a policy."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    scenario_type: str
    raw_scenario_type: str
    benchmark_partition: str
    contract_version: str = "v1"
    request_method: str
    request_path: str
    request_headers: dict[str, str] = Field(default_factory=dict)
    request_query: dict[str, Any] = Field(default_factory=dict)
    request_body: dict[str, Any] | None = None
    failure_code: int | None = None
    failure_message: str | None = None
    failure_signals: dict[str, Any] = Field(default_factory=dict)
    candidate_routes: list[CandidateRoute] = Field(default_factory=list)
    contract_hints: dict[str, Any] = Field(default_factory=dict)
    workflow_id: str | None = None
    app_stack: list[str] = Field(default_factory=list)
    visible_tools: list[dict[str, Any]] = Field(default_factory=list)
    step_index: int = Field(default=0, ge=0)
    remaining_steps: int = Field(default=0, ge=0)
    prior_actions: list[dict[str, Any]] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)


class ReMorphState(BaseModel):
    """Explicit OpenEnv-facing state object returned by the environment."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    scenario_type: str
    benchmark_partition: str
    failed_request: dict[str, Any] = Field(default_factory=dict)
    error_signal: dict[str, Any] = Field(default_factory=dict)
    contract_hints: dict[str, Any] = Field(default_factory=dict)
    candidate_routes: list[dict[str, Any]] = Field(default_factory=list)
    workflow_id: str | None = None
    app_stack: list[str] = Field(default_factory=list)
    visible_tools: list[dict[str, Any]] = Field(default_factory=list)
    step_index: int = Field(default=0, ge=0)
    remaining_steps: int = Field(default=0, ge=0)
    prior_actions: list[dict[str, Any]] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)


class PolicyAction(BaseModel):
    """Structured action emitted by an agent."""

    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    target_method: str | None = None
    target_path: str | None = None
    header_patch: dict[str, str] | None = None
    query_patch: dict[str, Any] | None = None
    body_patch: dict[str, Any] | None = None
    reason: str | None = None


class RewardBreakdown(BaseModel):
    """Structured reward terms for one environment step."""

    model_config = ConfigDict(extra="forbid")

    reward_total: float = 0.0
    reward_success: float = 0.0
    reward_progress: float = 0.0
    reward_efficiency: float = 0.0
    reward_route_accuracy: float = 0.0
    reward_payload_accuracy: float = 0.0
    reward_auth_safety: float = 0.0
    reward_abstention: float = 0.0
    reward_penalty_retries: float = 0.0
    reward_penalty_hallucination: float = 0.0


class TransitionOutcome(BaseModel):
    """Typed feedback after executing one action."""

    model_config = ConfigDict(extra="forbid")

    request_succeeded: bool
    http_status: int | None = None
    retry_count: int = Field(default=0, ge=0)
    selected_route_correct: bool = False
    payload_valid: bool = False
    used_hallucinated_auth: bool = False
    abstained: bool = False
    correct_abstention: bool = False
    max_retries_exceeded: bool = False
