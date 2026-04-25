"""OpenEnv-compatible ReMorph environment for API drift repair."""

from __future__ import annotations

from random import Random
from typing import Any

from remorph_openenv.live_local import LiveLocalServiceHarness
from remorph_openenv.models import PolicyAction, PolicyState, ReMorphState, TransitionOutcome
from remorph_openenv.rewards import normalize_reward, score_progress_transition, score_transition
from remorph_openenv.scenarios import ScenarioSpec, load_built_in_scenarios

try:
    from openenv import Environment as _OpenEnvEnvironment
except Exception:  # pragma: no cover - compatibility fallback
    class _OpenEnvEnvironment:
        """Fallback base when OpenEnv is unavailable locally."""

        pass


class ReMorphEnv(_OpenEnvEnvironment):
    """Minimal OpenEnv-style environment with deterministic built-in scenarios."""

    def __init__(
        self,
        *,
        scenarios: list[ScenarioSpec] | None = None,
        seed: int = 0,
        split: str = "all",
        execution_mode: str | None = "simulated",
        randomize: bool = True,
    ) -> None:
        self._default_seed = seed
        self._split = split
        self._execution_mode = execution_mode
        self._randomize = randomize
        self._custom_scenarios = scenarios is not None
        self._scenarios = scenarios or load_built_in_scenarios(
            seed=seed,
            split=split,
            execution_mode=execution_mode,
            randomize=randomize,
        )
        self._rng = Random(seed)
        self._current_scenario: ScenarioSpec | None = None
        self._current_phase_index = 0
        self._retry_count = 0
        self._action_history: list[PolicyAction] = []
        self._done = False
        self._live_harness = LiveLocalServiceHarness() if self._execution_mode == "live_local" else None

    def reset(self, scenario_id: str | None = None, seed: int | None = None) -> dict[str, Any]:
        """Load one scenario and return the initial observation."""

        if seed is not None:
            self._rng = Random(seed)
            if not self._custom_scenarios:
                self._scenarios = load_built_in_scenarios(
                    seed=seed,
                    split=self._split,
                    execution_mode=self._execution_mode,
                    randomize=self._randomize,
                )

        if scenario_id:
            selected = next(
                (scenario for scenario in self._scenarios if scenario.scenario_id == scenario_id),
                None,
            )
            if selected is None:
                raise ValueError(f"Unknown scenario_id: {scenario_id}")
        else:
            selected = self._rng.choice(self._scenarios)

        self._current_scenario = selected
        self._current_phase_index = 0
        self._retry_count = 0
        self._action_history = []
        self._done = False
        if self._live_harness is not None:
            self._live_harness.reset(selected, phase_index=0)
        return self.state()

    def state(self) -> dict[str, Any]:
        """Return the active observation in a Gym/OpenEnv-friendly shape."""

        if self._current_scenario is None:
            return {}
        return self._current_scenario.to_openenv_state(
            phase_index=self._current_phase_index,
            retry_count=self._retry_count,
            prior_actions=self._action_history,
        ).model_dump(mode="json")

    def get_state(self) -> dict[str, Any]:
        """Compatibility alias for validators that expect get_state()."""

        return self.state()

    def step(self, action: PolicyAction | dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute one structured repair action."""

        if self._current_scenario is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode already completed. Call reset() before another step().")

        current_state = self._current_scenario.build_policy_state(
            phase_index=self._current_phase_index,
            retry_count=self._retry_count,
            prior_actions=self._action_history,
        )
        validated_action = action if isinstance(action, PolicyAction) else PolicyAction.model_validate(action)
        expected_action = self._current_scenario.phases[self._current_phase_index].expected_action
        live_result = None
        if self._execution_mode == "live_local":
            if self._live_harness is None:
                raise RuntimeError("live_local execution requested without an initialized service harness.")
            live_result = self._live_harness.execute_action(
                validated_action,
                self._current_scenario,
                self._current_phase_index,
            )
            matched = bool(live_result["phase_completed"])
        else:
            matched = _actions_match(validated_action, expected_action)
        is_final_phase = self._current_phase_index >= len(self._current_scenario.phases) - 1
        self._action_history.append(validated_action)

        if matched and not is_final_phase:
            reward_breakdown = score_progress_transition(
                matched=True,
                retry_count=self._retry_count,
                revealed_signal_bonus=0.5,
            )
            self._current_phase_index += 1
            if self._live_harness is not None:
                self._live_harness.set_phase(self._current_phase_index)
            raw_reward = reward_breakdown.reward_total
            normalized = normalize_reward(raw_reward)
            info = {
                "success": False,
                "done_reason": "progressed",
                "status_code": int(live_result["status_code"]) if live_result is not None else 202,
                "raw_reward": raw_reward,
                "normalized_reward": normalized,
                "reward_breakdown": reward_breakdown.model_dump(mode="json"),
                "reference_action": expected_action.model_dump(mode="json"),
                "benchmark_partition": current_state.benchmark_partition,
                "raw_scenario_type": current_state.raw_scenario_type,
                "workflow_id": current_state.workflow_id,
                "phase_index": self._current_phase_index,
            }
            return self.state(), normalized, False, info

        if not matched and self._retry_count + 1 < self._current_scenario.max_steps:
            self._retry_count += 1
            reward_breakdown = score_progress_transition(
                matched=False,
                retry_count=self._retry_count,
                revealed_signal_bonus=0.0,
            )
            raw_reward = reward_breakdown.reward_total
            normalized = normalize_reward(raw_reward)
            info = {
                "success": False,
                "done_reason": "retry_available",
                "status_code": int(live_result["status_code"]) if live_result is not None else int(current_state.failure_code or 400),
                "raw_reward": raw_reward,
                "normalized_reward": normalized,
                "reward_breakdown": reward_breakdown.model_dump(mode="json"),
                "reference_action": expected_action.model_dump(mode="json"),
                "benchmark_partition": current_state.benchmark_partition,
                "raw_scenario_type": current_state.raw_scenario_type,
                "workflow_id": current_state.workflow_id,
                "phase_index": self._current_phase_index,
            }
            return self.state(), normalized, False, info

        outcome = self._evaluate_action(
            self._current_scenario,
            current_state,
            validated_action,
            matched=matched,
            live_result=live_result,
        )
        reward_breakdown = score_transition(current_state, validated_action, outcome)
        raw_reward = reward_breakdown.reward_total
        normalized = normalize_reward(raw_reward)
        self._done = True

        info = {
            "success": bool(outcome.request_succeeded or outcome.correct_abstention),
            "status_code": outcome.http_status,
            "raw_reward": raw_reward,
            "normalized_reward": normalized,
            "reward_breakdown": reward_breakdown.model_dump(mode="json"),
            "reference_action": expected_action.model_dump(mode="json"),
            "benchmark_partition": current_state.benchmark_partition,
            "raw_scenario_type": current_state.raw_scenario_type,
            "workflow_id": current_state.workflow_id,
            "phase_index": self._current_phase_index,
        }
        return self.state(), normalized, True, info

    def close(self) -> None:
        if self._live_harness is not None:
            self._live_harness.close()
        self._current_scenario = None
        self._current_phase_index = 0
        self._retry_count = 0
        self._action_history = []
        self._done = False

    def available_scenarios(self) -> list[str]:
        return [scenario.scenario_id for scenario in self._scenarios]

    def _evaluate_action(
        self,
        scenario: ScenarioSpec,
        state: PolicyState,
        action: PolicyAction,
        *,
        matched: bool,
        live_result: dict[str, Any] | None = None,
    ) -> TransitionOutcome:
        if self._execution_mode == "live_local":
            if live_result is None:
                raise RuntimeError("live_local evaluation requires a prior harness result.")
            result = live_result
            return TransitionOutcome(
                request_succeeded=matched,
                http_status=int(result["status_code"]),
                retry_count=state.retry_count,
                selected_route_correct=bool(result["route_match"]),
                payload_valid=bool(result["payload_valid"]),
                used_hallucinated_auth=bool(result["used_hallucinated_auth"]),
                abstained=action.action_type == "abstain",
                correct_abstention=bool(result["correct_abstention"]),
                max_retries_exceeded=state.retry_count + 1 >= scenario.max_steps and not matched,
            )

        if scenario.benchmark_partition == "unrecoverable":
            is_abstain = action.action_type == "abstain"
            used_hallucinated_auth = action.action_type == "repair_auth" and bool(action.header_patch)
            return TransitionOutcome(
                request_succeeded=False,
                http_status=401,
                retry_count=state.retry_count,
                selected_route_correct=False,
                payload_valid=False,
                used_hallucinated_auth=used_hallucinated_auth,
                abstained=is_abstain,
                correct_abstention=is_abstain,
                max_retries_exceeded=False,
            )

        return TransitionOutcome(
            request_succeeded=matched,
            http_status=scenario.phases[self._current_phase_index].success_status_code if matched else int(state.failure_code or 400),
            retry_count=state.retry_count,
            selected_route_correct=_route_match(action, scenario.phases[self._current_phase_index].expected_action),
            payload_valid=_payload_match(action, scenario.phases[self._current_phase_index].expected_action),
            used_hallucinated_auth=False,
            abstained=action.action_type == "abstain",
            correct_abstention=False,
            max_retries_exceeded=state.retry_count + 1 >= scenario.max_steps and not matched,
        )


def _actions_match(action: PolicyAction, reference_action: PolicyAction) -> bool:
    if action.action_type != reference_action.action_type:
        return False
    if action.action_type == "repair_route":
        return _route_match(action, reference_action)
    if action.action_type == "repair_payload":
        return _payload_match(action, reference_action)
    if action.action_type == "repair_auth":
        return dict(action.header_patch or {}) == dict(reference_action.header_patch or {})
    return True


def _route_match(action: PolicyAction, reference_action: PolicyAction) -> bool:
    return (
        (action.target_method or "").upper() == (reference_action.target_method or "").upper()
        and str(action.target_path or "") == str(reference_action.target_path or "")
    )


def _payload_match(action: PolicyAction, reference_action: PolicyAction) -> bool:
    return dict(action.body_patch or {}) == dict(reference_action.body_patch or {})


ReMorphEnvironment = ReMorphEnv
