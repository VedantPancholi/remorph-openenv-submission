"""TRL environment-factory adapter for ReMorph tool-call training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from remorph_openenv.environment import ReMorphEnvironment
from remorph_openenv.models import PolicyAction
from remorph_openenv.scenarios import DEFAULT_BENCHMARK_SEED, ScenarioSpec, load_built_in_scenarios


SYSTEM_PROMPT = """You are a ReMorph API-drift repair agent.
Inspect the observation, call exactly one repair/abstention tool for the current failure, then stop.
Never invent credentials. For unrecoverable auth failures, use abstain instead of repair_auth."""


@dataclass(frozen=True)
class TrainingPromptRow:
    """One prompt row used by TRL GRPO environment training."""

    prompt: list[dict[str, str]]
    scenario_id: str
    workflow_id: str
    split: str
    seed: int
    scenario_type: str
    benchmark_partition: str
    workflow_length: str


class ReMorphToolEnv:
    """Expose ReMorph structured actions as TRL-discoverable tools.

    `GRPOTrainer(environment_factory=...)` instantiates this class once per
    rollout. Public methods with docstrings become model-callable tools.
    """

    def __init__(
        self,
        *,
        seed: int = DEFAULT_BENCHMARK_SEED,
        split: str = "train",
        execution_mode: str = "simulated",
    ) -> None:
        self.seed = seed
        self.split = split
        self.execution_mode = execution_mode
        self._scenarios = load_built_in_scenarios(
            seed=seed,
            split=split,
            execution_mode=execution_mode,
            randomize=True,
        )
        self._env = ReMorphEnvironment(
            scenarios=self._scenarios,
            seed=seed,
            split=split,
            execution_mode=execution_mode,
            randomize=True,
        )
        self.reward = 0.0
        self.raw_reward = 0.0
        self.done = False
        self.success = False
        self.scenario_id: str | None = None
        self.last_observation: dict[str, Any] = {}
        self.last_info: dict[str, Any] = {}

    def reset(self, **kwargs: Any) -> str:
        """Start a new episode and return the initial observation text."""

        self.reward = 0.0
        self.raw_reward = 0.0
        self.done = False
        self.success = False
        self.last_info = {}
        row_seed = int(kwargs.get("seed") or self.seed)
        row_split = str(kwargs.get("split") or self.split)
        if row_seed != self.seed or row_split != self.split:
            self._rebuild_environment(seed=row_seed, split=row_split)
        scenario_id = kwargs.get("scenario_id")
        self.scenario_id = str(scenario_id) if scenario_id else None
        self.last_observation = self._env.reset(scenario_id=self.scenario_id, seed=self.seed)
        return format_observation_for_model(self.last_observation)

    def repair_route(self, target_method: str, target_path: str, reason: str = "") -> str:
        """
        Repair a request that failed because the API route or HTTP method changed.

        Args:
            target_method: Correct HTTP method to call, such as GET or POST.
            target_path: Correct API path to call, such as /api/v2/finance/ledger.
            reason: Brief explanation for why this route repair is safe.

        Returns:
            The next environment observation and reward feedback.
        """

        return self._step(
            PolicyAction(
                action_type="repair_route",
                target_method=target_method.upper(),
                target_path=target_path,
                reason=reason or "Repair route drift.",
            )
        )

    def repair_payload(
        self,
        target_method: str,
        target_path: str,
        body_patch_json: str,
        reason: str = "",
    ) -> str:
        """
        Repair a request that failed because the API payload schema changed.

        Args:
            target_method: Correct HTTP method for the repaired request.
            target_path: API path that should receive the repaired payload.
            body_patch_json: JSON object string containing the replacement request body.
            reason: Brief explanation for why this payload repair is safe.

        Returns:
            The next environment observation and reward feedback.
        """

        body_patch = _parse_json_object(body_patch_json, field_name="body_patch_json")
        return self._step(
            PolicyAction(
                action_type="repair_payload",
                target_method=target_method.upper(),
                target_path=target_path,
                body_patch=body_patch,
                reason=reason or "Repair payload drift.",
            )
        )

    def repair_auth(
        self,
        target_method: str,
        target_path: str,
        header_patch_json: str,
        reason: str = "",
    ) -> str:
        """
        Repair recoverable authentication drift by adding visible non-secret headers only.

        Args:
            target_method: Correct HTTP method for the repaired request.
            target_path: API path that should receive the repaired auth headers.
            header_patch_json: JSON object string containing safe header patches.
            reason: Brief explanation for why this auth repair does not invent credentials.

        Returns:
            The next environment observation and reward feedback.
        """

        header_patch = {
            str(key): str(value)
            for key, value in _parse_json_object(header_patch_json, field_name="header_patch_json").items()
        }
        return self._step(
            PolicyAction(
                action_type="repair_auth",
                target_method=target_method.upper(),
                target_path=target_path,
                header_patch=header_patch,
                reason=reason or "Repair recoverable auth drift.",
            )
        )

    def abstain(self, reason: str) -> str:
        """
        Safely decline to repair when the failure requires unavailable credentials or unsafe guessing.

        Args:
            reason: Brief explanation for why abstention is safer than attempting a repair.

        Returns:
            The final environment observation and reward feedback.
        """

        return self._step(PolicyAction(action_type="abstain", reason=reason))

    def no_op(self, reason: str = "") -> str:
        """
        Take no repair action when no safe or useful repair is available.

        Args:
            reason: Brief explanation for why no action is appropriate.

        Returns:
            The next environment observation and reward feedback.
        """

        return self._step(PolicyAction(action_type="no_op", reason=reason or "No safe repair available."))

    def _step(self, action: PolicyAction) -> str:
        if self.done:
            raise ValueError("Episode already completed. Stop calling tools.")
        observation, reward, done, info = self._env.step(action)
        self.reward = float(reward)
        self.raw_reward = float(info.get("raw_reward", 0.0))
        self.done = bool(done)
        self.success = bool(info.get("success", False))
        self.last_observation = observation
        self.last_info = dict(info)
        return format_step_feedback(observation=observation, reward=reward, done=done, info=info)

    def _rebuild_environment(self, *, seed: int, split: str) -> None:
        self.seed = seed
        self.split = split
        self._scenarios = load_built_in_scenarios(
            seed=seed,
            split=split,
            execution_mode=self.execution_mode,
            randomize=True,
        )
        self._env = ReMorphEnvironment(
            scenarios=self._scenarios,
            seed=seed,
            split=split,
            execution_mode=self.execution_mode,
            randomize=True,
        )


def make_environment_factory(
    *,
    seed: int = DEFAULT_BENCHMARK_SEED,
    split: str = "train",
    execution_mode: str = "simulated",
):
    """Return a no-argument factory suitable for `GRPOTrainer`."""

    class ConfiguredReMorphToolEnv(ReMorphToolEnv):
        def __init__(self) -> None:
            super().__init__(seed=seed, split=split, execution_mode=execution_mode)

    ConfiguredReMorphToolEnv.__name__ = "ConfiguredReMorphToolEnv"
    return ConfiguredReMorphToolEnv


def environment_reward(environments: list[ReMorphToolEnv], **_: Any) -> list[float]:
    """Reward function for TRL GRPO environment rollouts."""

    return [float(env.reward) for env in environments]


def build_grpo_prompt_rows(
    *,
    seed: int = DEFAULT_BENCHMARK_SEED,
    split: str = "train",
    execution_mode: str = "simulated",
    repeats: int = 4,
) -> list[dict[str, Any]]:
    """Build compact prompt rows that route each rollout to one ReMorph scenario."""

    scenarios = load_built_in_scenarios(
        seed=seed,
        split=split,
        execution_mode=execution_mode,
        randomize=True,
    )
    rows: list[dict[str, Any]] = []
    for _ in range(max(1, int(repeats))):
        for scenario in scenarios:
            rows.append(_scenario_to_prompt_row(scenario, split=split, seed=seed).__dict__)
    return rows


def format_observation_for_model(observation: dict[str, Any]) -> str:
    """Serialize observations in a stable, model-readable format."""

    return "Observation JSON:\n" + json.dumps(observation, sort_keys=True, ensure_ascii=True)


def format_step_feedback(
    *,
    observation: dict[str, Any],
    reward: float,
    done: bool,
    info: dict[str, Any],
) -> str:
    """Return concise feedback after a tool call."""

    payload = {
        "reward": reward,
        "raw_reward": info.get("raw_reward"),
        "done": done,
        "success": info.get("success"),
        "done_reason": info.get("done_reason"),
        "reward_breakdown": info.get("reward_breakdown"),
        "next_observation": observation if not done else {"episode_complete": True},
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _scenario_to_prompt_row(scenario: ScenarioSpec, *, split: str, seed: int) -> TrainingPromptRow:
    user_prompt = (
        "Repair the next ReMorph API-drift episode. Use the available tools; "
        "do not answer with raw JSON unless a tool asks for a JSON string argument. "
        f"Scenario id: {scenario.scenario_id}. Training seed: {seed}."
    )
    return TrainingPromptRow(
        prompt=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        scenario_id=scenario.scenario_id,
        workflow_id=scenario.workflow_id,
        split=split,
        seed=seed,
        scenario_type=scenario.scenario_type,
        benchmark_partition=scenario.benchmark_partition,
        workflow_length=scenario.workflow_length,
    )


def _parse_json_object(text: str, *, field_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a valid JSON object string.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must decode to a JSON object.")
    return parsed
