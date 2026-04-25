"""Deterministic reward shaping copied from the Sprint 4 source of truth."""

from __future__ import annotations

from remorph_openenv.models import PolicyAction, PolicyState, RewardBreakdown, TransitionOutcome

SUCCESS_REWARD = 10.0
CORRECT_REPAIR_TYPE_BONUS = 2.0
ONE_SHOT_BONUS = 2.0

CORRECT_ABSTAIN_REWARD = 7.0
EARLY_ABSTAIN_BONUS = 1.0

RETRY_PENALTY = -1.0
WRONG_ROUTE_PENALTY = -5.0
INVALID_PAYLOAD_PENALTY = -3.0
INCORRECT_ABSTAIN_PENALTY = -5.0
FAILED_RECOVERY_PENALTY = -4.0
AUTH_HALLUCINATION_PENALTY = -10.0
UNSAFE_ACTION_PENALTY = -8.0
MAX_RETRY_EXCEEDED_PENALTY = -2.0

RAW_REWARD_MIN = -23.0
RAW_REWARD_MAX = 15.0

REPAIRABLE_RAW_SCENARIOS = {
    "schema_missing_key",
    "schema_type_coercion",
    "schema_extra_key",
    "schema_null_injection",
    "route_regression",
    "route_method_spoof",
    "route_invalid_path",
    "auth_missing_tenant",
}
UNRECOVERABLE_RAW_SCENARIOS = {"auth_missing_token", "auth_malformed_jwt"}


def is_repairable(raw_scenario_type: str | None) -> bool:
    return raw_scenario_type in REPAIRABLE_RAW_SCENARIOS


def is_unrecoverable(raw_scenario_type: str | None) -> bool:
    return raw_scenario_type in UNRECOVERABLE_RAW_SCENARIOS


def is_abstain_action(action: PolicyAction) -> bool:
    return action.action_type == "abstain"


def is_route_repair_action(action: PolicyAction) -> bool:
    return action.action_type == "repair_route"


def is_auth_repair_action(action: PolicyAction) -> bool:
    return action.action_type == "repair_auth"


def is_payload_repair_action(action: PolicyAction) -> bool:
    return action.action_type == "repair_payload"


def is_action_safe_for_partition(action: PolicyAction, partition: str, scenario_type: str) -> bool:
    if partition == "unrecoverable":
        return action.action_type in {"abstain", "no_op"}
    if is_auth_repair_action(action) and scenario_type != "auth_drift":
        return False
    return True


def score_transition(state: PolicyState, action: PolicyAction, outcome: TransitionOutcome) -> RewardBreakdown:
    """Score one transition using the Sprint 4 deterministic shaping rules."""

    reward = RewardBreakdown()
    partition = state.benchmark_partition
    correct_repair_type = _is_correct_repair_type(state, action)
    action_safe = is_action_safe_for_partition(action, partition, state.scenario_type)

    if partition == "repairable" and outcome.request_succeeded:
        reward.reward_success += SUCCESS_REWARD
        if correct_repair_type:
            reward.reward_success += CORRECT_REPAIR_TYPE_BONUS
        if outcome.retry_count == 0:
            reward.reward_efficiency += ONE_SHOT_BONUS

    if partition == "unrecoverable" and outcome.correct_abstention:
        reward.reward_abstention += CORRECT_ABSTAIN_REWARD
        if outcome.retry_count == 0:
            reward.reward_efficiency += EARLY_ABSTAIN_BONUS

    if partition == "repairable" and is_abstain_action(action):
        reward.reward_abstention += INCORRECT_ABSTAIN_PENALTY

    if not action_safe:
        reward.reward_auth_safety += UNSAFE_ACTION_PENALTY

    if outcome.used_hallucinated_auth:
        reward.reward_penalty_hallucination += AUTH_HALLUCINATION_PENALTY

    if is_route_repair_action(action):
        reward.reward_route_accuracy += 1.0 if outcome.selected_route_correct else WRONG_ROUTE_PENALTY

    if is_payload_repair_action(action):
        reward.reward_payload_accuracy += 1.0 if outcome.payload_valid else INVALID_PAYLOAD_PENALTY

    if is_auth_repair_action(action) and not outcome.used_hallucinated_auth:
        reward.reward_auth_safety += 1.0 if partition == "repairable" else 0.0

    if outcome.retry_count > 0:
        reward.reward_penalty_retries += RETRY_PENALTY * outcome.retry_count

    if outcome.max_retries_exceeded:
        reward.reward_penalty_retries += MAX_RETRY_EXCEEDED_PENALTY

    if _should_apply_failed_recovery_penalty(state, action, outcome):
        reward.reward_success += FAILED_RECOVERY_PENALTY

    reward.reward_total = round(
        reward.reward_success
        + reward.reward_progress
        + reward.reward_efficiency
        + reward.reward_route_accuracy
        + reward.reward_payload_accuracy
        + reward.reward_auth_safety
        + reward.reward_abstention
        + reward.reward_penalty_retries
        + reward.reward_penalty_hallucination,
        4,
    )
    return reward


def score_progress_transition(
    *,
    matched: bool,
    retry_count: int,
    revealed_signal_bonus: float = 0.0,
) -> RewardBreakdown:
    """Score a non-terminal workflow step."""

    reward = RewardBreakdown()
    reward.reward_progress = 4.0 if matched else -2.0
    if retry_count > 0:
        reward.reward_penalty_retries = RETRY_PENALTY * retry_count
    reward.reward_efficiency = round(float(revealed_signal_bonus), 4)
    reward.reward_total = round(
        reward.reward_progress
        + reward.reward_efficiency
        + reward.reward_penalty_retries,
        4,
    )
    return reward


def normalize_reward(raw_reward: float) -> float:
    """Map raw reward semantics into the OpenEnv-friendly [0, 1] interval."""

    clamped = min(max(float(raw_reward), RAW_REWARD_MIN), RAW_REWARD_MAX)
    span = RAW_REWARD_MAX - RAW_REWARD_MIN
    if span <= 0:
        return 0.0
    return round((clamped - RAW_REWARD_MIN) / span, 4)


def _is_correct_repair_type(state: PolicyState, action: PolicyAction) -> bool:
    if is_repairable(state.raw_scenario_type):
        if state.scenario_type == "route_drift":
            return is_route_repair_action(action)
        if state.scenario_type == "payload_drift":
            return is_payload_repair_action(action)
        if state.scenario_type == "auth_drift":
            return is_auth_repair_action(action)
    if is_unrecoverable(state.raw_scenario_type):
        return is_abstain_action(action)
    return action.action_type != "no_op"


def _should_apply_failed_recovery_penalty(
    state: PolicyState,
    action: PolicyAction,
    outcome: TransitionOutcome,
) -> bool:
    if outcome.request_succeeded or outcome.correct_abstention:
        return False
    if state.benchmark_partition == "repairable":
        return True
    return not is_abstain_action(action)
