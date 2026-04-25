"""Split-aware training, telemetry, and benchmark reporting utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from remorph_openenv.environment import ReMorphEnvironment
from remorph_openenv.models import PolicyAction
from remorph_openenv.policies import (
    ReplayMemoryPolicy,
    SupervisedStructuredPolicy,
    baseline_action,
    infer_belief,
    observation_signature,
)
from remorph_openenv.scenarios import DEFAULT_BENCHMARK_SEED, ScenarioSpec, load_built_in_scenarios


@dataclass(frozen=True)
class EpisodeTrace:
    scenario_id: str
    workflow_id: str
    steps: list[dict[str, Any]]
    success: bool
    normalized_reward_sum: float
    raw_reward_sum: float
    episode_normalized_return_capped: float


def load_split_scenarios(*, seed: int = DEFAULT_BENCHMARK_SEED) -> dict[str, list[ScenarioSpec]]:
    """Return the fixed Phase 1 train/eval benchmark splits."""

    return {
        "train": load_built_in_scenarios(seed=seed, split="train"),
        "eval": load_built_in_scenarios(seed=seed, split="eval"),
        "all": load_built_in_scenarios(seed=seed, split="all"),
    }


def load_scenarios_from_manifest(manifest_path: Path, *, seed: int = DEFAULT_BENCHMARK_SEED) -> list[ScenarioSpec]:
    """Load a scenario subset from a checked-in manifest file."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario_ids = [row["scenario_id"] for row in manifest.get("scenarios", [])]
    scenario_map = {
        scenario.scenario_id: scenario
        for scenario in load_built_in_scenarios(seed=seed, split="all")
    }
    missing_ids = [scenario_id for scenario_id in scenario_ids if scenario_id not in scenario_map]
    if missing_ids:
        raise ValueError(f"Unknown scenario ids in manifest {manifest_path}: {missing_ids}")
    return [scenario_map[scenario_id] for scenario_id in scenario_ids]


def serialize_observation(observation: dict[str, Any]) -> str:
    """Serialize an observation in a stable TRL-ready text format."""

    return json.dumps(observation, sort_keys=True, ensure_ascii=True)


def serialize_action(action: dict[str, Any] | PolicyAction) -> str:
    """Serialize an action in a stable TRL-ready JSON format."""

    payload = action.model_dump(mode="json") if isinstance(action, PolicyAction) else dict(action)
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def rollout_policy(
    *,
    scenarios: list[ScenarioSpec],
    policy_fn: Any,
    policy_name: str = "policy",
) -> dict[str, Any]:
    """Evaluate a policy callable over a scenario pack and collect step telemetry."""

    env = ReMorphEnvironment(scenarios=scenarios, seed=0)
    traces: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    success_count = 0
    total_normalized_reward = 0.0
    total_raw_reward = 0.0
    total_capped_episode_return = 0.0
    total_steps = 0

    for scenario in scenarios:
        observation = env.reset(scenario_id=scenario.scenario_id)
        done = False
        trace_steps: list[dict[str, Any]] = []
        episode_success = False
        episode_normalized = 0.0
        episode_raw = 0.0
        previous_belief: str | None = None
        while not done:
            belief, confidence = infer_belief(observation)
            action = policy_fn(observation)
            next_observation, reward, done, info = env.step(action)
            trace_steps.append(
                {
                    "action": action.model_dump(mode="json"),
                    "normalized_reward": reward,
                    "raw_reward": info["raw_reward"],
                    "done": done,
                    "success": info["success"],
                    "phase_index": info.get("phase_index"),
                }
            )
            telemetry.append(
                {
                    "policy_name": policy_name,
                    "scenario_id": scenario.scenario_id,
                    "workflow_id": scenario.workflow_id,
                    "step_index": len(trace_steps) - 1,
                    "observation_signature": observation_signature(observation),
                    "belief": belief,
                    "confidence": confidence,
                    "belief_changed": previous_belief is not None and belief != previous_belief,
                    "action": action.model_dump(mode="json"),
                    "normalized_reward": reward,
                    "raw_reward": info["raw_reward"],
                    "done": done,
                    "success": info["success"],
                    "reward_breakdown": info["reward_breakdown"],
                }
            )
            previous_belief = belief
            observation = next_observation
            total_steps += 1
            episode_normalized += float(reward)
            episode_raw += float(info["raw_reward"])
            if done:
                episode_success = bool(info["success"])
        success_count += int(episode_success)
        capped_episode_return = min(1.0, round(episode_normalized, 4))
        total_normalized_reward += episode_normalized
        total_raw_reward += episode_raw
        total_capped_episode_return += capped_episode_return
        traces.append(
            EpisodeTrace(
                scenario_id=scenario.scenario_id,
                workflow_id=scenario.workflow_id,
                steps=trace_steps,
                success=episode_success,
                normalized_reward_sum=round(episode_normalized, 4),
                raw_reward_sum=round(episode_raw, 4),
                episode_normalized_return_capped=capped_episode_return,
            ).__dict__
        )

    scenario_count = len(scenarios)
    return {
        "scenario_count": scenario_count,
        "success_count": success_count,
        "success_rate": round(success_count / scenario_count, 4) if scenario_count else 0.0,
        "average_normalized_reward": round(total_normalized_reward / scenario_count, 4) if scenario_count else 0.0,
        "average_raw_reward": round(total_raw_reward / scenario_count, 4) if scenario_count else 0.0,
        "average_episode_normalized_return_capped": round(total_capped_episode_return / scenario_count, 4) if scenario_count else 0.0,
        "average_steps_per_episode": round(total_steps / scenario_count, 4) if scenario_count else 0.0,
        "traces": traces,
        "telemetry": telemetry,
    }


def collect_reference_dataset(scenarios: list[ScenarioSpec]) -> list[dict[str, Any]]:
    """Collect observation/action pairs from the built-in oracle plan."""

    env = ReMorphEnvironment(scenarios=scenarios, seed=0)
    dataset: list[dict[str, Any]] = []
    for scenario in scenarios:
        observation = env.reset(scenario_id=scenario.scenario_id)
        done = False
        phase_index = 0
        while not done:
            action = scenario.phases[phase_index].expected_action
            dataset.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "workflow_id": scenario.workflow_id,
                    "phase_index": phase_index,
                    "observation": observation,
                    "action": action.model_dump(mode="json"),
                }
            )
            observation, _, done, _ = env.step(action)
            phase_index += 1
    return dataset


def train_replay_policy(
    *,
    scenarios: list[ScenarioSpec] | None = None,
    epochs: int = 3,
) -> dict[str, Any]:
    """Train a tiny replay-memory policy and return metrics histories."""

    scenario_pack = scenarios or load_built_in_scenarios()
    replay_policy = ReplayMemoryPolicy()
    dataset = collect_reference_dataset(scenario_pack)

    loss_history: list[dict[str, Any]] = []
    reward_history: list[dict[str, Any]] = []

    baseline_summary = rollout_policy(scenarios=scenario_pack, policy_fn=baseline_action, policy_name="baseline")
    reward_history.append(
        {
            "epoch": 0,
            "average_normalized_reward": baseline_summary["average_normalized_reward"],
            "average_raw_reward": baseline_summary["average_raw_reward"],
            "success_rate": baseline_summary["success_rate"],
        }
    )

    for epoch in range(epochs):
        mismatch_count = 0
        for row in dataset:
            predicted = replay_policy.predict(row["observation"])
            target = PolicyAction.model_validate(row["action"])
            if predicted != target:
                mismatch_count += 1
            replay_policy.update(row["observation"], target)

        learned_summary = rollout_policy(scenarios=scenario_pack, policy_fn=replay_policy.predict, policy_name="replay")
        loss_history.append(
            {
                "epoch": epoch + 1,
                "mismatch_count": mismatch_count,
                "mismatch_rate": round(mismatch_count / len(dataset), 4) if dataset else 0.0,
            }
        )
        reward_history.append(
            {
                "epoch": epoch + 1,
                "average_normalized_reward": learned_summary["average_normalized_reward"],
                "average_raw_reward": learned_summary["average_raw_reward"],
                "success_rate": learned_summary["success_rate"],
            }
        )

    final_summary = rollout_policy(scenarios=scenario_pack, policy_fn=replay_policy.predict, policy_name="replay")
    oracle_summary = rollout_policy(
        scenarios=scenario_pack,
        policy_name="oracle",
        policy_fn=lambda observation: next(
            PolicyAction.model_validate(row["action"])
            for row in dataset
            if row["observation"] == observation
        ),
    )
    return {
        "trainer": "replay_memory_policy",
        "scenario_count": len(scenario_pack),
        "training_example_count": len(dataset),
        "loss_history": loss_history,
        "reward_history": reward_history,
        "baseline_summary": baseline_summary,
        "learned_summary": final_summary,
        "oracle_summary": oracle_summary,
        "model_memory_size": len(replay_policy.memory),
        "dataset": dataset,
    }


def train_supervised_policy(
    *,
    train_scenarios: list[ScenarioSpec] | None = None,
    eval_scenarios: list[ScenarioSpec] | None = None,
    seed: int = DEFAULT_BENCHMARK_SEED,
) -> dict[str, Any]:
    """Train the supervised structured policy on train split and evaluate on held-out eval."""

    splits = load_split_scenarios(seed=seed)
    train_pack = train_scenarios or splits["train"]
    eval_pack = eval_scenarios or splits["eval"]
    train_dataset = collect_reference_dataset(train_pack)
    eval_dataset = collect_reference_dataset(eval_pack)

    replay_policy = ReplayMemoryPolicy()
    for row in train_dataset:
        replay_policy.update(row["observation"], PolicyAction.model_validate(row["action"]))

    supervised_policy = SupervisedStructuredPolicy()
    model_config = supervised_policy.fit(train_dataset)

    baseline_train = rollout_policy(scenarios=train_pack, policy_fn=baseline_action, policy_name="baseline")
    baseline_eval = rollout_policy(scenarios=eval_pack, policy_fn=baseline_action, policy_name="baseline")
    replay_eval = rollout_policy(scenarios=eval_pack, policy_fn=replay_policy.predict, policy_name="replay")
    supervised_train = rollout_policy(scenarios=train_pack, policy_fn=supervised_policy.predict, policy_name="supervised")
    supervised_eval = rollout_policy(scenarios=eval_pack, policy_fn=supervised_policy.predict, policy_name="supervised")
    oracle_eval = rollout_policy(
        scenarios=eval_pack,
        policy_name="oracle",
        policy_fn=lambda observation: next(
            PolicyAction.model_validate(row["action"])
            for row in eval_dataset
            if row["observation"] == observation
        ),
    )

    loss_history = [
        {
            "epoch": 1,
            "mismatch_count": sum(
                int(supervised_policy.predict(row["observation"]) != PolicyAction.model_validate(row["action"]))
                for row in train_dataset
            ),
            "mismatch_rate": round(
                sum(
                    int(supervised_policy.predict(row["observation"]) != PolicyAction.model_validate(row["action"]))
                    for row in train_dataset
                )
                / len(train_dataset),
                4,
            )
            if train_dataset
            else 0.0,
        }
    ]

    reward_history = [
        {
            "epoch": 0,
            "split": "train",
            "policy_name": "baseline",
            "average_normalized_reward": baseline_train["average_normalized_reward"],
            "average_raw_reward": baseline_train["average_raw_reward"],
            "success_rate": baseline_train["success_rate"],
        },
        {
            "epoch": 0,
            "split": "eval",
            "policy_name": "baseline",
            "average_normalized_reward": baseline_eval["average_normalized_reward"],
            "average_raw_reward": baseline_eval["average_raw_reward"],
            "success_rate": baseline_eval["success_rate"],
        },
        {
            "epoch": 1,
            "split": "train",
            "policy_name": "supervised",
            "average_normalized_reward": supervised_train["average_normalized_reward"],
            "average_raw_reward": supervised_train["average_raw_reward"],
            "success_rate": supervised_train["success_rate"],
        },
        {
            "epoch": 1,
            "split": "eval",
            "policy_name": "supervised",
            "average_normalized_reward": supervised_eval["average_normalized_reward"],
            "average_raw_reward": supervised_eval["average_raw_reward"],
            "success_rate": supervised_eval["success_rate"],
        },
    ]

    return {
        "trainer": "supervised_structured_policy",
        "seed": seed,
        "train_scenario_count": len(train_pack),
        "eval_scenario_count": len(eval_pack),
        "training_example_count": len(train_dataset),
        "eval_example_count": len(eval_dataset),
        "loss_history": loss_history,
        "reward_history": reward_history,
        "baseline_train_summary": baseline_train,
        "baseline_eval_summary": baseline_eval,
        "replay_eval_summary": replay_eval,
        "supervised_train_summary": supervised_train,
        "supervised_eval_summary": supervised_eval,
        "oracle_eval_summary": oracle_eval,
        "model_config": model_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "telemetry": {
            "baseline_eval": baseline_eval["telemetry"],
            "replay_eval": replay_eval["telemetry"],
            "supervised_eval": supervised_eval["telemetry"],
            "oracle_eval": oracle_eval["telemetry"],
        },
    }


def build_benchmark_report(result: dict[str, Any]) -> dict[str, str]:
    """Create markdown and JSON benchmark reports from a supervised run."""

    report_json = {
        "trainer": result["trainer"],
        "seed": result["seed"],
        "train_scenario_count": result["train_scenario_count"],
        "eval_scenario_count": result["eval_scenario_count"],
        "training_example_count": result["training_example_count"],
        "model_config": result["model_config"],
        "metrics": {
            "baseline_eval_success_rate": result["baseline_eval_summary"]["success_rate"],
            "replay_eval_success_rate": result["replay_eval_summary"]["success_rate"],
            "supervised_eval_success_rate": result["supervised_eval_summary"]["success_rate"],
            "oracle_eval_success_rate": result["oracle_eval_summary"]["success_rate"],
            "baseline_eval_avg_raw_reward": result["baseline_eval_summary"]["average_raw_reward"],
            "replay_eval_avg_raw_reward": result["replay_eval_summary"]["average_raw_reward"],
            "supervised_eval_avg_raw_reward": result["supervised_eval_summary"]["average_raw_reward"],
            "oracle_eval_avg_raw_reward": result["oracle_eval_summary"]["average_raw_reward"],
        },
    }
    markdown = "\n".join(
        [
            "# ReMorph Benchmark Report",
            "",
            f"- Trainer: `{result['trainer']}`",
            f"- Seed: `{result['seed']}`",
            f"- Train scenarios: `{result['train_scenario_count']}`",
            f"- Eval scenarios: `{result['eval_scenario_count']}`",
            f"- Training examples: `{result['training_example_count']}`",
            "",
            "## Eval Metrics",
            "",
            "| Policy | Success Rate | Avg Raw Reward | Avg Capped Episode Return | Avg Steps |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| baseline | {result['baseline_eval_summary']['success_rate']:.4f} | {result['baseline_eval_summary']['average_raw_reward']:.4f} | {result['baseline_eval_summary']['average_episode_normalized_return_capped']:.4f} | {result['baseline_eval_summary']['average_steps_per_episode']:.4f} |",
            f"| replay | {result['replay_eval_summary']['success_rate']:.4f} | {result['replay_eval_summary']['average_raw_reward']:.4f} | {result['replay_eval_summary']['average_episode_normalized_return_capped']:.4f} | {result['replay_eval_summary']['average_steps_per_episode']:.4f} |",
            f"| supervised | {result['supervised_eval_summary']['success_rate']:.4f} | {result['supervised_eval_summary']['average_raw_reward']:.4f} | {result['supervised_eval_summary']['average_episode_normalized_return_capped']:.4f} | {result['supervised_eval_summary']['average_steps_per_episode']:.4f} |",
            f"| oracle | {result['oracle_eval_summary']['success_rate']:.4f} | {result['oracle_eval_summary']['average_raw_reward']:.4f} | {result['oracle_eval_summary']['average_episode_normalized_return_capped']:.4f} | {result['oracle_eval_summary']['average_steps_per_episode']:.4f} |",
            "",
            "## Model Config",
            "",
            f"```json\n{json.dumps(result['model_config'], indent=2)}\n```",
        ]
    )
    return {"markdown": markdown, "json": json.dumps(report_json, indent=2)}


def build_trl_dataset_rows(dataset: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    """Convert observation/action examples into a TRL-friendly JSONL payload."""

    rows: list[dict[str, Any]] = []
    for row in dataset:
        rows.append(
            {
                "split": split,
                "scenario_id": row["scenario_id"],
                "workflow_id": row["workflow_id"],
                "phase_index": row["phase_index"],
                "input_text": serialize_observation(row["observation"]),
                "target_text": serialize_action(row["action"]),
                "policy_io_schema": {
                    "input": "serialized_observation_json",
                    "output": "structured_action_json",
                },
            }
        )
    return rows


def build_reference_policy(dataset: list[dict[str, Any]]) -> Any:
    """Build an observation-exact reference policy from a dataset."""

    action_map = {
        observation_signature(row["observation"]): PolicyAction.model_validate(row["action"])
        for row in dataset
    }

    def policy(observation: dict[str, Any]) -> PolicyAction:
        signature = observation_signature(observation)
        if signature not in action_map:
            raise KeyError(f"No reference action for observation signature: {signature}")
        return action_map[signature]

    return policy


def build_supervised_policy_from_train_scenarios(train_scenarios: list[ScenarioSpec]) -> SupervisedStructuredPolicy:
    """Train and return the supervised structured policy from train scenarios."""

    dataset = collect_reference_dataset(train_scenarios)
    policy = SupervisedStructuredPolicy()
    policy.fit(dataset)
    return policy


def build_replay_policy_from_train_scenarios(train_scenarios: list[ScenarioSpec]) -> ReplayMemoryPolicy:
    """Train and return the replay-memory policy from train scenarios."""

    dataset = collect_reference_dataset(train_scenarios)
    policy = ReplayMemoryPolicy()
    for row in dataset:
        policy.update(row["observation"], PolicyAction.model_validate(row["action"]))
    return policy


def write_telemetry_jsonl(telemetry_groups: dict[str, list[dict[str, Any]]], output_path: Path) -> int:
    """Write grouped telemetry rows to a JSONL file."""

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for group_name, rows in telemetry_groups.items():
            for row in rows:
                payload = dict(row)
                payload["telemetry_group"] = group_name
                handle.write(json.dumps(payload) + "\n")
                count += 1
    return count
