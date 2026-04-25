"""Run the split-aware supervised training loop inside the clean submission repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.training import (
    build_benchmark_report,
    build_trl_dataset_rows,
    load_scenarios_from_manifest,
    train_supervised_policy,
    write_telemetry_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the clean ReMorph supervised policy.")
    parser.add_argument("--output-dir", default="artifacts/submission/training_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-manifest", default="artifacts/submission/splits/train_manifest.json")
    parser.add_argument("--eval-manifest", default="artifacts/submission/splits/eval_manifest.json")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    telemetry_dir = REPO_ROOT / "artifacts" / "submission" / "telemetry"
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    train_manifest_path = REPO_ROOT / args.train_manifest
    eval_manifest_path = REPO_ROOT / args.eval_manifest
    train_scenarios = load_scenarios_from_manifest(train_manifest_path, seed=args.seed)
    eval_scenarios = load_scenarios_from_manifest(eval_manifest_path, seed=args.seed)

    result = train_supervised_policy(
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        seed=args.seed,
    )
    report = build_benchmark_report(result)

    loss_history = result["loss_history"]
    reward_history = result["reward_history"]
    eval_summary = {
        "baseline": result["baseline_eval_summary"],
        "replay": result["replay_eval_summary"],
        "supervised": result["supervised_eval_summary"],
        "oracle": result["oracle_eval_summary"],
    }
    training_summary = {
        "trainer": result["trainer"],
        "seed": result["seed"],
        "train_scenario_count": result["train_scenario_count"],
        "eval_scenario_count": result["eval_scenario_count"],
        "training_example_count": result["training_example_count"],
        "final_train_success_rate": result["supervised_train_summary"]["success_rate"],
        "final_eval_success_rate": result["supervised_eval_summary"]["success_rate"],
        "final_eval_average_raw_reward": result["supervised_eval_summary"]["average_raw_reward"],
        "observe_act_reward_learn_repeat": True,
    }
    dataset_stats = {
        "train_example_count": result["training_example_count"],
        "eval_example_count": result["eval_example_count"],
        "train_scenario_count": result["train_scenario_count"],
        "eval_scenario_count": result["eval_scenario_count"],
    }
    checkpoint_metadata = {
        "policy_type": result["trainer"],
        "seed": result["seed"],
        "train_manifest": str(train_manifest_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "eval_manifest": str(eval_manifest_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "model_config": result["model_config"],
    }
    trl_dataset_rows = build_trl_dataset_rows(result["train_dataset"], split="train") + build_trl_dataset_rows(
        result["eval_dataset"],
        split="eval",
    )

    (output_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
    (output_dir / "loss_history.json").write_text(json.dumps(loss_history, indent=2), encoding="utf-8")
    (output_dir / "reward_history.json").write_text(json.dumps(reward_history, indent=2), encoding="utf-8")
    (output_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    (output_dir / "model_config.json").write_text(json.dumps(result["model_config"], indent=2), encoding="utf-8")
    (output_dir / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=2), encoding="utf-8")
    (output_dir / "checkpoint_metadata.json").write_text(json.dumps(checkpoint_metadata, indent=2), encoding="utf-8")
    (output_dir / "train_reference_dataset.json").write_text(json.dumps(result["train_dataset"], indent=2), encoding="utf-8")
    (output_dir / "eval_reference_dataset.json").write_text(json.dumps(result["eval_dataset"], indent=2), encoding="utf-8")
    with (output_dir / "trl_dataset.jsonl").open("w", encoding="utf-8") as handle:
        for row in trl_dataset_rows:
            handle.write(json.dumps(row) + "\n")
    (REPO_ROOT / "artifacts" / "submission" / "benchmark_report.md").write_text(report["markdown"], encoding="utf-8")
    (REPO_ROOT / "artifacts" / "submission" / "benchmark_report.json").write_text(report["json"], encoding="utf-8")

    telemetry_count = write_telemetry_jsonl(result["telemetry"], telemetry_dir / "rollouts.jsonl")

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "telemetry_rows": telemetry_count,
                "trl_dataset_rows": len(trl_dataset_rows),
                "training_summary": training_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
