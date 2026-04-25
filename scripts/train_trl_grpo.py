"""Train a ReMorph tool-calling policy with TRL GRPO.

The script intentionally supports a dependency-light dry run:

    python scripts/train_trl_grpo.py --dry-run

Real training requires the optional training stack:

    pip install trl transformers datasets accelerate torch peft
    python scripts/train_trl_grpo.py --train --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.trl_env import (  # noqa: E402
    build_grpo_prompt_rows,
    environment_reward,
    make_environment_factory,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def run_dry_run(
    *,
    output_dir: Path,
    seed: int,
    split: str,
    repeats: int,
    train_path: Path | None = None,
) -> dict[str, Any]:
    """Validate prompt rows and one environment rollout without importing TRL."""

    rows = _load_jsonl(train_path) if train_path else build_grpo_prompt_rows(seed=seed, split=split, repeats=repeats)
    if not rows:
        raise ValueError("No GRPO prompt rows were available for dry-run.")
    factory = make_environment_factory(seed=seed, split=split)
    env = factory()
    first_row = rows[0]
    observation_text = env.reset(**first_row)
    feedback = env.no_op("Dry run intentionally checks that low-quality actions receive feedback.")
    metrics = [
        {
            "step": 0,
            "train/reward": 0.0,
            "train/reward_std": 0.0,
            "train/frac_reward_zero_std": 1.0,
            "raw_reward": 0.0,
            "success": 0.0,
            "note": "episode_reset",
        },
        {
            "step": 1,
            "train/reward": env.reward,
            "train/reward_std": 0.0,
            "train/frac_reward_zero_std": 1.0,
            "raw_reward": env.raw_reward,
            "success": 1.0 if env.success else 0.0,
            "note": "intentional_no_op_probe",
        },
    ]
    payload = {
        "status": "dry_run_ok",
        "row_count": len(rows),
        "train_path": _display_path(train_path) if train_path else None,
        "first_scenario_id": first_row["scenario_id"] if rows else None,
        "first_seed": first_row.get("seed"),
        "first_prompt": first_row["prompt"] if rows else [],
        "initial_observation_preview": observation_text[:1000],
        "feedback_preview": feedback[:1000],
        "reward_after_no_op": env.reward,
        "raw_reward_after_no_op": env.raw_reward,
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trl_grpo_dry_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "trl_grpo_metrics.json").write_text(json.dumps({"metrics": metrics}, indent=2), encoding="utf-8")
    return payload


def run_training(
    *,
    output_dir: Path,
    model_name: str,
    seed: int,
    split: str,
    repeats: int,
    max_steps: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    max_prompt_length: int,
    max_completion_length: int,
    use_vllm: bool,
    train_path: Path | None = None,
    eval_path: Path | None = None,
    eval_steps: int = 25,
) -> dict[str, Any]:
    """Run GRPO with TRL using ReMorphToolEnv as the environment factory."""

    try:
        from datasets import Dataset  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing optional training dependencies. Install: "
            "pip install trl transformers datasets accelerate torch peft"
        ) from exc

    rows = _load_jsonl(train_path) if train_path else build_grpo_prompt_rows(seed=seed, split=split, repeats=repeats)
    if not rows:
        raise ValueError("No GRPO prompt rows were generated.")
    eval_rows = _load_jsonl(eval_path) if eval_path else []

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "grpo_prompt_rows.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_list(rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
    trainer_output_dir = output_dir / "trainer"
    model_output_dir = output_dir / "model"

    training_args = GRPOConfig(
        output_dir=str(trainer_output_dir),
        run_name="remorph-openenv-grpo",
        learning_rate=float(learning_rate),
        max_steps=max(1, int(max_steps)),
        per_device_train_batch_size=max(1, int(per_device_train_batch_size)),
        gradient_accumulation_steps=max(1, int(gradient_accumulation_steps)),
        num_generations=max(2, int(num_generations)),
        max_prompt_length=max(128, int(max_prompt_length)),
        max_completion_length=max(128, int(max_completion_length)),
        logging_steps=1,
        eval_strategy="steps" if eval_rows else "no",
        eval_steps=max(1, int(eval_steps)) if eval_rows else None,
        save_strategy="steps",
        save_steps=max(10, min(100, int(max_steps))),
        report_to="none",
        remove_unused_columns=False,
        seed=int(seed),
        use_vllm=bool(use_vllm),
        chat_template_kwargs={"enable_thinking": False},
    )

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=environment_reward,
        environment_factory=make_environment_factory(seed=seed, split=split),
    )
    train_result = trainer.train()
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    metrics = dict(getattr(train_result, "metrics", {}) or {})
    metrics_rows = _trainer_metrics_to_rows(
        log_history=list(getattr(getattr(trainer, "state", None), "log_history", []) or []),
        final_metrics=metrics,
    )
    summary = {
        "status": "completed",
        "trainer": "trl_grpo_environment_factory",
        "model_name": model_name,
        "seed": seed,
        "split": split,
        "prompt_row_count": len(rows),
        "eval_prompt_row_count": len(eval_rows),
        "dataset_path": str(dataset_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_train_path": _display_path(train_path) if train_path else None,
        "source_eval_path": _display_path(eval_path) if eval_path else None,
        "trainer_output_dir": str(trainer_output_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "model_output_dir": str(model_output_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "metrics": metrics,
        "metrics_rows": metrics_rows,
        "config": {
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_generations": num_generations,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            "use_vllm": use_vllm,
            "eval_steps": eval_steps,
        },
    }
    (output_dir / "trl_grpo_training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "trl_grpo_metrics.json").write_text(json.dumps({"metrics": metrics_rows}, indent=2), encoding="utf-8")
    return summary


def _trainer_metrics_to_rows(*, log_history: list[dict[str, Any]], final_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert trainer logs into plot-friendly step rows."""

    rows: list[dict[str, Any]] = []
    for index, log in enumerate(log_history):
        row: dict[str, Any] = {"step": int(log.get("step") or log.get("global_step") or index + 1)}
        for source_key, dest_key in {
            "loss": "train/loss",
            "train_loss": "train/loss",
            "reward": "train/reward",
            "rewards/mean": "train/reward",
            "reward_std": "train/reward_std",
            "rewards/std": "train/reward_std",
            "frac_reward_zero_std": "train/frac_reward_zero_std",
            "rewards/frac_zero_std": "train/frac_reward_zero_std",
            "eval_reward": "eval/reward",
            "eval_rewards/mean": "eval/reward",
        }.items():
            if source_key in log:
                row[dest_key] = log[source_key]
        if len(row) > 1:
            rows.append(row)

    if rows:
        return rows

    row = {"step": int(final_metrics.get("global_step") or final_metrics.get("train_steps") or 1)}
    for source_key, dest_key in {
        "train_loss": "train/loss",
        "loss": "train/loss",
        "reward": "train/reward",
        "mean_reward": "train/reward",
    }.items():
        if source_key in final_metrics:
            row[dest_key] = final_metrics[source_key]
    return [row]


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ReMorph with TRL GRPO environment_factory.")
    parser.add_argument("--output-dir", default="artifacts/submission/trl_grpo_run")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-path", default="")
    parser.add_argument("--eval-path", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "eval", "all"], default="train")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/env plumbing without TRL.")
    parser.add_argument("--train", action="store_true", help="Run real TRL GRPO training.")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    train_path = REPO_ROOT / args.train_path if args.train_path else None
    eval_path = REPO_ROOT / args.eval_path if args.eval_path else None
    if args.train and args.dry_run:
        raise ValueError("Choose only one of --train or --dry-run.")
    if not args.train:
        result = run_dry_run(
            output_dir=output_dir,
            seed=args.seed,
            split=args.split,
            repeats=args.repeats,
            train_path=train_path,
        )
    else:
        result = run_training(
            output_dir=output_dir,
            model_name=args.model,
            seed=args.seed,
            split=args.split,
            repeats=args.repeats,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_generations=args.num_generations,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            use_vllm=args.use_vllm,
            train_path=train_path,
            eval_path=eval_path,
            eval_steps=args.eval_steps,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
