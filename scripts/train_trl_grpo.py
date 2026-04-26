"""Train a ReMorph tool-calling policy with TRL GRPO.

The script intentionally supports a dependency-light dry run:

    python scripts/train_trl_grpo.py --dry-run

Real training requires the optional training stack (transformers>=5.2 for TRL
``environment_factory``):

    pip install "transformers>=5.2" trl datasets accelerate torch peft
    python scripts/train_trl_grpo.py --train --model Qwen/Qwen2.5-0.5B-Instruct

Faster local CPU setup (avoids multi-gigabyte CUDA wheels; use GPU wheels on HF/Colab):

    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python .venv/bin/python "transformers>=5.2" trl accelerate peft datasets
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.trl_env import (  # noqa: E402
    build_grpo_prompt_rows,
    environment_reward,
    make_environment_factory,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

_MIN_TRANSFORMERS_FOR_GRPO_ENV = (5, 2, 0)


def _transformers_version_tuple(version_str: str) -> tuple[int, int, int]:
    base = version_str.split("+")[0].split("rc")[0].strip()
    parts: list[int] = []
    for seg in base.split(".")[:3]:
        if seg.isdigit():
            parts.append(int(seg))
        else:
            break
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def _require_transformers_for_grpo_environment_factory() -> None:
    """TRL raises at GRPOTrainer init if transformers is too old; fail fast with a clear fix."""

    import transformers  # type: ignore

    ver = getattr(transformers, "__version__", "0")
    if _transformers_version_tuple(ver) < _MIN_TRANSFORMERS_FOR_GRPO_ENV:
        raise RuntimeError(
            f"transformers {ver} is too old for GRPOTrainer(environment_factory=...) "
            f"(need >= {_MIN_TRANSFORMERS_FOR_GRPO_ENV[0]}.{_MIN_TRANSFORMERS_FOR_GRPO_ENV[1]}). "
            "Run: pip install -U --force-reinstall 'transformers>=5.2.0,<6' "
            "and use pip_constraints.txt with requirements-training.txt, then retry. "
            "For tool-calling GRPO, also: pip install 'jmespath>=1.0,<2'."
        )




def _ensure_tool_response_schema(*, processing_class: Any, model_name: str) -> None:
    """Ensure TRL can parse tool calls for templates not yet recognized by add_response_schema."""

    if getattr(processing_class, "response_schema", None) is not None:
        return

    try:
        from trl.chat_template_utils import add_response_schema  # type: ignore

        add_response_schema(processing_class)
        return
    except Exception as exc:  # noqa: BLE001
        normalized = model_name.lower().replace("_", "")
        if "qwen2.5" in normalized or "qwen25" in normalized:
            # Qwen2.5 tool format is compatible with TRL's Qwen3 parser schema.
            from trl.chat_template_utils import qwen3_schema  # type: ignore

            processing_class.response_schema = qwen3_schema
            return
        raise RuntimeError(
            "Failed to configure tokenizer response schema for tool-calling GRPO. "
            "Either use a model with a recognized chat template, or manually set "
            "tokenizer.response_schema before creating GRPOTrainer."
        ) from exc
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


def _grpo_config_field_names(grpo_config_cls: type) -> set[str]:
    """Collect dataclass field names from GRPOConfig and its bases (TrainingArguments, etc.)."""

    names: set[str] = set()
    for cls in grpo_config_cls.__mro__:
        if cls is object:
            continue
        if dataclasses.is_dataclass(cls):
            names.update(f.name for f in dataclasses.fields(cls))
    return names


def _prompt_token_length(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    chat_template_kwargs: dict[str, Any] | None,
) -> int:
    kwargs = dict(chat_template_kwargs or {})
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("enable_thinking", None)
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
    if isinstance(ids, list):
        return len(ids)
    if hasattr(ids, "shape"):
        return int(ids.shape[-1])
    return len(ids)


def _trim_row_prompt_to_max_tokens(
    row: dict[str, Any],
    tokenizer: Any,
    max_tokens: int,
    *,
    chat_template_kwargs: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a row copy whose chat prompt fits in max_tokens, or None if it cannot be made to fit."""

    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        return dict(row)
    messages = [dict(m) for m in prompt]
    if _prompt_token_length(tokenizer, messages, chat_template_kwargs=chat_template_kwargs) <= max_tokens:
        return {**row, "prompt": messages}
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_indices:
        return None
    idx = user_indices[-1]
    original = str(messages[idx].get("content") or "")
    marker = "\n\n[... truncated for GRPO max prompt length ...]\n\n"
    best = 0
    lo, hi = 0, len(original)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = original[:mid]
        if mid < len(original):
            candidate = candidate.rstrip() + marker
        messages[idx]["content"] = candidate
        if _prompt_token_length(tokenizer, messages, chat_template_kwargs=chat_template_kwargs) <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best == 0:
        return None
    final_text = original[:best].rstrip()
    if best < len(original):
        final_text = final_text + marker
    messages[idx]["content"] = final_text
    if _prompt_token_length(tokenizer, messages, chat_template_kwargs=chat_template_kwargs) > max_tokens:
        return None
    return {**row, "prompt": messages}


def _trim_rows_for_max_prompt_tokens(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
    *,
    chat_template_kwargs: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Ensure each row's prompt encodes to at most max_tokens (TRL may omit max_prompt_length on GRPOConfig)."""

    out: list[dict[str, Any]] = []
    for row in rows:
        adjusted = _trim_row_prompt_to_max_tokens(
            row,
            tokenizer,
            max_tokens,
            chat_template_kwargs=chat_template_kwargs,
        )
        if adjusted is not None:
            out.append(adjusted)
    stats = {
        "input_rows": len(rows),
        "output_rows": len(out),
        "dropped": len(rows) - len(out),
    }
    return out, stats


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
    resume_from_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run GRPO with TRL using ReMorphToolEnv as the environment factory."""

    try:
        import torch  # type: ignore
        from datasets import Dataset  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        _require_transformers_for_grpo_environment_factory()
        try:
            import jmespath  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "TRL GRPO with tools requires jmespath. Install: pip install 'jmespath>=1.0,<2'"
            ) from exc
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    _ensure_tool_response_schema(processing_class=tokenizer, model_name=model_name)

    chat_template_kwargs: dict[str, Any] = {"enable_thinking": False}
    max_prompt_tokens = max(128, int(max_prompt_length))
    rows, trim_stats_train = _trim_rows_for_max_prompt_tokens(
        rows,
        tokenizer,
        max_prompt_tokens,
        chat_template_kwargs=chat_template_kwargs,
    )
    if eval_rows:
        eval_rows, trim_stats_eval = _trim_rows_for_max_prompt_tokens(
            eval_rows,
            tokenizer,
            max_prompt_tokens,
            chat_template_kwargs=chat_template_kwargs,
        )
    else:
        trim_stats_eval = {"input_rows": 0, "output_rows": 0, "dropped": 0}

    if not rows:
        raise ValueError(
            "All training rows were dropped after fitting prompts to --max-prompt-length. "
            "Increase --max-prompt-length or shorten scenario text."
        )

    dataset_path = output_dir / "grpo_prompt_rows.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    train_dataset = Dataset.from_list(rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
    trainer_output_dir = output_dir / "trainer"
    model_output_dir = output_dir / "model"

    allowed = _grpo_config_field_names(GRPOConfig)
    base_config_kwargs: dict[str, Any] = {
        "output_dir": str(trainer_output_dir),
        "run_name": "remorph-openenv-grpo",
        "learning_rate": float(learning_rate),
        "max_steps": max(1, int(max_steps)),
        "per_device_train_batch_size": max(1, int(per_device_train_batch_size)),
        "gradient_accumulation_steps": max(1, int(gradient_accumulation_steps)),
        "num_generations": max(2, int(num_generations)),
        "max_prompt_length": max_prompt_tokens,
        "max_completion_length": max(128, int(max_completion_length)),
        "logging_steps": 1,
        # Keep eval batch aligned with environment instances; TRL environment_factory
        # currently expects prompt batch size not to exceed env count.
        "per_device_eval_batch_size": max(1, int(per_device_train_batch_size)),
        "eval_strategy": "steps" if eval_rows else "no",
        "eval_steps": max(1, int(eval_steps)) if eval_rows else None,
        "save_strategy": "steps",
        "save_steps": max(10, min(100, int(max_steps))),
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": int(seed),
        "use_vllm": bool(use_vllm),
        "chat_template_kwargs": chat_template_kwargs,
    }
    if not torch.cuda.is_available():
        # TRL/transformers default to bf16 on GPU; CPU training requires explicit flags.
        for key, value in (
            ("use_cpu", True),
            ("bf16", False),
            ("fp16", False),
            ("use_vllm", False),
        ):
            if key in allowed:
                base_config_kwargs[key] = value
    config_kwargs = {key: value for key, value in base_config_kwargs.items() if key in allowed}
    training_args = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=environment_reward,
        environment_factory=make_environment_factory(seed=seed, split=split),
    )
    resume_checkpoint_value: str | bool | None
    if resume_from_checkpoint is None:
        resume_checkpoint_value = None
    elif str(resume_from_checkpoint).lower() == "latest":
        resume_checkpoint_value = True
    else:
        if not resume_from_checkpoint.exists():
            raise ValueError(f"Checkpoint path does not exist: {resume_from_checkpoint}")
        resume_checkpoint_value = str(resume_from_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint_value)
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
        "grpo_prompt_trim_stats": {"train": trim_stats_train, "eval": trim_stats_eval},
        "dataset_path": str(dataset_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_train_path": _display_path(train_path) if train_path else None,
        "source_eval_path": _display_path(eval_path) if eval_path else None,
        "resume_from_checkpoint": (
            str(resume_from_checkpoint)
            if resume_from_checkpoint is not None
            else None
        ),
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
    _update_run_ledger(output_dir=output_dir, summary=summary)
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


def _best_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and row.get(key) is not None]
    return max(values) if values else None


def _last_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and row.get(key) is not None]
    return values[-1] if values else None


def _update_run_ledger(*, output_dir: Path, summary: dict[str, Any]) -> None:
    """Append one TRL run result to persistent tabular ledgers."""

    metrics_rows = list(summary.get("metrics_rows") or [])
    metrics = dict(summary.get("metrics") or {})
    config = dict(summary.get("config") or {})
    ledger_row: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": summary.get("status"),
        "model_name": summary.get("model_name"),
        "seed": summary.get("seed"),
        "split": summary.get("split"),
        "prompt_row_count": summary.get("prompt_row_count"),
        "eval_prompt_row_count": summary.get("eval_prompt_row_count"),
        "source_train_path": summary.get("source_train_path"),
        "source_eval_path": summary.get("source_eval_path"),
        "trainer_output_dir": summary.get("trainer_output_dir"),
        "model_output_dir": summary.get("model_output_dir"),
        "resume_from_checkpoint": summary.get("resume_from_checkpoint"),
        "max_steps": config.get("max_steps"),
        "learning_rate": config.get("learning_rate"),
        "per_device_train_batch_size": config.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
        "num_generations": config.get("num_generations"),
        "max_prompt_length": config.get("max_prompt_length"),
        "max_completion_length": config.get("max_completion_length"),
        "use_vllm": config.get("use_vllm"),
        "train_loss_final": (
            metrics.get("train_loss")
            if metrics.get("train_loss") is not None
            else _last_value(metrics_rows, "train/loss")
        ),
        "reward_best": _best_value(metrics_rows, "train/reward"),
        "reward_last": _last_value(metrics_rows, "train/reward"),
        "reward_std_last": _last_value(metrics_rows, "train/reward_std"),
        "frac_reward_zero_std_last": _last_value(metrics_rows, "train/frac_reward_zero_std"),
        "eval_reward_best": _best_value(metrics_rows, "eval/reward"),
        "eval_reward_last": _last_value(metrics_rows, "eval/reward"),
    }

    root = output_dir.parent
    json_path = root / "trl_run_ledger.json"
    csv_path = root / "trl_run_ledger.csv"
    markdown_path = root / "trl_run_ledger.md"

    existing_rows: list[dict[str, Any]] = []
    if json_path.exists():
        existing_payload = json.loads(json_path.read_text(encoding="utf-8"))
        existing_rows = list(existing_payload.get("runs") or [])
    existing_rows.append(ledger_row)

    best_run = max(
        existing_rows,
        key=lambda row: (
            float(row.get("eval_reward_best") or row.get("reward_best") or -9999.0),
            -float(row.get("frac_reward_zero_std_last") or 1.0),
        ),
    )
    json_payload = {"runs": existing_rows, "best_run": best_run}
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    fieldnames = list(ledger_row.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

    markdown_lines = [
        "# TRL Run Ledger",
        "",
        f"- Total runs: `{len(existing_rows)}`",
        f"- Best run model: `{best_run.get('model_name')}`",
        f"- Best run eval_reward_best: `{best_run.get('eval_reward_best')}`",
        "",
        "| timestamp_utc | model_name | prompt_row_count | max_steps | reward_best | eval_reward_best | frac_reward_zero_std_last | model_output_dir |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in existing_rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("timestamp_utc") or ""),
                    str(row.get("model_name") or ""),
                    str(row.get("prompt_row_count") or ""),
                    str(row.get("max_steps") or ""),
                    str(row.get("reward_best") or ""),
                    str(row.get("eval_reward_best") or ""),
                    str(row.get("frac_reward_zero_std_last") or ""),
                    str(row.get("model_output_dir") or ""),
                ]
            )
            + " |"
        )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")


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
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Checkpoint path or 'latest' to continue an earlier GRPO run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate data/env plumbing without TRL.")
    parser.add_argument("--train", action="store_true", help="Run real TRL GRPO training.")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    train_path = REPO_ROOT / args.train_path if args.train_path else None
    eval_path = REPO_ROOT / args.eval_path if args.eval_path else None
    resume_from_checkpoint = (
        Path(args.resume_from_checkpoint)
        if args.resume_from_checkpoint and args.resume_from_checkpoint != "latest"
        else (Path("latest") if args.resume_from_checkpoint == "latest" else None)
    )
    if resume_from_checkpoint is not None and resume_from_checkpoint.name != "latest":
        resume_from_checkpoint = (
            REPO_ROOT / args.resume_from_checkpoint
            if not resume_from_checkpoint.is_absolute()
            else resume_from_checkpoint
        )
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
            resume_from_checkpoint=resume_from_checkpoint,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
