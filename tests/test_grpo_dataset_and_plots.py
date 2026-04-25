from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_grpo_dataset import generate_grpo_dataset
from scripts.plot_trl_grpo import plot_trl_metrics
from scripts.train_trl_grpo import run_dry_run


def test_generate_grpo_dataset_writes_seeded_train_eval_files(tmp_path: Path) -> None:
    summary = generate_grpo_dataset(
        output_dir=tmp_path,
        train_seed_start=1000,
        train_seed_count=50,
        eval_seed_start=2000,
        eval_seed_count=10,
    )

    train_path = tmp_path / "train_prompts.jsonl"
    eval_path = tmp_path / "eval_prompts.jsonl"
    summary_path = tmp_path / "dataset_summary.json"
    train_rows = _read_jsonl(train_path)
    eval_rows = _read_jsonl(eval_path)

    assert train_path.exists()
    assert eval_path.exists()
    assert summary_path.exists()
    assert summary["train_row_count"] >= 500
    assert summary["eval_row_count"] >= 60
    assert len(train_rows) == summary["train_row_count"]
    assert len(eval_rows) == summary["eval_row_count"]
    assert {row["seed"] for row in train_rows}.isdisjoint({row["seed"] for row in eval_rows})
    assert {"seed", "scenario_id", "prompt", "scenario_type", "benchmark_partition", "workflow_length"} <= set(
        train_rows[0]
    )


def test_train_trl_grpo_dry_run_accepts_generated_train_path(tmp_path: Path) -> None:
    generate_grpo_dataset(
        output_dir=tmp_path / "dataset",
        train_seed_start=1000,
        train_seed_count=1,
        eval_seed_start=2000,
        eval_seed_count=1,
    )

    result = run_dry_run(
        output_dir=tmp_path / "run",
        seed=42,
        split="train",
        repeats=1,
        train_path=tmp_path / "dataset" / "train_prompts.jsonl",
    )

    assert result["status"] == "dry_run_ok"
    assert result["row_count"] == 10
    assert result["first_seed"] == 1000
    assert (tmp_path / "run" / "trl_grpo_metrics.json").exists()


def test_plot_trl_grpo_writes_dark_grid_training_artifacts(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": [
                    {
                        "step": step,
                        "train/reward": step / 10,
                        "train/reward_std": 0.2 / (step + 1),
                        "train/frac_reward_zero_std": 1.0 / (step + 1),
                        "train/loss": 1.0 - step / 20,
                    }
                    for step in range(1, 8)
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = plot_trl_metrics(metrics_path=metrics_path, output_dir=tmp_path / "plots", smoothing_window=3)

    assert "train/reward" in summary["artifacts"]
    assert "train/reward_std" in summary["artifacts"]
    assert "train/frac_reward_zero_std" in summary["artifacts"]
    assert "train/loss" in summary["artifacts"]
    assert (tmp_path / "plots" / "trl_train_reward.png").exists()
    assert (tmp_path / "plots" / "trl_train_reward_std.png").exists()
    assert (tmp_path / "plots" / "trl_train_frac_reward_zero_std.png").exists()
    assert (tmp_path / "plots" / "trl_train_loss.png").exists()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
