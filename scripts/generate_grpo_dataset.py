"""Generate seeded GRPO prompt datasets for ReMorph environment training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.trl_env import build_grpo_prompt_rows  # noqa: E402


DEFAULT_OUTPUT_DIR = "artifacts/submission/grpo_dataset"


def generate_grpo_dataset(
    *,
    output_dir: Path,
    train_seed_start: int = 1000,
    train_seed_count: int = 50,
    eval_seed_start: int = 2000,
    eval_seed_count: int = 10,
    execution_mode: str = "simulated",
) -> dict[str, Any]:
    """Generate train/eval JSONL files with non-overlapping seeded variants."""

    train_seeds = list(range(train_seed_start, train_seed_start + train_seed_count))
    eval_seeds = list(range(eval_seed_start, eval_seed_start + eval_seed_count))
    overlap = sorted(set(train_seeds) & set(eval_seeds))
    if overlap:
        raise ValueError(f"Train and eval seed ranges overlap: {overlap[:5]}")

    train_rows = _rows_for_seeds(train_seeds, split="train", execution_mode=execution_mode)
    eval_rows = _rows_for_seeds(eval_seeds, split="eval", execution_mode=execution_mode)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_prompts.jsonl"
    eval_path = output_dir / "eval_prompts.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    summary = {
        "status": "ok",
        "execution_mode": execution_mode,
        "train_seed_start": train_seed_start,
        "train_seed_count": train_seed_count,
        "eval_seed_start": eval_seed_start,
        "eval_seed_count": eval_seed_count,
        "train_seed_min": min(train_seeds) if train_seeds else None,
        "train_seed_max": max(train_seeds) if train_seeds else None,
        "eval_seed_min": min(eval_seeds) if eval_seeds else None,
        "eval_seed_max": max(eval_seeds) if eval_seeds else None,
        "seed_overlap_count": len(overlap),
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "train_path": _display_path(train_path),
        "eval_path": _display_path(eval_path),
        "scenario_type_counts": {
            "train": _count_by_key(train_rows, "scenario_type"),
            "eval": _count_by_key(eval_rows, "scenario_type"),
        },
        "benchmark_partition_counts": {
            "train": _count_by_key(train_rows, "benchmark_partition"),
            "eval": _count_by_key(eval_rows, "benchmark_partition"),
        },
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _rows_for_seeds(seeds: list[int], *, split: str, execution_mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.extend(
            build_grpo_prompt_rows(
                seed=seed,
                split=split,
                execution_mode=execution_mode,
                repeats=1,
            )
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seeded ReMorph GRPO prompt datasets.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-seed-start", type=int, default=1000)
    parser.add_argument("--train-seed-count", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=2000)
    parser.add_argument("--eval-seed-count", type=int, default=10)
    parser.add_argument("--execution-mode", default="simulated")
    args = parser.parse_args()

    summary = generate_grpo_dataset(
        output_dir=REPO_ROOT / args.output_dir,
        train_seed_start=args.train_seed_start,
        train_seed_count=args.train_seed_count,
        eval_seed_start=args.eval_seed_start,
        eval_seed_count=args.eval_seed_count,
        execution_mode=args.execution_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
