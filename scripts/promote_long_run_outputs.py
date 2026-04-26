#!/usr/bin/env python3
"""Promote stronger long-run GRPO outputs and refresh best-run pointers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION_ROOT = REPO_ROOT / "artifacts" / "submission"
DEFAULT_PROMOTION_PATH = DEFAULT_SUBMISSION_ROOT / "best_run_promotion.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_summary(summary: dict[str, Any]) -> float:
    rows = list(summary.get("metrics_rows") or [])
    eval_rewards = [float(row["eval/reward"]) for row in rows if isinstance(row, dict) and "eval/reward" in row]
    if eval_rewards:
        return max(eval_rewards)
    fallback = summary.get("eval_reward_best")
    if isinstance(fallback, int | float):
        return float(fallback)
    return float("-inf")


def _latest_frac_zero_std(summary: dict[str, Any]) -> float | None:
    rows = list(summary.get("metrics_rows") or [])
    for row in reversed(rows):
        value = row.get("train/frac_reward_zero_std") if isinstance(row, dict) else None
        if isinstance(value, int | float):
            return float(value)
    return None


def promote(*, submission_root: Path, promotion_path: Path) -> dict[str, Any]:
    candidates = sorted(submission_root.glob("**/trl_grpo_training_summary.json"))
    if not candidates:
        payload = {
            "status": "warning",
            "promotion_rule": "choose highest eval_reward_best; tie-breaker lower frac_reward_zero_std_last",
            "message": f"No summaries found under {submission_root}",
            "best_run": None,
            "candidates": [],
        }
        promotion_path.parent.mkdir(parents=True, exist_ok=True)
        promotion_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    records: list[dict[str, Any]] = []
    for path in candidates:
        summary = _read_json(path)
        records.append(
            {
                "summary_path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "model_name": summary.get("model_name"),
                "output_dir": summary.get("model_output_dir") or summary.get("trainer_output_dir"),
                "eval_reward_best": _score_summary(summary),
                "frac_reward_zero_std_last": _latest_frac_zero_std(summary),
                "status": summary.get("status"),
            }
        )

    records.sort(
        key=lambda r: (
            float(r.get("eval_reward_best") or float("-inf")),
            -float(r.get("frac_reward_zero_std_last") or 1.0),
        ),
        reverse=True,
    )
    best = records[0]

    payload = {
        "status": "ok",
        "promotion_rule": "choose highest eval_reward_best; tie-breaker lower frac_reward_zero_std_last",
        "best_run": best,
        "candidates": records,
    }
    promotion_path.parent.mkdir(parents=True, exist_ok=True)
    promotion_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote best long-run output into canonical best-run manifest.")
    parser.add_argument("--submission-root", default=str(DEFAULT_SUBMISSION_ROOT))
    parser.add_argument("--promotion-path", default=str(DEFAULT_PROMOTION_PATH))
    args = parser.parse_args()

    payload = promote(submission_root=Path(args.submission_root), promotion_path=Path(args.promotion_path))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
