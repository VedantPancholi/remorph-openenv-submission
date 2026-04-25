from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_submission import evaluate_policy


def test_evaluate_submission_reports_supervised_eval_improvement() -> None:
    train_manifest = REPO_ROOT / "artifacts" / "submission" / "splits" / "train_manifest.json"
    eval_manifest = REPO_ROOT / "artifacts" / "submission" / "splits" / "eval_manifest.json"

    baseline = evaluate_policy(
        policy_name="baseline",
        split="eval",
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        seed=42,
    )
    supervised = evaluate_policy(
        policy_name="supervised",
        split="eval",
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        seed=42,
    )
    adaptive = evaluate_policy(
        policy_name="adaptive_reference",
        split="eval",
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        seed=42,
    )

    assert baseline["summary"]["success_rate"] == 0.0
    assert supervised["summary"]["success_rate"] == 1.0
    assert adaptive["summary"]["success_rate"] == 1.0
    assert supervised["summary"]["average_raw_reward"] > baseline["summary"]["average_raw_reward"]
