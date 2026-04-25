from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.training import build_benchmark_report, train_supervised_policy


def test_benchmark_report_contains_supervised_eval_section() -> None:
    result = train_supervised_policy(seed=42)
    report = build_benchmark_report(result)

    assert "ReMorph Benchmark Report" in report["markdown"]
    assert "supervised" in report["markdown"]
    assert "\"supervised_eval_success_rate\"" in report["json"]
    assert "baseline_eval_success_rate" in report["json"]
