"""Generate reproducible split manifests for the Phase 1 benchmark pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.scenarios import DEFAULT_BENCHMARK_SEED, scenario_catalog


def _manifest_rows(catalog: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": row["scenario_id"],
            "workflow_id": row["workflow_id"],
            "validation_tier": row["validation_tier"],
            "execution_mode": row["execution_mode"],
            "split": row["split"],
            "workflow_length": row["workflow_length"],
            "service_domain": row["service_domain"],
            "benchmark_partition": row["benchmark_partition"],
            "phase_count": row["phase_count"],
        }
        for row in catalog
        if row["split"] == split
    ]


def generate_split_manifests(*, seed: int = DEFAULT_BENCHMARK_SEED) -> dict[str, Any]:
    catalog = scenario_catalog(seed=seed, randomize=True)
    output_dir = REPO_ROOT / "artifacts" / "submission" / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = {
        "seed": seed,
        "split": "train",
        "scenarios": _manifest_rows(catalog, "train"),
    }
    train_manifest["scenario_count"] = len(train_manifest["scenarios"])
    eval_manifest = {
        "seed": seed,
        "split": "eval",
        "scenarios": _manifest_rows(catalog, "eval"),
    }
    eval_manifest["scenario_count"] = len(eval_manifest["scenarios"])
    catalog_payload = {
        "seed": seed,
        "scenario_count": len(catalog),
        "validation_scenario_count": len([row for row in catalog if row["validation_tier"] == "phase1"]),
        "benchmark_scenario_count": len([row for row in catalog if row["validation_tier"] != "phase1"]),
        "scenarios": catalog,
    }

    (output_dir / "train_manifest.json").write_text(json.dumps(train_manifest, indent=2), encoding="utf-8")
    (output_dir / "eval_manifest.json").write_text(json.dumps(eval_manifest, indent=2), encoding="utf-8")
    (output_dir / "scenario_catalog.json").write_text(json.dumps(catalog_payload, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "seed": seed,
        "output_dir": str(output_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "train_scenario_count": len(train_manifest["scenarios"]),
        "eval_scenario_count": len(eval_manifest["scenarios"]),
        "catalog_scenario_count": len(catalog),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 1 train/eval split manifests.")
    parser.add_argument("--seed", type=int, default=DEFAULT_BENCHMARK_SEED)
    args = parser.parse_args()
    print(json.dumps(generate_split_manifests(seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
