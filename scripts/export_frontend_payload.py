#!/usr/bin/env python3
"""Export deterministic frontend JSON payload from SQLite bridge."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "artifacts" / "submission" / "frontend_bridge.sqlite3"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "artifacts" / "submission" / "frontend_payload.json"
CONTRACT_VERSION = "remorph-ui-v1"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _rows_to_dicts(cursor: sqlite3.Cursor, rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    colnames = [d[0] for d in cursor.description]
    return [dict(zip(colnames, row, strict=False)) for row in rows]


def export_payload(*, db_path: Path, output_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        runs_cur = conn.execute(
            """
            SELECT
                run_id, timestamp_utc, model_name, stage_label, status,
                prompt_row_count, eval_prompt_row_count, max_steps,
                reward_best, reward_last, eval_reward_best, eval_reward_last,
                frac_reward_zero_std_last, model_output_dir, source_type
            FROM runs
            ORDER BY
                COALESCE(eval_reward_best, -1e18) DESC,
                run_id ASC
            """
        )
        runs = _rows_to_dicts(runs_cur, runs_cur.fetchall())

        metrics_cur = conn.execute(
            """
            SELECT run_id, step, metric_name, metric_value, source_type
            FROM metrics
            ORDER BY run_id ASC, step ASC, metric_name ASC
            """
        )
        metric_rows = _rows_to_dicts(metrics_cur, metrics_cur.fetchall())
        metrics_by_run: dict[str, list[dict[str, Any]]] = {}
        for row in metric_rows:
            run_id = str(row["run_id"])
            metrics_by_run.setdefault(run_id, []).append(
                {
                    "step": int(row["step"]),
                    "metric_name": str(row["metric_name"]),
                    "metric_value": float(row["metric_value"]),
                    "source_type": str(row["source_type"]),
                }
            )

        artifacts_cur = conn.execute(
            """
            SELECT run_id, artifact_type, artifact_path
            FROM artifacts
            ORDER BY run_id ASC, artifact_type ASC, artifact_path ASC
            """
        )
        artifact_rows = _rows_to_dicts(artifacts_cur, artifacts_cur.fetchall())
        artifacts_by_run: dict[str, list[dict[str, Any]]] = {}
        for row in artifact_rows:
            run_id = str(row["run_id"])
            artifacts_by_run.setdefault(run_id, []).append(
                {
                    "artifact_type": str(row["artifact_type"]),
                    "artifact_path": str(row["artifact_path"]),
                }
            )

        metric_names_cur = conn.execute("SELECT DISTINCT metric_name FROM metrics ORDER BY metric_name ASC")
        metric_names = [str(row[0]) for row in metric_names_cur.fetchall()]

        payload = {
            "contract_version": CONTRACT_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "db_path": _display_path(db_path),
            "summary": {
                "total_runs": len(runs),
                "total_metric_points": len(metric_rows),
                "total_artifacts": len(artifact_rows),
                "metric_names": metric_names,
            },
            "runs": runs,
            "metrics_by_run": metrics_by_run,
            "artifacts_by_run": artifacts_by_run,
            "best_run": runs[0] if runs else None,
        }
    finally:
        conn.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frontend payload JSON from bridge SQLite DB.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    payload = export_payload(db_path=Path(args.db_path), output_path=Path(args.output_path))
    print(
        json.dumps(
            {
                "status": "ok",
                "contract_version": payload["contract_version"],
                "output_path": _display_path(Path(args.output_path)),
                "summary": payload["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
