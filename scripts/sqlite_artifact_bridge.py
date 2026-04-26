#!/usr/bin/env python3
"""Build a lightweight SQLite bridge from submission artifacts for frontend use."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "artifacts" / "submission" / "frontend_bridge.sqlite3"
DEFAULT_SUBMISSION_ROOT = REPO_ROOT / "artifacts" / "submission"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            timestamp_utc TEXT,
            model_name TEXT,
            stage_label TEXT,
            status TEXT,
            prompt_row_count INTEGER,
            eval_prompt_row_count INTEGER,
            max_steps INTEGER,
            reward_best REAL,
            reward_last REAL,
            eval_reward_best REAL,
            eval_reward_last REAL,
            frac_reward_zero_std_last REAL,
            model_output_dir TEXT,
            source_type TEXT
        );

        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            step INTEGER,
            metric_name TEXT,
            metric_value REAL,
            source_type TEXT
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            artifact_type TEXT,
            artifact_path TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run_id, step);
        CREATE INDEX IF NOT EXISTS idx_metrics_metric_name ON metrics(metric_name);
        CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
        """
    )


def _upsert_run(conn: sqlite3.Connection, row: dict[str, Any], source_type: str) -> str:
    stage_label = "unknown"
    model_output_dir = str(row.get("model_output_dir") or "")
    if "run_v1" in model_output_dir:
        stage_label = "stage1"
    elif "run_v2" in model_output_dir:
        stage_label = "stage2"
    elif "run_v3" in model_output_dir:
        stage_label = "stage3"

    run_id = (
        str(row.get("timestamp_utc") or "").replace(":", "_")
        + "__"
        + str(row.get("model_name") or "unknown_model")
        + "__"
        + stage_label
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (
            run_id, timestamp_utc, model_name, stage_label, status,
            prompt_row_count, eval_prompt_row_count, max_steps,
            reward_best, reward_last, eval_reward_best, eval_reward_last,
            frac_reward_zero_std_last, model_output_dir, source_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            row.get("timestamp_utc"),
            row.get("model_name"),
            stage_label,
            row.get("status"),
            row.get("prompt_row_count"),
            row.get("eval_prompt_row_count"),
            row.get("max_steps"),
            row.get("reward_best"),
            row.get("reward_last"),
            row.get("eval_reward_best"),
            row.get("eval_reward_last"),
            row.get("frac_reward_zero_std_last"),
            row.get("model_output_dir"),
            source_type,
        ),
    )
    return run_id


def _insert_artifact(conn: sqlite3.Connection, run_id: str, artifact_type: str, artifact_path: str) -> None:
    conn.execute(
        "INSERT INTO artifacts (run_id, artifact_type, artifact_path) VALUES (?, ?, ?)",
        (run_id, artifact_type, artifact_path),
    )


def _insert_metric_rows(conn: sqlite3.Connection, run_id: str, metrics_rows: list[dict[str, Any]], source_type: str) -> None:
    for row in metrics_rows:
        step = int(row.get("step") or 0)
        for key, value in row.items():
            if key == "step":
                continue
            if isinstance(value, int | float):
                conn.execute(
                    """
                    INSERT INTO metrics (run_id, step, metric_name, metric_value, source_type)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, step, key, float(value), source_type),
                )


def build_sqlite_bridge(*, db_path: Path, submission_root: Path) -> dict[str, Any]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        conn.execute("DELETE FROM metrics")
        conn.execute("DELETE FROM artifacts")
        conn.execute("DELETE FROM runs")

        run_count = 0
        metric_count = 0

        ledger_path = submission_root / "trl_run_ledger.json"
        if ledger_path.exists():
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            for row in payload.get("runs", []):
                run_id = _upsert_run(conn, row, source_type="ledger")
                run_count += 1
                model_output_dir = str(row.get("model_output_dir") or "")
                if model_output_dir:
                    _insert_artifact(conn, run_id, "model_output_dir", model_output_dir)

        for summary_path in sorted(submission_root.glob("trl_grpo_run_v*/trl_grpo_training_summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            fake_row = {
                "timestamp_utc": summary.get("metrics", {}).get("train_runtime"),
                "model_name": summary.get("model_name"),
                "status": summary.get("status"),
                "prompt_row_count": summary.get("prompt_row_count"),
                "eval_prompt_row_count": summary.get("eval_prompt_row_count"),
                "max_steps": summary.get("config", {}).get("max_steps"),
                "reward_best": max(
                    [float(r.get("train/reward")) for r in summary.get("metrics_rows", []) if "train/reward" in r] or [0.0]
                ),
                "reward_last": (
                    float(summary.get("metrics_rows", [{}])[-1].get("train/reward"))
                    if summary.get("metrics_rows")
                    else None
                ),
                "eval_reward_best": max(
                    [float(r.get("eval/reward")) for r in summary.get("metrics_rows", []) if "eval/reward" in r] or [0.0]
                ),
                "eval_reward_last": (
                    float(summary.get("metrics_rows", [{}])[-1].get("eval/reward"))
                    if summary.get("metrics_rows") and "eval/reward" in summary.get("metrics_rows", [{}])[-1]
                    else None
                ),
                "frac_reward_zero_std_last": (
                    float(summary.get("metrics_rows", [{}])[-1].get("train/frac_reward_zero_std"))
                    if summary.get("metrics_rows") and "train/frac_reward_zero_std" in summary.get("metrics_rows", [{}])[-1]
                    else None
                ),
                "model_output_dir": summary.get("model_output_dir"),
            }
            run_id = _upsert_run(conn, fake_row, source_type="summary")
            run_count += 1
            _insert_metric_rows(conn, run_id, list(summary.get("metrics_rows") or []), source_type="summary")
            metric_count += sum(max(0, len(r) - 1) for r in list(summary.get("metrics_rows") or []))

            for artifact in [
                summary.get("dataset_path"),
                summary.get("trainer_output_dir"),
                summary.get("model_output_dir"),
                _display_path(summary_path),
            ]:
                if artifact:
                    _insert_artifact(conn, run_id, "summary_artifact", str(artifact))

        for plot_path in sorted((submission_root / "plots").rglob("*.png")):
            conn.execute(
                "INSERT INTO artifacts (run_id, artifact_type, artifact_path) VALUES (?, ?, ?)",
                ("global", "plot_png", _display_path(plot_path)),
            )

        conn.commit()

        run_rows = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        metric_rows = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        artifact_rows = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        summary = {
            "status": "ok",
            "db_path": _display_path(db_path),
            "submission_root": _display_path(submission_root),
            "imported_run_records": run_count,
            "imported_metric_records": metric_count,
            "sqlite_counts": {
                "runs": run_rows,
                "metrics": metric_rows,
                "artifacts": artifact_rows,
            },
        }
        (submission_root / "sqlite_bridge_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create frontend-ready SQLite bridge from submission artifacts.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--submission-root", default=str(DEFAULT_SUBMISSION_ROOT))
    args = parser.parse_args()

    summary = build_sqlite_bridge(
        db_path=Path(args.db_path),
        submission_root=Path(args.submission_root),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
