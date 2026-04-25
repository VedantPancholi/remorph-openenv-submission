"""Plot judge-readable TRL GRPO training metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SERIES_CONFIG = {
    "train/reward": ("TRL GRPO train/reward", "Reward", "trl_train_reward.png"),
    "train/reward_std": ("TRL GRPO train/reward_std", "Reward std", "trl_train_reward_std.png"),
    "train/frac_reward_zero_std": (
        "TRL GRPO train/frac_reward_zero_std",
        "Fraction reward zero std",
        "trl_train_frac_reward_zero_std.png",
    ),
    "train/loss": ("TRL GRPO train/loss", "Loss", "trl_train_loss.png"),
}


def plot_trl_metrics(*, metrics_path: Path, output_dir: Path, smoothing_window: int = 5) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = normalize_metric_rows(payload.get("metrics", []))
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    for key, (title, ylabel, filename) in SERIES_CONFIG.items():
        if _has_series(rows, key):
            artifacts[key] = _plot_series(
                rows=rows,
                y_key=key,
                title=title,
                ylabel=ylabel,
                output_path=output_dir / filename,
                smoothing_window=smoothing_window,
            )

    summary = {
        "status": "ok",
        "metrics_path": _display_path(metrics_path),
        "row_count": len(rows),
        "smoothing_window": smoothing_window,
        "artifacts": artifacts,
    }
    (output_dir / "trl_grpo_plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def normalize_metric_rows(metrics: Any) -> list[dict[str, Any]]:
    """Normalize trainer/dry-run metric rows to canonical plotting keys."""

    rows: list[dict[str, Any]] = []
    if not isinstance(metrics, list):
        return rows
    for index, item in enumerate(metrics):
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {"step": int(item.get("step") or item.get("global_step") or index)}
        aliases = {
            "reward": "train/reward",
            "raw_reward": "train/raw_reward",
            "mean_reward": "train/reward",
            "rewards/mean": "train/reward",
            "train/reward": "train/reward",
            "reward_std": "train/reward_std",
            "rewards/std": "train/reward_std",
            "train/reward_std": "train/reward_std",
            "frac_reward_zero_std": "train/frac_reward_zero_std",
            "rewards/frac_zero_std": "train/frac_reward_zero_std",
            "train/frac_reward_zero_std": "train/frac_reward_zero_std",
            "loss": "train/loss",
            "train_loss": "train/loss",
            "train/loss": "train/loss",
        }
        for source_key, dest_key in aliases.items():
            value = item.get(source_key)
            if isinstance(value, int | float):
                row[dest_key] = float(value)
        if len(row) > 1:
            rows.append(row)
    return rows


def _has_series(rows: list[dict[str, Any]], key: str) -> bool:
    return any(key in row and isinstance(row.get(key), int | float) for row in rows)


def _plot_series(
    *,
    rows: list[dict[str, Any]],
    y_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    smoothing_window: int,
) -> str:
    points = [
        (float(row.get("step", index)), float(row[y_key]))
        for index, row in enumerate(rows)
        if y_key in row and isinstance(row.get(y_key), int | float)
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    smoothed = _moving_average(ys, window=max(1, smoothing_window))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor("#05070A")
    ax.set_facecolor("#05070A")
    ax.plot(xs, ys, color="#2F80ED", linewidth=1.4, alpha=0.9, label=f"{y_key}_raw")
    ax.plot(xs, smoothed, color="#FF4B4B", linewidth=2.4, alpha=0.95, label=f"{y_key}_smooth")
    ax.set_title(title, color="#F5F7FA", fontsize=12, weight="bold")
    ax.set_xlabel("step", color="#F5F7FA", fontsize=10)
    ax.set_ylabel(ylabel, color="#F5F7FA", fontsize=10)
    ax.grid(color="#3F4654", linestyle="-", linewidth=1.1, alpha=0.9)
    ax.tick_params(colors="#F5F7FA", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#5B6372")
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, -0.27), ncol=2, frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color("#F5F7FA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return _display_path(output_path)


def _moving_average(values: list[float], *, window: int) -> list[float]:
    averaged: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TRL GRPO metrics.")
    parser.add_argument("--metrics-path", default="artifacts/submission/trl_grpo_run/trl_grpo_metrics.json")
    parser.add_argument("--output-dir", default="artifacts/submission/plots")
    parser.add_argument("--smoothing-window", type=int, default=5)
    args = parser.parse_args()

    summary = plot_trl_metrics(
        metrics_path=REPO_ROOT / args.metrics_path,
        output_dir=REPO_ROOT / args.output_dir,
        smoothing_window=args.smoothing_window,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
