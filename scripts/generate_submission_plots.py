"""Generate readable submission plots for the clean repo."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLOTS_DIR = REPO_ROOT / "artifacts" / "submission" / "plots"
TRAINING_DIR = REPO_ROOT / "artifacts" / "submission" / "training_run"
DPI = 220
POLICY_ORDER = ["baseline", "replay", "supervised", "oracle"]
POLICY_COLORS = {
    "baseline": "#3478F6",
    "replay": "#FF9F43",
    "supervised": "#28A745",
    "oracle": "#DC3545",
}


def _load_json(path: Path):
    if not path.exists():
        warnings.warn(f"Missing input file: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _figure(figsize: tuple[float, float] = (10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    return fig, ax


def _finalize(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _fallback_plot(path: Path, title: str, message: str, metrics: dict[str, float] | None = None) -> None:
    fig, ax = _figure()
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold")
    lines = [message]
    if metrics:
        lines.extend(f"{label}: {value:.4f}" for label, value in metrics.items())
    ax.text(
        0.5,
        0.5,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F6F8FA", "edgecolor": "#C7D0D9"},
        transform=ax.transAxes,
    )
    _finalize(fig, path)


def _format_ticks(ax) -> None:
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)


def _annotate_bars(ax, bars, fmt: str = "{:.2f}", y_offset_ratio: float = 0.02) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax != ymin else 1.0
    offset = span * y_offset_ratio
    for bar in bars:
        height = bar.get_height()
        y = height + offset if height >= 0 else height + offset * 0.35
        va = "bottom" if height >= 0 else "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=10,
            weight="bold",
        )


def _plot_line(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[tuple[str, list[float], list[float], str]],
    y_limits: tuple[float, float] | None = None,
    integer_x: bool = False,
    fallback_metrics: dict[str, float] | None = None,
) -> None:
    valid_series = [(label, xs, ys, color) for label, xs, ys, color in series if xs and ys and len(xs) == len(ys)]
    if not valid_series:
        warnings.warn(f"No usable data for {path.name}; creating fallback plot.")
        _fallback_plot(path, f"{title} (Fallback)", "No time-series data was available.", fallback_metrics)
        return

    fig, ax = _figure()
    for label, xs, ys, color in valid_series:
        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=6, label=label, color=color)

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if integer_x:
        unique_xs = sorted({x for _, xs, _, _ in valid_series for x in xs})
        if len(unique_xs) == 1:
            x_value = unique_xs[0]
            ax.set_xticks([x_value])
            ax.set_xlim(x_value - 0.5, x_value + 0.5)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if len(valid_series) > 1:
        ax.legend(frameon=False)
    _format_ticks(ax)
    _finalize(fig, path)


def _plot_bars(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    y_limits: tuple[float, float] | None = None,
    rotation: int = 0,
    fallback_metrics: dict[str, float] | None = None,
    value_fmt: str = "{:.2f}",
) -> None:
    if not labels or not values or len(labels) != len(values):
        warnings.warn(f"No usable bar data for {path.name}; creating fallback plot.")
        _fallback_plot(path, f"{title} (Fallback)", "Comparison values were unavailable.", fallback_metrics)
        return

    fig, ax = _figure()
    bars = ax.bar(labels, values, color=colors, edgecolor="#243447", linewidth=0.8)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        ymin = min(0.0, min(values))
        ymax = max(values) if values else 1.0
        span = max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - span * 0.08, ymax + span * 0.18)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right" if rotation else "center")
    _annotate_bars(ax, bars, fmt=value_fmt)
    _format_ticks(ax)
    _finalize(fig, path)


def _copy_plot(src: Path, dest: Path) -> None:
    if src.exists():
        dest.write_bytes(src.read_bytes())


def _get_eval_policy_metrics(eval_summary: dict | None, metric_name: str) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    values: list[float] = []
    if not isinstance(eval_summary, dict):
        return labels, values

    for policy in POLICY_ORDER:
        metrics = eval_summary.get(policy)
        if isinstance(metrics, dict) and metric_name in metrics:
            labels.append(policy.title())
            values.append(float(metrics[metric_name]))
    return labels, values


def _get_reward_history_rows(reward_history: list | None) -> list[dict]:
    return [row for row in reward_history or [] if isinstance(row, dict)]


def _generate_loss_curve(loss_history: list | None, training_summary: dict | None) -> None:
    rows = [row for row in loss_history or [] if isinstance(row, dict) and "epoch" in row and "mismatch_rate" in row]
    fallback_metrics = {}
    if isinstance(training_summary, dict):
        if "final_train_success_rate" in training_summary:
            fallback_metrics["Final Train Success"] = float(training_summary["final_train_success_rate"])
        if "final_eval_success_rate" in training_summary:
            fallback_metrics["Final Eval Success"] = float(training_summary["final_eval_success_rate"])

    _plot_line(
        PLOTS_DIR / "loss_curve.png",
        "Loss Curve (Mismatch Rate Proxy)",
        "Epoch",
        "Loss",
        [("Mismatch Rate", [float(row["epoch"]) for row in rows], [float(row["mismatch_rate"]) for row in rows], POLICY_COLORS["baseline"])],
        y_limits=(0.0, max(1.05, max((float(row["mismatch_rate"]) for row in rows), default=1.0) * 1.05)),
        integer_x=True,
        fallback_metrics=fallback_metrics or None,
    )


def _generate_reward_curve(reward_history: list | None, training_summary: dict | None) -> None:
    rows = _get_reward_history_rows(reward_history)
    train_rows = [row for row in rows if row.get("split") == "train" and "average_normalized_reward" in row and "epoch" in row]
    eval_rows = [row for row in rows if row.get("split") == "eval" and "average_normalized_reward" in row and "epoch" in row]
    fallback_metrics = {}
    if isinstance(training_summary, dict):
        if "final_eval_average_raw_reward" in training_summary:
            fallback_metrics["Final Eval Raw Reward"] = float(training_summary["final_eval_average_raw_reward"])
        if "final_eval_success_rate" in training_summary:
            fallback_metrics["Final Eval Success"] = float(training_summary["final_eval_success_rate"])

    _plot_line(
        PLOTS_DIR / "reward_curve.png",
        "Normalized Reward Curve",
        "Epoch",
        "Average Reward (Normalized)",
        [
            ("Train", [float(row["epoch"]) for row in train_rows], [float(row["average_normalized_reward"]) for row in train_rows], POLICY_COLORS["supervised"]),
            ("Eval", [float(row["epoch"]) for row in eval_rows], [float(row["average_normalized_reward"]) for row in eval_rows], POLICY_COLORS["replay"]),
        ],
        integer_x=True,
        fallback_metrics=fallback_metrics or None,
    )


def _generate_success_rate_comparison(eval_summary: dict | None) -> None:
    labels, values = _get_eval_policy_metrics(eval_summary, "success_rate")
    _plot_bars(
        PLOTS_DIR / "success_rate_comparison.png",
        "Success Rate Comparison",
        "Policy",
        "Success Rate",
        labels,
        values,
        [POLICY_COLORS[label.lower()] for label in labels],
        y_limits=(0.0, 1.05),
        value_fmt="{:.2f}",
    )


def _generate_avg_reward_comparison(eval_summary: dict | None) -> None:
    labels, values = _get_eval_policy_metrics(eval_summary, "average_episode_normalized_return_capped")
    _plot_bars(
        PLOTS_DIR / "avg_reward_comparison.png",
        "Average Episode Return Comparison (Normalized)",
        "Policy",
        "Average Episode Return (Normalized)",
        labels,
        values,
        [POLICY_COLORS[label.lower()] for label in labels],
        value_fmt="{:.4f}",
    )


def _generate_train_vs_eval_success(reward_history: list | None, training_summary: dict | None) -> None:
    rows = _get_reward_history_rows(reward_history)
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for policy in ("baseline", "supervised"):
        for split in ("train", "eval"):
            match = next(
                (
                    row for row in rows
                    if row.get("policy_name") == policy and row.get("split") == split and "success_rate" in row
                ),
                None,
            )
            if match is not None:
                labels.append(f"{policy.title()} {split.title()}")
                values.append(float(match["success_rate"]))
                colors.append(POLICY_COLORS[policy])

    fallback_metrics = {}
    if isinstance(training_summary, dict):
        if "final_train_success_rate" in training_summary:
            fallback_metrics["Final Train Success"] = float(training_summary["final_train_success_rate"])
        if "final_eval_success_rate" in training_summary:
            fallback_metrics["Final Eval Success"] = float(training_summary["final_eval_success_rate"])

    output_path = PLOTS_DIR / "train_vs_eval_success.png"
    _plot_bars(
        output_path,
        "Train vs Eval Success Rate",
        "Policy / Split",
        "Success Rate",
        labels,
        values,
        colors,
        y_limits=(0.0, 1.05),
        rotation=15,
        fallback_metrics=fallback_metrics or None,
        value_fmt="{:.2f}",
    )
    _copy_plot(output_path, PLOTS_DIR / "train_eval_success_comparison.png")


def _generate_episode_return_comparison(reward_history: list | None, training_summary: dict | None) -> None:
    rows = _get_reward_history_rows(reward_history)
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for policy in ("baseline", "supervised"):
        for split in ("train", "eval"):
            match = next(
                (
                    row for row in rows
                    if row.get("policy_name") == policy and row.get("split") == split and "average_raw_reward" in row
                ),
                None,
            )
            if match is not None:
                labels.append(f"{policy.title()} {split.title()}")
                values.append(float(match["average_raw_reward"]))
                colors.append(POLICY_COLORS[policy])

    fallback_metrics = {}
    if isinstance(training_summary, dict) and "final_eval_average_raw_reward" in training_summary:
        fallback_metrics["Final Eval Raw Reward"] = float(training_summary["final_eval_average_raw_reward"])

    output_path = PLOTS_DIR / "episode_return_comparison.png"
    _plot_bars(
        output_path,
        "Episode Return Comparison (Raw Reward)",
        "Policy / Split",
        "Average Episode Return (Raw Reward)",
        labels,
        values,
        colors,
        rotation=15,
        fallback_metrics=fallback_metrics or None,
        value_fmt="{:.2f}",
    )
    _copy_plot(output_path, PLOTS_DIR / "train_eval_raw_reward_comparison.png")


def main() -> None:
    warnings.simplefilter("always", UserWarning)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    loss_history = _load_json(TRAINING_DIR / "loss_history.json")
    reward_history = _load_json(TRAINING_DIR / "reward_history.json")
    eval_summary = _load_json(TRAINING_DIR / "eval_summary.json")
    training_summary = _load_json(TRAINING_DIR / "training_summary.json")

    _generate_loss_curve(loss_history, training_summary)
    _generate_reward_curve(reward_history, training_summary)
    _generate_success_rate_comparison(eval_summary)
    _generate_avg_reward_comparison(eval_summary)
    _generate_train_vs_eval_success(reward_history, training_summary)
    _generate_episode_return_comparison(reward_history, training_summary)

    print(json.dumps({"status": "ok", "plots_dir": str(PLOTS_DIR)}, indent=2))


if __name__ == "__main__":
    main()
