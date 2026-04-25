"""Overlay TRL GRPO stage metrics (2- or 3-way) for judge-facing comparison charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_COLORS = ("#2F80ED", "#FF4B4B", "#2ECC71")


def _series(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for i, r in enumerate(rows):
        if key in r and isinstance(r.get(key), int | float):
            xs.append(float(r.get("step", i + 1)))
            ys.append(float(r[key]))
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay TRL training summaries for master charts.")
    parser.add_argument(
        "--summary-1",
        default="artifacts/submission/trl_grpo_run_v1/trl_grpo_training_summary.json",
    )
    parser.add_argument(
        "--summary-2",
        default="artifacts/submission/trl_grpo_run_v2/trl_grpo_training_summary.json",
    )
    parser.add_argument(
        "--summary-3",
        default="",
        help="Optional third training summary (e.g. 3000-row stage) for 3-way plots.",
    )
    parser.add_argument("--label-1", default="stage1_500")
    parser.add_argument("--label-2", default="stage2_1500")
    parser.add_argument("--label-3", default="stage3_3000")
    parser.add_argument("--output-dir", default="artifacts/submission/plots/master")
    args = parser.parse_args()

    root = REPO_ROOT
    paths = [args.summary_1, args.summary_2]
    labels = [args.label_1, args.label_2]
    if args.summary_3.strip():
        paths.append(args.summary_3.strip())
        labels.append(args.label_3)

    summaries: list[dict[str, Any]] = []
    all_rows: list[list[dict[str, Any]]] = []
    for p in paths:
        ap = root / p if not Path(p).is_absolute() else Path(p)
        data = json.loads(ap.read_text(encoding="utf-8"))
        summaries.append(data)
        all_rows.append(list(data.get("metrics_rows") or []))

    n_stages = len(summaries)
    stage_tag = f"{n_stages}-way" if n_stages > 2 else "STAGE1 vs STAGE2"

    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    for key, metric_name, ylabel, fname in [
        ("train/reward", "Train reward", "Reward", "master_reward_compare.png"),
        ("train/loss", "Train loss", "Loss", "master_loss_compare.png"),
        (
            "train/frac_reward_zero_std",
            "frac_reward_zero_std",
            "Fraction",
            "master_zero_std_compare.png",
        ),
        ("eval/reward", "Eval reward", "Eval reward", "master_eval_reward_compare.png"),
    ]:
        series_list = [_series(rows, key) for rows in all_rows]
        if not any(y for _, y in series_list):
            continue
        fig, ax = plt.subplots(figsize=(10, 4.8))
        fig.patch.set_facecolor("#05070A")
        ax.set_facecolor("#05070A")
        for idx, (pair, label) in enumerate(zip(series_list, labels, strict=True)):
            xs, ys = pair
            if ys:
                color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                ax.plot(xs, ys, color=color, linewidth=2, label=label)
        ax.set_title(f"{metric_name} ({stage_tag})", color="#F5F7FA", fontsize=12, weight="bold")
        ax.set_xlabel("step", color="#F5F7FA")
        ax.set_ylabel(ylabel, color="#F5F7FA")
        ax.grid(color="#3F4654", linestyle="-", linewidth=1.0, alpha=0.9)
        ax.tick_params(colors="#F5F7FA")
        for sp in ax.spines.values():
            sp.set_color("#5B6372")
        lg = ax.legend(frameon=False, fontsize=9)
        for t in lg.get_texts():
            t.set_color("#F5F7FA")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    summary_out: dict[str, Any] = {
        "status": "ok",
        "stages": n_stages,
        "output_dir": str((root / args.output_dir).relative_to(root)).replace("\\", "/"),
    }
    for i, s in enumerate(summaries, start=1):
        summary_out[f"stage{i}"] = {
            "prompt_row_count": s.get("prompt_row_count"),
            "train_loss": s.get("metrics", {}).get("train_loss"),
        }

    (root / "artifacts" / "submission" / "master_stage_summary.json").write_text(
        json.dumps(summary_out, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
