"""Overlay Stage-1 vs Stage-2 TRL metrics for judge-facing comparison charts."""

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


def _series(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for i, r in enumerate(rows):
        if key in r and isinstance(r.get(key), int | float):
            xs.append(float(r.get("step", i + 1)))
            ys.append(float(r[key]))
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-1",
        default="artifacts/submission/trl_grpo_run_v1/trl_grpo_training_summary.json",
    )
    parser.add_argument(
        "--summary-2",
        default="artifacts/submission/trl_grpo_run_v2/trl_grpo_training_summary.json",
    )
    parser.add_argument("--label-1", default="stage1_500")
    parser.add_argument("--label-2", default="stage2_1500")
    parser.add_argument("--output-dir", default="artifacts/submission/plots/master")
    args = parser.parse_args()

    root = REPO_ROOT
    s1 = json.loads((root / args.summary_1).read_text(encoding="utf-8"))
    s2 = json.loads((root / args.summary_2).read_text(encoding="utf-8"))
    rows1 = list(s1.get("metrics_rows") or [])
    rows2 = list(s2.get("metrics_rows") or [])

    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    for key, title, ylabel, fname in [
        ("train/reward", "Train reward (STAGE1 vs STAGE2)", "Reward", "master_reward_compare.png"),
        ("train/loss", "Train loss (STAGE1 vs STAGE2)", "Loss", "master_loss_compare.png"),
        (
            "train/frac_reward_zero_std",
            "frac_reward_zero_std (STAGE1 vs STAGE2)",
            "Fraction",
            "master_zero_std_compare.png",
        ),
        ("eval/reward", "Eval reward (STAGE1 vs STAGE2)", "Eval reward", "master_eval_reward_compare.png"),
    ]:
        x1, y1 = _series(rows1, key)
        x2, y2 = _series(rows2, key)
        if not y1 and not y2:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.8))
        fig.patch.set_facecolor("#05070A")
        ax.set_facecolor("#05070A")
        if y1:
            ax.plot(x1, y1, color="#2F80ED", linewidth=2, label=args.label_1)
        if y2:
            ax.plot(x2, y2, color="#FF4B4B", linewidth=2, label=args.label_2)
        ax.set_title(title, color="#F5F7FA", fontsize=12, weight="bold")
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

    summary = {
        "status": "ok",
        "output_dir": str((root / args.output_dir).relative_to(root)).replace("\\", "/"),
        "stage1": {
            "prompt_row_count": s1.get("prompt_row_count"),
            "train_loss": s1.get("metrics", {}).get("train_loss"),
        },
        "stage2": {
            "prompt_row_count": s2.get("prompt_row_count"),
            "train_loss": s2.get("metrics", {}).get("train_loss"),
        },
    }
    (root / "artifacts" / "submission" / "master_stage_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
