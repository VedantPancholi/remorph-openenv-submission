"""Evaluate one submission policy on a manifest-selected split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.policies import baseline_action
from remorph_openenv.training import (
    build_reference_policy,
    build_replay_policy_from_train_scenarios,
    build_supervised_policy_from_train_scenarios,
    collect_reference_dataset,
    load_scenarios_from_manifest,
    rollout_policy,
)


def evaluate_policy(
    *,
    policy_name: str,
    split: str,
    train_manifest: Path,
    eval_manifest: Path,
    seed: int,
) -> dict[str, Any]:
    train_scenarios = load_scenarios_from_manifest(train_manifest, seed=seed)
    eval_scenarios = load_scenarios_from_manifest(eval_manifest, seed=seed)
    scenarios = train_scenarios if split == "train" else eval_scenarios

    if policy_name == "baseline":
        policy_fn = baseline_action
    elif policy_name == "supervised":
        policy_fn = build_supervised_policy_from_train_scenarios(train_scenarios).predict
    elif policy_name == "replay":
        policy_fn = build_replay_policy_from_train_scenarios(train_scenarios).predict
    elif policy_name in {"adaptive_reference", "oracle"}:
        reference_dataset = collect_reference_dataset(scenarios)
        policy_fn = build_reference_policy(reference_dataset)
    else:
        raise ValueError(f"Unsupported policy: {policy_name}")

    summary = rollout_policy(scenarios=scenarios, policy_fn=policy_fn, policy_name=policy_name)
    return {
        "policy_name": policy_name,
        "split": split,
        "seed": seed,
        "train_manifest": str(train_manifest.relative_to(REPO_ROOT)).replace("\\", "/"),
        "eval_manifest": str(eval_manifest.relative_to(REPO_ROOT)).replace("\\", "/"),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one policy on a selected manifest split.")
    parser.add_argument("--policy", choices=["baseline", "supervised", "replay", "adaptive_reference", "oracle"], required=True)
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-manifest", default="artifacts/submission/splits/train_manifest.json")
    parser.add_argument("--eval-manifest", default="artifacts/submission/splits/eval_manifest.json")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    result = evaluate_policy(
        policy_name=args.policy,
        split=args.split,
        train_manifest=REPO_ROOT / args.train_manifest,
        eval_manifest=REPO_ROOT / args.eval_manifest,
        seed=args.seed,
    )
    if args.output_path:
        output_path = REPO_ROOT / args.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
