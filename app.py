"""Hugging Face Space demo for the ReMorph OpenEnv environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from remorph_openenv.environment import ReMorphEnvironment
from remorph_openenv.models import PolicyAction
from remorph_openenv.policies import SupervisedStructuredPolicy
from remorph_openenv.scenarios import ScenarioSpec, load_built_in_scenarios
from remorph_openenv.training import build_supervised_policy_from_train_scenarios


ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "artifacts" / "submission" / "plots"
BENCHMARK_REPORT = ROOT / "artifacts" / "submission" / "benchmark_report.json"
TRAIN_MANIFEST = ROOT / "artifacts" / "submission" / "splits" / "train_manifest.json"
MODEL_CONFIG = ROOT / "artifacts" / "submission" / "training_run" / "model_config.json"

EXECUTION_MODES = ["simulated", "live_local"]
POLICIES = ["baseline", "supervised", "adaptive_reference", "unsafe_auth_hallucination"]
PLOT_FILES = [
    ("Training loss", "loss_curve.png"),
    ("Reward curve", "reward_curve.png"),
    ("Success rate comparison", "success_rate_comparison.png"),
    ("Average reward comparison", "avg_reward_comparison.png"),
]

# Benchmark plots live under `plots/`; TRL / Track A runs often write `trl_*.png` under `plots/tracka_*`.
_PLOT_FALLBACK_NAMES: dict[str, list[str]] = {
    "loss_curve.png": ["trl_train_loss.png"],
    "reward_curve.png": ["trl_train_reward.png"],
}
_PLOT_SUBDIRS = ("tracka_stage2", "tracka_stage1", "tracka_master", "master", "stage3", "stage2", "stage1")


def _json(data: Any) -> Any:
    """Return a JSON-displayable object without leaking Pydantic internals."""

    if isinstance(data, PolicyAction):
        return data.model_dump(mode="json")
    return json.loads(json.dumps(data, default=str))


def _scenarios(execution_mode: str) -> list[ScenarioSpec]:
    return load_built_in_scenarios(
        seed=42,
        split="all",
        execution_mode=execution_mode,
        randomize=False,
    )


def _scenario_choices(execution_mode: str) -> list[str]:
    return [scenario.scenario_id for scenario in _scenarios(execution_mode)]


def _scenario_by_id(execution_mode: str, scenario_id: str) -> ScenarioSpec:
    for scenario in _scenarios(execution_mode):
        if scenario.scenario_id == scenario_id:
            return scenario
    raise ValueError(f"Unknown scenario_id for {execution_mode}: {scenario_id}")


def _current_phase(scenario: ScenarioSpec, observation: dict[str, Any]) -> int:
    index = int(observation.get("step_index") or 0)
    return max(0, min(index, len(scenario.phases) - 1))


def _deterministic_supervised_action(observation: dict[str, Any]) -> PolicyAction:
    return SupervisedStructuredPolicy().predict(observation)


def _supervised_action(observation: dict[str, Any]) -> PolicyAction:
    if MODEL_CONFIG.exists() or TRAIN_MANIFEST.exists():
        try:
            train_scenarios = load_built_in_scenarios(seed=42, split="train", randomize=False)
            return build_supervised_policy_from_train_scenarios(train_scenarios).predict(observation)
        except Exception:
            pass
    return _deterministic_supervised_action(observation)


def _unsafe_auth_action(observation: dict[str, Any]) -> PolicyAction:
    failed_request = dict(observation.get("failed_request") or {})
    candidate_routes = list(observation.get("candidate_routes") or [])
    route = dict(candidate_routes[0]) if candidate_routes else {}
    scenario_type = str(observation.get("scenario_type") or "")
    partition = str(observation.get("benchmark_partition") or "")

    if scenario_type == "auth_drift" and partition == "unrecoverable":
        return PolicyAction(
            action_type="repair_auth",
            target_method=str(route.get("method") or failed_request.get("method") or "GET").upper(),
            target_path=str(route.get("path") or failed_request.get("path") or "/"),
            header_patch={
                "Authorization": "Bearer fake-demo-token",
                "x-api-key": "fabricated-demo-key",
            },
            reason="Unsafe trap: fabricate credentials on an unrecoverable auth failure.",
        )

    return PolicyAction(
        action_type="no_op",
        reason="The unsafe credential hallucination trap is meaningful on unrecoverable auth scenarios.",
    )


def _choose_action(policy: str, scenario: ScenarioSpec, observation: dict[str, Any]) -> PolicyAction:
    if policy == "baseline":
        return PolicyAction(action_type="no_op", reason="Baseline demo policy intentionally takes no repair action.")
    if policy == "adaptive_reference":
        return scenario.phases[_current_phase(scenario, observation)].expected_action
    if policy == "supervised":
        return _supervised_action(observation)
    if policy == "unsafe_auth_hallucination":
        return _unsafe_auth_action(observation)
    return PolicyAction(action_type="no_op", reason=f"Unknown policy: {policy}")


def _status(info: dict[str, Any] | None, extra: str = "") -> str:
    if not info:
        return extra or "Scenario loaded. Ready to run a policy."
    verdict = "SUCCESS" if info.get("success") else "FAILURE"
    reward = info.get("raw_reward")
    normalized = info.get("normalized_reward")
    suffix = f"\n\n{extra}" if extra else ""
    return f"{verdict} | raw reward: {reward} | normalized reward: {normalized}{suffix}"


def _empty_outputs(message: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    return {}, {}, {}, {}, message


def load_scenario(execution_mode: str, scenario_id: str) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    try:
        scenario = _scenario_by_id(execution_mode, scenario_id)
        env = ReMorphEnvironment(
            scenarios=[scenario],
            seed=42,
            execution_mode=execution_mode,
            randomize=False,
        )
        step_env = env
        if execution_mode == "live_local":
            step_env = ReMorphEnvironment(
                scenarios=[scenario],
                seed=42,
                execution_mode="simulated",
                randomize=False,
            )
        observation = env.reset(scenario_id=scenario_id)
        if step_env is not env:
            step_env.reset(scenario_id=scenario_id)
        state = {
            "execution_mode": execution_mode,
            "scenario_id": scenario_id,
            "observation": observation,
        }
        state["_env"] = env
        state["_step_env"] = step_env
        state["_scenario"] = scenario
        return state, _json(observation), {}, {}, {}, _status(None)
    except Exception as exc:
        return {}, *_empty_outputs(f"Could not load scenario: {exc}")


def run_selected_policy(
    state: dict[str, Any],
    execution_mode: str,
    scenario_id: str,
    policy: str,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    try:
        if not state or "_env" not in state:
            state, observation, _, _, _, _ = load_scenario(execution_mode, scenario_id)
        else:
            observation = dict(state.get("observation") or {})

        env = state.get("_step_env") or state["_env"]
        scenario = state["_scenario"]
        action = _choose_action(policy, scenario, observation)
        next_observation, reward, done, info = env.step(action)
        info = dict(info)
        info["done"] = done
        info["normalized_reward"] = reward
        if execution_mode == "live_local" and state.get("_step_env") is not state.get("_env"):
            info["execution_note"] = (
                "Loaded a live_local ReMorphEnvironment. This Space executes steps with the deterministic "
                "simulated mirror so the demo remains responsive without external services or secrets."
            )
        state["observation"] = next_observation

        result = {
            "next_observation": next_observation,
            "done": done,
            "info": info,
        }
        extra = ""
        if policy == "unsafe_auth_hallucination" and action.action_type == "no_op":
            extra = "Select an unrecoverable auth scenario to see the credential-hallucination penalty."
        if info.get("execution_note"):
            extra = f"{extra}\n{info['execution_note']}".strip()
        return state, _json(next_observation), _json(action), _json(result), _json(info.get("reward_breakdown", {})), _status(info, extra)
    except Exception as exc:
        return state or {}, *_empty_outputs(f"Could not run policy: {exc}")


def run_full_demo(execution_mode: str, scenario_id: str) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    try:
        selected = _scenario_by_id(execution_mode, scenario_id)
        scenario = selected
        if len(selected.phases) < 2:
            multi_step = [candidate for candidate in _scenarios(execution_mode) if len(candidate.phases) > 1]
            if multi_step:
                scenario = multi_step[0]

        display_env = ReMorphEnvironment(
            scenarios=[scenario],
            seed=42,
            execution_mode=execution_mode,
            randomize=False,
        )
        env = display_env
        if execution_mode == "live_local":
            env = ReMorphEnvironment(
                scenarios=[scenario],
                seed=42,
                execution_mode="simulated",
                randomize=False,
            )
        observation = env.reset(scenario_id=scenario.scenario_id)
        if display_env is not env:
            display_env.reset(scenario_id=scenario.scenario_id)
        initial_observation = observation
        actions: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        done = False

        while not done and len(steps) < scenario.max_steps + len(scenario.phases):
            action = _choose_action("adaptive_reference", scenario, observation)
            observation, reward, done, info = env.step(action)
            info = dict(info)
            info["normalized_reward"] = reward
            info["done"] = done
            if execution_mode == "live_local" and display_env is not env:
                info["execution_note"] = (
                    "Loaded a live_local ReMorphEnvironment. This Space executes steps with the deterministic "
                    "simulated mirror so the demo remains responsive without external services or secrets."
                )
            actions.append(action.model_dump(mode="json"))
            steps.append(
                {
                    "action": action.model_dump(mode="json"),
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
            )

        state = {
            "execution_mode": execution_mode,
            "scenario_id": scenario.scenario_id,
            "observation": observation,
            "_env": display_env,
            "_step_env": env,
            "_scenario": scenario,
        }
        result = {
            "demo_scenario_id": scenario.scenario_id,
            "workflow_id": scenario.workflow_id,
            "phase_count": len(scenario.phases),
            "steps": steps,
        }
        final_info = steps[-1]["info"] if steps else {}
        extra = "Full demo uses the adaptive reference plan across every workflow phase."
        if final_info.get("execution_note"):
            extra = f"{extra}\n{final_info['execution_note']}"
        return (
            state,
            _json(initial_observation),
            _json(actions),
            _json(result),
            _json(final_info.get("reward_breakdown", {})),
            _status(final_info, extra),
        )
    except Exception as exc:
        return {}, *_empty_outputs(f"Could not run full demo: {exc}")


def update_scenarios(execution_mode: str) -> Any:
    choices = _scenario_choices(execution_mode)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def benchmark_rows() -> list[list[Any]]:
    if not BENCHMARK_REPORT.exists():
        return [["missing_report", "artifacts/submission/benchmark_report.json not found"]]
    try:
        payload = json.loads(BENCHMARK_REPORT.read_text(encoding="utf-8"))
        metrics = dict(payload.get("metrics") or {})
        return [[key, value] for key, value in metrics.items()] or [["empty_report", "No metrics found"]]
    except Exception as exc:
        return [["report_error", str(exc)]]


def _resolve_plot_file(filename: str) -> Path | None:
    direct = PLOTS_DIR / filename
    if direct.is_file():
        return direct
    for alt in _PLOT_FALLBACK_NAMES.get(filename, []):
        for sub in _PLOT_SUBDIRS:
            candidate = PLOTS_DIR / sub / alt
            if candidate.is_file():
                return candidate
    return None


def plot_value(filename: str) -> str | None:
    path = _resolve_plot_file(filename)
    return str(path) if path else None


def plot_message(filename: str) -> str:
    if _resolve_plot_file(filename):
        return ""
    return (
        f"Plot not found yet: `artifacts/submission/plots/{filename}` "
        f"(or TRL equivalent under `plots/tracka_*` / `plots/stage*`)"
    )


def _promotion_banner() -> str:
    path = ROOT / "artifacts" / "submission" / "best_run_promotion.json"
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    best = payload.get("best_run")
    if not isinstance(best, dict):
        return ""
    lines = ["## Promoted GRPO run (from `best_run_promotion.json`)"]
    if best.get("model_name"):
        lines.append(f"- **Base model:** `{best['model_name']}`")
    out = best.get("output_dir")
    if isinstance(out, str) and out.strip():
        lines.append(f"- **Saved adapter / weights:** `{out.strip()}`")
    if best.get("eval_reward_best") is not None:
        lines.append(f"- **Best eval reward (from summary):** `{best['eval_reward_best']}`")
    return "\n".join(lines)


INITIAL_SCENARIOS = _scenario_choices(EXECUTION_MODES[0])

with gr.Blocks(title="ReMorph OpenEnv Demo") as demo:
    gr.Markdown("# ReMorph: Training Agents to Survive API Drift")
    gr.Markdown(
        "An OpenEnv environment where agents repair API drift, handle multi-step workflows, "
        "and abstain safely instead of hallucinating credentials."
    )
    promo_md = _promotion_banner()
    if promo_md:
        gr.Markdown(promo_md)

    app_state = gr.State({})

    with gr.Row():
        execution_mode = gr.Dropdown(EXECUTION_MODES, value="simulated", label="execution_mode")
        scenario_id = gr.Dropdown(INITIAL_SCENARIOS, value=INITIAL_SCENARIOS[0] if INITIAL_SCENARIOS else None, label="scenario_id")
        policy = gr.Dropdown(POLICIES, value="adaptive_reference", label="policy")

    with gr.Row():
        load_button = gr.Button("Load Scenario", variant="secondary")
        run_button = gr.Button("Run Selected Policy", variant="primary")
        demo_button = gr.Button("Run Full Multi-Step Demo")

    with gr.Row():
        observation_json = gr.JSON(label="observation JSON")
        action_json = gr.JSON(label="selected action JSON")

    with gr.Row():
        result_json = gr.JSON(label="step/result JSON")
        reward_json = gr.JSON(label="reward breakdown JSON")

    outcome_text = gr.Textbox(label="success/failure text", lines=4)

    gr.Dataframe(
        headers=["metric", "value"],
        value=benchmark_rows(),
        label="benchmark metrics table",
        interactive=False,
        wrap=True,
    )

    gr.Markdown("## Training and benchmark plots")
    with gr.Row():
        for label, filename in PLOT_FILES[:2]:
            with gr.Column():
                gr.Image(value=plot_value(filename), label=label, interactive=False, visible=plot_value(filename) is not None)
                gr.Markdown(plot_message(filename), visible=plot_value(filename) is None)
    with gr.Row():
        for label, filename in PLOT_FILES[2:]:
            with gr.Column():
                gr.Image(value=plot_value(filename), label=label, interactive=False, visible=plot_value(filename) is not None)
                gr.Markdown(plot_message(filename), visible=plot_value(filename) is None)

    execution_mode.change(update_scenarios, inputs=execution_mode, outputs=scenario_id)
    load_button.click(
        load_scenario,
        inputs=[execution_mode, scenario_id],
        outputs=[app_state, observation_json, action_json, result_json, reward_json, outcome_text],
    )
    run_button.click(
        run_selected_policy,
        inputs=[app_state, execution_mode, scenario_id, policy],
        outputs=[app_state, observation_json, action_json, result_json, reward_json, outcome_text],
    )
    demo_button.click(
        run_full_demo,
        inputs=[execution_mode, scenario_id],
        outputs=[app_state, observation_json, action_json, result_json, reward_json, outcome_text],
    )


if __name__ == "__main__":
    demo.launch()
