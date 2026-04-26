#!/usr/bin/env python3
"""Submission checks for the HF Space demo and docs alignment."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
APP_PATH = REPO_ROOT / "app.py"
SERVER_APP_PATH = REPO_ROOT / "server" / "app.py"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_python_symbols(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.With):
            for item in node.items:
                optional_vars = getattr(item, "optional_vars", None)
                if isinstance(optional_vars, ast.Name):
                    names.add(optional_vars.id)
    return names


def run_checks(*, space_url: str = "", timeout_s: int = 20) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "status": "ok",
        "checks": {},
    }

    checks["checks"]["readme_exists"] = README_PATH.exists()
    checks["checks"]["app_py_exists"] = APP_PATH.exists()
    checks["checks"]["server_app_py_exists"] = SERVER_APP_PATH.exists()
    checks["checks"]["readme_mentions_space_phase"] = (
        "Hugging Face Space" in README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else False
    )
    checks["checks"]["readme_mentions_colab"] = (
        "Colab" in README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else False
    )
    checks["checks"]["readme_mentions_trl"] = (
        "TRL GRPO" in README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else False
    )

    app_symbols = _parse_python_symbols(APP_PATH)
    checks["checks"]["gradio_app_object_present"] = "demo" in app_symbols
    checks["checks"]["load_scenario_function_present"] = "load_scenario" in app_symbols
    checks["checks"]["run_selected_policy_function_present"] = "run_selected_policy" in app_symbols
    checks["checks"]["run_full_demo_function_present"] = "run_full_demo" in app_symbols

    server_symbols = _parse_python_symbols(SERVER_APP_PATH)
    checks["checks"]["server_main_present"] = "main" in server_symbols
    checks["checks"]["server_demo_episode_present"] = "run_demo_episode" in server_symbols

    # Best-effort import check; if optional UI deps are absent, keep warning-level signal only.
    try:
        _load_module(APP_PATH, "space_app_import_check")
        checks["checks"]["app_import_check"] = True
    except Exception as exc:  # noqa: BLE001
        checks["checks"]["app_import_check"] = False
        checks["checks"]["app_import_error"] = str(exc)

    if space_url:
        try:
            url = space_url.rstrip("/") + "/"
            response = requests.get(url, timeout=timeout_s)
            checks["checks"]["space_http_status"] = response.status_code
            checks["checks"]["space_http_ok"] = response.status_code < 500
            checks["checks"]["space_html_received"] = "text/html" in response.headers.get("content-type", "")
        except Exception as exc:  # noqa: BLE001
            checks["checks"]["space_http_ok"] = False
            checks["checks"]["space_http_error"] = str(exc)
            checks["status"] = "warning"

    if not all(
        bool(v)
        for k, v in checks["checks"].items()
        if k
        in {
            "readme_exists",
            "app_py_exists",
            "server_app_py_exists",
            "gradio_app_object_present",
            "load_scenario_function_present",
            "run_selected_policy_function_present",
            "run_full_demo_function_present",
            "server_main_present",
            "server_demo_episode_present",
        }
    ):
        checks["status"] = "warning"

    output_path = REPO_ROOT / "artifacts" / "submission" / "space_submission_check.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    checks["output_path"] = str(output_path.relative_to(REPO_ROOT)).replace("\\", "/")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HF Space submission checks.")
    parser.add_argument("--space-url", default="", help="Optional deployed HF Space URL to health-check.")
    parser.add_argument("--timeout-s", type=int, default=20)
    args = parser.parse_args()
    payload = run_checks(space_url=args.space_url, timeout_s=args.timeout_s)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
