#!/usr/bin/env bash
# HF Jobs: install git first, then run this, OR use the README one-liner that clones + runs.
# Usage (inside container, after git is installed):
#   bash scripts/hf_job_bootstrap_two_stage.sh
set -euo pipefail

REPO_URL="${REPO_URL:?Set REPO_URL (e.g. https://github.com/VedantPancholi/remorph-openenv-submission.git)}"
REPO_DIR="${REPO_DIR:-remorph-openenv-submission}"

rm -rf "$REPO_DIR"
git clone --depth 1 "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
exec bash scripts/hf_run_staged_grpo_two_stage_persist.sh
