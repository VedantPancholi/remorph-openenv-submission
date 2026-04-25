"""Serve the Phase 2 live-local gateway app with Uvicorn."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remorph_openenv.live_local import create_live_local_gateway_app


def main() -> None:
    app = create_live_local_gateway_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
