#!/usr/bin/env python3
"""Fail fast before a long GRPO run: TRL checks for environment_factory + tools."""

from __future__ import annotations

import argparse
import sys

from packaging.version import Version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    import transformers  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tv = Version(transformers.__version__)
    if tv < Version("5.2.0"):
        print(f"FAIL: transformers {transformers.__version__} < 5.2.0", file=sys.stderr)
        return 1
    print(f"OK: transformers {transformers.__version__}")

    try:
        import jmespath  # noqa: F401
    except ImportError:
        print("FAIL: jmespath not installed (pip install 'jmespath>=1.0,<2')", file=sys.stderr)
        return 1
    print("OK: jmespath import")

    from trl.chat_template_utils import supports_tool_calling  # noqa: PLC0415
    from trl.import_utils import is_jmespath_available  # noqa: PLC0415

    if not is_jmespath_available():
        print("FAIL: TRL reports jmespath unavailable", file=sys.stderr)
        return 1
    print("OK: TRL is_jmespath_available")

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        from trl.chat_template_utils import add_response_schema  # noqa: PLC0415

        if getattr(tok, "response_schema", None) is None:
            add_response_schema(tok)
    except Exception:
        normalized = args.model.lower().replace("_", "")
        if "qwen2.5" in normalized or "qwen25" in normalized:
            from trl.chat_template_utils import qwen3_schema  # noqa: PLC0415

            tok.response_schema = qwen3_schema
        else:
            raise
    print("OK: response_schema configured")

    if not supports_tool_calling(tok):
        print(
            "FAIL: tokenizer chat template does not support tool calling per TRL. "
            "Try another --model with a tools-capable template.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: supports_tool_calling for {args.model}")

    print("All GRPOTrainer pre-checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
