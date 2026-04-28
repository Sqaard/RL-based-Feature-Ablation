from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence


CANONICAL_TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "Latent Actions"
    / "latent_action_phase2_tools.py"
)


def _load_canonical() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_latent_action_phase2_tools_canonical",
        CANONICAL_TOOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load canonical latent-action tool: {CANONICAL_TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_canonical = _load_canonical()

action_trace_to_matrix = _canonical.action_trace_to_matrix
build_simple_action_codes = _canonical.build_simple_action_codes
summarize_action_codes = _canonical.summarize_action_codes
run_teacher_action_audit = _canonical.run_teacher_action_audit


def main(argv: Optional[Sequence[str]] = None) -> None:
    _canonical.main(argv)


if __name__ == "__main__":
    main()
