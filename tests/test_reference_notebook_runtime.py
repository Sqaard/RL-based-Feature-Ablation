from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZON_A_ROOT = PROJECT_ROOT / "Ablation Ladder v2"
if str(HORIZON_A_ROOT) not in sys.path:
    sys.path.insert(0, str(HORIZON_A_ROOT))

from dow30_reference_notebook_runtime import (
    DEFAULT_REFERENCE_NOTEBOOK_PATH,
    REFERENCE_RUNTIME_CELL_PATTERNS,
    plan_reference_notebook_runtime_cells,
)


class ReferenceNotebookRuntimeTests(unittest.TestCase):
    def test_runtime_plan_finds_all_required_cells(self) -> None:
        self.assertTrue(DEFAULT_REFERENCE_NOTEBOOK_PATH.exists())
        plan = plan_reference_notebook_runtime_cells(DEFAULT_REFERENCE_NOTEBOOK_PATH)

        self.assertEqual(len(plan), len(REFERENCE_RUNTIME_CELL_PATTERNS))
        self.assertEqual(
            [item["key"] for item in plan],
            list(REFERENCE_RUNTIME_CELL_PATTERNS.keys()),
        )
        self.assertTrue(all(item["source"].strip() for item in plan))
        self.assertTrue(all("%matplotlib inline" not in item["source"] for item in plan))


if __name__ == "__main__":
    unittest.main()
