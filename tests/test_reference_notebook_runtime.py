from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZON_A_ROOT = PROJECT_ROOT / "src" / "feature_ablation"
if str(HORIZON_A_ROOT) not in sys.path:
    sys.path.insert(0, str(HORIZON_A_ROOT))

from dow30_reference_notebook_runtime import (
    DEFAULT_REFERENCE_NOTEBOOK_PATH,
    REFERENCE_RUNTIME_CELL_PATTERNS,
    plan_reference_notebook_runtime_cells,
)


class ReferenceNotebookRuntimeTests(unittest.TestCase):
    def test_runtime_plan_finds_all_required_cells(self) -> None:
        tmp_dir = Path("C:/tmp") / f"reference_notebook_runtime_test_{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = tmp_dir / "reference.ipynb"
        try:
            notebook_path.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "code",
                                "source": [pattern + "\n%matplotlib inline\n"],
                            }
                            for pattern in REFERENCE_RUNTIME_CELL_PATTERNS.values()
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plan = plan_reference_notebook_runtime_cells(notebook_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(len(plan), len(REFERENCE_RUNTIME_CELL_PATTERNS))
        self.assertEqual(
            [item["key"] for item in plan],
            list(REFERENCE_RUNTIME_CELL_PATTERNS.keys()),
        )
        self.assertTrue(all(item["source"].strip() for item in plan))
        self.assertTrue(all("%matplotlib inline" not in item["source"] for item in plan))


if __name__ == "__main__":
    unittest.main()
