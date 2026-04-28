from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZON_A_ROOT = PROJECT_ROOT / "Ablation Ladder v2"
LATENT_ACTION_ROOT = PROJECT_ROOT / "Latent Actions"
if str(LATENT_ACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(LATENT_ACTION_ROOT))

from latent_action_phase2_tools import run_teacher_action_audit


class LatentActionPhase2ToolTests(unittest.TestCase):
    def test_teacher_action_audit_writes_core_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            actions_path = tmp_path / "walk_forward_test_actions.csv"
            output_dir = tmp_path / "audit"
            pd.DataFrame(
                {
                    "run_key": ["base_macro__fold_01__seed42"] * 4,
                    "feature_set": ["base_macro"] * 4,
                    "feature_family": ["macro_context"] * 4,
                    "fold_id": ["fold_01"] * 4,
                    "seed": [42] * 4,
                    "split_name": ["test"] * 4,
                    "action_row_id": [0, 1, 2, 3],
                    "action_step": [0, 1, 2, 3],
                    "date": pd.bdate_range("2020-01-01", periods=4),
                    "AAA": [0.0, 0.2, 0.1, 0.0],
                    "BBB": [0.0, 0.0, -0.1, 0.3],
                }
            ).to_csv(actions_path, index=False)

            result = run_teacher_action_audit(
                actions_path=actions_path,
                output_dir=output_dir,
            )

            self.assertEqual(result["artifact_index"]["action_dim"], 2)
            self.assertFalse(result["summary"].empty)
            self.assertTrue((output_dir / "latent_action_teacher_matrix.csv").exists())
            self.assertTrue((output_dir / "latent_action_teacher_simple_codes.csv").exists())
            self.assertTrue((output_dir / "artifact_index.json").exists())


if __name__ == "__main__":
    unittest.main()
