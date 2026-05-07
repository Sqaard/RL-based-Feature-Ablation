from __future__ import annotations

import importlib.util
import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "behavior_primitives"
    / "behavior_interpretability_audit.py"
)
SPEC = importlib.util.spec_from_file_location("behavior_interpretability_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


class BehaviorInterpretabilityAuditTests(unittest.TestCase):
    def test_kmeans_is_deterministic_and_shape_safe(self) -> None:
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [9.9, 10.0],
                [10.0, 10.1],
            ],
            dtype=float,
        )

        labels_a, centers_a, inertia_a = audit._kmeans(features, k=2, seed=17, n_iter=20, n_init=4)
        labels_b, centers_b, inertia_b = audit._kmeans(features, k=2, seed=17, n_iter=20, n_init=4)

        self.assertEqual(labels_a.shape, (4,))
        self.assertEqual(centers_a.shape, (2, 2))
        self.assertTrue(np.array_equal(labels_a, labels_b))
        self.assertTrue(np.allclose(centers_a, centers_b))
        self.assertTrue(np.isfinite(inertia_a))
        self.assertEqual(inertia_a, inertia_b)

    def test_behavior_audit_end_to_end_on_tiny_exports(self) -> None:
        tmp_base = Path("C:/tmp")
        tmp_base.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_base / f"behavior_audit_test_{os.getpid()}"
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            tickers = ["AAPL", "MSFT", "WMT"]
            dates = pd.date_range("2020-01-01", periods=8, freq="D")
            run_key = "base_macro__fold_01__seed42"
            meta = {
                "run_key": run_key,
                "feature_set": "base_macro",
                "feature_family": "macro_context",
                "is_negative_control": False,
                "fold_id": "fold_01",
                "seed": 42,
                "selected_model_type": "best_model",
                "selection_rule": "checkpoint_robust_score",
                "split_name": "test",
            }

            action_rows = []
            observation_rows = []
            daily_rows = []
            processed_rows = []
            portfolio_value = 1000.0
            for idx, date in enumerate(dates):
                date_s = date.strftime("%Y-%m-%d")
                action_row = {**meta, "action_row_id": idx, "action_step": idx, "date": date_s}
                action_row.update(
                    {
                        "AAPL": 10.0 + idx,
                        "MSFT": -5.0 if idx % 2 else 2.0,
                        "WMT": 1.0 + (idx % 3),
                    }
                )
                action_rows.append(action_row)

                obs_row = {**meta, "observation_row_id": idx, "date": date_s}
                obs_row["obs_0000"] = 50.0
                for price_idx, price in enumerate([100.0, 200.0, 80.0], start=1):
                    obs_row[f"obs_{price_idx:04d}"] = price
                for share_idx, shares in enumerate([1.0 + idx * 0.1, 0.6, 1.2 + (idx % 2) * 0.2], start=4):
                    obs_row[f"obs_{share_idx:04d}"] = shares
                for raw_idx, value in enumerate([0.1 * idx, -0.05 * idx, 0.03 * idx]):
                    obs_row[f"raw_policy_action_{raw_idx:03d}"] = value
                observation_rows.append(obs_row)

                daily_return = 0.001 * ((idx % 3) - 1)
                benchmark_return = 0.0005 * ((idx % 2) * 2 - 1)
                portfolio_value *= 1.0 + daily_return
                daily_rows.append(
                    {
                        "run_key": run_key,
                        "date": date_s,
                        "daily_return": daily_return,
                        "benchmark_return": benchmark_return,
                        "excess_return_vs_benchmark": daily_return - benchmark_return,
                        "turnover": 0.1 + idx * 0.01,
                        "portfolio_value": portfolio_value,
                        "regime_label_exogenous": "unknown",
                    }
                )

                for tic_idx, ticker in enumerate(tickers):
                    processed_rows.append(
                        {
                            "date": date_s,
                            "tic": ticker,
                            "daily_return": daily_return + tic_idx * 0.0001,
                            "VIX": 20.0 + idx,
                            "10Y_Yield": 3.0,
                            "SP500_Trend": 1.0 if idx % 2 else -1.0,
                            "turbulence": 0.2 * idx,
                            "Market_Regime": idx % 2,
                            "volume_ratio": 1.0,
                            "atr_rel": 0.02,
                            "rsi_30": 50.0,
                            "cci_30": 0.0,
                            "dx_30": 20.0,
                        }
                    )

            actions_path = tmp_path / "actions.csv"
            observations_path = tmp_path / "observations.csv"
            daily_path = tmp_path / "daily.csv"
            processed_path = tmp_path / "processed.csv"
            output_dir = tmp_path / "audit_out"

            pd.DataFrame(action_rows).to_csv(actions_path, index=False)
            pd.DataFrame(observation_rows).to_csv(observations_path, index=False)
            pd.DataFrame(daily_rows).to_csv(daily_path, index=False)
            pd.DataFrame(processed_rows).to_csv(processed_path, index=False)

            metadata = audit.run_behavior_interpretability_audit(
                actions_path=actions_path,
                observations_path=observations_path,
                daily_returns_path=daily_path,
                processed_dataset_path=processed_path,
                output_dir=output_dir,
                n_primitives=2,
                window=3,
                min_periods=2,
                seed=11,
                make_figures=False,
            )

            self.assertEqual(metadata["behavior_windows"], 7)
            summary = pd.read_csv(output_dir / "behavior_primitive_summary.csv")
            assignments = pd.read_csv(output_dir / "behavior_primitive_assignments.csv")
            report = pd.read_json(output_dir / "audit_report.json", typ="series")

            self.assertGreaterEqual(len(summary), 1)
            self.assertLessEqual(len(summary), 2)
            self.assertIn("primitive_reliability_score", summary.columns)
            self.assertIn("analysis_regime", assignments.columns)
            self.assertTrue(assignments["analysis_regime"].str.startswith("market_regime_").any())
            self.assertEqual(report["status"], "completed")
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
