from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZON_A_ROOT = PROJECT_ROOT / "Ablation Ladder v2"
if str(HORIZON_A_ROOT) not in sys.path:
    sys.path.insert(0, str(HORIZON_A_ROOT))

from dow30_horizon_a import PRIMARY_BENCHMARK_ID, build_benchmark_suite_frame
from dow30_reporting import merge_research_output_dirs


class ReportingMergeTests(unittest.TestCase):
    @staticmethod
    def _make_panel() -> pd.DataFrame:
        dates = pd.bdate_range("2020-01-01", periods=45)
        rows: list[dict[str, object]] = []
        for ticker, drift in (("AAA", 0.0010), ("BBB", 0.0005), ("CCC", -0.0002)):
            close = 100.0
            for idx, date in enumerate(dates):
                daily_return = drift + 0.004 * np.sin(idx / 4.0)
                close *= 1.0 + daily_return
                rows.append(
                    {
                        "date": date,
                        "tic": ticker,
                        "close": close,
                        "daily_return": daily_return,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _make_results_frame(
        *,
        run_key: str,
        feature_set: str,
        feature_family: str,
        is_negative_control: bool,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "run_key": run_key,
                    "model_name": "custom_custom",
                    "feature_set": feature_set,
                    "feature_family": feature_family,
                    "is_negative_control": is_negative_control,
                    "feature_set_description": "Synthetic test row.",
                    "fold_id": "fold_01",
                    "seed": 42,
                    "n_features": 12,
                    "generalization_ratio": 0.9,
                    "retention_ratio": 0.8,
                    "selected_artifact_type": "best_model",
                    "selection_rule": "checkpoint_robust_score",
                    "checkpoint_selection_rule": "checkpoint_robust_score",
                    "train_sharpe": 1.0,
                    "train_return_pct": 12.0,
                    "train_max_drawdown": -0.10,
                    "train_turnover": 0.12,
                    "validation_sharpe": 0.8,
                    "validation_return_pct": 8.0,
                    "validation_max_drawdown": -0.12,
                    "validation_turnover": 0.10,
                    "test_sharpe": 0.7,
                    "test_return_pct": 7.0,
                    "test_max_drawdown": -0.15,
                    "test_turnover": 0.08,
                    "robust_selection_score": 0.5,
                }
            ]
        )

    @staticmethod
    def _make_daily_frame(
        *,
        run_key: str,
        feature_set: str,
        feature_family: str,
        is_negative_control: bool,
        benchmark_suite: pd.DataFrame,
        include_benchmark_metadata: bool = False,
    ) -> pd.DataFrame:
        primary = benchmark_suite[
            benchmark_suite["benchmark_id"] == PRIMARY_BENCHMARK_ID
        ][["date", "benchmark_id", "benchmark_return", "benchmark_turnover"]].reset_index(drop=True)
        primary["date"] = pd.to_datetime(primary["date"])
        returns = pd.Series(np.full(len(primary), 0.0011))
        values = 1_000_000.0 * (1.0 + returns).cumprod()
        daily = pd.DataFrame(
            {
                "date": primary["date"],
                "run_key": run_key,
                "feature_set": feature_set,
                "feature_family": feature_family,
                "is_negative_control": is_negative_control,
                "fold_id": "fold_01",
                "seed": 42,
                "daily_return": returns,
                "portfolio_value": values,
                "turnover": np.full(len(primary), 0.05),
                "benchmark_return": primary["benchmark_return"].values,
                "selected_model_type": "best_model",
                "selection_rule": "checkpoint_robust_score",
                "regime_label_exogenous": "neutral",
                "excess_return_vs_benchmark": returns - primary["benchmark_return"].values,
            }
        )
        if include_benchmark_metadata:
            daily.insert(10, "benchmark_id", primary["benchmark_id"].values)
            daily.insert(12, "benchmark_turnover", primary["benchmark_turnover"].values)
            daily.insert(13, "benchmark_transaction_cost", 0.0)
        return daily

    @staticmethod
    def _make_actions_frame(*, run_key: str, feature_set: str) -> pd.DataFrame:
        dates = pd.bdate_range("2020-01-29", periods=3)
        return pd.DataFrame(
            {
                "run_key": run_key,
                "feature_set": feature_set,
                "feature_family": "macro_context",
                "is_negative_control": False,
                "fold_id": "fold_01",
                "seed": 42,
                "selected_model_type": "best_model",
                "selection_rule": "checkpoint_robust_score",
                "split_name": "test",
                "action_row_id": [0, 1, 2],
                "action_step": [0, 1, 2],
                "date": dates,
                "AAA": [0.1, 0.2, 0.0],
                "BBB": [0.0, 0.1, 0.3],
            }
        )

    def test_merge_research_output_dirs(self) -> None:
        panel = self._make_panel()
        dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        benchmark_suite = build_benchmark_suite_frame(
            panel,
            fold_id="fold_01",
            test_start=dates[20],
            test_end=dates[-1],
        )
        folds = pd.DataFrame(
            [
                {
                    "fold_id": "fold_01",
                    "train_start": "2020-01-01",
                    "train_end": "2020-01-20",
                    "validation_start": "2020-01-21",
                    "validation_end": "2020-01-24",
                    "test_start": str(dates[20].date()),
                    "test_end": str(dates[-1].date()),
                    "embargo_days": 5,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_dir = tmp_path / "baseline"
            candidate_dir = tmp_path / "candidate"
            merged_dir = tmp_path / "merged"
            baseline_dir.mkdir()
            candidate_dir.mkdir()

            self._make_results_frame(
                run_key="base_macro__fold_01__seed42",
                feature_set="base_macro",
                feature_family="macro_context",
                is_negative_control=False,
            ).to_csv(baseline_dir / "walk_forward_results.csv", index=False)
            self._make_results_frame(
                run_key="base_macro_xsec_dispersion_correlation_regime__fold_01__seed42",
                feature_set="base_macro_xsec_dispersion_correlation_regime",
                feature_family="market_internal_dispersion_correlation",
                is_negative_control=False,
            ).to_csv(candidate_dir / "walk_forward_results.csv", index=False)

            self._make_daily_frame(
                run_key="base_macro__fold_01__seed42",
                feature_set="base_macro",
                feature_family="macro_context",
                is_negative_control=False,
                benchmark_suite=benchmark_suite,
            ).to_csv(baseline_dir / "walk_forward_daily_test_returns.csv", index=False)
            self._make_daily_frame(
                run_key="base_macro_xsec_dispersion_correlation_regime__fold_01__seed42",
                feature_set="base_macro_xsec_dispersion_correlation_regime",
                feature_family="market_internal_dispersion_correlation",
                is_negative_control=False,
                benchmark_suite=benchmark_suite,
                include_benchmark_metadata=True,
            ).to_csv(candidate_dir / "walk_forward_daily_test_returns.csv", index=False)

            self._make_actions_frame(
                run_key="base_macro__fold_01__seed42",
                feature_set="base_macro",
            ).to_csv(baseline_dir / "walk_forward_test_actions.csv", index=False)
            self._make_actions_frame(
                run_key="base_macro_xsec_dispersion_correlation_regime__fold_01__seed42",
                feature_set="base_macro_xsec_dispersion_correlation_regime",
            ).to_csv(candidate_dir / "walk_forward_test_actions.csv", index=False)

            folds.to_csv(baseline_dir / "walk_forward_folds.csv", index=False)
            folds.to_csv(candidate_dir / "walk_forward_folds.csv", index=False)
            benchmark_suite.to_csv(baseline_dir / "benchmark_suite_daily.csv", index=False)
            benchmark_suite.to_csv(candidate_dir / "benchmark_suite_daily.csv", index=False)

            result = merge_research_output_dirs(
                [baseline_dir, candidate_dir],
                output_dir=merged_dir,
            )

            self.assertEqual(result["merged_results"]["run_key"].nunique(), 2)
            self.assertEqual(result["merged_daily"]["run_key"].nunique(), 2)
            self.assertEqual(result["merged_actions"]["run_key"].nunique(), 2)
            self.assertIn("benchmark_id", result["merged_daily"].columns)
            self.assertIn("benchmark_transaction_cost", result["merged_daily"].columns)
            self.assertFalse(result["merged_benchmark"].empty)
            self.assertTrue((merged_dir / "walk_forward_results_merged.csv").exists())
            self.assertTrue((merged_dir / "walk_forward_test_actions_merged.csv").exists())
            self.assertTrue((merged_dir / "analysis" / "corrected_walk_forward_summary.csv").exists())
            self.assertTrue((merged_dir / "analysis" / "walk_forward_test_actions.csv").exists())
            self.assertEqual(
                set(result["merged_results"]["feature_set"].unique()),
                {"base_macro", "base_macro_xsec_dispersion_correlation_regime"},
            )


if __name__ == "__main__":
    unittest.main()
