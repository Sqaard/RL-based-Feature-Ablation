from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZON_A_ROOT = PROJECT_ROOT / "Ablation Ladder v2"
if str(HORIZON_A_ROOT) not in sys.path:
    sys.path.insert(0, str(HORIZON_A_ROOT))

from dow30_horizon_a import PRIMARY_BENCHMARK_ID, build_benchmark_suite_frame
from dow30_reporting import (
    build_benchmark_comparison_reports,
    build_primary_benchmark_enriched_summary,
)


class HorizonABenchmarkTests(unittest.TestCase):
    @staticmethod
    def _make_panel() -> pd.DataFrame:
        dates = pd.bdate_range("2020-01-01", periods=45)
        rows: list[dict[str, object]] = []
        for ticker, drift in (("AAA", 0.0010), ("BBB", 0.0005), ("CCC", -0.0002)):
            close = 100.0
            for idx, date in enumerate(dates):
                daily_return = drift + 0.005 * np.sin(idx / 4.0)
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

    def test_build_benchmark_suite_frame(self) -> None:
        panel = self._make_panel()
        dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        suite = build_benchmark_suite_frame(
            panel,
            fold_id="fold_01",
            test_start=dates[20],
            test_end=dates[-1],
        )

        self.assertFalse(suite.empty)
        self.assertEqual(set(suite["benchmark_id"].unique()), {
            "dow30_equal_weight_rebalance_matched",
            "dow30_market_proxy_buy_hold",
            "dow30_equal_weight_vol_target",
            "dow30_trend_filter_overlay",
        })
        self.assertTrue((suite.groupby("benchmark_id")["date"].nunique() > 0).all())
        primary_flags = suite[suite["benchmark_id"] == PRIMARY_BENCHMARK_ID]["is_primary_benchmark"]
        self.assertTrue(primary_flags.all())

    def test_benchmark_reporting_outputs(self) -> None:
        panel = self._make_panel()
        dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        test_dates = dates[20:]
        suite = build_benchmark_suite_frame(
            panel,
            fold_id="fold_01",
            test_start=test_dates[0],
            test_end=test_dates[-1],
        )

        agent_returns = pd.Series(np.full(len(test_dates), 0.0012), index=test_dates)
        agent = pd.DataFrame(
            {
                "date": test_dates,
                "run_key": ["base_macro__fold_01__seed42"] * len(test_dates),
                "feature_set": ["base_macro"] * len(test_dates),
                "feature_family": ["macro_context"] * len(test_dates),
                "is_negative_control": [False] * len(test_dates),
                "fold_id": ["fold_01"] * len(test_dates),
                "seed": [42] * len(test_dates),
                "daily_return": agent_returns.values,
                "portfolio_value": 1_000_000.0 * (1.0 + agent_returns).cumprod().values,
                "turnover": np.full(len(test_dates), 0.05),
                "selection_rule": ["checkpoint_robust_score"] * len(test_dates),
                "selected_model_type": ["best_model"] * len(test_dates),
            }
        )

        run_df, feature_df, fold_df = build_benchmark_comparison_reports(agent, suite)
        self.assertFalse(run_df.empty)
        self.assertFalse(feature_df.empty)
        self.assertFalse(fold_df.empty)

        corrected_summary = pd.DataFrame(
            [
                {
                    "feature_set": "base_macro",
                    "feature_family": "macro_context",
                    "is_negative_control": False,
                    "test_sharpe_median": 0.5,
                    "retention_ratio_median": 0.8,
                }
            ]
        )
        enriched = build_primary_benchmark_enriched_summary(corrected_summary, feature_df)
        self.assertIn("primary_benchmark_excess_sharpe_median", enriched.columns)

    def test_enriched_summary_backfills_feature_metadata(self) -> None:
        panel = self._make_panel()
        dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        test_dates = dates[20:]
        suite = build_benchmark_suite_frame(
            panel,
            fold_id="fold_01",
            test_start=test_dates[0],
            test_end=test_dates[-1],
        )

        agent_returns = pd.Series(np.full(len(test_dates), 0.0012), index=test_dates)
        agent = pd.DataFrame(
            {
                "date": test_dates,
                "run_key": ["base_macro__fold_01__seed42"] * len(test_dates),
                "feature_set": ["base_macro"] * len(test_dates),
                "feature_family": ["macro_context"] * len(test_dates),
                "is_negative_control": [False] * len(test_dates),
                "fold_id": ["fold_01"] * len(test_dates),
                "seed": [42] * len(test_dates),
                "daily_return": agent_returns.values,
                "portfolio_value": 1_000_000.0 * (1.0 + agent_returns).cumprod().values,
                "turnover": np.full(len(test_dates), 0.05),
                "selection_rule": ["checkpoint_robust_score"] * len(test_dates),
                "selected_model_type": ["best_model"] * len(test_dates),
            }
        )

        _, feature_df, _ = build_benchmark_comparison_reports(agent, suite)
        corrected_summary = pd.DataFrame(
            [
                {
                    "feature_set": "base_macro",
                    "test_sharpe_median": 0.5,
                    "retention_ratio_median": 0.8,
                }
            ]
        )

        enriched = build_primary_benchmark_enriched_summary(corrected_summary, feature_df)

        self.assertEqual(enriched.loc[0, "feature_family"], "macro_context")
        self.assertFalse(bool(enriched.loc[0, "is_negative_control"]))
        self.assertIn("primary_benchmark_excess_sharpe_median", enriched.columns)


if __name__ == "__main__":
    unittest.main()
