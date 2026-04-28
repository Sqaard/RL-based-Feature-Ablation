from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZON_A_ROOT = PROJECT_ROOT / "Ablation Ladder v2"
if str(HORIZON_A_ROOT) not in sys.path:
    sys.path.insert(0, str(HORIZON_A_ROOT))

from dow30_next_cycle_launch import (
    build_launch_kwargs,
    load_launch_config,
    run_bootstrapped_notebook_launch_from_csv,
    run_notebook_launch_from_dataframe,
    run_launch_preflight,
    run_launch_preflight_from_dataframe,
)


class NextCycleLaunchTests(unittest.TestCase):
    @staticmethod
    def _make_dataset() -> pd.DataFrame:
        dates = pd.bdate_range("2010-01-01", "2023-03-01")
        rows: list[dict[str, object]] = []
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        for ticker_idx, ticker in enumerate(tickers):
            close = 100.0 + 5.0 * ticker_idx
            for idx, date in enumerate(dates):
                daily_return = 0.0002 * (ticker_idx + 1) + 0.005 * np.sin(idx / 17.0 + ticker_idx)
                close *= 1.0 + daily_return
                rows.append(
                    {
                        "date": date,
                        "date_available": date,
                        "tic": ticker,
                        "close": close,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "open": close * 1.001,
                        "volume": 1_000_000 + 10_000 * ticker_idx,
                        "daily_return": daily_return,
                        "atr_rel": 0.02,
                        "macd": 0.1,
                        "rsi_30": 50.0,
                        "cci_30": 100.0,
                        "dx_30": 20.0,
                        "volume_ratio": 1.0,
                        "obv_pct_change": 0.0,
                        "turbulence": 0.5,
                        "10Y_Yield": 2.0,
                        "VIX": 20.0,
                        "SP500_Trend": 0.01,
                        "Market_Regime": 0.0,
                        "Regime_0_Prob": 0.5,
                        "Regime_1_Prob": 0.5,
                        "gru_return_forecast_1d": 0.0,
                        "gru_return_forecast_2d": 0.0,
                        "gru_return_forecast_3d": 0.0,
                        "gru_return_forecast_4d": 0.0,
                        "gru_return_forecast_5d": 0.0,
                        "forecast_mean": 0.0,
                        "forecast_std": 0.0,
                        "forecast_trend": 0.0,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _make_config(config_path: Path) -> None:
        payload = {
            "experiment_name": "test_next_cycle_launch",
            "runner_kwargs": {
                "base_config_name": "custom_custom",
                "candidate_feature_families": ["xsec_dispersion_correlation_regime"],
                "feature_set_filter": [
                    "base_macro",
                    "base_macro_hmm",
                    "base_macro_gru",
                    "base_macro_xsec_dispersion_correlation_regime",
                ],
                "seeds": [42, 123, 999],
                "total_timesteps": 200000,
                "max_folds": None,
                "es_mode": "relaxed",
                "dropout_p": 0.1,
                "eval_freq": 8192,
                "checkpoint_freq": 4096,
                "verbose": 0,
            },
        }
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def test_load_and_build_launch_kwargs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "launch.yaml"
            self._make_config(config_path)

            config = load_launch_config(config_path)
            kwargs = build_launch_kwargs(config)

            self.assertEqual(kwargs["candidate_feature_families"], ("xsec_dispersion_correlation_regime",))
            self.assertEqual(
                kwargs["feature_set_filter"],
                (
                    "base_macro",
                    "base_macro_hmm",
                    "base_macro_gru",
                    "base_macro_xsec_dispersion_correlation_regime",
                ),
            )

    def test_build_launch_kwargs_can_switch_to_candidate_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "launch.yaml"
            self._make_config(config_path)

            config = load_launch_config(config_path)
            kwargs = build_launch_kwargs(
                config,
                selected_candidate_family="xsec_dispersion_correlation_regime",
                panel_scope="candidate_only",
            )

            self.assertEqual(kwargs["candidate_feature_families"], ("xsec_dispersion_correlation_regime",))
            self.assertEqual(
                kwargs["feature_set_filter"],
                ("base_macro_xsec_dispersion_correlation_regime",),
            )
            self.assertEqual(kwargs["selected_candidate_family"], "xsec_dispersion_correlation_regime")
            self.assertEqual(kwargs["panel_scope"], "candidate_only")

    def test_build_launch_kwargs_can_switch_between_new_candidate_families(self) -> None:
        expected = {
            "rates_term_structure_lsc": ("base_macro_rates_term_structure_lsc",),
            "credit_stress_proxies": ("base_macro_credit_stress_proxies",),
            "breadth_internal_structure": ("base_macro_breadth_internal_structure",),
            "sector_relative_context": ("base_macro_sector_relative_context",),
            "xsec_sector_gated_context": ("base_macro_xsec_sector_gated_context",),
            "xsec_sector_complementarity_v2": ("base_macro_xsec_sector_complementarity_v2",),
            "vol_term_or_implied_vol_proxy": ("base_macro_vol_term_or_implied_vol_proxy",),
            "rates_credit_vol_risk_state_context": ("base_macro_rates_credit_vol_risk_state_context",),
            "analyst_or_fund_revision_features": ("base_macro_analyst_or_fund_revision_features",),
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "launch.yaml"
            self._make_config(config_path)
            config = load_launch_config(config_path)

            for candidate_family, feature_set_filter in expected.items():
                with self.subTest(candidate_family=candidate_family):
                    kwargs = build_launch_kwargs(
                        config,
                        selected_candidate_family=candidate_family,
                        panel_scope="candidate_only",
                    )
                    self.assertEqual(kwargs["candidate_feature_families"], (candidate_family,))
                    self.assertEqual(kwargs["feature_set_filter"], feature_set_filter)

    def test_run_launch_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "launch.yaml"
            dataset_path = tmp_path / "processed.csv"
            output_dir = tmp_path / "preflight_output"
            self._make_config(config_path)
            self._make_dataset().to_csv(dataset_path, index=False)

            result = run_launch_preflight(
                config_path=config_path,
                dataset_path=dataset_path,
                output_dir=output_dir,
            )

            self.assertEqual(result["summary"]["status"], "ready_to_launch")
            self.assertTrue(result["summary"]["user_action_required_now"])
            self.assertGreater(result["summary"]["fold_count"], 0)
            self.assertTrue((output_dir / "launch_preflight_report.json").exists())
            self.assertTrue((output_dir / "launch_notebook_cell.py").exists())
            self.assertTrue((output_dir / "launch_kwargs.json").exists())
            self.assertTrue((output_dir / "launch_config_snapshot.yaml").exists())
            self.assertTrue((output_dir / "post_run_rebuild_commands.json").exists())
            self.assertIn("run_bootstrapped_notebook_launch_from_csv", result["notebook_snippet"])

    def test_run_launch_preflight_from_dataframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "launch.yaml"
            output_dir = tmp_path / "preflight_df_output"
            self._make_config(config_path)

            result = run_launch_preflight_from_dataframe(
                self._make_dataset(),
                config_path=config_path,
                output_dir=output_dir,
            )

            self.assertEqual(result["summary"]["status"], "ready_to_launch")
            self.assertTrue(result["summary"]["dataset_label"].endswith("processed_dataset_snapshot.csv"))
            self.assertTrue((output_dir / "launch_preflight_report.json").exists())
            self.assertTrue((output_dir / "processed_dataset_snapshot.csv").exists())

    def test_run_notebook_launch_from_dataframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "launch.yaml"
            output_dir = tmp_path / "launch_output"
            self._make_config(config_path)

            captured_kwargs: dict[str, object] = {}

            def fake_research_runner(**kwargs):
                captured_kwargs.update(kwargs)
                return {"results": {"artifact_index": {"outputs": {}}}}

            with patch(
                "dow30_next_cycle_launch.build_notebook_research_runner",
                return_value=fake_research_runner,
            ) as patched_builder:
                result = run_notebook_launch_from_dataframe(
                    self._make_dataset(),
                    notebook_ns={"processed": "available"},
                    config_path=config_path,
                    output_dir=output_dir,
                    selected_candidate_family="xsec_dispersion_correlation_regime",
                    panel_scope="candidate_only",
                )

            patched_builder.assert_called_once_with({"processed": "available"})
            self.assertEqual(result["execution_summary"]["status"], "completed")
            self.assertTrue((output_dir / "launch_execution_report.json").exists())
            self.assertEqual(captured_kwargs["base_config_name"], "custom_custom")
            self.assertEqual(
                captured_kwargs["candidate_feature_families"],
                ("xsec_dispersion_correlation_regime",),
            )
            self.assertEqual(
                captured_kwargs["feature_set_filter"],
                ("base_macro_xsec_dispersion_correlation_regime",),
            )
            self.assertEqual(captured_kwargs["seeds"], (42, 123, 999))

    def test_run_bootstrapped_notebook_launch_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "launch.yaml"
            dataset_path = tmp_path / "processed.csv"
            self._make_config(config_path)
            self._make_dataset().to_csv(dataset_path, index=False)

            with patch(
                "dow30_next_cycle_launch.load_reference_notebook_runtime",
                return_value={"bootstrapped": True},
            ) as patched_runtime, patch(
                "dow30_next_cycle_launch.run_notebook_launch_from_dataframe",
                return_value={"execution_summary": {"status": "completed"}},
            ) as patched_run:
                result = run_bootstrapped_notebook_launch_from_csv(
                    config_path=config_path,
                    dataset_path=dataset_path,
                    output_dir=tmp_path / "launch_output",
                    selected_candidate_family="xsec_dispersion_correlation_regime",
                    panel_scope="candidate_only",
                )

            patched_runtime.assert_called_once()
            patched_run.assert_called_once()
            self.assertEqual(result["execution_summary"]["status"], "completed")
            run_call_args = patched_run.call_args
            self.assertIsInstance(run_call_args.args[0], pd.DataFrame)
            self.assertEqual(run_call_args.kwargs["notebook_ns"], {"bootstrapped": True})
            self.assertEqual(
                run_call_args.kwargs["selected_candidate_family"],
                "xsec_dispersion_correlation_regime",
            )
            self.assertEqual(run_call_args.kwargs["panel_scope"], "candidate_only")


if __name__ == "__main__":
    unittest.main()
