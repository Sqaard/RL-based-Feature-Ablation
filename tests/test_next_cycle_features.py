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

from dow30_horizon_a import (
    ANALYST_OR_FUND_REVISION_PROXY_FEATURES,
    BREADTH_INTERNAL_STRUCTURE_FEATURES,
    CREDIT_STRESS_PROXIES_FEATURES,
    NEXT_CYCLE_PANEL_SCOPE_CANDIDATE_ONLY,
    NEXT_CYCLE_PANEL_SCOPE_REFERENCE_PANEL,
    RATES_CREDIT_VOL_RISK_STATE_CONTEXT_FEATURES,
    RATES_TERM_STRUCTURE_LSC_FEATURES,
    SECTOR_RELATIVE_CONTEXT_FEATURES,
    XSEC_SECTOR_COMPLEMENTARITY_V2_FEATURES,
    VOL_TERM_OR_IMPLIED_VOL_PROXY_FEATURES,
    XSEC_SECTOR_GATED_CONTEXT_FEATURES,
    XSEC_DISPERSION_CORRELATION_FEATURES,
    build_next_cycle_candidate_experiment_spec,
    build_next_cycle_feature_set_filter,
    build_controlled_feature_registry,
    build_next_cycle_a1_xsec_experiment_spec,
    ensure_candidate_feature_families,
)
from dow30_research_support import DEFAULT_FEATURE_GROUPS


class NextCycleFeatureTests(unittest.TestCase):
    @staticmethod
    def _make_panel() -> pd.DataFrame:
        dates = pd.bdate_range("2018-01-01", periods=120)
        rows: list[dict[str, object]] = []
        for ticker, drift in (("AAA", 0.0010), ("BBB", 0.0007), ("CCC", -0.0001), ("DDD", 0.0004), ("EEE", 0.0002)):
            close = 100.0
            for idx, date in enumerate(dates):
                daily_return = drift + 0.01 * np.sin(idx / 7.0 + len(ticker))
                close *= 1.0 + daily_return
                rows.append(
                    {
                        "date": date,
                        "tic": ticker,
                        "close": close,
                        "daily_return": daily_return,
                        "10Y_Yield": 2.0,
                        "VIX": 20.0,
                        "SP500_Trend": 0.01,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _with_precomputed_external_features(panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        for idx, column in enumerate(
            RATES_TERM_STRUCTURE_LSC_FEATURES
            + CREDIT_STRESS_PROXIES_FEATURES
            + VOL_TERM_OR_IMPLIED_VOL_PROXY_FEATURES
            + ANALYST_OR_FUND_REVISION_PROXY_FEATURES,
            start=1,
        ):
            out[column] = float(idx) / 100.0
        return out

    def test_ensure_candidate_feature_families_adds_xsec_features(self) -> None:
        panel = self._make_panel()
        enriched = ensure_candidate_feature_families(
            panel,
            candidate_feature_families=("xsec_dispersion_correlation_regime",),
        )

        for column in XSEC_DISPERSION_CORRELATION_FEATURES:
            self.assertIn(column, enriched.columns)
            self.assertGreater(int(enriched[column].notna().sum()), 0)

    def test_ensure_candidate_feature_families_adds_breadth_features(self) -> None:
        panel = self._make_panel()
        enriched = ensure_candidate_feature_families(
            panel,
            candidate_feature_families=("breadth_internal_structure",),
        )

        for column in BREADTH_INTERNAL_STRUCTURE_FEATURES:
            self.assertIn(column, enriched.columns)
            self.assertGreater(int(enriched[column].notna().sum()), 0)

    def test_ensure_candidate_feature_families_adds_sector_relative_features(self) -> None:
        panel = self._make_panel()
        enriched = ensure_candidate_feature_families(
            panel,
            candidate_feature_families=("sector_relative_context",),
        )

        for column in SECTOR_RELATIVE_CONTEXT_FEATURES:
            self.assertIn(column, enriched.columns)
            self.assertGreater(int(enriched[column].notna().sum()), 0)

    def test_ensure_candidate_feature_families_adds_xsec_sector_gated_features(self) -> None:
        panel = self._make_panel()
        enriched = ensure_candidate_feature_families(
            panel,
            candidate_feature_families=("xsec_sector_gated_context",),
        )

        for column in (
            XSEC_DISPERSION_CORRELATION_FEATURES
            + SECTOR_RELATIVE_CONTEXT_FEATURES
            + XSEC_SECTOR_GATED_CONTEXT_FEATURES
        ):
            self.assertIn(column, enriched.columns)
            self.assertGreater(int(enriched[column].notna().sum()), 0)

    def test_ensure_candidate_feature_families_adds_xsec_sector_complementarity_v2_features(self) -> None:
        panel = self._make_panel()
        enriched = ensure_candidate_feature_families(
            panel,
            candidate_feature_families=("xsec_sector_complementarity_v2",),
        )

        for column in (
            XSEC_DISPERSION_CORRELATION_FEATURES
            + SECTOR_RELATIVE_CONTEXT_FEATURES
            + XSEC_SECTOR_COMPLEMENTARITY_V2_FEATURES
        ):
            self.assertIn(column, enriched.columns)
            self.assertGreater(int(enriched[column].notna().sum()), 0)

    def test_ensure_candidate_feature_families_accepts_precomputed_external_features(self) -> None:
        panel = self._with_precomputed_external_features(self._make_panel())
        enriched = ensure_candidate_feature_families(
            panel,
            candidate_feature_families=(
                "rates_term_structure_lsc",
                "credit_stress_proxies",
                "vol_term_or_implied_vol_proxy",
                "analyst_or_fund_revision_features",
            ),
        )

        for column in (
            RATES_TERM_STRUCTURE_LSC_FEATURES
            + CREDIT_STRESS_PROXIES_FEATURES
            + VOL_TERM_OR_IMPLIED_VOL_PROXY_FEATURES
            + ANALYST_OR_FUND_REVISION_PROXY_FEATURES
        ):
            self.assertIn(column, enriched.columns)
            self.assertGreater(int(enriched[column].notna().sum()), 0)

    def test_ensure_candidate_feature_families_adds_rates_credit_vol_risk_state_context(self) -> None:
        panel = self._with_precomputed_external_features(self._make_panel())
        enriched = ensure_candidate_feature_families(
            panel,
            candidate_feature_families=("rates_credit_vol_risk_state_context",),
        )

        for column in (
            RATES_TERM_STRUCTURE_LSC_FEATURES
            + CREDIT_STRESS_PROXIES_FEATURES
            + VOL_TERM_OR_IMPLIED_VOL_PROXY_FEATURES
            + RATES_CREDIT_VOL_RISK_STATE_CONTEXT_FEATURES
        ):
            self.assertIn(column, enriched.columns)
            self.assertGreater(int(enriched[column].notna().sum()), 0)

    def test_ensure_candidate_feature_families_rejects_missing_external_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires precomputed lag-clean columns"):
            ensure_candidate_feature_families(
                self._make_panel(),
                candidate_feature_families=("rates_term_structure_lsc",),
            )

    def test_registry_can_include_and_filter_next_cycle_candidate(self) -> None:
        registry = build_controlled_feature_registry(
            DEFAULT_FEATURE_GROUPS,
            candidate_feature_families=("xsec_dispersion_correlation_regime",),
            include_feature_sets=(
                "base_macro",
                "base_macro_hmm",
                "base_macro_gru",
                "base_macro_xsec_dispersion_correlation_regime",
            ),
        )

        self.assertEqual(
            list(registry.keys()),
            [
                "base_macro",
                "base_macro_hmm",
                "base_macro_gru",
                "base_macro_xsec_dispersion_correlation_regime",
            ],
        )
        candidate_cols = registry["base_macro_xsec_dispersion_correlation_regime"].columns
        for column in XSEC_DISPERSION_CORRELATION_FEATURES:
            self.assertIn(column, candidate_cols)

    def test_registry_can_include_new_panel_derived_candidates(self) -> None:
        registry = build_controlled_feature_registry(
            DEFAULT_FEATURE_GROUPS,
            candidate_feature_families=(
                "breadth_internal_structure",
                "sector_relative_context",
            ),
            include_feature_sets=(
                "base_macro_breadth_internal_structure",
                "base_macro_sector_relative_context",
            ),
        )

        self.assertEqual(
            list(registry.keys()),
            [
                "base_macro_breadth_internal_structure",
                "base_macro_sector_relative_context",
            ],
        )
        breadth_cols = registry["base_macro_breadth_internal_structure"].columns
        sector_cols = registry["base_macro_sector_relative_context"].columns
        for column in BREADTH_INTERNAL_STRUCTURE_FEATURES:
            self.assertIn(column, breadth_cols)
        for column in SECTOR_RELATIVE_CONTEXT_FEATURES:
            self.assertIn(column, sector_cols)

    def test_registry_can_include_xsec_sector_gated_candidate(self) -> None:
        registry = build_controlled_feature_registry(
            DEFAULT_FEATURE_GROUPS,
            candidate_feature_families=("xsec_sector_gated_context",),
            include_feature_sets=("base_macro_xsec_sector_gated_context",),
        )

        self.assertEqual(list(registry.keys()), ["base_macro_xsec_sector_gated_context"])
        candidate_cols = registry["base_macro_xsec_sector_gated_context"].columns
        for column in (
            XSEC_DISPERSION_CORRELATION_FEATURES
            + SECTOR_RELATIVE_CONTEXT_FEATURES
            + XSEC_SECTOR_GATED_CONTEXT_FEATURES
        ):
            self.assertIn(column, candidate_cols)

    def test_registry_can_include_xsec_sector_complementarity_v2_candidate(self) -> None:
        registry = build_controlled_feature_registry(
            DEFAULT_FEATURE_GROUPS,
            candidate_feature_families=("xsec_sector_complementarity_v2",),
            include_feature_sets=("base_macro_xsec_sector_complementarity_v2",),
        )

        self.assertEqual(list(registry.keys()), ["base_macro_xsec_sector_complementarity_v2"])
        candidate_cols = registry["base_macro_xsec_sector_complementarity_v2"].columns
        for column in (
            XSEC_DISPERSION_CORRELATION_FEATURES
            + SECTOR_RELATIVE_CONTEXT_FEATURES
            + XSEC_SECTOR_COMPLEMENTARITY_V2_FEATURES
        ):
            self.assertIn(column, candidate_cols)

    def test_registry_can_include_external_macro_and_revision_candidates(self) -> None:
        expected = {
            "rates_term_structure_lsc": (
                "base_macro_rates_term_structure_lsc",
                RATES_TERM_STRUCTURE_LSC_FEATURES,
            ),
            "credit_stress_proxies": (
                "base_macro_credit_stress_proxies",
                CREDIT_STRESS_PROXIES_FEATURES,
            ),
            "vol_term_or_implied_vol_proxy": (
                "base_macro_vol_term_or_implied_vol_proxy",
                VOL_TERM_OR_IMPLIED_VOL_PROXY_FEATURES,
            ),
            "rates_credit_vol_risk_state_context": (
                "base_macro_rates_credit_vol_risk_state_context",
                RATES_TERM_STRUCTURE_LSC_FEATURES
                + CREDIT_STRESS_PROXIES_FEATURES
                + VOL_TERM_OR_IMPLIED_VOL_PROXY_FEATURES
                + RATES_CREDIT_VOL_RISK_STATE_CONTEXT_FEATURES,
            ),
            "analyst_or_fund_revision_features": (
                "base_macro_analyst_or_fund_revision_features",
                ANALYST_OR_FUND_REVISION_PROXY_FEATURES,
            ),
        }
        for candidate_family, (feature_set, feature_columns) in expected.items():
            with self.subTest(candidate_family=candidate_family):
                registry = build_controlled_feature_registry(
                    DEFAULT_FEATURE_GROUPS,
                    candidate_feature_families=(candidate_family,),
                    include_feature_sets=(feature_set,),
                )
                self.assertEqual(list(registry.keys()), [feature_set])
                candidate_cols = registry[feature_set].columns
                for column in feature_columns:
                    self.assertIn(column, candidate_cols)

    def test_next_cycle_a1_spec_matches_runner_contract(self) -> None:
        spec = build_next_cycle_a1_xsec_experiment_spec()
        self.assertEqual(
            spec["candidate_feature_families"],
            ["xsec_dispersion_correlation_regime"],
        )
        self.assertEqual(
            spec["feature_set_filter"],
            [
                "base_macro",
                "base_macro_hmm",
                "base_macro_gru",
                "base_macro_xsec_dispersion_correlation_regime",
            ],
        )
        self.assertEqual(spec["required_comparable_seeds"], [42, 123, 999])

    def test_candidate_only_feature_filter(self) -> None:
        feature_filter = build_next_cycle_feature_set_filter(
            "xsec_dispersion_correlation_regime",
            panel_scope=NEXT_CYCLE_PANEL_SCOPE_CANDIDATE_ONLY,
        )
        self.assertEqual(
            feature_filter,
            ["base_macro_xsec_dispersion_correlation_regime"],
        )

    def test_candidate_only_feature_filter_supports_all_implemented_candidates(self) -> None:
        expected = {
            "rates_term_structure_lsc": "base_macro_rates_term_structure_lsc",
            "credit_stress_proxies": "base_macro_credit_stress_proxies",
            "xsec_dispersion_correlation_regime": "base_macro_xsec_dispersion_correlation_regime",
            "breadth_internal_structure": "base_macro_breadth_internal_structure",
            "sector_relative_context": "base_macro_sector_relative_context",
            "xsec_sector_gated_context": "base_macro_xsec_sector_gated_context",
            "xsec_sector_complementarity_v2": "base_macro_xsec_sector_complementarity_v2",
            "vol_term_or_implied_vol_proxy": "base_macro_vol_term_or_implied_vol_proxy",
            "rates_credit_vol_risk_state_context": "base_macro_rates_credit_vol_risk_state_context",
            "analyst_or_fund_revision_features": "base_macro_analyst_or_fund_revision_features",
        }
        for candidate_family, feature_set_name in expected.items():
            with self.subTest(candidate_family=candidate_family):
                self.assertEqual(
                    build_next_cycle_feature_set_filter(
                        candidate_family,
                        panel_scope=NEXT_CYCLE_PANEL_SCOPE_CANDIDATE_ONLY,
                    ),
                    [feature_set_name],
                )

    def test_generic_candidate_spec_can_build_candidate_only_mode(self) -> None:
        spec = build_next_cycle_candidate_experiment_spec(
            "xsec_dispersion_correlation_regime",
            panel_scope=NEXT_CYCLE_PANEL_SCOPE_REFERENCE_PANEL,
        )
        self.assertEqual(spec["panel_scope"], NEXT_CYCLE_PANEL_SCOPE_REFERENCE_PANEL)
        self.assertEqual(
            spec["feature_set_filter"],
            [
                "base_macro",
                "base_macro_hmm",
                "base_macro_gru",
                "base_macro_xsec_dispersion_correlation_regime",
            ],
        )


if __name__ == "__main__":
    unittest.main()
