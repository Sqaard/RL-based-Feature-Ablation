from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


REFERENCE_SEEDS: tuple[int, ...] = (42, 123, 999, 2024, 2025)

REFERENCE_WALK_FORWARD_SCHEDULE = {
    "start_date": "2010-01-01",
    "first_test_start": "2016-01-04",
    "end_date": "2023-03-01",
    "min_train_months": 60,
    "inner_validation_months": 3,
    "test_window_months": 3,
    "step_months": 6,
    "embargo_days": 5,
}


EVENT_CALENDAR_FEATURES: tuple[str, ...] = (
    "cal_day_of_week_sin",
    "cal_day_of_week_cos",
    "cal_is_month_start",
    "cal_is_month_end",
    "cal_is_quarter_start",
    "cal_is_quarter_end",
    "cal_is_turn_of_month",
    "cal_is_year_start",
    "cal_is_year_end",
    "cal_is_option_expiry_week",
    "cal_is_option_expiry_day",
    "cal_trading_day_of_month",
    "cal_trading_days_left_in_month",
    "cal_trading_day_of_quarter",
    "cal_trading_days_left_in_quarter",
)

XSEC_DISPERSION_CORRELATION_FEATURES: tuple[str, ...] = (
    "xsec_ret_dispersion_lag1",
    "xsec_ret_dispersion_20d_mean",
    "xsec_ret_dispersion_60d_zscore",
    "xsec_mean_pairwise_corr_20d",
    "xsec_mean_pairwise_corr_60d_zscore",
    "xsec_dispersion_minus_corr_regime_score",
)

BREADTH_INTERNAL_STRUCTURE_FEATURES: tuple[str, ...] = (
    "breadth_advancing_share_lag1",
    "breadth_advancing_share_20d_mean",
    "breadth_declining_share_lag1",
    "breadth_above_20d_sma_share_lag1",
    "breadth_above_60d_sma_share_lag1",
    "breadth_new_20d_high_share_lag1",
    "breadth_new_20d_low_share_lag1",
    "breadth_participation_regime_score",
)

SECTOR_RELATIVE_CONTEXT_FEATURES: tuple[str, ...] = (
    "sector_ret_lag1",
    "sector_ret_20d_mean",
    "sector_ret_60d_zscore",
    "sector_rel_market_ret_20d",
    "sector_leadership_rank_20d",
    "stock_rel_sector_ret_lag1",
    "stock_rel_sector_ret_20d_mean",
    "stock_rel_sector_ret_60d_zscore",
)

XSEC_SECTOR_GATED_CONTEXT_FEATURES: tuple[str, ...] = (
    "xsec_sector_stockpick_gate",
    "xsec_sector_leadership_gate",
    "xsec_sector_rel_market_gate",
    "xsec_stock_rel_sector_momentum_gate",
    "xsec_sector_corr_risk_gate",
    "xsec_sector_dispersion_leadership_alignment",
)

RATES_TERM_STRUCTURE_LSC_FEATURES: tuple[str, ...] = (
    "rates_lsc_level_lag1",
    "rates_lsc_slope_10y_3mo_lag1",
    "rates_lsc_slope_10y_2y_lag1",
    "rates_lsc_slope_30y_5y_lag1",
    "rates_lsc_curvature_2y10y30y_lag1",
    "rates_lsc_curvature_3mo5y30y_lag1",
    "rates_lsc_level_20d_change",
    "rates_lsc_slope_10y_3mo_20d_change",
    "rates_lsc_level_60d_zscore",
    "rates_lsc_slope_10y_3mo_60d_zscore",
    "rates_lsc_curvature_2y10y30y_60d_zscore",
    "rates_lsc_curve_inversion_flag_lag1",
    "rates_lsc_policy_pressure_score",
)

CREDIT_STRESS_PROXIES_FEATURES: tuple[str, ...] = (
    "credit_baa_spread_lag1",
    "credit_aaa_spread_lag1",
    "credit_baa_aaa_quality_spread_lag1",
    "credit_baa_spread_20d_change",
    "credit_quality_spread_20d_change",
    "credit_baa_spread_60d_zscore",
    "credit_aaa_spread_60d_zscore",
    "credit_quality_spread_60d_zscore",
    "credit_nfci_lag5",
    "credit_nfci_126d_zscore",
    "credit_stress_regime_score",
)

VOL_TERM_OR_IMPLIED_VOL_PROXY_FEATURES: tuple[str, ...] = (
    "vol_vix_lag1",
    "vol_vxv_lag1",
    "vol_term_slope_vxv_vix_lag1",
    "vol_term_ratio_vxv_vix_lag1",
    "vol_vix_20d_change",
    "vol_term_slope_20d_change",
    "vol_vix_60d_zscore",
    "vol_term_slope_60d_zscore",
    "vol_term_backwardation_flag_lag1",
    "vol_implied_stress_regime_score",
)

ANALYST_OR_FUND_REVISION_PROXY_FEATURES: tuple[str, ...] = (
    "fundrev_new_statement_flag_lag1",
    "fundrev_days_since_statement_lag1",
    "fundrev_revenue_growth_lag1",
    "fundrev_eps_growth_lag1",
    "fundrev_op_margin_lag1",
    "fundrev_net_margin_lag1",
    "fundrev_debt_ratio_lag1",
    "fundrev_revenue_qoq_revision_lag1",
    "fundrev_eps_qoq_revision_lag1",
    "fundrev_net_income_qoq_revision_lag1",
    "fundrev_profitability_revision_score_lag1",
    "fundrev_balance_sheet_stress_revision_score_lag1",
    "fundrev_valuation_reset_score_lag1",
)

DOW30_STATIC_SECTOR_MAP: Mapping[str, str] = {
    "AAPL": "information_technology",
    "AMGN": "health_care",
    "AMZN": "consumer_discretionary",
    "AXP": "financials",
    "BA": "industrials",
    "CAT": "industrials",
    "CRM": "information_technology",
    "CSCO": "information_technology",
    "CVX": "energy",
    "DIS": "communication_services",
    "GS": "financials",
    "HD": "consumer_discretionary",
    "HON": "industrials",
    "IBM": "information_technology",
    "INTC": "information_technology",
    "JNJ": "health_care",
    "JPM": "financials",
    "KO": "consumer_staples",
    "MCD": "consumer_discretionary",
    "MMM": "industrials",
    "MRK": "health_care",
    "MSFT": "information_technology",
    "NKE": "consumer_discretionary",
    "PG": "consumer_staples",
    "TRV": "financials",
    "UNH": "health_care",
    "V": "financials",
    "VZ": "communication_services",
    "WMT": "consumer_staples",
}


IMPLEMENTED_NEXT_CYCLE_CANDIDATE_FAMILIES: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    {
        "rates_term_structure_lsc": {
            "rank": 1,
            "family_type": "macro_exogenous",
            "planned_status": "implemented_candidate_available",
            "suitability": "NOW",
            "needs_external_data": True,
            "derivable_from_existing_panel": False,
            "feature_set_name": "base_macro_rates_term_structure_lsc",
            "feature_columns": list(RATES_TERM_STRUCTURE_LSC_FEATURES),
            "economic_intuition": (
                "Treasury level, slope, and curvature can proxy policy stance, growth expectations, "
                "and discount-rate pressure that matter for broad large-cap equity regimes."
            ),
            "notes": (
                "Implemented from lag-clean FRED Treasury curve features in "
                "`processed_final_fixed_external_lagclean_full.csv`."
            ),
        },
        "credit_stress_proxies": {
            "rank": 2,
            "family_type": "macro_exogenous",
            "planned_status": "implemented_candidate_available",
            "suitability": "NOW",
            "needs_external_data": True,
            "derivable_from_existing_panel": False,
            "feature_set_name": "base_macro_credit_stress_proxies",
            "feature_columns": list(CREDIT_STRESS_PROXIES_FEATURES),
            "economic_intuition": (
                "Credit spread and financial-condition stress can capture funding conditions and "
                "risk appetite that often lead equity drawdowns and rotation in Dow 30 names."
            ),
            "notes": (
                "Implemented with long-history FRED proxies BAA10Y, AAA10Y, and NFCI. ICE BofA OAS "
                "was not used because the current FRED download does not cover the 2010-2023 test window."
            ),
        },
        "xsec_dispersion_correlation_regime": {
            "rank": 3,
            "family_type": "market_internal",
            "planned_status": "implemented_candidate_available",
            "suitability": "NOW",
            "needs_external_data": False,
            "derivable_from_existing_panel": True,
            "feature_set_name": "base_macro_xsec_dispersion_correlation_regime",
            "feature_columns": list(XSEC_DISPERSION_CORRELATION_FEATURES),
            "economic_intuition": (
                "Lagged cross-sectional dispersion and average pairwise correlation can distinguish "
                "stock-picking-friendly regimes from index-dominated correlation spikes."
            ),
            "notes": (
                "Implemented from the existing Dow 30 panel with causal lagging. This is the first "
                "next-cycle candidate that can be run without new external data."
            ),
        },
        "breadth_internal_structure": {
            "rank": 4,
            "family_type": "market_internal",
            "planned_status": "implemented_candidate_available",
            "suitability": "NOW",
            "needs_external_data": False,
            "derivable_from_existing_panel": True,
            "feature_set_name": "base_macro_breadth_internal_structure",
            "feature_columns": list(BREADTH_INTERNAL_STRUCTURE_FEATURES),
            "economic_intuition": (
                "Lagged participation and breadth measures can identify whether index moves have broad "
                "internal support or are being carried by narrow leadership."
            ),
            "notes": (
                "Implemented from the existing Dow 30 panel with prior-day returns and prior-day close "
                "structure only. No external data is required."
            ),
        },
        "sector_relative_context": {
            "rank": 5,
            "family_type": "relative_context",
            "planned_status": "implemented_candidate_available",
            "suitability": "NOW",
            "needs_external_data": False,
            "derivable_from_existing_panel": True,
            "feature_set_name": "base_macro_sector_relative_context",
            "feature_columns": list(SECTOR_RELATIVE_CONTEXT_FEATURES),
            "economic_intuition": (
                "Sector-relative return context can encode leadership, rotation, and stock-vs-sector "
                "positioning without changing the policy architecture."
            ),
            "notes": (
                "Implemented with a fixed Dow 30 ticker-to-sector map and causal lagged return features. "
                "Unmapped tickers fall back to an unknown sector bucket."
            ),
        },
        "xsec_sector_gated_context": {
            "rank": 6,
            "family_type": "market_internal_relative_interaction",
            "planned_status": "implemented_diagnostic_candidate_available",
            "suitability": "NOW",
            "needs_external_data": False,
            "derivable_from_existing_panel": True,
            "feature_set_name": "base_macro_xsec_sector_gated_context",
            "feature_columns": list(
                XSEC_DISPERSION_CORRELATION_FEATURES
                + SECTOR_RELATIVE_CONTEXT_FEATURES
                + XSEC_SECTOR_GATED_CONTEXT_FEATURES
            ),
            "economic_intuition": (
                "Cross-sectional stock-picking regimes can condition whether sector leadership and "
                "stock-vs-sector momentum should matter to the policy."
            ),
            "notes": (
                "Implemented as a post-analysis diagnostic branch combining the stable xsec signal with "
                "the episodic sector-relative signal. It uses only lagged/rolling panel-derived features."
            ),
        },
        "vol_term_or_implied_vol_proxy": {
            "rank": 7,
            "family_type": "macro_exogenous",
            "planned_status": "implemented_candidate_available",
            "suitability": "NOW",
            "needs_external_data": True,
            "derivable_from_existing_panel": False,
            "feature_set_name": "base_macro_vol_term_or_implied_vol_proxy",
            "feature_columns": list(VOL_TERM_OR_IMPLIED_VOL_PROXY_FEATURES),
            "economic_intuition": (
                "VIX/VXV term structure can separate temporary spot-volatility spikes from more "
                "persistent implied-volatility stress regimes."
            ),
            "notes": (
                "Implemented from lag-clean FRED VIXCLS and VXVCLS features. The one-day lag prevents "
                "same-day close leakage into the trading decision."
            ),
        },
        "analyst_or_fund_revision_features": {
            "rank": 8,
            "family_type": "fundamental_exogenous_proxy",
            "planned_status": "implemented_proxy_candidate_available",
            "suitability": "DIAGNOSTIC_ONLY",
            "needs_external_data": False,
            "derivable_from_existing_panel": True,
            "feature_set_name": "base_macro_analyst_or_fund_revision_features",
            "feature_columns": list(ANALYST_OR_FUND_REVISION_PROXY_FEATURES),
            "economic_intuition": (
                "Point-in-time fundamental release changes can proxy changing company expectations when "
                "true analyst estimate revisions are unavailable."
            ),
            "notes": (
                "This is not a true analyst-estimate dataset. It is a conservative lagged fundamental "
                "revision proxy built from the existing point-in-time fundamental block."
            ),
        },
    }
)

NEXT_CYCLE_A1_XSEC_FEATURE_SET_FILTER: tuple[str, ...] = (
    "base_macro",
    "base_macro_hmm",
    "base_macro_gru",
    "base_macro_xsec_dispersion_correlation_regime",
)

NEXT_CYCLE_REFERENCE_PANEL_FEATURE_SETS: tuple[str, ...] = (
    "base_macro",
    "base_macro_hmm",
    "base_macro_gru",
)

NEXT_CYCLE_PANEL_SCOPE_CANDIDATE_ONLY = "candidate_only"
NEXT_CYCLE_PANEL_SCOPE_REFERENCE_PANEL = "candidate_plus_reference_panel"
NEXT_CYCLE_PANEL_SCOPE_REFERENCE_PANEL_WITH_BASE = "candidate_plus_reference_with_base_anchor"


UNIMPLEMENTED_CANDIDATE_FEATURE_FAMILIES: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    {}
)


def get_implemented_next_cycle_candidate_families() -> "OrderedDict[str, dict[str, Any]]":
    return OrderedDict(
        (name, dict(spec))
        for name, spec in IMPLEMENTED_NEXT_CYCLE_CANDIDATE_FAMILIES.items()
    )


def get_implemented_candidate_family_spec(candidate_family: str) -> dict[str, Any]:
    family_id = str(candidate_family)
    registry = get_implemented_next_cycle_candidate_families()
    if family_id not in registry:
        available = ", ".join(registry.keys()) or "<none>"
        raise KeyError(
            f"Unknown implemented candidate family `{family_id}`. Available implemented families: {available}."
        )
    return dict(registry[family_id])


def get_candidate_feature_set_name(candidate_family: str) -> str:
    spec = get_implemented_candidate_family_spec(candidate_family)
    feature_set_name = str(spec.get("feature_set_name", "")).strip()
    if not feature_set_name:
        raise ValueError(
            f"Implemented candidate family `{candidate_family}` is missing `feature_set_name` metadata."
        )
    return feature_set_name


def build_next_cycle_feature_set_filter(
    candidate_family: str,
    *,
    panel_scope: str = NEXT_CYCLE_PANEL_SCOPE_CANDIDATE_ONLY,
) -> list[str]:
    feature_set_name = get_candidate_feature_set_name(candidate_family)
    scope = str(panel_scope)
    if scope == NEXT_CYCLE_PANEL_SCOPE_CANDIDATE_ONLY:
        return [feature_set_name]
    if scope == NEXT_CYCLE_PANEL_SCOPE_REFERENCE_PANEL:
        return list(NEXT_CYCLE_REFERENCE_PANEL_FEATURE_SETS) + [feature_set_name]
    if scope == NEXT_CYCLE_PANEL_SCOPE_REFERENCE_PANEL_WITH_BASE:
        return ["base"] + list(NEXT_CYCLE_REFERENCE_PANEL_FEATURE_SETS) + [feature_set_name]
    raise ValueError(f"Unsupported next-cycle panel_scope: {panel_scope}")


def build_next_cycle_candidate_experiment_spec(
    candidate_family: str,
    *,
    panel_scope: str = NEXT_CYCLE_PANEL_SCOPE_CANDIDATE_ONLY,
) -> dict[str, Any]:
    candidate_spec = get_implemented_candidate_family_spec(candidate_family)
    return {
        "candidate_feature_families": [str(candidate_family)],
        "feature_set_filter": build_next_cycle_feature_set_filter(
            candidate_family,
            panel_scope=panel_scope,
        ),
        "required_comparable_seeds": [42, 123, 999],
        "optional_stability_extension_seeds": [2024, 2025],
        "panel_scope": str(panel_scope),
        "candidate_feature_set_name": str(candidate_spec["feature_set_name"]),
        "reference_baseline": {
            "config_name": "custom_custom",
            "algorithm": "PPO",
            "reward_mode": "custom_reward",
            "policy_mode": "custom_mlp_policy",
            "checkpoint_selection_rule": "checkpoint_robust_score",
            "configuration_selection_rule": "robust_q25_retention",
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "action_constraint": "long_only_box_[-1,1]_scaled_by_hmax",
        },
    }


def build_next_cycle_a1_xsec_experiment_spec(
    *,
    include_base_anchor: bool = False,
) -> dict[str, Any]:
    panel_scope = (
        NEXT_CYCLE_PANEL_SCOPE_REFERENCE_PANEL_WITH_BASE
        if include_base_anchor
        else NEXT_CYCLE_PANEL_SCOPE_REFERENCE_PANEL
    )
    return build_next_cycle_candidate_experiment_spec(
        "xsec_dispersion_correlation_regime",
        panel_scope=panel_scope,
    )


def get_unimplemented_candidate_feature_families() -> "OrderedDict[str, dict[str, Any]]":
    return OrderedDict(
        (name, dict(spec))
        for name, spec in UNIMPLEMENTED_CANDIDATE_FEATURE_FAMILIES.items()
    )


@dataclass(frozen=True)
class ReferenceExperimentConfig:
    config_name: str
    agent_name: str
    algorithm: str
    reward_mode: str
    policy_mode: str
    environment_mode: str
    buy_cost_pct: float
    sell_cost_pct: float
    hmax: int
    reward_scaling: float
    action_constraint: str
    checkpoint_selection_rule: str
    configuration_selection_rule: str
    seed_list: tuple[int, ...]
    walk_forward_schedule: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "agent_name": self.agent_name,
            "algorithm": self.algorithm,
            "reward_mode": self.reward_mode,
            "policy_mode": self.policy_mode,
            "environment_mode": self.environment_mode,
            "buy_cost_pct": self.buy_cost_pct,
            "sell_cost_pct": self.sell_cost_pct,
            "hmax": self.hmax,
            "reward_scaling": self.reward_scaling,
            "action_constraint": self.action_constraint,
            "checkpoint_selection_rule": self.checkpoint_selection_rule,
            "configuration_selection_rule": self.configuration_selection_rule,
            "seed_list": list(self.seed_list),
            "walk_forward_schedule": dict(self.walk_forward_schedule),
        }


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    columns: tuple[str, ...]
    feature_family: str
    is_negative_control: bool
    feature_set_description: str
    source_groups: tuple[str, ...]

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "feature_set": self.name,
            "feature_family": self.feature_family,
            "is_negative_control": bool(self.is_negative_control),
            "feature_set_description": self.feature_set_description,
            "n_features": len(self.columns),
            "source_groups": list(self.source_groups),
            "feature_columns": list(self.columns),
        }


@dataclass(frozen=True)
class SelectionRuleSpec:
    name: str
    description: str
    validation_sharpe_quantile: float
    turnover_weight: float = 0.0
    max_drawdown_weight: float = 0.0
    retention_weight: float = 0.0
    generalization_weight: float = 0.0
    is_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "validation_sharpe_quantile": self.validation_sharpe_quantile,
            "turnover_weight": self.turnover_weight,
            "max_drawdown_weight": self.max_drawdown_weight,
            "retention_weight": self.retention_weight,
            "generalization_weight": self.generalization_weight,
            "is_default": self.is_default,
        }


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    benchmark_name: str
    description: str
    family: str
    uses_transaction_costs: bool
    requires_pretest_history: bool
    is_primary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "description": self.description,
            "family": self.family,
            "uses_transaction_costs": self.uses_transaction_costs,
            "requires_pretest_history": self.requires_pretest_history,
            "is_primary": self.is_primary,
        }


def build_reference_experiment_config(
    config_name: str = "custom_custom",
    *,
    seed_list: Sequence[int] = REFERENCE_SEEDS,
    walk_forward_schedule: Optional[Mapping[str, Any]] = None,
) -> ReferenceExperimentConfig:
    config_map = {
        "finrl_finrl": {
            "reward_mode": "finrl_reward",
            "policy_mode": "finrl_mlp_policy",
            "environment_mode": "finrl_env",
        },
        "finrl_custom": {
            "reward_mode": "finrl_reward",
            "policy_mode": "custom_mlp_policy",
            "environment_mode": "finrl_env",
        },
        "zhang_finrl": {
            "reward_mode": "zhang_risk_adjusted_reward",
            "policy_mode": "finrl_mlp_policy",
            "environment_mode": "zhang_env",
        },
        "zhang_custom": {
            "reward_mode": "zhang_risk_adjusted_reward",
            "policy_mode": "custom_mlp_policy",
            "environment_mode": "zhang_env",
        },
        "custom_finrl": {
            "reward_mode": "custom_reward",
            "policy_mode": "finrl_mlp_policy",
            "environment_mode": "custom_env",
        },
        "custom_custom": {
            "reward_mode": "custom_reward",
            "policy_mode": "custom_mlp_policy",
            "environment_mode": "custom_env",
        },
    }
    if config_name not in config_map:
        raise ValueError(f"Unknown reference config: {config_name}")

    spec = config_map[config_name]
    return ReferenceExperimentConfig(
        config_name=config_name,
        agent_name="DRLAgent",
        algorithm="PPO",
        reward_mode=spec["reward_mode"],
        policy_mode=spec["policy_mode"],
        environment_mode=spec["environment_mode"],
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        hmax=100,
        reward_scaling=1e-4,
        action_constraint="long_only_box_[-1,1]_scaled_by_hmax",
        checkpoint_selection_rule="checkpoint_robust_score",
        configuration_selection_rule="robust_q25_retention",
        seed_list=tuple(int(seed) for seed in seed_list),
        walk_forward_schedule=dict(walk_forward_schedule or REFERENCE_WALK_FORWARD_SCHEDULE),
    )


def add_event_calendar_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    unique_dates = pd.DataFrame({date_col: pd.DatetimeIndex(sorted(out[date_col].dropna().unique()))})
    if unique_dates.empty:
        for col in EVENT_CALENDAR_FEATURES:
            out[col] = np.nan
        return out

    dates = unique_dates[date_col]
    day_of_week = dates.dt.dayofweek.astype(float)
    unique_dates["cal_day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 5.0)
    unique_dates["cal_day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 5.0)

    month_period = dates.dt.to_period("M")
    quarter_period = dates.dt.to_period("Q")
    year_period = dates.dt.year

    month_rank = unique_dates.groupby(month_period).cumcount() + 1
    month_total = unique_dates.groupby(month_period)[date_col].transform("size")
    quarter_rank = unique_dates.groupby(quarter_period).cumcount() + 1
    quarter_total = unique_dates.groupby(quarter_period)[date_col].transform("size")
    year_rank = unique_dates.groupby(year_period).cumcount() + 1
    year_total = unique_dates.groupby(year_period)[date_col].transform("size")

    weekday_in_month = unique_dates.groupby([month_period, dates.dt.day_name()]).cumcount() + 1
    is_third_friday = (dates.dt.dayofweek == 4) & (weekday_in_month == 3)

    unique_dates["cal_is_month_start"] = (month_rank == 1).astype(float)
    unique_dates["cal_is_month_end"] = (month_rank == month_total).astype(float)
    unique_dates["cal_is_quarter_start"] = (quarter_rank == 1).astype(float)
    unique_dates["cal_is_quarter_end"] = (quarter_rank == quarter_total).astype(float)
    unique_dates["cal_is_turn_of_month"] = ((month_rank <= 2) | ((month_total - month_rank) < 2)).astype(float)
    unique_dates["cal_is_year_start"] = (year_rank <= 3).astype(float)
    unique_dates["cal_is_year_end"] = ((year_total - year_rank) < 3).astype(float)
    unique_dates["cal_is_option_expiry_day"] = is_third_friday.astype(float)
    unique_dates["cal_is_option_expiry_week"] = dates.dt.isocalendar().week.isin(
        dates[is_third_friday].dt.isocalendar().week.tolist()
    ).astype(float)
    unique_dates["cal_trading_day_of_month"] = month_rank.astype(float)
    unique_dates["cal_trading_days_left_in_month"] = (month_total - month_rank).astype(float)
    unique_dates["cal_trading_day_of_quarter"] = quarter_rank.astype(float)
    unique_dates["cal_trading_days_left_in_quarter"] = (quarter_total - quarter_rank).astype(float)

    return out.merge(unique_dates, on=date_col, how="left")


def ensure_event_calendar_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    if set(EVENT_CALENDAR_FEATURES).issubset(df.columns):
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        return out
    return add_event_calendar_features(df, date_col=date_col)


def _rolling_zscore(
    series: pd.Series,
    *,
    window: int,
) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0).replace(0.0, np.nan)
    return (series - rolling_mean) / rolling_std


def _mean_pairwise_corr(
    window_frame: pd.DataFrame,
    *,
    min_tickers: int = 5,
    min_periods: int = 10,
) -> float:
    clean = window_frame.dropna(axis=1, how="all")
    if clean.shape[1] < min_tickers or len(clean) < min_periods:
        return float("nan")

    corr = clean.corr(min_periods=min_periods)
    if corr.empty:
        return float("nan")

    upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper_values = corr.where(upper_mask).stack().dropna()
    if upper_values.empty:
        return float("nan")
    return float(upper_values.mean())


def add_xsec_dispersion_correlation_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "tic",
    return_col: str = "daily_return",
    price_col: str = "close",
    corr_window: int = 20,
    z_window: int = 60,
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    if return_col not in out.columns and {ticker_col, price_col}.issubset(out.columns):
        out = out.sort_values([ticker_col, date_col]).reset_index(drop=True)
        out[return_col] = out.groupby(ticker_col)[price_col].pct_change()

    panel = (
        out[[date_col, ticker_col, return_col]]
        .drop_duplicates(subset=[date_col, ticker_col], keep="last")
        .pivot(index=date_col, columns=ticker_col, values=return_col)
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
    )
    if panel.empty:
        for col in XSEC_DISPERSION_CORRELATION_FEATURES:
            out[col] = np.nan
        return out

    dispersion_raw = panel.std(axis=1, ddof=0)
    dispersion_lag1 = dispersion_raw.shift(1)
    dispersion_20d_mean = dispersion_lag1.rolling(window=corr_window, min_periods=corr_window).mean()
    dispersion_60d_zscore = _rolling_zscore(dispersion_lag1, window=z_window)

    lagged_panel = panel.shift(1)
    corr_rows: list[tuple[pd.Timestamp, float]] = []
    for idx, current_date in enumerate(lagged_panel.index):
        window = lagged_panel.iloc[max(0, idx - corr_window + 1) : idx + 1]
        corr_rows.append((pd.Timestamp(current_date), _mean_pairwise_corr(window)))
    mean_pairwise_corr_20d = pd.Series(
        {date_value: corr_value for date_value, corr_value in corr_rows},
        name="xsec_mean_pairwise_corr_20d",
    ).sort_index()
    mean_pairwise_corr_60d_zscore = _rolling_zscore(mean_pairwise_corr_20d, window=z_window)

    xsec_features = pd.DataFrame(
        {
            date_col: panel.index,
            "xsec_ret_dispersion_lag1": dispersion_lag1.values,
            "xsec_ret_dispersion_20d_mean": dispersion_20d_mean.values,
            "xsec_ret_dispersion_60d_zscore": dispersion_60d_zscore.values,
            "xsec_mean_pairwise_corr_20d": mean_pairwise_corr_20d.reindex(panel.index).values,
            "xsec_mean_pairwise_corr_60d_zscore": mean_pairwise_corr_60d_zscore.reindex(panel.index).values,
        }
    )
    xsec_features["xsec_dispersion_minus_corr_regime_score"] = (
        xsec_features["xsec_ret_dispersion_60d_zscore"]
        - xsec_features["xsec_mean_pairwise_corr_60d_zscore"]
    )
    return out.merge(xsec_features, on=date_col, how="left")


def ensure_xsec_dispersion_correlation_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    if set(XSEC_DISPERSION_CORRELATION_FEATURES).issubset(df.columns):
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        return out
    return add_xsec_dispersion_correlation_features(df, date_col=date_col)


def add_breadth_internal_structure_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "tic",
    return_col: str = "daily_return",
    price_col: str = "close",
    fast_window: int = 20,
    slow_window: int = 60,
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    if return_col not in out.columns and {ticker_col, price_col}.issubset(out.columns):
        out = out.sort_values([ticker_col, date_col]).reset_index(drop=True)
        out[return_col] = out.groupby(ticker_col)[price_col].pct_change()

    required_cols = {date_col, ticker_col, return_col, price_col}
    if not required_cols.issubset(out.columns):
        for col in BREADTH_INTERNAL_STRUCTURE_FEATURES:
            out[col] = np.nan
        return out

    returns_panel = (
        out[[date_col, ticker_col, return_col]]
        .drop_duplicates(subset=[date_col, ticker_col], keep="last")
        .pivot(index=date_col, columns=ticker_col, values=return_col)
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
    )
    price_panel = (
        out[[date_col, ticker_col, price_col]]
        .drop_duplicates(subset=[date_col, ticker_col], keep="last")
        .pivot(index=date_col, columns=ticker_col, values=price_col)
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
    )
    if returns_panel.empty or price_panel.empty:
        for col in BREADTH_INTERNAL_STRUCTURE_FEATURES:
            out[col] = np.nan
        return out

    lagged_returns = returns_panel.shift(1)
    lagged_prices = price_panel.shift(1)
    advancing_share = (lagged_returns > 0.0).sum(axis=1) / lagged_returns.notna().sum(axis=1).replace(0, np.nan)
    declining_share = (lagged_returns < 0.0).sum(axis=1) / lagged_returns.notna().sum(axis=1).replace(0, np.nan)

    fast_sma = lagged_prices.rolling(window=fast_window, min_periods=fast_window).mean()
    slow_sma = lagged_prices.rolling(window=slow_window, min_periods=slow_window).mean()
    above_fast_share = (lagged_prices > fast_sma).sum(axis=1) / lagged_prices.notna().sum(axis=1).replace(0, np.nan)
    above_slow_share = (lagged_prices > slow_sma).sum(axis=1) / lagged_prices.notna().sum(axis=1).replace(0, np.nan)

    rolling_high = lagged_prices.rolling(window=fast_window, min_periods=fast_window).max()
    rolling_low = lagged_prices.rolling(window=fast_window, min_periods=fast_window).min()
    new_high_share = (lagged_prices >= rolling_high).sum(axis=1) / lagged_prices.notna().sum(axis=1).replace(0, np.nan)
    new_low_share = (lagged_prices <= rolling_low).sum(axis=1) / lagged_prices.notna().sum(axis=1).replace(0, np.nan)

    advancing_mean = advancing_share.rolling(window=fast_window, min_periods=fast_window).mean()
    participation_score = (
        _rolling_zscore(advancing_mean, window=slow_window)
        + _rolling_zscore(above_slow_share, window=slow_window)
        - _rolling_zscore(new_low_share, window=slow_window)
    )

    breadth_features = pd.DataFrame(
        {
            date_col: returns_panel.index,
            "breadth_advancing_share_lag1": advancing_share.values,
            "breadth_advancing_share_20d_mean": advancing_mean.values,
            "breadth_declining_share_lag1": declining_share.values,
            "breadth_above_20d_sma_share_lag1": above_fast_share.reindex(returns_panel.index).values,
            "breadth_above_60d_sma_share_lag1": above_slow_share.reindex(returns_panel.index).values,
            "breadth_new_20d_high_share_lag1": new_high_share.reindex(returns_panel.index).values,
            "breadth_new_20d_low_share_lag1": new_low_share.reindex(returns_panel.index).values,
            "breadth_participation_regime_score": participation_score.reindex(returns_panel.index).values,
        }
    )
    return out.merge(breadth_features, on=date_col, how="left")


def ensure_breadth_internal_structure_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    if set(BREADTH_INTERNAL_STRUCTURE_FEATURES).issubset(df.columns):
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        return out
    return add_breadth_internal_structure_features(df, date_col=date_col)


def _stack_wide_feature(
    frame: pd.DataFrame,
    *,
    value_name: str,
    date_col: str,
    key_col: str,
) -> pd.DataFrame:
    try:
        stacked = frame.stack(future_stack=True)
    except TypeError:
        stacked = frame.stack(dropna=False)
    return (
        stacked.rename(value_name)
        .reset_index()
        .rename(columns={"level_0": date_col, "level_1": key_col})
    )


def add_sector_relative_context_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "tic",
    return_col: str = "daily_return",
    price_col: str = "close",
    sector_map: Mapping[str, str] = DOW30_STATIC_SECTOR_MAP,
    fast_window: int = 20,
    slow_window: int = 60,
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    if return_col not in out.columns and {ticker_col, price_col}.issubset(out.columns):
        out = out.sort_values([ticker_col, date_col]).reset_index(drop=True)
        out[return_col] = out.groupby(ticker_col)[price_col].pct_change()

    required_cols = {date_col, ticker_col, return_col}
    if not required_cols.issubset(out.columns):
        for col in SECTOR_RELATIVE_CONTEXT_FEATURES:
            out[col] = np.nan
        return out

    returns_panel = (
        out[[date_col, ticker_col, return_col]]
        .drop_duplicates(subset=[date_col, ticker_col], keep="last")
        .pivot(index=date_col, columns=ticker_col, values=return_col)
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
    )
    if returns_panel.empty:
        for col in SECTOR_RELATIVE_CONTEXT_FEATURES:
            out[col] = np.nan
        return out

    ticker_to_sector = pd.Series(
        {
            str(ticker): str(sector_map.get(str(ticker), "unknown"))
            for ticker in returns_panel.columns
        }
    )
    sector_returns = returns_panel.T.groupby(ticker_to_sector).mean().T.sort_index()
    sector_lag1 = sector_returns.shift(1)
    sector_20d_mean = sector_lag1.rolling(window=fast_window, min_periods=fast_window).mean()
    sector_60d_zscore = sector_lag1.apply(lambda series: _rolling_zscore(series, window=slow_window))

    market_lag1 = returns_panel.mean(axis=1, skipna=True).shift(1)
    market_20d_mean = market_lag1.rolling(window=fast_window, min_periods=fast_window).mean()
    sector_rel_market_20d = sector_20d_mean.sub(market_20d_mean, axis=0)
    sector_leadership_rank_20d = sector_20d_mean.rank(axis=1, pct=True)

    key_frame = out[[date_col, ticker_col]].drop_duplicates().copy()
    key_frame["_sector_bucket"] = key_frame[ticker_col].astype(str).map(sector_map).fillna("unknown")
    sector_features = key_frame.copy()
    sector_feature_frames = {
        "sector_ret_lag1": sector_lag1,
        "sector_ret_20d_mean": sector_20d_mean,
        "sector_ret_60d_zscore": sector_60d_zscore,
        "sector_rel_market_ret_20d": sector_rel_market_20d,
        "sector_leadership_rank_20d": sector_leadership_rank_20d,
    }
    for feature_name, wide_frame in sector_feature_frames.items():
        long_frame = _stack_wide_feature(
            wide_frame,
            value_name=feature_name,
            date_col=date_col,
            key_col="_sector_bucket",
        )
        sector_features = sector_features.merge(
            long_frame,
            on=[date_col, "_sector_bucket"],
            how="left",
        )

    ticker_sector_lag1 = pd.DataFrame(index=returns_panel.index, columns=returns_panel.columns, dtype=float)
    for ticker in returns_panel.columns:
        sector = ticker_to_sector[str(ticker)]
        if sector in sector_lag1.columns:
            ticker_sector_lag1[ticker] = sector_lag1[sector]
    stock_lag1 = returns_panel.shift(1)
    stock_rel_sector_lag1 = stock_lag1 - ticker_sector_lag1
    stock_rel_sector_20d_mean = stock_rel_sector_lag1.rolling(
        window=fast_window,
        min_periods=fast_window,
    ).mean()
    stock_rel_sector_60d_zscore = stock_rel_sector_lag1.apply(
        lambda series: _rolling_zscore(series, window=slow_window)
    )

    stock_features = key_frame[[date_col, ticker_col]].copy()
    stock_feature_frames = {
        "stock_rel_sector_ret_lag1": stock_rel_sector_lag1,
        "stock_rel_sector_ret_20d_mean": stock_rel_sector_20d_mean,
        "stock_rel_sector_ret_60d_zscore": stock_rel_sector_60d_zscore,
    }
    for feature_name, wide_frame in stock_feature_frames.items():
        long_frame = _stack_wide_feature(
            wide_frame,
            value_name=feature_name,
            date_col=date_col,
            key_col=ticker_col,
        )
        stock_features = stock_features.merge(
            long_frame,
            on=[date_col, ticker_col],
            how="left",
        )

    feature_frame = sector_features.merge(stock_features, on=[date_col, ticker_col], how="left")
    feature_frame = feature_frame.drop(columns=["_sector_bucket"], errors="ignore")
    return out.merge(feature_frame, on=[date_col, ticker_col], how="left")


def ensure_sector_relative_context_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    if set(SECTOR_RELATIVE_CONTEXT_FEATURES).issubset(df.columns):
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        return out
    return add_sector_relative_context_features(df, date_col=date_col)


def _bounded_regime_gate(series: pd.Series, *, scale: float = 2.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series(np.tanh(values.clip(lower=-6.0, upper=6.0) / scale), index=series.index)


def add_xsec_sector_gated_context_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    out = ensure_xsec_dispersion_correlation_features(df, date_col=date_col)
    out = ensure_sector_relative_context_features(out, date_col=date_col)
    out[date_col] = pd.to_datetime(out[date_col])

    stockpick_gate = _bounded_regime_gate(out["xsec_dispersion_minus_corr_regime_score"])
    corr_risk_gate = _bounded_regime_gate(out["xsec_mean_pairwise_corr_60d_zscore"])
    dispersion_gate = _bounded_regime_gate(out["xsec_ret_dispersion_60d_zscore"])
    leadership_centered = (
        pd.to_numeric(out["sector_leadership_rank_20d"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        - 0.5
    )
    sector_rel_market = pd.to_numeric(
        out["sector_rel_market_ret_20d"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    stock_rel_sector = pd.to_numeric(
        out["stock_rel_sector_ret_20d_mean"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)

    out["xsec_sector_stockpick_gate"] = stockpick_gate
    out["xsec_sector_leadership_gate"] = stockpick_gate * leadership_centered
    out["xsec_sector_rel_market_gate"] = stockpick_gate * sector_rel_market
    out["xsec_stock_rel_sector_momentum_gate"] = stockpick_gate * stock_rel_sector
    out["xsec_sector_corr_risk_gate"] = corr_risk_gate * (-sector_rel_market)
    out["xsec_sector_dispersion_leadership_alignment"] = dispersion_gate * leadership_centered
    return out


def ensure_xsec_sector_gated_context_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    required = set(
        XSEC_DISPERSION_CORRELATION_FEATURES
        + SECTOR_RELATIVE_CONTEXT_FEATURES
        + XSEC_SECTOR_GATED_CONTEXT_FEATURES
    )
    if required.issubset(df.columns):
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        return out
    return add_xsec_sector_gated_context_features(df, date_col=date_col)


def ensure_precomputed_candidate_features(
    df: pd.DataFrame,
    *,
    candidate_family: str,
    required_columns: Sequence[str],
    date_col: str = "date",
) -> pd.DataFrame:
    missing = [str(col) for col in required_columns if str(col) not in df.columns]
    if missing:
        raise ValueError(
            "Candidate family "
            f"`{candidate_family}` requires precomputed lag-clean columns missing from the dataset: "
            f"{missing}. Rebuild the dataset with `Preprocessing/build_external_macro_dataset.py` "
            "and, for revision proxy features, `Preprocessing/build_revision_proxy_dataset.py`."
        )
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out


def ensure_candidate_feature_families(
    df: pd.DataFrame,
    *,
    candidate_feature_families: Optional[Sequence[str]] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    out = ensure_event_calendar_features(df, date_col=date_col)
    requested = tuple(str(name) for name in (candidate_feature_families or ()))
    if "xsec_dispersion_correlation_regime" in requested:
        out = ensure_xsec_dispersion_correlation_features(out, date_col=date_col)
    if "breadth_internal_structure" in requested:
        out = ensure_breadth_internal_structure_features(out, date_col=date_col)
    if "sector_relative_context" in requested:
        out = ensure_sector_relative_context_features(out, date_col=date_col)
    if "xsec_sector_gated_context" in requested:
        out = ensure_xsec_sector_gated_context_features(out, date_col=date_col)
    if "rates_term_structure_lsc" in requested:
        out = ensure_precomputed_candidate_features(
            out,
            candidate_family="rates_term_structure_lsc",
            required_columns=RATES_TERM_STRUCTURE_LSC_FEATURES,
            date_col=date_col,
        )
    if "credit_stress_proxies" in requested:
        out = ensure_precomputed_candidate_features(
            out,
            candidate_family="credit_stress_proxies",
            required_columns=CREDIT_STRESS_PROXIES_FEATURES,
            date_col=date_col,
        )
    if "vol_term_or_implied_vol_proxy" in requested:
        out = ensure_precomputed_candidate_features(
            out,
            candidate_family="vol_term_or_implied_vol_proxy",
            required_columns=VOL_TERM_OR_IMPLIED_VOL_PROXY_FEATURES,
            date_col=date_col,
        )
    if "analyst_or_fund_revision_features" in requested:
        out = ensure_precomputed_candidate_features(
            out,
            candidate_family="analyst_or_fund_revision_features",
            required_columns=ANALYST_OR_FUND_REVISION_PROXY_FEATURES,
            date_col=date_col,
        )
    return out


def build_controlled_feature_registry(
    feature_groups: Mapping[str, Sequence[str]],
    *,
    include_event_calendar: bool = True,
    candidate_feature_families: Optional[Sequence[str]] = None,
    include_feature_sets: Optional[Sequence[str]] = None,
) -> "OrderedDict[str, FeatureSetSpec]":
    groups = {
        name: tuple(dict.fromkeys(str(col) for col in cols))
        for name, cols in feature_groups.items()
    }

    base = groups.get("base", ())
    macro = groups.get("macro", ())
    hmm = groups.get("hmm", ())
    gru = groups.get("gru", ())
    xsec_regime = groups.get("xsec_dispersion_correlation_regime", ())
    breadth_internal = groups.get("breadth_internal_structure", ())
    sector_relative = groups.get("sector_relative_context", ())
    xsec_sector_gated = groups.get("xsec_sector_gated_context", ())
    rates_term_structure = groups.get("rates_term_structure_lsc", ())
    credit_stress = groups.get("credit_stress_proxies", ())
    vol_term = groups.get("vol_term_or_implied_vol_proxy", ())
    revision_proxy = groups.get("analyst_or_fund_revision_features", ())
    exogenous_plus = EVENT_CALENDAR_FEATURES if include_event_calendar else ()
    requested_candidates = {str(name) for name in (candidate_feature_families or ())}

    registry: "OrderedDict[str, FeatureSetSpec]" = OrderedDict()
    registry["base"] = FeatureSetSpec(
        name="base",
        columns=tuple(base),
        feature_family="technical_base",
        is_negative_control=False,
        feature_set_description="Prices, returns, and baseline technical risk features.",
        source_groups=("base",),
    )
    registry["base_macro"] = FeatureSetSpec(
        name="base_macro",
        columns=tuple(dict.fromkeys(base + macro)),
        feature_family="macro_context",
        is_negative_control=False,
        feature_set_description="Baseline technical features plus existing macro context features.",
        source_groups=("base", "macro"),
    )
    registry["base_macro_exogenous_plus"] = FeatureSetSpec(
        name="base_macro_exogenous_plus",
        columns=tuple(dict.fromkeys(base + macro + exogenous_plus)),
        feature_family="calendar_exogenous",
        is_negative_control=False,
        feature_set_description=(
            "Baseline technical and macro features plus a causal calendar/event exogenous layer."
        ),
        source_groups=("base", "macro", "event_calendar"),
    )
    registry["base_macro_hmm"] = FeatureSetSpec(
        name="base_macro_hmm",
        columns=tuple(dict.fromkeys(base + macro + hmm)),
        feature_family="negative_control_hmm",
        is_negative_control=True,
        feature_set_description="Baseline technical and macro features plus HMM state features as a negative control.",
        source_groups=("base", "macro", "hmm"),
    )
    registry["base_macro_gru"] = FeatureSetSpec(
        name="base_macro_gru",
        columns=tuple(dict.fromkeys(base + macro + gru)),
        feature_family="negative_control_gru",
        is_negative_control=True,
        feature_set_description="Baseline technical and macro features plus GRU forecast features as a negative control.",
        source_groups=("base", "macro", "gru"),
    )
    if "rates_term_structure_lsc" in requested_candidates and rates_term_structure:
        registry["base_macro_rates_term_structure_lsc"] = FeatureSetSpec(
            name="base_macro_rates_term_structure_lsc",
            columns=tuple(dict.fromkeys(base + macro + rates_term_structure)),
            feature_family="rates_term_structure_lsc",
            is_negative_control=False,
            feature_set_description=(
                "Baseline technical and macro features plus lag-clean Treasury level, slope, and curvature context."
            ),
            source_groups=("base", "macro", "rates_term_structure_lsc"),
        )
    if "credit_stress_proxies" in requested_candidates and credit_stress:
        registry["base_macro_credit_stress_proxies"] = FeatureSetSpec(
            name="base_macro_credit_stress_proxies",
            columns=tuple(dict.fromkeys(base + macro + credit_stress)),
            feature_family="credit_stress_proxies",
            is_negative_control=False,
            feature_set_description=(
                "Baseline technical and macro features plus lag-clean credit and financial-stress proxy context."
            ),
            source_groups=("base", "macro", "credit_stress_proxies"),
        )
    if "xsec_dispersion_correlation_regime" in requested_candidates and xsec_regime:
        registry["base_macro_xsec_dispersion_correlation_regime"] = FeatureSetSpec(
            name="base_macro_xsec_dispersion_correlation_regime",
            columns=tuple(dict.fromkeys(base + macro + xsec_regime)),
            feature_family="market_internal_dispersion_correlation",
            is_negative_control=False,
            feature_set_description=(
                "Baseline technical and macro features plus causal cross-sectional dispersion/correlation regime context."
            ),
            source_groups=("base", "macro", "xsec_dispersion_correlation_regime"),
        )
    if "breadth_internal_structure" in requested_candidates and breadth_internal:
        registry["base_macro_breadth_internal_structure"] = FeatureSetSpec(
            name="base_macro_breadth_internal_structure",
            columns=tuple(dict.fromkeys(base + macro + breadth_internal)),
            feature_family="market_internal_breadth",
            is_negative_control=False,
            feature_set_description=(
                "Baseline technical and macro features plus causal internal breadth and participation context."
            ),
            source_groups=("base", "macro", "breadth_internal_structure"),
        )
    if "sector_relative_context" in requested_candidates and sector_relative:
        registry["base_macro_sector_relative_context"] = FeatureSetSpec(
            name="base_macro_sector_relative_context",
            columns=tuple(dict.fromkeys(base + macro + sector_relative)),
            feature_family="sector_relative_context",
            is_negative_control=False,
            feature_set_description=(
                "Baseline technical and macro features plus causal sector-relative return context."
            ),
            source_groups=("base", "macro", "sector_relative_context"),
        )
    if (
        "xsec_sector_gated_context" in requested_candidates
        and xsec_regime
        and sector_relative
        and xsec_sector_gated
    ):
        registry["base_macro_xsec_sector_gated_context"] = FeatureSetSpec(
            name="base_macro_xsec_sector_gated_context",
            columns=tuple(
                dict.fromkeys(
                    base
                    + macro
                    + xsec_regime
                    + sector_relative
                    + xsec_sector_gated
                )
            ),
            feature_family="xsec_sector_gated_context",
            is_negative_control=False,
            feature_set_description=(
                "Baseline technical and macro features plus causal xsec-sector gated interaction context."
            ),
            source_groups=(
                "base",
                "macro",
                "xsec_dispersion_correlation_regime",
                "sector_relative_context",
                "xsec_sector_gated_context",
            ),
        )
    if "vol_term_or_implied_vol_proxy" in requested_candidates and vol_term:
        registry["base_macro_vol_term_or_implied_vol_proxy"] = FeatureSetSpec(
            name="base_macro_vol_term_or_implied_vol_proxy",
            columns=tuple(dict.fromkeys(base + macro + vol_term)),
            feature_family="vol_term_or_implied_vol_proxy",
            is_negative_control=False,
            feature_set_description=(
                "Baseline technical and macro features plus lag-clean implied-volatility term-structure context."
            ),
            source_groups=("base", "macro", "vol_term_or_implied_vol_proxy"),
        )
    if "analyst_or_fund_revision_features" in requested_candidates and revision_proxy:
        registry["base_macro_analyst_or_fund_revision_features"] = FeatureSetSpec(
            name="base_macro_analyst_or_fund_revision_features",
            columns=tuple(dict.fromkeys(base + macro + revision_proxy)),
            feature_family="analyst_or_fund_revision_features",
            is_negative_control=False,
            feature_set_description=(
                "Baseline technical and macro features plus lagged point-in-time fundamental revision proxy features."
            ),
            source_groups=("base", "macro", "analyst_or_fund_revision_features"),
        )
    if include_feature_sets is not None:
        requested_feature_sets = tuple(str(name) for name in include_feature_sets)
        registry = OrderedDict(
            (name, spec)
            for name, spec in registry.items()
            if name in requested_feature_sets
        )
    return registry


def infer_feature_metadata(feature_set_name: Any) -> dict[str, Any]:
    name = str(feature_set_name)
    if name == "base":
        family = "technical_base"
        description = "Prices, returns, and baseline technical risk features."
        is_negative = False
    elif name == "base_macro":
        family = "macro_context"
        description = "Baseline technical features plus macro context."
        is_negative = False
    elif name == "base_macro_exogenous_plus":
        family = "calendar_exogenous"
        description = "Baseline technical and macro features plus event/calendar exogenous features."
        is_negative = False
    elif name == "base_macro_hmm":
        family = "negative_control_hmm"
        description = "HMM-enhanced feature set treated as a negative control."
        is_negative = True
    elif name == "base_macro_gru":
        family = "negative_control_gru"
        description = "GRU-enhanced feature set treated as a negative control."
        is_negative = True
    elif name == "base_macro_rates_term_structure_lsc":
        family = "rates_term_structure_lsc"
        description = "Macro baseline plus lag-clean Treasury level, slope, and curvature features."
        is_negative = False
    elif name == "base_macro_credit_stress_proxies":
        family = "credit_stress_proxies"
        description = "Macro baseline plus lag-clean credit and financial-stress proxy features."
        is_negative = False
    elif name == "base_macro_xsec_dispersion_correlation_regime":
        family = "market_internal_dispersion_correlation"
        description = "Macro baseline plus causal cross-sectional dispersion/correlation regime features."
        is_negative = False
    elif name == "base_macro_breadth_internal_structure":
        family = "market_internal_breadth"
        description = "Macro baseline plus causal breadth and participation structure features."
        is_negative = False
    elif name == "base_macro_sector_relative_context":
        family = "sector_relative_context"
        description = "Macro baseline plus causal sector-relative return context features."
        is_negative = False
    elif name == "base_macro_xsec_sector_gated_context":
        family = "xsec_sector_gated_context"
        description = "Macro baseline plus causal xsec-sector gated interaction context features."
        is_negative = False
    elif name == "base_macro_vol_term_or_implied_vol_proxy":
        family = "vol_term_or_implied_vol_proxy"
        description = "Macro baseline plus lag-clean implied-volatility term-structure proxy features."
        is_negative = False
    elif name == "base_macro_analyst_or_fund_revision_features":
        family = "analyst_or_fund_revision_features"
        description = "Macro baseline plus lagged point-in-time fundamental revision proxy features."
        is_negative = False
    elif name == "base_macro_hmm_gru":
        family = "legacy_hmm_gru_combo"
        description = "Legacy combined HMM+GRU feature set from the previous cycle."
        is_negative = True
    elif name == "full":
        family = "legacy_full_stack"
        description = "Legacy full feature stack including fundamentals."
        is_negative = False
    else:
        family = "unknown"
        description = "Feature set metadata was inferred heuristically."
        is_negative = False
    return {
        "feature_set": name,
        "feature_family": family,
        "is_negative_control": bool(is_negative),
        "feature_set_description": description,
    }


def build_selection_rule_registry() -> "OrderedDict[str, SelectionRuleSpec]":
    rules: "OrderedDict[str, SelectionRuleSpec]" = OrderedDict()
    rules["sharpe_only"] = SelectionRuleSpec(
        name="sharpe_only",
        description="Baseline rule: median validation Sharpe only.",
        validation_sharpe_quantile=0.50,
        is_default=False,
    )
    rules["robust_q25"] = SelectionRuleSpec(
        name="robust_q25",
        description="25th percentile validation Sharpe with turnover and drawdown penalties.",
        validation_sharpe_quantile=0.25,
        turnover_weight=0.05,
        max_drawdown_weight=2.0,
        is_default=False,
    )
    rules["robust_q25_retention"] = SelectionRuleSpec(
        name="robust_q25_retention",
        description=(
            "25th percentile validation Sharpe with retention/generalization uplift and "
            "turnover/max-drawdown penalties."
        ),
        validation_sharpe_quantile=0.25,
        turnover_weight=0.05,
        max_drawdown_weight=2.0,
        retention_weight=0.25,
        generalization_weight=0.15,
        is_default=True,
    )
    return rules


PRIMARY_BENCHMARK_ID = "dow30_equal_weight_rebalance_matched"
LEGACY_PRIMARY_BENCHMARK_ID = "legacy_market_proxy_equal_weight_daily"


def build_benchmark_registry() -> "OrderedDict[str, BenchmarkSpec]":
    benchmarks: "OrderedDict[str, BenchmarkSpec]" = OrderedDict()
    benchmarks["dow30_equal_weight_rebalance_matched"] = BenchmarkSpec(
        benchmark_id="dow30_equal_weight_rebalance_matched",
        benchmark_name="Equal Weight Rebalance Matched",
        description=(
            "Equal-weight benchmark across the tradable Dow 30 panel, rebalanced on the same "
            "daily cadence as the agent and charged the same transaction-cost convention."
        ),
        family="passive_rebalanced",
        uses_transaction_costs=True,
        requires_pretest_history=False,
        is_primary=True,
    )
    benchmarks["dow30_market_proxy_buy_hold"] = BenchmarkSpec(
        benchmark_id="dow30_market_proxy_buy_hold",
        benchmark_name="Market Proxy Buy and Hold",
        description=(
            "Equal-weight market proxy established at the start of the test window and then held "
            "without adaptive rebalancing."
        ),
        family="passive_buy_hold",
        uses_transaction_costs=True,
        requires_pretest_history=False,
        is_primary=False,
    )
    benchmarks["dow30_equal_weight_vol_target"] = BenchmarkSpec(
        benchmark_id="dow30_equal_weight_vol_target",
        benchmark_name="Equal Weight Vol Target",
        description=(
            "Equal-weight portfolio with causal trailing-volatility targeting and daily "
            "transaction costs when exposure changes."
        ),
        family="structured_risk_control",
        uses_transaction_costs=True,
        requires_pretest_history=True,
        is_primary=False,
    )
    benchmarks["dow30_trend_filter_overlay"] = BenchmarkSpec(
        benchmark_id="dow30_trend_filter_overlay",
        benchmark_name="Trend Filter Overlay",
        description=(
            "Equal-weight portfolio with a causal market trend filter that removes exposure in "
            "negative-trend states."
        ),
        family="structured_trend_filter",
        uses_transaction_costs=True,
        requires_pretest_history=True,
        is_primary=False,
    )
    return benchmarks


def get_benchmark_registry() -> "OrderedDict[str, dict[str, Any]]":
    return OrderedDict(
        (benchmark_id, spec.to_dict())
        for benchmark_id, spec in build_benchmark_registry().items()
    )


def compute_selection_score_from_frame(
    frame: pd.DataFrame,
    rule: SelectionRuleSpec,
) -> dict[str, Any]:
    validation_sharpe = pd.to_numeric(frame.get("validation_sharpe"), errors="coerce").dropna()
    validation_turnover = pd.to_numeric(frame.get("validation_turnover"), errors="coerce").dropna()
    validation_mdd = pd.to_numeric(frame.get("validation_max_drawdown"), errors="coerce").dropna()
    retention_ratio = pd.to_numeric(frame.get("retention_ratio"), errors="coerce").dropna()
    generalization_ratio = pd.to_numeric(frame.get("generalization_ratio"), errors="coerce").dropna()

    validation_quantile = (
        float(validation_sharpe.quantile(rule.validation_sharpe_quantile))
        if not validation_sharpe.empty
        else np.nan
    )
    turnover_median = float(validation_turnover.median()) if not validation_turnover.empty else np.nan
    mdd_median = float(validation_mdd.median()) if not validation_mdd.empty else np.nan
    retention_median = float(retention_ratio.median()) if not retention_ratio.empty else np.nan
    generalization_median = (
        float(generalization_ratio.median()) if not generalization_ratio.empty else np.nan
    )

    score = validation_quantile
    if np.isnan(score):
        score = -np.inf
    score += rule.retention_weight * (0.0 if np.isnan(retention_median) else retention_median)
    score += rule.generalization_weight * (
        0.0 if np.isnan(generalization_median) else generalization_median
    )
    score -= rule.turnover_weight * (0.0 if np.isnan(turnover_median) else turnover_median)
    score -= rule.max_drawdown_weight * (0.0 if np.isnan(mdd_median) else abs(mdd_median))

    return {
        "selection_rule": rule.name,
        "selection_rule_description": rule.description,
        "score": float(score),
        "validation_sharpe_quantile": validation_quantile,
        "validation_sharpe_quantile_level": rule.validation_sharpe_quantile,
        "validation_turnover_median": turnover_median,
        "validation_max_drawdown_median": mdd_median,
        "retention_ratio_median": retention_median,
        "generalization_ratio_median": generalization_median,
        "n_runs_in_score": int(len(frame)),
        "n_folds_in_score": int(frame["fold_id"].nunique()) if "fold_id" in frame.columns else 0,
        "n_seeds_in_score": int(frame["seed"].nunique()) if "seed" in frame.columns else 0,
    }


def build_market_proxy_frame(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    return_col: str = "daily_return",
    price_col: str = "close",
) -> pd.DataFrame:
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    if return_col not in data.columns and {"tic", price_col}.issubset(data.columns):
        data = data.sort_values(["tic", date_col]).reset_index(drop=True)
        data[return_col] = data.groupby("tic")[price_col].pct_change()
    market = (
        data.groupby(date_col, dropna=False)[return_col]
        .mean()
        .reset_index()
        .rename(columns={return_col: "benchmark_return"})
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    market["benchmark_return"] = pd.to_numeric(market["benchmark_return"], errors="coerce").fillna(0.0)
    return market


def build_exogenous_regime_frame(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    return_col: str = "daily_return",
    price_col: str = "close",
    trend_window: int = 20,
    vol_window: int = 20,
    min_history: int = 40,
) -> pd.DataFrame:
    market = build_market_proxy_frame(df, date_col=date_col, return_col=return_col, price_col=price_col)
    market = market.sort_values(date_col).reset_index(drop=True)

    shifted_returns = market["benchmark_return"].shift(1)
    trend_signal = shifted_returns.rolling(window=trend_window, min_periods=trend_window).sum()
    realized_vol = shifted_returns.rolling(window=vol_window, min_periods=vol_window).std(ddof=0) * np.sqrt(252.0)
    vol_threshold = realized_vol.expanding(min_periods=max(min_history, vol_window)).median()

    bull_mask = trend_signal >= 0.0
    high_vol_mask = realized_vol >= vol_threshold
    enough_history = trend_signal.notna() & realized_vol.notna() & vol_threshold.notna()

    labels = np.full(len(market), "unknown", dtype=object)
    labels[enough_history & bull_mask & ~high_vol_mask] = "bull_low_vol"
    labels[enough_history & bull_mask & high_vol_mask] = "bull_high_vol"
    labels[enough_history & ~bull_mask & ~high_vol_mask] = "bear_low_vol"
    labels[enough_history & ~bull_mask & high_vol_mask] = "bear_high_vol"

    market["market_trend_signal"] = trend_signal
    market["realized_volatility"] = realized_vol
    market["volatility_threshold"] = vol_threshold
    market["regime_label_exogenous"] = labels
    return market


def _prepare_benchmark_returns_wide(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "tic",
    return_col: str = "daily_return",
    price_col: str = "close",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    if return_col not in data.columns and {ticker_col, price_col}.issubset(data.columns):
        data = data.sort_values([ticker_col, date_col]).reset_index(drop=True)
        data[return_col] = data.groupby(ticker_col)[price_col].pct_change()

    raw_wide = (
        data.pivot_table(
            index=date_col,
            columns=ticker_col,
            values=return_col,
            aggfunc="last",
        )
        .sort_index()
    )
    raw_wide = raw_wide.replace([np.inf, -np.inf], np.nan)
    filled_wide = raw_wide.fillna(0.0)
    return raw_wide, filled_wide


def _equal_weight_target(
    active_row: pd.Series,
    *,
    exposure: float = 1.0,
) -> pd.Series:
    active = active_row.fillna(False).astype(bool)
    weights = pd.Series(0.0, index=active.index, dtype=float)
    n_active = int(active.sum())
    if n_active <= 0 or exposure <= 0.0:
        return weights
    weights.loc[active] = float(exposure) / float(n_active)
    return weights


def _simulate_benchmark_path(
    returns_raw_wide: pd.DataFrame,
    returns_filled_wide: pd.DataFrame,
    *,
    benchmark_id: str,
    benchmark_name: str,
    benchmark_description: str,
    benchmark_family: str,
    target_weight_fn,
    buy_cost_pct: float,
    sell_cost_pct: float,
    initial_value: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active_mask = returns_raw_wide.notna()
    prev_weights = pd.Series(0.0, index=returns_filled_wide.columns, dtype=float)
    portfolio_value = float(initial_value)

    for idx, date in enumerate(returns_filled_wide.index):
        asset_returns = returns_filled_wide.loc[date]
        target_weights = pd.Series(
            target_weight_fn(date=date, idx=idx, prev_weights=prev_weights.copy(), active_mask=active_mask.loc[date]),
            index=returns_filled_wide.columns,
            dtype=float,
        ).fillna(0.0)
        target_weights = target_weights.clip(lower=0.0)
        weight_sum = float(target_weights.sum())
        if weight_sum > 1.0 + 1e-12:
            target_weights = target_weights / weight_sum

        buys = float((target_weights - prev_weights).clip(lower=0.0).sum())
        sells = float((prev_weights - target_weights).clip(lower=0.0).sum())
        turnover = float((target_weights - prev_weights).abs().sum() / 2.0)
        transaction_cost = float((buy_cost_pct * buys) + (sell_cost_pct * sells))
        gross_return = float((target_weights * asset_returns).sum())
        net_return = gross_return - transaction_cost
        portfolio_value = float(portfolio_value * (1.0 + net_return))

        gross_denominator = 1.0 + gross_return
        if abs(gross_denominator) <= 1e-12:
            prev_weights = pd.Series(0.0, index=prev_weights.index, dtype=float)
        else:
            prev_weights = ((target_weights * (1.0 + asset_returns)) / gross_denominator).fillna(0.0)

        rows.append(
            {
                "date": pd.Timestamp(date),
                "benchmark_id": benchmark_id,
                "benchmark_name": benchmark_name,
                "benchmark_description": benchmark_description,
                "benchmark_family": benchmark_family,
                "benchmark_return": net_return,
                "benchmark_gross_return": gross_return,
                "benchmark_turnover": turnover,
                "benchmark_transaction_cost": transaction_cost,
                "benchmark_portfolio_value": portfolio_value,
            }
        )

    return pd.DataFrame(rows)


def build_benchmark_suite_frame(
    df: pd.DataFrame,
    *,
    fold_id: Optional[str] = None,
    test_start: Optional[pd.Timestamp] = None,
    test_end: Optional[pd.Timestamp] = None,
    date_col: str = "date",
    ticker_col: str = "tic",
    return_col: str = "daily_return",
    price_col: str = "close",
    buy_cost_pct: float = 0.001,
    sell_cost_pct: float = 0.001,
    initial_value: float = 1_000_000.0,
    target_annual_vol: float = 0.15,
    trend_window: int = 20,
    vol_window: int = 20,
) -> pd.DataFrame:
    returns_raw_wide, returns_filled_wide = _prepare_benchmark_returns_wide(
        df,
        date_col=date_col,
        ticker_col=ticker_col,
        return_col=return_col,
        price_col=price_col,
    )
    if returns_filled_wide.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "fold_id",
                "benchmark_id",
                "benchmark_name",
                "benchmark_description",
                "benchmark_family",
                "benchmark_return",
                "benchmark_gross_return",
                "benchmark_turnover",
                "benchmark_transaction_cost",
                "benchmark_portfolio_value",
                "is_primary_benchmark",
            ]
        )

    market_proxy = build_market_proxy_frame(
        df,
        date_col=date_col,
        return_col=return_col,
        price_col=price_col,
    ).set_index(date_col)["benchmark_return"].reindex(returns_filled_wide.index)
    lagged_market_returns = market_proxy.shift(1)
    realized_vol = lagged_market_returns.rolling(window=vol_window, min_periods=vol_window).std(ddof=0)
    annualized_vol = realized_vol * np.sqrt(252.0)
    vol_target_exposure = (float(target_annual_vol) / annualized_vol).clip(lower=0.0, upper=1.0)
    vol_target_exposure = vol_target_exposure.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    trend_signal = lagged_market_returns.rolling(window=trend_window, min_periods=trend_window).sum()
    trend_exposure = pd.Series(
        np.where(trend_signal.fillna(0.0) >= 0.0, 1.0, 0.0),
        index=returns_filled_wide.index,
        dtype=float,
    )

    benchmarks = build_benchmark_registry()
    benchmark_frames: list[pd.DataFrame] = []
    first_date = pd.Timestamp(returns_filled_wide.index[0])

    benchmark_frames.append(
        _simulate_benchmark_path(
            returns_raw_wide,
            returns_filled_wide,
            benchmark_id="dow30_equal_weight_rebalance_matched",
            benchmark_name=benchmarks["dow30_equal_weight_rebalance_matched"].benchmark_name,
            benchmark_description=benchmarks["dow30_equal_weight_rebalance_matched"].description,
            benchmark_family=benchmarks["dow30_equal_weight_rebalance_matched"].family,
            target_weight_fn=lambda *, active_mask, **_: _equal_weight_target(active_mask, exposure=1.0),
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            initial_value=initial_value,
        )
    )
    benchmark_frames.append(
        _simulate_benchmark_path(
            returns_raw_wide,
            returns_filled_wide,
            benchmark_id="dow30_market_proxy_buy_hold",
            benchmark_name=benchmarks["dow30_market_proxy_buy_hold"].benchmark_name,
            benchmark_description=benchmarks["dow30_market_proxy_buy_hold"].description,
            benchmark_family=benchmarks["dow30_market_proxy_buy_hold"].family,
            target_weight_fn=lambda *, date, prev_weights, active_mask, **_: (
                _equal_weight_target(active_mask, exposure=1.0)
                if pd.Timestamp(date) == first_date or float(prev_weights.sum()) <= 1e-12
                else prev_weights
            ),
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            initial_value=initial_value,
        )
    )
    benchmark_frames.append(
        _simulate_benchmark_path(
            returns_raw_wide,
            returns_filled_wide,
            benchmark_id="dow30_equal_weight_vol_target",
            benchmark_name=benchmarks["dow30_equal_weight_vol_target"].benchmark_name,
            benchmark_description=benchmarks["dow30_equal_weight_vol_target"].description,
            benchmark_family=benchmarks["dow30_equal_weight_vol_target"].family,
            target_weight_fn=lambda *, date, active_mask, **_: _equal_weight_target(
                active_mask,
                exposure=float(vol_target_exposure.loc[pd.Timestamp(date)]),
            ),
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            initial_value=initial_value,
        )
    )
    benchmark_frames.append(
        _simulate_benchmark_path(
            returns_raw_wide,
            returns_filled_wide,
            benchmark_id="dow30_trend_filter_overlay",
            benchmark_name=benchmarks["dow30_trend_filter_overlay"].benchmark_name,
            benchmark_description=benchmarks["dow30_trend_filter_overlay"].description,
            benchmark_family=benchmarks["dow30_trend_filter_overlay"].family,
            target_weight_fn=lambda *, date, active_mask, **_: _equal_weight_target(
                active_mask,
                exposure=float(trend_exposure.loc[pd.Timestamp(date)]),
            ),
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            initial_value=initial_value,
        )
    )

    suite = pd.concat(benchmark_frames, ignore_index=True)
    if fold_id is not None:
        suite["fold_id"] = str(fold_id)
    else:
        suite["fold_id"] = np.nan

    suite["is_primary_benchmark"] = suite["benchmark_id"] == PRIMARY_BENCHMARK_ID
    suite["date"] = pd.to_datetime(suite["date"])

    if test_start is not None:
        suite = suite[suite["date"] >= pd.Timestamp(test_start)].copy()
    if test_end is not None:
        suite = suite[suite["date"] <= pd.Timestamp(test_end)].copy()

    return suite.reset_index(drop=True)
