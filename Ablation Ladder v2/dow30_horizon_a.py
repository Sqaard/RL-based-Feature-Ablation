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


def build_controlled_feature_registry(
    feature_groups: Mapping[str, Sequence[str]],
    *,
    include_event_calendar: bool = True,
) -> "OrderedDict[str, FeatureSetSpec]":
    groups = {
        name: tuple(dict.fromkeys(str(col) for col in cols))
        for name, cols in feature_groups.items()
    }

    base = groups.get("base", ())
    macro = groups.get("macro", ())
    hmm = groups.get("hmm", ())
    gru = groups.get("gru", ())
    exogenous_plus = EVENT_CALENDAR_FEATURES if include_event_calendar else ()

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
