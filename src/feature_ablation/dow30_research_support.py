from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from dow30_horizon_a import (
    LEGACY_PRIMARY_BENCHMARK_ID,
    PRIMARY_BENCHMARK_ID,
    build_benchmark_suite_frame,
    build_controlled_feature_registry,
    build_exogenous_regime_frame,
    build_market_proxy_frame,
    ensure_candidate_feature_families,
    ensure_event_calendar_features,
)
from dow30_reporting import (
    build_benchmark_comparison_reports,
    build_corrected_walk_forward_summary,
    build_primary_benchmark_enriched_summary,
    build_regime_reports_from_daily,
    build_statistical_credibility_report,
    build_selection_rule_comparison,
    compute_turnover_series_from_actions,
    deduplicate_run_level_results,
    recompute_pairwise_permutation_tests,
)


DEFAULT_FEATURE_GROUPS = OrderedDict(
    {
        "base": [
            "daily_return",
            "atr_rel",
            "macd",
            "rsi_30",
            "cci_30",
            "dx_30",
            "volume_ratio",
            "obv_pct_change",
            "turbulence",
        ],
        "macro": [
            "10Y_Yield",
            "VIX",
            "SP500_Trend",
        ],
        "hmm": [
            "Market_Regime",
            "Regime_0_Prob",
            "Regime_1_Prob",
        ],
        "gru": [
            "gru_return_forecast_1d",
            "gru_return_forecast_2d",
            "gru_return_forecast_3d",
            "gru_return_forecast_4d",
            "gru_return_forecast_5d",
            "forecast_mean",
            "forecast_std",
            "forecast_trend",
        ],
        "xsec_dispersion_correlation_regime": [
            "xsec_ret_dispersion_lag1",
            "xsec_ret_dispersion_20d_mean",
            "xsec_ret_dispersion_60d_zscore",
            "xsec_mean_pairwise_corr_20d",
            "xsec_mean_pairwise_corr_60d_zscore",
            "xsec_dispersion_minus_corr_regime_score",
        ],
        "breadth_internal_structure": [
            "breadth_advancing_share_lag1",
            "breadth_advancing_share_20d_mean",
            "breadth_declining_share_lag1",
            "breadth_above_20d_sma_share_lag1",
            "breadth_above_60d_sma_share_lag1",
            "breadth_new_20d_high_share_lag1",
            "breadth_new_20d_low_share_lag1",
            "breadth_participation_regime_score",
        ],
        "sector_relative_context": [
            "sector_ret_lag1",
            "sector_ret_20d_mean",
            "sector_ret_60d_zscore",
            "sector_rel_market_ret_20d",
            "sector_leadership_rank_20d",
            "stock_rel_sector_ret_lag1",
            "stock_rel_sector_ret_20d_mean",
            "stock_rel_sector_ret_60d_zscore",
        ],
        "xsec_sector_gated_context": [
            "xsec_sector_stockpick_gate",
            "xsec_sector_leadership_gate",
            "xsec_sector_rel_market_gate",
            "xsec_stock_rel_sector_momentum_gate",
            "xsec_sector_corr_risk_gate",
            "xsec_sector_dispersion_leadership_alignment",
        ],
        "xsec_sector_complementarity_v2": [
            "xsec_sector_v2_stockpick_regime_strength",
            "xsec_sector_v2_leadership_concentration",
            "xsec_sector_v2_stockpick_leadership_strength",
            "xsec_sector_v2_stockpick_residual_strength",
            "xsec_sector_v2_corr_leadership_mismatch",
            "xsec_sector_v2_sector_stock_confirmation",
            "xsec_sector_v2_rotation_pressure",
            "xsec_sector_v2_complementarity_score",
        ],
        "rates_term_structure_lsc": [
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
        ],
        "credit_stress_proxies": [
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
        ],
        "vol_term_or_implied_vol_proxy": [
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
        ],
        "rates_credit_vol_risk_state_context": [
            "risk_state_rates_credit_stress_gate",
            "risk_state_rates_vol_stress_gate",
            "risk_state_credit_vol_stress_gate",
            "risk_state_curve_inversion_credit_gate",
            "risk_state_curve_inversion_vol_gate",
            "risk_state_vol_backwardation_credit_gate",
            "risk_state_policy_credit_vol_composite",
            "risk_state_discount_stress_alignment",
        ],
        "analyst_or_fund_revision_features": [
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
        ],
        "fundamental": [
            "PE_ratio",
            "PB_ratio",
            "dividend_yield",
            "debt_ratio",
            "revenue_growth",
            "EV_EBITDA",
        ],
    }
)


def _iqr(values: pd.Series) -> float:
    clean = pd.Series(values).dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except TypeError:
        pass
    return float(value)


def _safe_ratio(numerator: Any, denominator: Any, default: float = np.nan) -> float:
    numerator = _safe_float(numerator)
    denominator = _safe_float(denominator)
    if np.isnan(numerator) or np.isnan(denominator):
        return default
    if abs(denominator) < 1e-12:
        return default
    return float(numerator / denominator)


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    if isinstance(value, tuple):
        return [_to_native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Series,)):
        return {str(k): _to_native(v) for k, v in value.to_dict().items()}
    if isinstance(value, (pd.DataFrame,)):
        return value.to_dict(orient="records")
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _serialize_json(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_native(dict(payload)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _align_to_calendar(
    trading_calendar: pd.DatetimeIndex,
    target: pd.Timestamp,
    side: str,
) -> Optional[pd.Timestamp]:
    if len(trading_calendar) == 0:
        return None

    if side == "left":
        idx = trading_calendar.searchsorted(target, side="left")
        if idx >= len(trading_calendar):
            return None
        return pd.Timestamp(trading_calendar[idx])

    if side == "right":
        idx = trading_calendar.searchsorted(target, side="right") - 1
        if idx < 0:
            return None
        return pd.Timestamp(trading_calendar[idx])

    raise ValueError(f"Unsupported side: {side}")


def build_feature_ladder(
    feature_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> "OrderedDict[str, list[str]]":
    groups = OrderedDict(
        (name, list(dict.fromkeys(cols)))
        for name, cols in (feature_groups or DEFAULT_FEATURE_GROUPS).items()
    )

    ladder = OrderedDict()
    ladder["base"] = list(groups.get("base", []))
    ladder["base_macro"] = list(dict.fromkeys(ladder["base"] + groups.get("macro", [])))
    ladder["base_macro_hmm"] = list(
        dict.fromkeys(ladder["base_macro"] + groups.get("hmm", []))
    )
    ladder["base_macro_hmm_gru"] = list(
        dict.fromkeys(ladder["base_macro_hmm"] + groups.get("gru", []))
    )
    ladder["full"] = list(
        dict.fromkeys(ladder["base_macro_hmm_gru"] + groups.get("fundamental", []))
    )
    return ladder


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_days: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _to_native(asdict(self))


@dataclass
class TrainOnlyPreprocessor:
    feature_cols: Sequence[str]
    date_col: str = "date"
    ticker_col: str = "tic"
    fill_strategy: str = "train_median"
    scale_strategy: str = "zscore"
    clip_quantiles: Optional[tuple[float, float]] = None
    fitted_: bool = False
    medians_: Optional[pd.Series] = None
    means_: Optional[pd.Series] = None
    stds_: Optional[pd.Series] = None
    mins_: Optional[pd.Series] = None
    maxs_: Optional[pd.Series] = None
    fit_summary_: Optional[dict[str, Any]] = None

    def fit(self, train_df: pd.DataFrame) -> "TrainOnlyPreprocessor":
        frame = train_df.copy()
        cols = [col for col in self.feature_cols if col in frame.columns]
        if not cols:
            raise ValueError("No feature columns available for preprocessing.")

        frame = frame.sort_values([self.ticker_col, self.date_col]).reset_index(drop=True)
        frame[cols] = frame.groupby(self.ticker_col)[cols].ffill()

        numeric = frame[cols].replace([np.inf, -np.inf], np.nan)
        self.medians_ = numeric.median()
        self.means_ = numeric.mean()
        self.stds_ = numeric.std(ddof=0).replace(0, 1.0).fillna(1.0)
        self.mins_ = numeric.min()
        self.maxs_ = numeric.max()
        self.fit_summary_ = {
            "feature_cols": cols,
            "n_rows": int(len(frame)),
            "fit_start": pd.to_datetime(frame[self.date_col]).min(),
            "fit_end": pd.to_datetime(frame[self.date_col]).max(),
            "fill_strategy": self.fill_strategy,
            "scale_strategy": self.scale_strategy,
            "clip_quantiles": self.clip_quantiles,
        }
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("TrainOnlyPreprocessor must be fit before transform.")

        out = frame.copy()
        cols = list(self.fit_summary_["feature_cols"])
        out = out.sort_values([self.ticker_col, self.date_col]).reset_index(drop=True)
        out[cols] = out.groupby(self.ticker_col)[cols].ffill()
        out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
        out[cols] = out[cols].fillna(self.medians_)

        if self.clip_quantiles is not None:
            lower_q, upper_q = self.clip_quantiles
            lower = out[cols].quantile(lower_q)
            upper = out[cols].quantile(upper_q)
            out[cols] = out[cols].clip(lower=lower, upper=upper, axis=1)

        if self.scale_strategy == "zscore":
            out[cols] = (out[cols] - self.means_) / self.stds_
        elif self.scale_strategy == "minmax":
            denominator = (self.maxs_ - self.mins_).replace(0, 1.0).fillna(1.0)
            out[cols] = (out[cols] - self.mins_) / denominator
        elif self.scale_strategy not in {"none", None}:
            raise ValueError(f"Unsupported scale_strategy: {self.scale_strategy}")

        return out

    def fit_transform_splits(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        self.fit(train_df)
        cols = list(self.fit_summary_["feature_cols"])

        tagged_frames = []
        for split_name, frame in (
            ("train", train_df),
            ("validation", validation_df),
            ("test", test_df),
        ):
            tagged = frame.copy()
            tagged["__split__"] = split_name
            tagged_frames.append(tagged)

        stacked = pd.concat(tagged_frames, ignore_index=True)
        stacked = stacked.sort_values([self.ticker_col, self.date_col]).reset_index(drop=True)
        stacked[cols] = stacked.groupby(self.ticker_col)[cols].ffill()
        stacked[cols] = stacked[cols].replace([np.inf, -np.inf], np.nan)
        stacked[cols] = stacked[cols].fillna(self.medians_)

        if self.scale_strategy == "zscore":
            stacked[cols] = (stacked[cols] - self.means_) / self.stds_
        elif self.scale_strategy == "minmax":
            denominator = (self.maxs_ - self.mins_).replace(0, 1.0).fillna(1.0)
            stacked[cols] = (stacked[cols] - self.mins_) / denominator

        transformed_train = stacked[stacked["__split__"] == "train"].drop(columns="__split__").reset_index(drop=True)
        transformed_validation = stacked[stacked["__split__"] == "validation"].drop(columns="__split__").reset_index(drop=True)
        transformed_test = stacked[stacked["__split__"] == "test"].drop(columns="__split__").reset_index(drop=True)
        return transformed_train, transformed_validation, transformed_test, _to_native(
            self.fit_summary_ or {}
        )


def generate_walk_forward_folds(
    df: pd.DataFrame,
    date_col: str = "date",
    start_date: Optional[str] = None,
    first_test_start: Optional[str] = None,
    end_date: Optional[str] = None,
    min_train_months: int = 60,
    inner_validation_months: int = 3,
    test_window_months: int = 3,
    step_months: int = 6,
    embargo_days: int = 0,
) -> list[WalkForwardFold]:
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    calendar = pd.DatetimeIndex(sorted(data[date_col].dropna().unique()))
    if len(calendar) == 0:
        raise ValueError("Dataset has no valid dates for fold generation.")

    global_start = pd.Timestamp(start_date) if start_date else pd.Timestamp(calendar.min())
    global_end = pd.Timestamp(end_date) if end_date else pd.Timestamp(calendar.max())
    cursor = (
        pd.Timestamp(first_test_start)
        if first_test_start
        else global_start + pd.DateOffset(months=min_train_months)
    )

    folds: list[WalkForwardFold] = []
    fold_idx = 1
    while cursor <= global_end:
        raw_test_start = cursor
        raw_test_end = cursor + pd.DateOffset(months=test_window_months) - pd.Timedelta(days=1)
        raw_validation_end = raw_test_start - pd.Timedelta(days=embargo_days + 1)
        raw_validation_start = raw_validation_end - pd.DateOffset(months=inner_validation_months) + pd.Timedelta(days=1)
        raw_train_end = raw_validation_start - pd.Timedelta(days=embargo_days + 1)

        if raw_train_end < global_start + pd.DateOffset(months=min_train_months) - pd.Timedelta(days=1):
            cursor = cursor + pd.DateOffset(months=step_months)
            continue
        if raw_test_end > global_end:
            break

        train_start = _align_to_calendar(calendar, global_start, "left")
        train_end = _align_to_calendar(calendar, raw_train_end, "right")
        validation_start = _align_to_calendar(calendar, raw_validation_start, "left")
        validation_end = _align_to_calendar(calendar, raw_validation_end, "right")
        test_start = _align_to_calendar(calendar, raw_test_start, "left")
        test_end = _align_to_calendar(calendar, raw_test_end, "right")

        if not all([train_start, train_end, validation_start, validation_end, test_start, test_end]):
            cursor = cursor + pd.DateOffset(months=step_months)
            continue

        if not (train_start <= train_end < validation_start <= validation_end < test_start <= test_end):
            cursor = cursor + pd.DateOffset(months=step_months)
            continue

        folds.append(
            WalkForwardFold(
                fold_id=f"fold_{fold_idx:02d}",
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
                embargo_days=embargo_days,
            )
        )
        fold_idx += 1
        cursor = cursor + pd.DateOffset(months=step_months)

    return folds


def folds_to_frame(folds: Sequence[WalkForwardFold]) -> pd.DataFrame:
    return pd.DataFrame([fold.to_dict() for fold in folds])


def split_frame_by_fold(
    df: pd.DataFrame,
    fold: WalkForwardFold,
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    train = data[(data[date_col] >= fold.train_start) & (data[date_col] <= fold.train_end)].copy()
    validation = data[(data[date_col] >= fold.validation_start) & (data[date_col] <= fold.validation_end)].copy()
    test = data[(data[date_col] >= fold.test_start) & (data[date_col] <= fold.test_end)].copy()
    return {"train": train, "validation": validation, "test": test}


def audit_dataset_integrity(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    constituent_history: Optional[pd.DataFrame] = None,
    date_col: str = "date",
    ticker_col: str = "tic",
    price_col: str = "close",
    fundamental_date_col: str = "date_available",
    gru_prefix: str = "gru_",
) -> dict[str, Any]:
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    issues: list[dict[str, Any]] = []
    warnings: list[str] = []
    checks: MutableMapping[str, Any] = {}

    duplicate_mask = data.duplicated([date_col, ticker_col])
    checks["duplicate_date_ticker_rows"] = int(duplicate_mask.sum())
    if checks["duplicate_date_ticker_rows"] > 0:
        issues.append(
            {
                "severity": "error",
                "check": "duplicate_date_ticker_rows",
                "message": "Found duplicate date/ticker observations.",
            }
        )

    monotonic_violations = 0
    for _, group in data.sort_values([ticker_col, date_col]).groupby(ticker_col):
        monotonic_violations += int((group[date_col].diff().dt.days.fillna(1) < 0).sum())
    checks["ticker_date_monotonicity_violations"] = monotonic_violations

    if fundamental_date_col in data.columns:
        fundamental_violations = int(
            (pd.to_datetime(data[fundamental_date_col]) > data[date_col]).fillna(False).sum()
        )
        checks["fundamental_future_leakage_rows"] = fundamental_violations
        if fundamental_violations > 0:
            issues.append(
                {
                    "severity": "error",
                    "check": "fundamental_future_leakage_rows",
                    "message": "Some rows use fundamentals before the report became available.",
                }
            )
    else:
        warnings.append(
            f"Column `{fundamental_date_col}` is missing, so fundamental-lag leakage was not checked."
        )

    if feature_cols:
        missing_features = sorted(set(feature_cols) - set(data.columns))
        checks["missing_feature_columns"] = missing_features
        if missing_features:
            issues.append(
                {
                    "severity": "error",
                    "check": "missing_feature_columns",
                    "message": "Some requested features are absent from the dataset.",
                    "details": missing_features,
                }
            )

    if constituent_history is None:
        warnings.append(
            "Historical Dow 30 membership was not verified because no constituent history table was provided."
        )
    else:
        required_cols = {ticker_col, "start_date", "end_date"}
        if not required_cols.issubset(set(constituent_history.columns)):
            issues.append(
                {
                    "severity": "error",
                    "check": "constituent_history_columns",
                    "message": "Constituent history must contain tic, start_date, and end_date columns.",
                }
            )
        else:
            history = constituent_history.copy()
            history["start_date"] = pd.to_datetime(history["start_date"])
            history["end_date"] = pd.to_datetime(history["end_date"])
            merged = data[[date_col, ticker_col]].merge(history, on=ticker_col, how="left")
            membership_valid = (
                (merged[date_col] >= merged["start_date"]) & (merged[date_col] <= merged["end_date"])
            ).fillna(False)
            membership_violations = int((~membership_valid).sum())
            checks["historical_constituent_violations"] = membership_violations
            if membership_violations > 0:
                issues.append(
                    {
                        "severity": "error",
                        "check": "historical_constituent_violations",
                        "message": "Some rows fall outside the supplied Dow 30 membership windows.",
                    }
                )

    gru_features = [col for col in data.columns if col.startswith(gru_prefix)]
    checks["gru_feature_columns"] = gru_features
    if gru_features:
        warnings.append(
            "GRU columns are present. For walk-forward runs they must be re-fit per fold or rebuilt with a rolling feature pipeline."
        )

    missing_prices = int(data[price_col].isna().sum()) if price_col in data.columns else None
    checks["missing_price_rows"] = missing_prices
    checks["row_count"] = int(len(data))
    checks["ticker_count"] = int(data[ticker_col].nunique()) if ticker_col in data.columns else None
    checks["date_start"] = data[date_col].min()
    checks["date_end"] = data[date_col].max()

    return {
        "ok": len([issue for issue in issues if issue["severity"] == "error"]) == 0,
        "checks": _to_native(checks),
        "issues": issues,
        "warnings": warnings,
    }


def build_data_card(
    df: pd.DataFrame,
    audit_report: Mapping[str, Any],
    feature_ladder: Optional[Mapping[str, Sequence[str]]] = None,
    candidate_feature_families: Optional[Sequence[str]] = None,
    include_feature_sets: Optional[Sequence[str]] = None,
    dataset_name: str = "dow30_processed",
    date_col: str = "date",
    ticker_col: str = "tic",
    output_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    data = ensure_candidate_feature_families(
        df.copy(),
        candidate_feature_families=candidate_feature_families,
        date_col=date_col,
    )
    data[date_col] = pd.to_datetime(data[date_col])

    registry = build_controlled_feature_registry(
        feature_ladder or DEFAULT_FEATURE_GROUPS,
        candidate_feature_families=candidate_feature_families,
        include_feature_sets=include_feature_sets,
    )
    card = {
        "dataset_name": dataset_name,
        "summary": {
            "rows": int(len(data)),
            "tickers": int(data[ticker_col].nunique()),
            "date_start": data[date_col].min(),
            "date_end": data[date_col].max(),
            "columns": list(map(str, data.columns)),
        },
        "feature_sets": {
            name: spec.to_metadata_dict() for name, spec in registry.items()
        },
        "integrity_gate": audit_report,
    }

    if output_path is not None:
        _serialize_json(card, output_path)
    return _to_native(card)


def build_model_card(
    model_name: str,
    feature_set_name: str,
    feature_cols: Sequence[str],
    fold: WalkForwardFold,
    preprocessing_summary: Optional[Mapping[str, Any]] = None,
    training_config: Optional[Mapping[str, Any]] = None,
    selection_config: Optional[Mapping[str, Any]] = None,
    results: Optional[Mapping[str, Any]] = None,
    output_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    card = {
        "model_name": model_name,
        "feature_set_name": feature_set_name,
        "n_features": len(feature_cols),
        "feature_cols": list(feature_cols),
        "fold": fold.to_dict(),
        "preprocessing": _to_native(preprocessing_summary or {}),
        "training_config": _to_native(training_config or {}),
        "selection_config": _to_native(selection_config or {}),
        "results": _to_native(results or {}),
    }
    if output_path is not None:
        _serialize_json(card, output_path)
    return _to_native(card)


def normalize_account_value_df(
    df_account_value: pd.DataFrame,
    value_col_name: str = "account_value",
    date_col: str = "date",
) -> pd.DataFrame:
    df = df_account_value.copy()

    if date_col not in df.columns:
        df = df.reset_index()
        if "index" in df.columns and date_col not in df.columns:
            df = df.rename(columns={"index": date_col})

    if value_col_name not in df.columns:
        candidates = [col for col in df.columns if col.lower() in {"account_value", "portfolio_value", "value"}]
        if candidates:
            df = df.rename(columns={candidates[0]: value_col_name})
        elif df.shape[1] >= 2:
            df = df.rename(columns={df.columns[1]: value_col_name})
        else:
            raise ValueError("Could not infer the account value column.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, value_col_name]].copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def add_curve_features(
    df_account_value: pd.DataFrame,
    value_col: str = "account_value",
    date_col: str = "date",
) -> pd.DataFrame:
    out = normalize_account_value_df(df_account_value, value_col_name=value_col, date_col=date_col)
    out["daily_return"] = out[value_col].pct_change().fillna(0.0)
    out["running_max"] = out[value_col].cummax()
    out["drawdown"] = out[value_col] / out["running_max"] - 1.0
    out["cumulative_return"] = out[value_col] / out[value_col].iloc[0] - 1.0
    return out


def estimate_turnover_from_actions(
    df_actions: Optional[pd.DataFrame],
    date_col: str = "date",
) -> float:
    if df_actions is None or len(df_actions) == 0:
        return float("nan")

    actions = df_actions.copy()
    if date_col not in actions.columns:
        actions = actions.reset_index()
        if "index" in actions.columns and date_col not in actions.columns:
            actions = actions.rename(columns={"index": date_col})

    if {"tic", "weight"}.issubset(actions.columns):
        wide = actions.pivot_table(index=date_col, columns="tic", values="weight", aggfunc="last")
    elif {"tic", "action"}.issubset(actions.columns):
        wide = actions.pivot_table(index=date_col, columns="tic", values="action", aggfunc="last")
    else:
        numeric_cols = [col for col in actions.columns if col != date_col and pd.api.types.is_numeric_dtype(actions[col])]
        wide = actions[[date_col] + numeric_cols].set_index(date_col)

    if wide.empty:
        return float("nan")

    wide = wide.sort_index().fillna(0.0)
    turnover = wide.diff().abs().sum(axis=1).dropna()
    if turnover.empty:
        return float("nan")
    return float(turnover.mean() / 2.0)


def compute_regime_breakdown(
    curve_df: pd.DataFrame,
    regime_frame: Optional[pd.DataFrame],
    date_col: str = "date",
    regime_col: str = "Market_Regime",
) -> pd.DataFrame:
    if regime_frame is None or regime_col not in regime_frame.columns:
        return pd.DataFrame(columns=[regime_col, "n_days", "mean_return", "volatility", "sharpe"])

    regimes = regime_frame[[date_col, regime_col]].copy()
    regimes[date_col] = pd.to_datetime(regimes[date_col])
    regimes = regimes.drop_duplicates(date_col)

    merged = add_curve_features(curve_df, date_col=date_col).merge(regimes, on=date_col, how="left")
    grouped = (
        merged.groupby(regime_col, dropna=False)["daily_return"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n_days", "mean": "mean_return", "std": "volatility"})
    )
    grouped["sharpe"] = np.where(
        grouped["volatility"].fillna(0.0).abs() > 1e-12,
        np.sqrt(252.0) * grouped["mean_return"] / grouped["volatility"],
        np.nan,
    )
    return grouped


def evaluate_equity_curve(
    df_account_value: pd.DataFrame,
    df_actions: Optional[pd.DataFrame] = None,
    regime_frame: Optional[pd.DataFrame] = None,
    date_col: str = "date",
    value_col: str = "account_value",
) -> dict[str, Any]:
    curve = add_curve_features(df_account_value, value_col=value_col, date_col=date_col)
    returns = curve["daily_return"]

    annual_return = (1.0 + returns).prod() ** (252.0 / max(len(returns), 1)) - 1.0
    annual_volatility = returns.std(ddof=0) * np.sqrt(252.0)
    sharpe = (
        np.sqrt(252.0) * returns.mean() / returns.std(ddof=0)
        if returns.std(ddof=0) > 1e-12
        else np.nan
    )

    metrics = {
        "initial_value": float(curve[value_col].iloc[0]),
        "final_value": float(curve[value_col].iloc[-1]),
        "return_pct": float((curve[value_col].iloc[-1] / curve[value_col].iloc[0] - 1.0) * 100.0),
        "cumulative_return": float(curve["cumulative_return"].iloc[-1]),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility) if not np.isnan(annual_volatility) else np.nan,
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "max_drawdown": float(curve["drawdown"].min()),
        "turnover": estimate_turnover_from_actions(df_actions, date_col=date_col),
    }

    regime_breakdown = compute_regime_breakdown(curve, regime_frame, date_col=date_col)
    return {
        "curve": curve,
        "metrics": metrics,
        "regime_breakdown": regime_breakdown,
    }


def _annualized_sharpe_from_returns(returns: pd.Series) -> float:
    numeric = pd.to_numeric(returns, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    volatility = numeric.std(ddof=0)
    if volatility <= 1e-12:
        return float("nan")
    return float(np.sqrt(252.0) * numeric.mean() / volatility)


def compute_temporal_subdomain_metrics(
    curve_df: pd.DataFrame,
    *,
    return_col: str = "daily_return",
    min_periods: int = 10,
) -> dict[str, float]:
    """Measure whether a candidate's validation performance survives temporal sub-splits."""

    if curve_df is None or curve_df.empty or return_col not in curve_df.columns:
        return {
            "first_half_sharpe": float("nan"),
            "second_half_sharpe": float("nan"),
            "subdomain_sharpe_floor": float("nan"),
            "subdomain_sharpe_gap": float("nan"),
        }

    returns = pd.to_numeric(curve_df[return_col], errors="coerce").dropna().reset_index(drop=True)
    if len(returns) < 2 * min_periods:
        return {
            "first_half_sharpe": float("nan"),
            "second_half_sharpe": float("nan"),
            "subdomain_sharpe_floor": float("nan"),
            "subdomain_sharpe_gap": float("nan"),
        }

    split_idx = len(returns) // 2
    first_sharpe = _annualized_sharpe_from_returns(returns.iloc[:split_idx])
    second_sharpe = _annualized_sharpe_from_returns(returns.iloc[split_idx:])
    sharpe_values = [value for value in (first_sharpe, second_sharpe) if not np.isnan(value)]
    floor = float(min(sharpe_values)) if sharpe_values else float("nan")
    gap = float(abs(first_sharpe - second_sharpe)) if not (np.isnan(first_sharpe) or np.isnan(second_sharpe)) else float("nan")

    return {
        "first_half_sharpe": first_sharpe,
        "second_half_sharpe": second_sharpe,
        "subdomain_sharpe_floor": floor,
        "subdomain_sharpe_gap": gap,
    }


def _clip_and_normalize_domain_multipliers(
    frame: pd.DataFrame,
    *,
    multiplier_col: str,
    min_multiplier: float,
    max_multiplier: float,
) -> pd.Series:
    clipped = pd.to_numeric(frame[multiplier_col], errors="coerce").fillna(1.0)
    clipped = clipped.clip(lower=min_multiplier, upper=max_multiplier)
    mean_value = float(clipped.mean()) if len(clipped) else 1.0
    if abs(mean_value) <= 1e-12:
        return clipped
    return clipped / mean_value


def build_domain_reward_scale_by_date(
    train_df: pd.DataFrame,
    config: Optional[Mapping[str, Any]] = None,
    *,
    date_col: str = "date",
    return_col: str = "daily_return",
) -> dict[str, Any]:
    """Build training-only reward multipliers by market domain/stress bucket."""

    cfg = dict(config or {})
    if not cfg.get("enabled", False):
        return {
            "enabled": False,
            "scale_by_date": {},
            "report": {"enabled": False},
        }
    if train_df.empty:
        return {
            "enabled": True,
            "scale_by_date": {},
            "report": {"enabled": True, "status": "empty_train_frame"},
        }

    derived_domain = str(cfg.get("derived_domain", "benchmark_abs_return_tercile"))
    mode = str(cfg.get("mode", "stress_upweight"))
    min_multiplier = float(cfg.get("min_multiplier", 0.75))
    max_multiplier = float(cfg.get("max_multiplier", 1.25))

    daily = train_df.copy()
    daily[date_col] = pd.to_datetime(daily[date_col])
    if return_col in daily.columns:
        daily_return = pd.to_numeric(daily[return_col], errors="coerce")
    elif "close" in daily.columns:
        close = pd.to_numeric(daily["close"], errors="coerce")
        daily_return = close.groupby(daily["tic"]).pct_change().fillna(0.0)
    else:
        daily_return = pd.Series(0.0, index=daily.index)
    daily = (
        pd.DataFrame({date_col: daily[date_col], "asset_return": daily_return})
        .groupby(date_col, as_index=False)["asset_return"]
        .mean()
        .rename(columns={"asset_return": "benchmark_return"})
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    if derived_domain == "benchmark_vol_tercile":
        daily["domain_value"] = daily["benchmark_return"].rolling(21, min_periods=5).std().fillna(
            daily["benchmark_return"].abs()
        )
        labels = ["low_vol", "mid_vol", "high_vol"]
    elif derived_domain == "benchmark_signed_return_tercile":
        daily["domain_value"] = daily["benchmark_return"]
        labels = ["low_return", "mid_return", "high_return"]
    elif derived_domain == "benchmark_drawdown_tercile":
        benchmark_curve = (1.0 + daily["benchmark_return"].fillna(0.0)).cumprod()
        benchmark_drawdown = benchmark_curve / benchmark_curve.cummax() - 1.0
        daily["domain_value"] = benchmark_drawdown.abs()
        labels = ["shallow_drawdown", "mid_drawdown", "deep_drawdown"]
    elif derived_domain == "market_regime" and "Market_Regime" in train_df.columns:
        regimes = (
            train_df[[date_col, "Market_Regime"]]
            .drop_duplicates(date_col)
            .assign(**{date_col: lambda x: pd.to_datetime(x[date_col])})
        )
        daily = daily.merge(regimes, on=date_col, how="left")
        daily["domain_label"] = daily["Market_Regime"].fillna("unknown").map(lambda value: f"market_regime_{value}")
        labels = []
    else:
        daily["domain_value"] = daily["benchmark_return"].abs()
        labels = ["low_abs_return", "mid_abs_return", "high_abs_return"]

    if "domain_label" not in daily.columns:
        try:
            daily["domain_label"] = pd.qcut(
                daily["domain_value"].rank(method="first"),
                q=3,
                labels=labels,
            ).astype(str)
        except ValueError:
            fallback_label = labels[1] if len(labels) >= 2 else "default_domain"
            daily["domain_label"] = fallback_label

    if mode == "inverse_frequency":
        counts = daily["domain_label"].value_counts(dropna=False)
        mean_count = float(counts.mean()) if not counts.empty else 1.0
        weight_by_domain = {
            str(domain): float(mean_count / count) if count else 1.0
            for domain, count in counts.items()
        }
    else:
        default_weights = {
            "low_abs_return": 0.85,
            "mid_abs_return": 1.0,
            "high_abs_return": 1.20,
            "low_vol": 0.85,
            "mid_vol": 1.0,
            "high_vol": 1.20,
            "low_return": 1.20,
            "mid_return": 1.0,
            "high_return": 0.85,
            "shallow_drawdown": 0.85,
            "mid_drawdown": 1.0,
            "deep_drawdown": 1.20,
        }
        weight_by_domain = {
            str(key): float(value)
            for key, value in dict(cfg.get("domain_weights", default_weights)).items()
        }

    daily["raw_multiplier"] = daily["domain_label"].map(weight_by_domain).fillna(1.0)
    daily["multiplier"] = _clip_and_normalize_domain_multipliers(
        daily,
        multiplier_col="raw_multiplier",
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )
    daily["date_key"] = daily[date_col].dt.strftime("%Y-%m-%d")
    scale_by_date = {
        str(row["date_key"]): float(row["multiplier"])
        for row in daily[["date_key", "multiplier"]].to_dict(orient="records")
    }
    report = {
        "enabled": True,
        "derived_domain": derived_domain,
        "mode": mode,
        "min_multiplier": min_multiplier,
        "max_multiplier": max_multiplier,
        "date_count": int(len(daily)),
        "domain_counts": {str(k): int(v) for k, v in daily["domain_label"].value_counts().items()},
        "weight_by_domain": weight_by_domain,
        "multiplier_min": float(daily["multiplier"].min()) if not daily.empty else np.nan,
        "multiplier_max": float(daily["multiplier"].max()) if not daily.empty else np.nan,
        "multiplier_mean": float(daily["multiplier"].mean()) if not daily.empty else np.nan,
    }
    return {
        "enabled": True,
        "scale_by_date": scale_by_date,
        "report": report,
    }


def _domain_reward_date_key(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def make_domain_reward_scaled_env_cls(env_cls):
    """Wrap a FinRL-style env class so only returned training rewards are reweighted."""

    class DomainRewardScaledEnv(env_cls):
        def __init__(
            self,
            *args,
            domain_reward_scale_by_date: Optional[Mapping[str, float]] = None,
            domain_reward_scaling_report: Optional[Mapping[str, Any]] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.domain_reward_scale_by_date = dict(domain_reward_scale_by_date or {})
            self.domain_reward_scaling_report = dict(domain_reward_scaling_report or {})

        def _current_domain_reward_multiplier(self) -> float:
            if not self.domain_reward_scale_by_date:
                return 1.0
            date_value = None
            if hasattr(self, "_get_date"):
                try:
                    date_value = self._get_date()
                except Exception:
                    date_value = None
            if date_value is None and hasattr(self, "date_memory") and self.date_memory:
                date_value = self.date_memory[-1]
            return float(self.domain_reward_scale_by_date.get(_domain_reward_date_key(date_value), 1.0))

        def step(self, actions):
            result = super().step(actions)
            if not self.domain_reward_scale_by_date:
                return result
            multiplier = self._current_domain_reward_multiplier()
            if len(result) == 5:
                state, reward, done, truncated, info = result
                scaled_reward = reward * multiplier
                self.reward = scaled_reward
                return state, scaled_reward, done, truncated, info
            state, reward, done, info = result
            scaled_reward = reward * multiplier
            self.reward = scaled_reward
            return state, scaled_reward, done, info

    DomainRewardScaledEnv.__name__ = f"DomainRewardScaled{env_cls.__name__}"
    return DomainRewardScaledEnv


def normalize_action_regularization_config(config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", False))
    normalized = {
        "enabled": enabled,
        "turnover_penalty": float(cfg.get("turnover_penalty", 0.0)),
        "smoothness_penalty": float(cfg.get("smoothness_penalty", 0.0)),
        "concentration_penalty": float(cfg.get("concentration_penalty", 0.0)),
        "max_weight_penalty": float(cfg.get("max_weight_penalty", 0.0)),
        "max_weight_target": float(cfg.get("max_weight_target", 0.20)),
        "kl_to_previous_penalty": float(cfg.get("kl_to_previous_penalty", 0.0)),
        "normalize_penalties": bool(cfg.get("normalize_penalties", True)),
        "train_only": bool(cfg.get("train_only", True)),
        "eps": float(cfg.get("eps", 1e-12)),
    }
    if not any(
        normalized[key] > 0.0
        for key in (
            "turnover_penalty",
            "smoothness_penalty",
            "concentration_penalty",
            "max_weight_penalty",
            "kl_to_previous_penalty",
        )
    ):
        normalized["enabled"] = False
    return normalized


def _safe_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def compute_action_regularization_terms(
    current_weights: Any,
    previous_weights: Any,
    previous_previous_weights: Any = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, float]:
    cfg = normalize_action_regularization_config(config)
    current = _safe_array(current_weights)
    previous = _safe_array(previous_weights)
    if current.shape != previous.shape:
        raise ValueError("current_weights and previous_weights must have the same shape")

    n_assets = max(1, int(current.size))
    turnover_raw = float(np.abs(current - previous).sum())
    if previous_previous_weights is None:
        smoothness_raw = 0.0
    else:
        previous_previous = _safe_array(previous_previous_weights)
        if previous_previous.shape != current.shape:
            raise ValueError("previous_previous_weights must have the same shape as current_weights")
        smoothness_raw = float(np.square(current - (2.0 * previous) + previous_previous).sum())

    current_clipped = np.clip(current, 0.0, None)
    hhi_raw = float(np.square(current_clipped).sum())
    max_weight_raw = float(current_clipped.max()) if current_clipped.size else 0.0
    max_weight_excess_raw = float(max(0.0, max_weight_raw - cfg["max_weight_target"]))

    eps = float(cfg["eps"])
    current_dist = np.r_[max(0.0, 1.0 - float(current_clipped.sum())), current_clipped] + eps
    previous_clipped = np.clip(previous, 0.0, None)
    previous_dist = np.r_[max(0.0, 1.0 - float(previous_clipped.sum())), previous_clipped] + eps
    current_dist = current_dist / current_dist.sum()
    previous_dist = previous_dist / previous_dist.sum()
    kl_to_previous_raw = float(np.sum(current_dist * np.log(current_dist / previous_dist)))

    if cfg["normalize_penalties"]:
        turnover_value = turnover_raw / 2.0
        smoothness_value = smoothness_raw / 4.0
        equal_weight_hhi = 1.0 / n_assets
        hhi_value = max(0.0, (hhi_raw - equal_weight_hhi) / max(eps, 1.0 - equal_weight_hhi))
        max_weight_excess_value = max_weight_excess_raw
        kl_to_previous_value = kl_to_previous_raw
    else:
        turnover_value = turnover_raw
        smoothness_value = smoothness_raw
        hhi_value = hhi_raw
        max_weight_excess_value = max_weight_excess_raw
        kl_to_previous_value = kl_to_previous_raw

    turnover_penalty = cfg["turnover_penalty"] * turnover_value
    smoothness_penalty = cfg["smoothness_penalty"] * smoothness_value
    concentration_penalty = cfg["concentration_penalty"] * hhi_value
    max_weight_penalty = cfg["max_weight_penalty"] * max_weight_excess_value
    kl_penalty = cfg["kl_to_previous_penalty"] * kl_to_previous_value
    total_penalty = float(
        turnover_penalty
        + smoothness_penalty
        + concentration_penalty
        + max_weight_penalty
        + kl_penalty
    )

    return {
        "turnover_raw": turnover_raw,
        "turnover_value": float(turnover_value),
        "smoothness_raw": smoothness_raw,
        "smoothness_value": float(smoothness_value),
        "herfindahl_concentration": hhi_raw,
        "concentration_value": float(hhi_value),
        "max_weight": max_weight_raw,
        "max_weight_excess": max_weight_excess_raw,
        "max_weight_excess_value": float(max_weight_excess_value),
        "kl_to_previous_raw": kl_to_previous_raw,
        "kl_to_previous_value": float(kl_to_previous_value),
        "turnover_penalty_contribution": float(turnover_penalty),
        "smoothness_penalty_contribution": float(smoothness_penalty),
        "concentration_penalty_contribution": float(concentration_penalty),
        "max_weight_penalty_contribution": float(max_weight_penalty),
        "kl_to_previous_penalty_contribution": float(kl_penalty),
        "total_action_regularization_penalty": total_penalty,
    }


def make_action_regularized_env_cls(env_cls):
    """Wrap a FinRL-style env class so action-behavior penalties affect training rewards only."""

    class ActionRegularizedEnv(env_cls):
        def __init__(
            self,
            *args,
            action_regularization_config: Optional[Mapping[str, Any]] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.action_regularization_config = normalize_action_regularization_config(
                action_regularization_config
            )
            self.action_regularization_audit_records: list[dict[str, Any]] = []
            self._action_reg_previous_weights: Optional[np.ndarray] = None
            self._action_reg_previous_previous_weights: Optional[np.ndarray] = None

        def _portfolio_stock_weights(self) -> np.ndarray:
            stock_dim = int(getattr(self, "stock_dim", getattr(self, "stock_dimension", 0)))
            if stock_dim <= 0:
                return np.asarray([], dtype=float)
            state = _safe_array(getattr(self, "state", []))
            if state.size < 1 + (2 * stock_dim):
                return np.zeros(stock_dim, dtype=float)
            cash = float(state[0])
            prices = state[1 : 1 + stock_dim]
            shares = state[1 + stock_dim : 1 + (2 * stock_dim)]
            stock_values = np.clip(prices, 0.0, None) * np.clip(shares, 0.0, None)
            total_value = float(cash + stock_values.sum())
            if total_value <= 0.0 or not np.isfinite(total_value):
                return np.zeros(stock_dim, dtype=float)
            return np.nan_to_num(stock_values / total_value, nan=0.0, posinf=0.0, neginf=0.0)

        def _current_action_regularization_date(self) -> str:
            date_value = None
            if hasattr(self, "_get_date"):
                try:
                    date_value = self._get_date()
                except Exception:
                    date_value = None
            if date_value is None and hasattr(self, "date_memory") and self.date_memory:
                date_value = self.date_memory[-1]
            return _domain_reward_date_key(date_value)

        def get_action_regularization_audit_frame(self) -> pd.DataFrame:
            return pd.DataFrame(self.action_regularization_audit_records)

        def get_action_regularization_summary(self) -> dict[str, Any]:
            frame = self.get_action_regularization_audit_frame()
            if frame.empty:
                return {"enabled": bool(self.action_regularization_config.get("enabled", False)), "steps": 0}
            summary: dict[str, Any] = {
                "enabled": bool(self.action_regularization_config.get("enabled", False)),
                "steps": int(len(frame)),
            }
            for col in [
                "turnover_raw",
                "smoothness_raw",
                "herfindahl_concentration",
                "max_weight",
                "total_action_regularization_penalty",
                "gross_reward_before_action_regularization",
                "training_reward_after_action_regularization",
            ]:
                if col in frame.columns:
                    values = pd.to_numeric(frame[col], errors="coerce")
                    summary[f"{col}_mean"] = float(values.mean())
                    summary[f"{col}_median"] = float(values.median())
            return summary

        def step(self, actions):
            if not self.action_regularization_config.get("enabled", False):
                return super().step(actions)

            previous_weights = self._portfolio_stock_weights()
            previous_previous_weights = self._action_reg_previous_previous_weights
            result = super().step(actions)
            current_weights = self._portfolio_stock_weights()
            terms = compute_action_regularization_terms(
                current_weights,
                previous_weights,
                previous_previous_weights,
                self.action_regularization_config,
            )
            penalty = terms["total_action_regularization_penalty"]

            if len(result) == 5:
                state, reward, done, truncated, info = result
                gross_reward = float(reward)
                adjusted_reward = gross_reward - penalty
                info = dict(info or {})
                info["action_regularization_penalty"] = penalty
                info["gross_reward_before_action_regularization"] = gross_reward
                self.reward = adjusted_reward
                out = (state, adjusted_reward, done, truncated, info)
            else:
                state, reward, done, info = result
                gross_reward = float(reward)
                adjusted_reward = gross_reward - penalty
                info = dict(info or {})
                info["action_regularization_penalty"] = penalty
                info["gross_reward_before_action_regularization"] = gross_reward
                self.reward = adjusted_reward
                out = (state, adjusted_reward, done, info)

            self.action_regularization_audit_records.append(
                {
                    "date": self._current_action_regularization_date(),
                    "gross_reward_before_action_regularization": gross_reward,
                    "training_reward_after_action_regularization": adjusted_reward,
                    **terms,
                }
            )
            self._action_reg_previous_previous_weights = previous_weights.copy()
            self._action_reg_previous_weights = current_weights.copy()
            return out

    ActionRegularizedEnv.__name__ = f"ActionRegularized{env_cls.__name__}"
    return ActionRegularizedEnv


def build_fold_benchmark_suite_export(
    raw_test_df: pd.DataFrame,
    *,
    fold_id: str,
    benchmark_source_df: Optional[pd.DataFrame] = None,
    date_col: str = "date",
    buy_cost_pct: float = 0.001,
    sell_cost_pct: float = 0.001,
    initial_value: float = 1_000_000.0,
) -> pd.DataFrame:
    if raw_test_df.empty:
        return pd.DataFrame()

    benchmark_source = (
        benchmark_source_df.copy()
        if benchmark_source_df is not None and not benchmark_source_df.empty
        else raw_test_df.copy()
    )
    benchmark_source[date_col] = pd.to_datetime(benchmark_source[date_col])
    raw_test = raw_test_df.copy()
    raw_test[date_col] = pd.to_datetime(raw_test[date_col])

    return build_benchmark_suite_frame(
        benchmark_source,
        fold_id=fold_id,
        test_start=raw_test[date_col].min(),
        test_end=raw_test[date_col].max(),
        date_col=date_col,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        initial_value=initial_value,
    )


def build_daily_test_export(
    curve_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    *,
    run_key: str,
    feature_set: str,
    feature_family: str,
    is_negative_control: bool,
    fold_id: str,
    seed: int,
    selected_model_type: Optional[str] = None,
    selection_rule: str = "checkpoint_robust_score",
    df_actions: Optional[pd.DataFrame] = None,
    benchmark_frame: Optional[pd.DataFrame] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    curve = add_curve_features(curve_df, date_col=date_col)
    if benchmark_frame is not None and not benchmark_frame.empty:
        benchmark = benchmark_frame.copy()
        benchmark[date_col] = pd.to_datetime(benchmark[date_col])
    else:
        benchmark = build_market_proxy_frame(raw_test_df, date_col=date_col).rename(
            columns={"benchmark_return": "benchmark_return"}
        )
        benchmark["benchmark_id"] = LEGACY_PRIMARY_BENCHMARK_ID
        benchmark["benchmark_turnover"] = np.nan
        benchmark["benchmark_transaction_cost"] = np.nan
    for col in ("benchmark_id", "benchmark_return", "benchmark_turnover", "benchmark_transaction_cost"):
        if col not in benchmark.columns:
            benchmark[col] = np.nan
    regime = build_exogenous_regime_frame(raw_test_df, date_col=date_col)[
        [date_col, "regime_label_exogenous"]
    ].copy()
    turnover_frame = compute_turnover_series_from_actions(df_actions, date_col=date_col)

    daily = curve.merge(benchmark, on=date_col, how="left")
    daily = daily.merge(turnover_frame, on=date_col, how="left")
    daily = daily.merge(regime, on=date_col, how="left")
    daily["run_key"] = run_key
    daily["feature_set"] = feature_set
    daily["feature_family"] = feature_family
    daily["is_negative_control"] = bool(is_negative_control)
    daily["fold_id"] = fold_id
    daily["seed"] = int(seed)
    daily["selected_model_type"] = selected_model_type
    daily["selection_rule"] = selection_rule
    daily["excess_return_vs_benchmark"] = daily["daily_return"] - daily["benchmark_return"].fillna(0.0)
    daily = daily.rename(columns={"account_value": "portfolio_value"})
    expected_cols = [
        date_col,
        "run_key",
        "feature_set",
        "feature_family",
        "is_negative_control",
        "fold_id",
        "seed",
        "daily_return",
        "portfolio_value",
        "turnover",
        "benchmark_id",
        "benchmark_return",
        "benchmark_turnover",
        "benchmark_transaction_cost",
        "selected_model_type",
        "selection_rule",
        "regime_label_exogenous",
        "excess_return_vs_benchmark",
    ]
    return daily[expected_cols].copy()


def build_test_action_export(
    df_actions: Optional[pd.DataFrame],
    *,
    run_key: str,
    feature_set: str,
    feature_family: str,
    is_negative_control: bool,
    fold_id: str,
    seed: int,
    selected_model_type: Optional[str] = None,
    selection_rule: str = "checkpoint_robust_score",
    split_name: str = "test",
    date_col: str = "date",
) -> pd.DataFrame:
    """Persist policy action traces for Phase-2 latent-action diagnostics."""

    if df_actions is None or df_actions.empty:
        return pd.DataFrame()

    actions = df_actions.copy()
    if date_col not in actions.columns:
        actions = actions.reset_index()
        if "index" in actions.columns:
            actions = actions.rename(columns={"index": "action_step"})
    if date_col in actions.columns:
        parsed_dates = pd.to_datetime(actions[date_col], errors="coerce")
        if parsed_dates.notna().any():
            actions[date_col] = parsed_dates
        elif "action_step" not in actions.columns:
            actions = actions.rename(columns={date_col: "action_step"})
    if "action_step" not in actions.columns:
        actions.insert(0, "action_step", np.arange(len(actions), dtype=int))

    actions = actions.reset_index(drop=True)
    if "action_row_id" in actions.columns:
        actions["action_row_id"] = np.arange(len(actions), dtype=int)
    else:
        actions.insert(0, "action_row_id", np.arange(len(actions), dtype=int))
    metadata = {
        "run_key": run_key,
        "feature_set": feature_set,
        "feature_family": feature_family,
        "is_negative_control": bool(is_negative_control),
        "fold_id": fold_id,
        "seed": int(seed),
        "selected_model_type": selected_model_type,
        "selection_rule": selection_rule,
        "split_name": split_name,
    }
    for col, value in reversed(list(metadata.items())):
        if col in actions.columns:
            actions[col] = value
        else:
            actions.insert(0, col, value)
    return actions


def build_test_observation_export(
    df_observations: Optional[pd.DataFrame],
    *,
    run_key: str,
    feature_set: str,
    feature_family: str,
    is_negative_control: bool,
    fold_id: str,
    seed: int,
    selected_model_type: Optional[str] = None,
    selection_rule: str = "checkpoint_robust_score",
    split_name: str = "test",
    date_col: str = "date",
) -> pd.DataFrame:
    """Persist exact policy observation traces for Phase-2 latent-action diagnostics."""

    if df_observations is None or df_observations.empty:
        return pd.DataFrame()

    observations = df_observations.copy().reset_index(drop=True)
    if "observation_row_id" in observations.columns:
        observations["observation_row_id"] = np.arange(len(observations), dtype=int)
    else:
        observations.insert(0, "observation_row_id", np.arange(len(observations), dtype=int))
    if date_col in observations.columns:
        observations[date_col] = pd.to_datetime(observations[date_col], errors="coerce")

    metadata = {
        "run_key": run_key,
        "feature_set": feature_set,
        "feature_family": feature_family,
        "is_negative_control": bool(is_negative_control),
        "fold_id": fold_id,
        "seed": int(seed),
        "selected_model_type": selected_model_type,
        "selection_rule": selection_rule,
        "split_name": split_name,
    }
    for col, value in reversed(list(metadata.items())):
        if col in observations.columns:
            observations[col] = value
        else:
            observations.insert(0, col, value)
    return observations


def compute_generalization_ratio(train_metric: Any, validation_metric: Any) -> float:
    return _safe_ratio(validation_metric, train_metric)


def compute_retention_ratio(validation_metric: Any, test_metric: Any) -> float:
    return _safe_ratio(test_metric, validation_metric)


def score_artifact_candidates(
    candidate_df: pd.DataFrame,
    objective_col: str = "validation_sharpe",
    train_metric_col: str = "train_sharpe",
    drawdown_col: str = "validation_max_drawdown",
    turnover_col: str = "validation_turnover",
    generalization_weight: float = 0.35,
    drawdown_weight: float = 0.10,
    turnover_weight: float = 0.05,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df.copy()

    scored = candidate_df.copy()
    scored["generalization_ratio"] = scored.apply(
        lambda row: compute_generalization_ratio(row.get(train_metric_col), row.get(objective_col)),
        axis=1,
    )
    scored["train_validation_gap"] = (
        pd.to_numeric(scored[train_metric_col], errors="coerce")
        - pd.to_numeric(scored[objective_col], errors="coerce")
    ).abs()
    scored["validation_drawdown_penalty"] = pd.to_numeric(scored.get(drawdown_col, np.nan), errors="coerce").abs()
    scored["validation_turnover_penalty"] = pd.to_numeric(scored.get(turnover_col, np.nan), errors="coerce").fillna(0.0)

    scored["robust_selection_score"] = (
        pd.to_numeric(scored[objective_col], errors="coerce").fillna(-np.inf)
        + generalization_weight * scored["generalization_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        - generalization_weight * scored["train_validation_gap"].fillna(0.0)
        - drawdown_weight * scored["validation_drawdown_penalty"].fillna(0.0)
        - turnover_weight * scored["validation_turnover_penalty"].fillna(0.0)
    )
    scored = scored.sort_values(
        ["robust_selection_score", objective_col],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    return scored


def score_artifact_candidates_by_rule(
    candidate_df: pd.DataFrame,
    selection_rule: str = "checkpoint_robust_score",
    objective_col: str = "validation_sharpe",
    train_metric_col: str = "train_sharpe",
    drawdown_col: str = "validation_max_drawdown",
    turnover_col: str = "validation_turnover",
    generalization_weight: float = 0.35,
    drawdown_weight: float = 0.10,
    turnover_weight: float = 0.05,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df.copy()

    def numeric_candidate_column(scored_frame: pd.DataFrame, col: str) -> pd.Series:
        if col not in scored_frame.columns:
            return pd.Series(np.nan, index=scored_frame.index, dtype=float)
        return pd.to_numeric(scored_frame[col], errors="coerce")

    if selection_rule == "legacy_validation_sharpe":
        scored = candidate_df.copy()
        scored["artifact_selection_rule"] = selection_rule
        scored["artifact_selection_score"] = pd.to_numeric(scored[objective_col], errors="coerce")
        if "validation_return_pct" not in scored.columns:
            scored["validation_return_pct"] = np.nan
        return scored.sort_values(
            [objective_col, "validation_return_pct"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

    if selection_rule == "checkpoint_generalization_score":
        scored = candidate_df.copy()
        scored["generalization_ratio"] = scored.apply(
            lambda row: compute_generalization_ratio(row.get(train_metric_col), row.get(objective_col)),
            axis=1,
        )
        if "validation_return_pct" not in scored.columns:
            scored["validation_return_pct"] = np.nan
        scored["artifact_selection_rule"] = selection_rule
        scored["artifact_selection_score"] = (
            pd.to_numeric(scored["generalization_ratio"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(-np.inf)
        )
        return scored.sort_values(
            ["artifact_selection_score", objective_col, "validation_return_pct"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)

    if selection_rule == "checkpoint_temporal_robust_score":
        scored = score_artifact_candidates(
            candidate_df,
            objective_col=objective_col,
            train_metric_col=train_metric_col,
            drawdown_col=drawdown_col,
            turnover_col=turnover_col,
            generalization_weight=generalization_weight,
            drawdown_weight=drawdown_weight,
            turnover_weight=turnover_weight,
        )
        floor = numeric_candidate_column(scored, "validation_subdomain_sharpe_floor")
        gap = numeric_candidate_column(scored, "validation_subdomain_sharpe_gap").abs()
        floor_term = np.tanh(floor.fillna(0.0) / 3.0)
        gap_penalty = np.tanh(gap.fillna(0.0) / 5.0)
        scored["temporal_robust_selection_score"] = (
            pd.to_numeric(scored["robust_selection_score"], errors="coerce").fillna(-np.inf)
            + 0.20 * floor_term
            - 0.10 * gap_penalty
        )
        scored["artifact_selection_rule"] = selection_rule
        scored["artifact_selection_score"] = scored["temporal_robust_selection_score"]
        return scored.sort_values(
            ["artifact_selection_score", objective_col],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

    if selection_rule != "checkpoint_robust_score":
        raise ValueError(f"Unsupported checkpoint selection rule: {selection_rule}")

    scored = score_artifact_candidates(
        candidate_df,
        objective_col=objective_col,
        train_metric_col=train_metric_col,
        drawdown_col=drawdown_col,
        turnover_col=turnover_col,
        generalization_weight=generalization_weight,
        drawdown_weight=drawdown_weight,
        turnover_weight=turnover_weight,
    )
    scored["artifact_selection_rule"] = selection_rule
    scored["artifact_selection_score"] = scored["robust_selection_score"]
    return scored


def select_best_artifact(
    candidate_df: pd.DataFrame,
    selection_rule: str = "checkpoint_robust_score",
    **score_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored = score_artifact_candidates_by_rule(
        candidate_df,
        selection_rule=selection_rule,
        **score_kwargs,
    )
    if scored.empty:
        return scored, {}
    selected = scored.iloc[0].to_dict()
    selected["artifact_selection_rule"] = selection_rule
    selected["artifact_selection_score"] = selected.get(
        "artifact_selection_score",
        selected.get("robust_selection_score"),
    )
    return scored, selected


def select_best_artifact_by_robust_score(
    candidate_df: pd.DataFrame,
    **score_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return select_best_artifact(
        candidate_df,
        selection_rule="checkpoint_robust_score",
        **score_kwargs,
    )


def build_walk_forward_report(
    results_df: pd.DataFrame,
    group_cols: Sequence[str] = ("feature_set", "feature_family", "is_negative_control"),
    output_dir: Optional[str | Path] = None,
) -> dict[str, pd.DataFrame]:
    if results_df.empty:
        empty = pd.DataFrame()
        return {"summary": empty, "folds": empty, "regime": empty, "diagnostics": {}}

    unique_results_df, diagnostics = deduplicate_run_level_results(results_df)
    grouped = build_corrected_walk_forward_summary(
        unique_results_df,
        group_cols=tuple(group_cols),
    )
    report = {
        "summary": grouped,
        "folds": unique_results_df.copy(),
        "regime": pd.DataFrame(),
        "diagnostics": diagnostics,
    }

    if output_dir is not None:
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(base / "walk_forward_summary.csv", index=False)
        grouped.to_csv(base / "corrected_walk_forward_summary.csv", index=False)
        unique_results_df.to_csv(base / "walk_forward_results.csv", index=False)
        unique_results_df.to_csv(base / "unique_run_level_results.csv", index=False)
        _serialize_json(diagnostics, base / "walk_forward_report_diagnostics.json")

    return report


def paired_permutation_test(
    results_df: pd.DataFrame,
    left_label: str,
    right_label: str,
    strategy_col: str = "feature_set",
    value_col: str = "test_sharpe",
    pair_cols: Sequence[str] = ("fold_id", "seed"),
    n_permutations: int = 10_000,
    random_state: int = 42,
) -> dict[str, Any]:
    pivot = (
        results_df.pivot_table(
            index=list(pair_cols),
            columns=strategy_col,
            values=value_col,
            aggfunc="mean",
        )
        .dropna(subset=[left_label, right_label], how="any")
    )
    if pivot.empty:
        return {
            "left": left_label,
            "right": right_label,
            "n_pairs": 0,
            "observed_diff": np.nan,
            "p_value": np.nan,
        }

    diffs = (pivot[left_label] - pivot[right_label]).to_numpy(dtype=float)
    observed = float(diffs.mean())
    rng = np.random.default_rng(random_state)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(diffs)))
    permuted = (signs * diffs).mean(axis=1)
    p_value = float((np.abs(permuted) >= abs(observed)).mean())

    return {
        "left": left_label,
        "right": right_label,
        "n_pairs": int(len(diffs)),
        "observed_diff": observed,
        "p_value": p_value,
        "mean_left": float(pivot[left_label].mean()),
        "mean_right": float(pivot[right_label].mean()),
    }


def build_pairwise_permutation_suite(
    results_df: pd.DataFrame,
    strategy_col: str = "feature_set",
    value_col: str = "test_sharpe",
    pair_cols: Sequence[str] = ("fold_id", "seed"),
    n_permutations: int = 10_000,
    random_state: int = 42,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    labels = list(dict.fromkeys(results_df[strategy_col].dropna().tolist()))
    rows = []
    for idx, left in enumerate(labels):
        for right in labels[idx + 1 :]:
            rows.append(
                paired_permutation_test(
                    results_df=results_df,
                    left_label=left,
                    right_label=right,
                    strategy_col=strategy_col,
                    value_col=value_col,
                    pair_cols=pair_cols,
                    n_permutations=n_permutations,
                    random_state=random_state,
                )
            )

    suite = pd.DataFrame(rows)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suite.to_csv(path, index=False)
    return suite


def run_feature_ablation_ladder(
    df: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    run_fold_fn: Callable[..., Mapping[str, Any]],
    feature_ladder: Optional[Mapping[str, Sequence[str]]] = None,
    candidate_feature_families: Optional[Sequence[str]] = None,
    feature_set_filter: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (42, 123, 999, 2024, 2025),
    date_col: str = "date",
    output_dir: Optional[str | Path] = None,
    selection_config: Optional[Mapping[str, Any]] = None,
    model_name: str = "ppo_dow30",
) -> dict[str, Any]:
    prepared_df = ensure_candidate_feature_families(
        df,
        candidate_feature_families=candidate_feature_families,
        date_col=date_col,
    )
    feature_registry = build_controlled_feature_registry(
        feature_ladder or DEFAULT_FEATURE_GROUPS,
        candidate_feature_families=candidate_feature_families,
        include_feature_sets=feature_set_filter,
    )
    rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    daily_test_rows: list[pd.DataFrame] = []
    test_action_rows: list[pd.DataFrame] = []
    test_observation_rows: list[pd.DataFrame] = []
    benchmark_suite_rows: list[pd.DataFrame] = []
    artifacts: dict[str, Any] = {}

    base_dir = Path(output_dir) if output_dir is not None else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    for feature_set_name, feature_spec in feature_registry.items():
        feature_cols = list(feature_spec.columns)
        for fold in folds:
            splits = split_frame_by_fold(prepared_df, fold, date_col=date_col)
            for seed in seeds:
                run_key = f"{feature_set_name}__{fold.fold_id}__seed{seed}"
                result = dict(
                    run_fold_fn(
                        train_df=splits["train"].copy(),
                        validation_df=splits["validation"].copy(),
                        test_df=splits["test"].copy(),
                        feature_cols=list(feature_cols),
                        fold=fold,
                        seed=seed,
                        feature_set_name=feature_set_name,
                    )
                )

                train_metrics = dict(result.get("train_metrics", {}))
                validation_metrics = dict(result.get("validation_metrics", {}))
                test_metrics = dict(result.get("test_metrics", {}))

                generalization_ratio = compute_generalization_ratio(
                    train_metrics.get("sharpe"),
                    validation_metrics.get("sharpe"),
                )
                retention_ratio = compute_retention_ratio(
                    validation_metrics.get("sharpe"),
                    test_metrics.get("sharpe"),
                )

                row = {
                    "run_key": run_key,
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "feature_family": feature_spec.feature_family,
                    "is_negative_control": feature_spec.is_negative_control,
                    "feature_set_description": feature_spec.feature_set_description,
                    "fold_id": fold.fold_id,
                    "seed": seed,
                    "n_features": len(feature_cols),
                    "generalization_ratio": generalization_ratio,
                    "retention_ratio": retention_ratio,
                    "selected_artifact_type": result.get("selected_artifact_type"),
                    "selection_rule": result.get("selection_rule", "checkpoint_robust_score"),
                    "checkpoint_selection_rule": result.get(
                        "checkpoint_selection_rule",
                        "checkpoint_robust_score",
                    ),
                }
                for prefix, metrics in (
                    ("train", train_metrics),
                    ("validation", validation_metrics),
                    ("test", test_metrics),
                ):
                    row[f"{prefix}_sharpe"] = metrics.get("sharpe")
                    row[f"{prefix}_return_pct"] = metrics.get("return_pct")
                    row[f"{prefix}_max_drawdown"] = metrics.get("max_drawdown")
                    row[f"{prefix}_turnover"] = metrics.get("turnover")
                selected_artifact = result.get("selected_artifact", {})
                if isinstance(selected_artifact, Mapping):
                    row["robust_selection_score"] = selected_artifact.get("robust_selection_score")
                    row["selected_artifact_score"] = selected_artifact.get(
                        "artifact_selection_score",
                        selected_artifact.get("robust_selection_score"),
                    )
                    for score_col in (
                        "artifact_selection_score",
                        "temporal_robust_selection_score",
                        "validation_first_half_sharpe",
                        "validation_second_half_sharpe",
                        "validation_subdomain_sharpe_floor",
                        "validation_subdomain_sharpe_gap",
                    ):
                        if score_col in selected_artifact:
                            row[score_col] = selected_artifact.get(score_col)

                rows.append(row)

                regime_breakdown = result.get("regime_breakdown")
                if isinstance(regime_breakdown, pd.DataFrame) and not regime_breakdown.empty:
                    tmp = regime_breakdown.copy()
                    tmp["run_key"] = run_key
                    tmp["feature_set"] = feature_set_name
                    tmp["fold_id"] = fold.fold_id
                    tmp["seed"] = seed
                    if "mean_return" in tmp.columns:
                        tmp["regime_mean_return"] = tmp["mean_return"]
                    regime_rows.extend(tmp.to_dict(orient="records"))

                daily_test_frame = result.get("daily_test_frame")
                if isinstance(daily_test_frame, pd.DataFrame) and not daily_test_frame.empty:
                    daily_test_rows.append(daily_test_frame.copy())

                test_action_frame = result.get("test_action_frame")
                if isinstance(test_action_frame, pd.DataFrame) and not test_action_frame.empty:
                    test_action_rows.append(test_action_frame.copy())

                test_observation_frame = result.get("test_observation_frame")
                if isinstance(test_observation_frame, pd.DataFrame) and not test_observation_frame.empty:
                    test_observation_rows.append(test_observation_frame.copy())

                benchmark_suite_frame = result.get("benchmark_suite_frame")
                if isinstance(benchmark_suite_frame, pd.DataFrame) and not benchmark_suite_frame.empty:
                    benchmark_suite_rows.append(benchmark_suite_frame.copy())

                artifacts[run_key] = result

                if base_dir is not None:
                    model_cards_dir = base_dir / "model_cards"
                    model_cards_dir.mkdir(exist_ok=True)
                    build_model_card(
                        model_name=model_name,
                        feature_set_name=feature_set_name,
                        feature_cols=feature_cols,
                        fold=fold,
                        preprocessing_summary=result.get("preprocessing_summary"),
                        training_config=result.get("training_config"),
                        selection_config=selection_config,
                        results={
                            "train_metrics": train_metrics,
                            "validation_metrics": validation_metrics,
                            "test_metrics": test_metrics,
                            "generalization_ratio": generalization_ratio,
                            "retention_ratio": retention_ratio,
                            "selection_rule": row["selection_rule"],
                            "checkpoint_selection_rule": row["checkpoint_selection_rule"],
                        },
                        output_path=model_cards_dir / f"{run_key}.json",
                    )

    raw_results_df = pd.DataFrame(rows)
    legacy_regime_df = pd.DataFrame(regime_rows) if regime_rows else pd.DataFrame()
    if not legacy_regime_df.empty and "Market_Regime" in legacy_regime_df.columns and "regime" not in legacy_regime_df.columns:
        legacy_regime_df = legacy_regime_df.rename(columns={"Market_Regime": "regime"})
    daily_test_df = pd.concat(daily_test_rows, ignore_index=True) if daily_test_rows else pd.DataFrame()
    test_actions_df = pd.concat(test_action_rows, ignore_index=True) if test_action_rows else pd.DataFrame()
    test_observations_df = (
        pd.concat(test_observation_rows, ignore_index=True) if test_observation_rows else pd.DataFrame()
    )
    benchmark_suite_df = (
        pd.concat(benchmark_suite_rows, ignore_index=True)
        if benchmark_suite_rows
        else pd.DataFrame()
    )
    if not benchmark_suite_df.empty:
        benchmark_suite_df["date"] = pd.to_datetime(benchmark_suite_df["date"])
        benchmark_suite_df = benchmark_suite_df.drop_duplicates(
            subset=["fold_id", "date", "benchmark_id"],
            keep="first",
        ).reset_index(drop=True)

    report = build_walk_forward_report(
        results_df=raw_results_df,
        group_cols=("feature_set",),
        output_dir=base_dir,
    )
    unique_results_df = report["folds"]
    pairwise_suite = recompute_pairwise_permutation_tests(unique_results_df)
    selection_comparison_df, selection_summary_df, validation_vs_test_df = build_selection_rule_comparison(
        unique_results_df
    )
    regime_run_level_df = pd.DataFrame()
    regime_summary_by_feature_df = pd.DataFrame()
    regime_summary_by_fold_df = pd.DataFrame()
    benchmark_run_level_df = pd.DataFrame()
    benchmark_summary_by_feature_df = pd.DataFrame()
    benchmark_summary_by_fold_df = pd.DataFrame()
    enriched_summary_df = report["summary"].copy()
    if not daily_test_df.empty:
        regime_run_level_df, regime_summary_by_feature_df, regime_summary_by_fold_df = (
            build_regime_reports_from_daily(daily_test_df)
        )
    if not daily_test_df.empty and not benchmark_suite_df.empty:
        benchmark_run_level_df, benchmark_summary_by_feature_df, benchmark_summary_by_fold_df = (
            build_benchmark_comparison_reports(
                daily_test_df,
                benchmark_suite_df,
            )
        )
        enriched_summary_df = build_primary_benchmark_enriched_summary(
            report["summary"],
            benchmark_summary_by_feature_df,
        )
    statistical_credibility = build_statistical_credibility_report(
        unique_results_df,
        selection_summary_df=selection_summary_df,
        benchmark_summary_by_feature_df=benchmark_summary_by_feature_df,
    )
    if base_dir is not None:
        pairwise_suite.to_csv(base_dir / "pairwise_permutation_tests.csv", index=False)
        pairwise_suite.to_csv(base_dir / "pairwise_permutation_tests_recomputed.csv", index=False)
        selection_comparison_df.to_csv(base_dir / "selection_rule_comparison.csv", index=False)
        selection_summary_df.to_csv(base_dir / "selection_rule_summary.csv", index=False)
        validation_vs_test_df.to_csv(base_dir / "validation_vs_test_winner_by_fold.csv", index=False)
        enriched_summary_df.to_csv(
            base_dir / "corrected_walk_forward_summary_with_primary_benchmark.csv",
            index=False,
        )
        if not legacy_regime_df.empty:
            legacy_regime_df.to_csv(base_dir / "walk_forward_regime_breakdown.csv", index=False)
        if not daily_test_df.empty:
            daily_test_df.to_csv(base_dir / "walk_forward_daily_test_returns.csv", index=False)
            regime_run_level_df.to_csv(base_dir / "regime_run_level_metrics.csv", index=False)
            regime_summary_by_feature_df.to_csv(base_dir / "regime_summary_by_feature_set.csv", index=False)
            regime_summary_by_fold_df.to_csv(base_dir / "regime_summary_by_fold.csv", index=False)
        if not test_actions_df.empty:
            test_actions_df.to_csv(base_dir / "walk_forward_test_actions.csv", index=False)
        if not test_observations_df.empty:
            test_observations_df.to_csv(base_dir / "walk_forward_test_observations.csv", index=False)
        if not benchmark_suite_df.empty:
            benchmark_suite_df.to_csv(base_dir / "benchmark_suite_daily.csv", index=False)
        if not benchmark_run_level_df.empty:
            benchmark_run_level_df.to_csv(base_dir / "benchmark_run_level_metrics.csv", index=False)
            benchmark_summary_by_feature_df.to_csv(base_dir / "benchmark_summary_by_feature_set.csv", index=False)
            benchmark_summary_by_fold_df.to_csv(base_dir / "benchmark_summary_by_fold.csv", index=False)
        warnings = []
        if daily_test_df.empty:
            warnings.append("Daily test export was empty, so regime diagnostics were not written.")
        if benchmark_suite_df.empty:
            warnings.append("Benchmark suite export was empty, so multi-benchmark reports were not written.")
        artifact_index = {
            "raw_row_count": int(len(raw_results_df)),
            "unique_run_key_count": int(unique_results_df["run_key"].nunique()) if not unique_results_df.empty else 0,
            "regime_expanded_row_count": int(len(raw_results_df) - len(unique_results_df)),
            "legacy_regime_rows": int(len(legacy_regime_df)),
            "daily_test_rows": int(len(daily_test_df)),
            "test_action_rows": int(len(test_actions_df)),
            "test_observation_rows": int(len(test_observations_df)),
            "benchmark_suite_rows": int(len(benchmark_suite_df)),
            "primary_benchmark_id": PRIMARY_BENCHMARK_ID,
            "warnings": warnings,
            "outputs": {
                "walk_forward_results": str(base_dir / "walk_forward_results.csv"),
                "corrected_walk_forward_summary": str(base_dir / "corrected_walk_forward_summary.csv"),
                "corrected_walk_forward_summary_with_primary_benchmark": str(
                    base_dir / "corrected_walk_forward_summary_with_primary_benchmark.csv"
                ),
                "selection_rule_comparison": str(base_dir / "selection_rule_comparison.csv"),
                "selection_rule_summary": str(base_dir / "selection_rule_summary.csv"),
                "validation_vs_test_winner_by_fold": str(base_dir / "validation_vs_test_winner_by_fold.csv"),
                "pairwise_permutation_tests_recomputed": str(
                    base_dir / "pairwise_permutation_tests_recomputed.csv"
                ),
                "statistical_credibility_report": str(base_dir / "statistical_credibility_report.json"),
            },
        }
        if not daily_test_df.empty:
            artifact_index["outputs"]["walk_forward_daily_test_returns"] = str(
                base_dir / "walk_forward_daily_test_returns.csv"
            )
            artifact_index["outputs"]["regime_run_level_metrics"] = str(base_dir / "regime_run_level_metrics.csv")
            artifact_index["outputs"]["regime_summary_by_feature_set"] = str(
                base_dir / "regime_summary_by_feature_set.csv"
            )
            artifact_index["outputs"]["regime_summary_by_fold"] = str(base_dir / "regime_summary_by_fold.csv")
        if not test_actions_df.empty:
            artifact_index["outputs"]["walk_forward_test_actions"] = str(
                base_dir / "walk_forward_test_actions.csv"
            )
        if not test_observations_df.empty:
            artifact_index["outputs"]["walk_forward_test_observations"] = str(
                base_dir / "walk_forward_test_observations.csv"
            )
        if not benchmark_suite_df.empty:
            artifact_index["outputs"]["benchmark_suite_daily"] = str(base_dir / "benchmark_suite_daily.csv")
        if not benchmark_run_level_df.empty:
            artifact_index["outputs"]["benchmark_run_level_metrics"] = str(
                base_dir / "benchmark_run_level_metrics.csv"
            )
            artifact_index["outputs"]["benchmark_summary_by_feature_set"] = str(
                base_dir / "benchmark_summary_by_feature_set.csv"
            )
            artifact_index["outputs"]["benchmark_summary_by_fold"] = str(
                base_dir / "benchmark_summary_by_fold.csv"
            )
        _serialize_json(artifact_index, base_dir / "artifact_index.json")
        _serialize_json(statistical_credibility, base_dir / "statistical_credibility_report.json")
    else:
        artifact_index = {
            "raw_row_count": int(len(raw_results_df)),
            "unique_run_key_count": int(unique_results_df["run_key"].nunique()) if not unique_results_df.empty else 0,
            "test_action_rows": int(len(test_actions_df)),
            "test_observation_rows": int(len(test_observations_df)),
            "benchmark_suite_rows": int(len(benchmark_suite_df)),
            "primary_benchmark_id": PRIMARY_BENCHMARK_ID,
        }

    return {
        "results_df": unique_results_df,
        "summary_df": report["summary"],
        "summary_with_primary_benchmark_df": enriched_summary_df,
        "regime_df": legacy_regime_df,
        "pairwise_tests": pairwise_suite,
        "selection_rule_comparison": selection_comparison_df,
        "selection_rule_summary": selection_summary_df,
        "validation_vs_test_winner_by_fold": validation_vs_test_df,
        "daily_test_df": daily_test_df,
        "test_actions_df": test_actions_df,
        "test_observations_df": test_observations_df,
        "benchmark_suite_df": benchmark_suite_df,
        "benchmark_run_level_metrics": benchmark_run_level_df,
        "benchmark_summary_by_feature_set": benchmark_summary_by_feature_df,
        "benchmark_summary_by_fold": benchmark_summary_by_fold_df,
        "regime_run_level_metrics": regime_run_level_df,
        "regime_summary_by_feature_set": regime_summary_by_feature_df,
        "regime_summary_by_fold": regime_summary_by_fold_df,
        "statistical_credibility_report": statistical_credibility,
        "artifact_index": artifact_index,
        "artifacts": artifacts,
    }
