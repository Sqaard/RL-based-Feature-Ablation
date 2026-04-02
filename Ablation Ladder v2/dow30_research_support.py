from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from dow30_horizon_a import (
    build_controlled_feature_registry,
    build_exogenous_regime_frame,
    build_market_proxy_frame,
    ensure_event_calendar_features,
)
from dow30_reporting import (
    build_corrected_walk_forward_summary,
    build_regime_reports_from_daily,
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
    dataset_name: str = "dow30_processed",
    date_col: str = "date",
    ticker_col: str = "tic",
    output_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    data = ensure_event_calendar_features(df.copy(), date_col=date_col)
    data[date_col] = pd.to_datetime(data[date_col])

    registry = build_controlled_feature_registry(feature_ladder or DEFAULT_FEATURE_GROUPS)
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
    date_col: str = "date",
) -> pd.DataFrame:
    curve = add_curve_features(curve_df, date_col=date_col)
    benchmark = build_market_proxy_frame(raw_test_df, date_col=date_col).rename(
        columns={"benchmark_return": "benchmark_return"}
    )
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
        "benchmark_return",
        "selected_model_type",
        "selection_rule",
        "regime_label_exogenous",
        "excess_return_vs_benchmark",
    ]
    return daily[expected_cols].copy()


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
    group_cols: Sequence[str] = ("feature_set",),
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
    seeds: Sequence[int] = (42, 123, 999, 2024, 2025),
    date_col: str = "date",
    output_dir: Optional[str | Path] = None,
    selection_config: Optional[Mapping[str, Any]] = None,
    model_name: str = "ppo_dow30",
) -> dict[str, Any]:
    prepared_df = ensure_event_calendar_features(df, date_col=date_col)
    feature_registry = build_controlled_feature_registry(feature_ladder or DEFAULT_FEATURE_GROUPS)
    rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    daily_test_rows: list[pd.DataFrame] = []
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
                    row["selected_artifact_score"] = selected_artifact.get("robust_selection_score")

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
    if not daily_test_df.empty:
        regime_run_level_df, regime_summary_by_feature_df, regime_summary_by_fold_df = (
            build_regime_reports_from_daily(daily_test_df)
        )
    if base_dir is not None:
        pairwise_suite.to_csv(base_dir / "pairwise_permutation_tests.csv", index=False)
        pairwise_suite.to_csv(base_dir / "pairwise_permutation_tests_recomputed.csv", index=False)
        selection_comparison_df.to_csv(base_dir / "selection_rule_comparison.csv", index=False)
        selection_summary_df.to_csv(base_dir / "selection_rule_summary.csv", index=False)
        validation_vs_test_df.to_csv(base_dir / "validation_vs_test_winner_by_fold.csv", index=False)
        if not legacy_regime_df.empty:
            legacy_regime_df.to_csv(base_dir / "walk_forward_regime_breakdown.csv", index=False)
        if not daily_test_df.empty:
            daily_test_df.to_csv(base_dir / "walk_forward_daily_test_returns.csv", index=False)
            regime_run_level_df.to_csv(base_dir / "regime_run_level_metrics.csv", index=False)
            regime_summary_by_feature_df.to_csv(base_dir / "regime_summary_by_feature_set.csv", index=False)
            regime_summary_by_fold_df.to_csv(base_dir / "regime_summary_by_fold.csv", index=False)
        diagnostics = {
            "raw_row_count": int(len(raw_results_df)),
            "unique_run_key_count": int(unique_results_df["run_key"].nunique()) if not unique_results_df.empty else 0,
            "regime_expanded_row_count": int(len(raw_results_df) - len(unique_results_df)),
            "legacy_regime_rows": int(len(legacy_regime_df)),
            "daily_test_rows": int(len(daily_test_df)),
        }
        _serialize_json(diagnostics, base_dir / "artifact_index.json")

    return {
        "results_df": unique_results_df,
        "summary_df": report["summary"],
        "regime_df": legacy_regime_df,
        "pairwise_tests": pairwise_suite,
        "selection_rule_comparison": selection_comparison_df,
        "selection_rule_summary": selection_summary_df,
        "validation_vs_test_winner_by_fold": validation_vs_test_df,
        "daily_test_df": daily_test_df,
        "regime_run_level_metrics": regime_run_level_df,
        "regime_summary_by_feature_set": regime_summary_by_feature_df,
        "regime_summary_by_fold": regime_summary_by_fold_df,
        "artifacts": artifacts,
    }
