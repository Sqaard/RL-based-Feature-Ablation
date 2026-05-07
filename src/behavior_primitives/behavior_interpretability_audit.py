from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_DIR = PROJECT_ROOT / "Latent Actions" / "research_outputs_phase2_base_macro_teacher"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Behavior Interpretability Audit" / "research_outputs_behavior_interpretability_base_macro"
DEFAULT_PROCESSED_DATASET = PROJECT_ROOT / "processed_final_fixed_external_lagclean_full.csv"

META_COLS = {
    "run_key",
    "feature_set",
    "feature_family",
    "is_negative_control",
    "fold_id",
    "seed",
    "selected_model_type",
    "selection_rule",
    "split_name",
    "action_row_id",
    "action_step",
    "date",
}

TICKER_SECTOR = {
    "AAPL": "Information Technology",
    "AMGN": "Health Care",
    "AMZN": "Consumer Discretionary",
    "AXP": "Financials",
    "BA": "Industrials",
    "CAT": "Industrials",
    "CRM": "Information Technology",
    "CSCO": "Information Technology",
    "CVX": "Energy",
    "DIS": "Communication Services",
    "GS": "Financials",
    "HD": "Consumer Discretionary",
    "HON": "Industrials",
    "IBM": "Information Technology",
    "INTC": "Information Technology",
    "JNJ": "Health Care",
    "JPM": "Financials",
    "KO": "Consumer Staples",
    "MCD": "Consumer Discretionary",
    "MMM": "Industrials",
    "MRK": "Health Care",
    "MSFT": "Information Technology",
    "NKE": "Consumer Discretionary",
    "PG": "Consumer Staples",
    "TRV": "Financials",
    "UNH": "Health Care",
    "V": "Financials",
    "VZ": "Communication Services",
    "WMT": "Consumer Staples",
}

PROCESSED_USECOLS = [
    "date",
    "tic",
    "daily_return",
    "VIX",
    "10Y_Yield",
    "SP500_Trend",
    "turbulence",
    "Market_Regime",
    "volume_ratio",
    "atr_rel",
    "rsi_30",
    "cci_30",
    "dx_30",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if pd.isna(value):
        return None
    return value


def _read_csv(path: str | Path, *, parse_date: bool = True, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=usecols)
    if parse_date and "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _numeric_cols(frame: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(frame[col])
    ]


def _action_matrix(actions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = actions.copy()
    if {"tic", "weight"}.issubset(frame.columns):
        value_col = "weight"
    elif {"tic", "action"}.issubset(frame.columns):
        value_col = "action"
    else:
        value_col = ""
    index_cols = [col for col in ["run_key", "feature_set", "feature_family", "fold_id", "seed", "split_name", "date", "action_step", "action_row_id"] if col in frame.columns]
    if value_col:
        matrix = (
            frame.pivot_table(index=index_cols, columns="tic", values=value_col, aggfunc="last")
            .sort_index()
            .reset_index()
        )
        matrix.columns = [str(col) for col in matrix.columns]
        action_cols = [col for col in matrix.columns if col not in index_cols]
        return matrix, action_cols
    action_cols = _numeric_cols(frame, META_COLS)
    return frame[index_cols + action_cols].sort_values(index_cols).reset_index(drop=True), action_cols


def _safe_sharpe(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) < 3:
        return float("nan")
    std = float(vals.std(ddof=1))
    if std <= 1e-12:
        return float("nan")
    return float(vals.mean() / std * np.sqrt(252.0))


def _max_drawdown_from_returns(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if vals.empty:
        return float("nan")
    curve = (1.0 + vals).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    return float(drawdown.min())


def _analysis_regime_label(exogenous: Any, market_regime: Any) -> str:
    exogenous_label = str(exogenous) if pd.notna(exogenous) else ""
    if exogenous_label and exogenous_label.lower() not in {"unknown", "nan", "none"}:
        return exogenous_label
    if pd.notna(market_regime):
        try:
            numeric = float(market_regime)
            if numeric.is_integer():
                return f"market_regime_{int(numeric)}"
        except (TypeError, ValueError):
            pass
        market_label = str(market_regime)
        if market_label and market_label.lower() not in {"unknown", "nan", "none"}:
            return f"market_regime_{market_label}"
    return "unknown"


def _standardize_features(features: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.Series, pd.Series]:
    numeric = features.apply(pd.to_numeric, errors="coerce")
    medians = numeric.median(axis=0, skipna=True).fillna(0.0)
    numeric = numeric.fillna(medians)
    means = numeric.mean(axis=0)
    stds = numeric.std(axis=0, ddof=0).replace(0.0, np.nan)
    keep = stds[stds.notna()].index.tolist()
    if not keep:
        raise ValueError("No non-constant numeric behavior features were available for clustering.")
    scaled = ((numeric[keep] - means[keep]) / stds[keep]).to_numpy(dtype=float)
    return np.nan_to_num(scaled), keep, means[keep], stds[keep]


def _kmeans(X: np.ndarray, *, k: int, seed: int = 42, n_iter: int = 120, n_init: int = 8) -> tuple[np.ndarray, np.ndarray, float]:
    if len(X) == 0:
        raise ValueError("Cannot cluster an empty feature matrix.")
    k = max(1, min(int(k), len(X)))
    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")
    for _ in range(max(1, n_init)):
        init_idx = rng.choice(len(X), size=k, replace=False)
        centers = X[init_idx].copy()
        labels = np.zeros(len(X), dtype=int)
        for _step in range(n_iter):
            distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _step > 0:
                break
            labels = new_labels
            for cluster in range(k):
                members = X[labels == cluster]
                if members.size:
                    centers[cluster] = members.mean(axis=0)
                else:
                    centers[cluster] = X[rng.integers(0, len(X))]
        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers, best_inertia


def build_market_context(processed_dataset_path: str | Path, tickers: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = _read_csv(processed_dataset_path, usecols=lambda col: col in set(PROCESSED_USECOLS))
    panel = panel[panel["tic"].isin(tickers)].copy()
    panel["sector"] = panel["tic"].map(TICKER_SECTOR).fillna("Unknown")
    for col in [c for c in PROCESSED_USECOLS if c not in {"date", "tic", "Market_Regime"}]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    context = (
        panel.groupby("date", dropna=False)
        .agg(
            market_daily_return_mean=("daily_return", "mean"),
            market_daily_return_std=("daily_return", "std"),
            vix=("VIX", "first"),
            yield_10y=("10Y_Yield", "first"),
            sp500_trend=("SP500_Trend", "first"),
            turbulence=("turbulence", "first"),
            volume_ratio_mean=("volume_ratio", "mean"),
            atr_rel_mean=("atr_rel", "mean"),
            rsi_30_mean=("rsi_30", "mean"),
            cci_30_mean=("cci_30", "mean"),
            dx_30_mean=("dx_30", "mean"),
            market_regime=("Market_Regime", "first"),
        )
        .reset_index()
    )

    panel = panel.sort_values(["tic", "date"]).reset_index(drop=True)
    panel["prior20_return"] = (
        panel.groupby("tic")["daily_return"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).sum())
    )
    panel["prior20_vol"] = (
        panel.groupby("tic")["daily_return"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
    )

    strategy_rows: list[dict[str, Any]] = []
    defensive = {"Health Care", "Consumer Staples"}
    cyclical = {"Consumer Discretionary", "Financials", "Industrials", "Energy"}
    for date, day in panel.groupby("date", sort=True):
        returns = pd.to_numeric(day["daily_return"], errors="coerce")
        equal_weight = float(returns.mean())
        row = {
            "date": date,
            "synthetic_equal_weight_return": equal_weight,
        }
        ranked_mom = day.dropna(subset=["prior20_return"])
        if len(ranked_mom) >= 6:
            top_n = max(3, len(ranked_mom) // 5)
            top = ranked_mom.nlargest(top_n, "prior20_return")["daily_return"].mean()
            bottom = ranked_mom.nsmallest(top_n, "prior20_return")["daily_return"].mean()
            row["synthetic_momentum_tilt_return"] = float(top - equal_weight)
            row["synthetic_momentum_long_short_return"] = float(top - bottom)
        ranked_vol = day.dropna(subset=["prior20_vol"])
        if len(ranked_vol) >= 6:
            low_n = max(3, len(ranked_vol) // 5)
            low_vol = ranked_vol.nsmallest(low_n, "prior20_vol")["daily_return"].mean()
            high_vol = ranked_vol.nlargest(low_n, "prior20_vol")["daily_return"].mean()
            row["synthetic_low_vol_defensive_return"] = float(low_vol - equal_weight)
            row["synthetic_low_minus_high_vol_return"] = float(low_vol - high_vol)
        def_ret = day[day["sector"].isin(defensive)]["daily_return"].mean()
        cyc_ret = day[day["sector"].isin(cyclical)]["daily_return"].mean()
        row["synthetic_defensive_minus_cyclical_return"] = float(def_ret - cyc_ret) if pd.notna(def_ret) and pd.notna(cyc_ret) else np.nan
        sector_returns = day.groupby("sector")["daily_return"].mean()
        row["synthetic_sector_dispersion_return"] = float(sector_returns.max() - sector_returns.min()) if len(sector_returns) > 1 else np.nan
        strategy_rows.append(row)

    strategies = pd.DataFrame(strategy_rows)
    return context, strategies


def build_observation_diagnostics(observations_path: str | Path, tickers: Sequence[str]) -> pd.DataFrame:
    n = len(tickers)
    header = pd.read_csv(observations_path, nrows=0).columns.tolist()
    obs_cols = [f"obs_{idx:04d}" for idx in range(1 + (2 * n)) if f"obs_{idx:04d}" in header]
    raw_cols = [col for col in header if col.startswith("raw_policy_action_")]
    meta_cols = [col for col in ["run_key", "feature_set", "feature_family", "fold_id", "seed", "split_name", "date"] if col in header]
    frame = _read_csv(observations_path, usecols=meta_cols + obs_cols + raw_cols)
    if len(obs_cols) < 1 + (2 * n):
        raise ValueError("Observation export does not contain enough state columns to recover portfolio weights.")

    cash = pd.to_numeric(frame["obs_0000"], errors="coerce").fillna(0.0).to_numpy()
    prices = frame[[f"obs_{idx:04d}" for idx in range(1, 1 + n)]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    shares = frame[[f"obs_{idx:04d}" for idx in range(1 + n, 1 + (2 * n))]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    stock_values = np.clip(prices, 0.0, None) * np.clip(shares, 0.0, None)
    total_value = cash + stock_values.sum(axis=1)
    weights = np.divide(stock_values, total_value[:, None], out=np.zeros_like(stock_values), where=total_value[:, None] > 0)

    out = frame[meta_cols].copy()
    weight_cols = [f"weight_{ticker}" for ticker in tickers]
    for idx, col in enumerate(weight_cols):
        out[col] = weights[:, idx]
    out["portfolio_hhi"] = np.square(weights).sum(axis=1)
    out["portfolio_max_weight"] = weights.max(axis=1)
    out["portfolio_benchmark_deviation_l1"] = np.abs(weights - (1.0 / n)).sum(axis=1)
    sector_frame = pd.DataFrame(weights, columns=tickers)
    sector_map = pd.Series({ticker: TICKER_SECTOR.get(ticker, "Unknown") for ticker in tickers})
    for sector in sorted(sector_map.unique()):
        members = [ticker for ticker in tickers if sector_map[ticker] == sector]
        out[f"sector_weight_{sector}"] = sector_frame[members].sum(axis=1)
    sector_cols = [col for col in out.columns if col.startswith("sector_weight_")]
    out["portfolio_sector_max_weight"] = out[sector_cols].max(axis=1) if sector_cols else np.nan
    if raw_cols:
        raw = frame[raw_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["raw_policy_action_l1"] = raw.abs().sum(axis=1)
        out["raw_policy_action_l2"] = np.sqrt(np.square(raw).sum(axis=1))
        out["raw_policy_action_max_abs"] = raw.abs().max(axis=1)
        out["raw_policy_action_active_dims"] = raw.abs().gt(1e-8).sum(axis=1)
    return out


def build_decision_rows(
    *,
    actions_path: str | Path,
    observations_path: str | Path,
    daily_returns_path: str | Path,
    processed_dataset_path: str | Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    actions = _read_csv(actions_path)
    matrix, action_cols = _action_matrix(actions)
    tickers = list(action_cols)
    action_values = matrix[action_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix["action_l1"] = action_values.abs().sum(axis=1)
    matrix["action_l2"] = np.sqrt(np.square(action_values).sum(axis=1))
    matrix["action_max_abs"] = action_values.abs().max(axis=1)
    matrix["action_active_dims"] = action_values.abs().gt(1e-8).sum(axis=1)
    matrix["action_buy_dims"] = action_values.gt(1e-8).sum(axis=1)
    matrix["action_sell_dims"] = action_values.lt(-1e-8).sum(axis=1)

    obs = build_observation_diagnostics(observations_path, tickers)
    daily = _read_csv(daily_returns_path)
    daily = daily.sort_values(["run_key", "date"]).reset_index(drop=True)
    for col in ["daily_return", "benchmark_return", "excess_return_vs_benchmark", "turnover", "portfolio_value"]:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily["next_daily_return"] = daily.groupby("run_key")["daily_return"].shift(-1)
    daily["next_benchmark_return"] = daily.groupby("run_key")["benchmark_return"].shift(-1)
    daily["next_excess_return_vs_benchmark"] = daily.groupby("run_key")["excess_return_vs_benchmark"].shift(-1)
    daily["portfolio_running_peak"] = daily.groupby("run_key")["portfolio_value"].cummax()
    daily["current_drawdown"] = daily["portfolio_value"] / daily["portfolio_running_peak"] - 1.0

    context, synthetic = build_market_context(processed_dataset_path, tickers)
    join_keys = ["run_key", "fold_id", "seed", "date"]
    rows = matrix.merge(obs, on=[key for key in join_keys if key in matrix.columns and key in obs.columns], how="left", suffixes=("", "_obs"))
    rows = rows.merge(
        daily[
            [
                col
                for col in [
                    "run_key",
                    "date",
                    "daily_return",
                    "benchmark_return",
                    "excess_return_vs_benchmark",
                    "turnover",
                    "portfolio_value",
                    "next_daily_return",
                    "next_benchmark_return",
                    "next_excess_return_vs_benchmark",
                    "regime_label_exogenous",
                    "current_drawdown",
                ]
                if col in daily.columns
            ]
        ],
        on=["run_key", "date"],
        how="left",
    )
    rows = rows.merge(context, on="date", how="left")
    rows = rows.merge(synthetic, on="date", how="left")
    rows = rows.sort_values(["run_key", "date", "action_step"]).reset_index(drop=True)

    weight_cols = [f"weight_{ticker}" for ticker in tickers]
    sector_cols = [col for col in rows.columns if col.startswith("sector_weight_")]
    for group_col_prefix, cols, out_col in [
        ("weight", weight_cols, "portfolio_weight_turnover_l1"),
        ("sector", sector_cols, "portfolio_sector_change_l1"),
        ("action", action_cols, "action_change_l1"),
    ]:
        if not cols:
            continue
        diffs = rows.groupby("run_key")[cols].diff().abs().sum(axis=1)
        rows[out_col] = diffs.fillna(0.0)
    if action_cols:
        prev = rows.groupby("run_key")[action_cols].diff()
        accel = prev.groupby(rows["run_key"]).diff()
        rows["action_jitter_l2"] = np.sqrt(np.square(accel).sum(axis=1)).fillna(0.0)
    return rows, tickers, action_cols


def build_behavior_windows(rows: pd.DataFrame, tickers: Sequence[str], *, window: int = 5, min_periods: int = 3) -> tuple[pd.DataFrame, list[str]]:
    feature_cols: list[str] = []
    window_rows: list[dict[str, Any]] = []
    sector_cols = [col for col in rows.columns if col.startswith("sector_weight_")]
    synthetic_cols = [col for col in rows.columns if col.startswith("synthetic_") and col.endswith("_return")]
    base_features = [
        "action_l1",
        "action_l2",
        "action_active_dims",
        "action_buy_dims",
        "action_sell_dims",
        "action_change_l1",
        "action_jitter_l2",
        "raw_policy_action_l1",
        "raw_policy_action_l2",
        "raw_policy_action_active_dims",
        "portfolio_weight_turnover_l1",
        "portfolio_hhi",
        "portfolio_max_weight",
        "portfolio_benchmark_deviation_l1",
        "portfolio_sector_max_weight",
        "portfolio_sector_change_l1",
        "turnover",
        "daily_return",
        "benchmark_return",
        "excess_return_vs_benchmark",
        "current_drawdown",
        "vix",
        "yield_10y",
        "sp500_trend",
        "turbulence",
        "market_daily_return_mean",
        "market_daily_return_std",
        "atr_rel_mean",
        "rsi_30_mean",
        "dx_30_mean",
    ]
    base_features = [col for col in base_features if col in rows.columns]
    feature_cols = [f"{col}_wmean" for col in base_features] + [f"{col}_last" for col in base_features]
    feature_cols += [f"{col}_wmean" for col in sector_cols]

    for run_key, group in rows.groupby("run_key", sort=False):
        group = group.sort_values(["date", "action_step"]).reset_index(drop=True)
        for idx in range(len(group)):
            start = max(0, idx - window + 1)
            chunk = group.iloc[start : idx + 1]
            if len(chunk) < min_periods:
                continue
            last = group.iloc[idx]
            row: dict[str, Any] = {
                "behavior_window_id": f"{run_key}__w{idx:04d}",
                "run_key": run_key,
                "feature_set": last.get("feature_set"),
                "feature_family": last.get("feature_family"),
                "fold_id": last.get("fold_id"),
                "seed": last.get("seed"),
                "date": last.get("date"),
                "window_start_date": chunk["date"].iloc[0],
                "window_end_date": last.get("date"),
                "window_length": int(len(chunk)),
                "reward_daily_return": last.get("next_daily_return"),
                "reward_benchmark_return": last.get("next_benchmark_return"),
                "reward_excess_return_vs_benchmark": last.get("next_excess_return_vs_benchmark"),
                "same_day_excess_return_vs_benchmark": last.get("excess_return_vs_benchmark"),
                "daily_return": last.get("daily_return"),
                "benchmark_return": last.get("benchmark_return"),
                "excess_return_vs_benchmark": last.get("excess_return_vs_benchmark"),
                "current_drawdown": last.get("current_drawdown"),
                "regime_label_exogenous": last.get("regime_label_exogenous"),
                "market_regime": last.get("market_regime"),
                "analysis_regime": _analysis_regime_label(last.get("regime_label_exogenous"), last.get("market_regime")),
            }
            for col in base_features:
                values = pd.to_numeric(chunk[col], errors="coerce")
                row[f"{col}_wmean"] = float(values.mean())
                row[f"{col}_last"] = float(pd.to_numeric(pd.Series([last.get(col)]), errors="coerce").iloc[0])
            for col in sector_cols:
                row[f"{col}_wmean"] = float(pd.to_numeric(chunk[col], errors="coerce").mean())
            for col in synthetic_cols:
                row[col] = last.get(col)
            window_rows.append(row)

    windows = pd.DataFrame(window_rows)
    feature_cols = [col for col in feature_cols if col in windows.columns]
    return windows, feature_cols


def assign_primitives(windows: pd.DataFrame, feature_cols: Sequence[str], *, n_primitives: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    X, used_cols, means, stds = _standardize_features(windows[list(feature_cols)])
    labels, centers, inertia = _kmeans(X, k=n_primitives, seed=seed)
    out = windows.copy()
    out["_cluster_label"] = labels
    order = (
        out.groupby("_cluster_label")["reward_excess_return_vs_benchmark"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    mapping = {cluster: f"primitive_{rank:02d}" for rank, cluster in enumerate(order)}
    out["primitive_id"] = out["_cluster_label"].map(mapping)
    metadata = {
        "n_primitives_requested": int(n_primitives),
        "n_primitives_actual": int(len(order)),
        "kmeans_inertia": inertia,
        "feature_columns_used": used_cols,
        "feature_means": means.to_dict(),
        "feature_stds": stds.to_dict(),
        "cluster_to_primitive": {str(key): value for key, value in mapping.items()},
    }
    return out.drop(columns=["_cluster_label"]), metadata


def build_primitive_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = max(1, len(assignments))
    regime_col = "analysis_regime" if "analysis_regime" in assignments.columns else "regime_label_exogenous"
    regime_base = assignments[regime_col].value_counts(normalize=True, dropna=False)
    for primitive, group in assignments.groupby("primitive_id", sort=True):
        fold_means = group.groupby("fold_id")["reward_excess_return_vs_benchmark"].mean()
        regime_counts = group[regime_col].value_counts(normalize=True, dropna=False)
        dominant_regime = str(regime_counts.index[0]) if not regime_counts.empty else ""
        dominant_lift = float(regime_counts.iloc[0] / regime_base.get(regime_counts.index[0], np.nan)) if not regime_counts.empty else np.nan
        row = {
            "primitive_id": primitive,
            "windows": int(len(group)),
            "share": float(len(group) / total),
            "folds": int(group["fold_id"].nunique()),
            "seeds": int(group["seed"].nunique()),
            "dominant_regime": dominant_regime,
            "dominant_regime_share": float(regime_counts.iloc[0]) if not regime_counts.empty else np.nan,
            "dominant_regime_lift": dominant_lift,
            "reward_daily_return_mean": float(group["reward_daily_return"].mean()),
            "reward_excess_return_mean": float(group["reward_excess_return_vs_benchmark"].mean()),
            "reward_excess_return_median": float(group["reward_excess_return_vs_benchmark"].median()),
            "reward_excess_sharpe": _safe_sharpe(group["reward_excess_return_vs_benchmark"]),
            "reward_excess_hit_rate": float((group["reward_excess_return_vs_benchmark"] > 0).mean()),
            "reward_excess_max_drawdown": _max_drawdown_from_returns(group["reward_excess_return_vs_benchmark"]),
            "worst_fold_excess_return_mean": float(fold_means.min()) if not fold_means.empty else np.nan,
            "fold_excess_return_std": float(fold_means.std(ddof=0)) if len(fold_means) > 1 else 0.0,
            "action_l1_mean": float(group.get("action_l1_wmean", pd.Series(dtype=float)).mean()),
            "action_change_l1_mean": float(group.get("action_change_l1_wmean", pd.Series(dtype=float)).mean()),
            "action_jitter_l2_mean": float(group.get("action_jitter_l2_wmean", pd.Series(dtype=float)).mean()),
            "test_turnover_mean": float(group.get("turnover_wmean", pd.Series(dtype=float)).mean()),
            "portfolio_hhi_mean": float(group.get("portfolio_hhi_wmean", pd.Series(dtype=float)).mean()),
            "portfolio_max_weight_mean": float(group.get("portfolio_max_weight_wmean", pd.Series(dtype=float)).mean()),
            "portfolio_benchmark_deviation_l1_mean": float(group.get("portfolio_benchmark_deviation_l1_wmean", pd.Series(dtype=float)).mean()),
            "portfolio_sector_max_weight_mean": float(group.get("portfolio_sector_max_weight_wmean", pd.Series(dtype=float)).mean()),
            "current_drawdown_mean": float(group.get("current_drawdown_wmean", pd.Series(dtype=float)).mean()),
            "current_drawdown_worst": float(group.get("current_drawdown_last", pd.Series(dtype=float)).min()),
            "vix_mean": float(group.get("vix_wmean", pd.Series(dtype=float)).mean()),
            "sp500_trend_mean": float(group.get("sp500_trend_wmean", pd.Series(dtype=float)).mean()),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    for col in [
        "reward_excess_return_mean",
        "reward_excess_sharpe",
        "reward_excess_hit_rate",
        "worst_fold_excess_return_mean",
    ]:
        std = summary[col].std(ddof=0)
        summary[f"z_{col}"] = 0.0 if std <= 1e-12 or pd.isna(std) else (summary[col] - summary[col].mean()) / std
    for col in [
        "test_turnover_mean",
        "portfolio_hhi_mean",
        "portfolio_benchmark_deviation_l1_mean",
        "fold_excess_return_std",
    ]:
        std = summary[col].std(ddof=0)
        summary[f"z_{col}"] = 0.0 if std <= 1e-12 or pd.isna(std) else (summary[col] - summary[col].mean()) / std
    summary["primitive_reliability_score"] = (
        0.35 * summary["z_reward_excess_return_mean"]
        + 0.25 * summary["z_reward_excess_sharpe"]
        + 0.15 * summary["z_reward_excess_hit_rate"]
        + 0.10 * summary["z_worst_fold_excess_return_mean"]
        - 0.05 * summary["z_test_turnover_mean"].clip(lower=0.0)
        - 0.05 * summary["z_portfolio_hhi_mean"].clip(lower=0.0)
        - 0.03 * summary["z_portfolio_benchmark_deviation_l1_mean"].clip(lower=0.0)
        - 0.02 * summary["z_fold_excess_return_std"].clip(lower=0.0)
    )
    summary["primitive_type"] = summary.apply(_primitive_type, axis=1)
    return summary.sort_values("primitive_reliability_score", ascending=False).reset_index(drop=True)


def _primitive_type(row: pd.Series) -> str:
    if row["reward_excess_return_mean"] > 0 and row["reward_excess_sharpe"] > 0 and row["worst_fold_excess_return_mean"] > -0.001:
        return "profitable_reliable_candidate"
    if row["reward_excess_return_mean"] < 0 and row["action_change_l1_mean"] > 0:
        return "bad_or_noisy_primitive"
    if row["dominant_regime_lift"] > 1.5:
        return "regime_specific_primitive"
    if row["portfolio_benchmark_deviation_l1_mean"] < 0.35:
        return "benchmark_tracking_primitive"
    if row["portfolio_hhi_mean"] > 0.15 or row["portfolio_max_weight_mean"] > 0.25:
        return "concentrated_risk_primitive"
    return "mixed_primitive"


def build_regime_alignment(assignments: pd.DataFrame) -> pd.DataFrame:
    regime_col = "analysis_regime" if "analysis_regime" in assignments.columns else "regime_label_exogenous"
    counts = (
        assignments.groupby(["primitive_id", regime_col], dropna=False)
        .size()
        .rename("windows")
        .reset_index()
    )
    if counts.empty:
        return counts
    counts["primitive_share"] = counts["windows"] / counts.groupby("primitive_id")["windows"].transform("sum")
    regime_base = assignments[regime_col].value_counts(normalize=True, dropna=False).rename("base_regime_share")
    counts = counts.merge(regime_base.reset_index().rename(columns={"index": regime_col}), on=regime_col, how="left")
    if regime_col != "regime_label_exogenous":
        counts = counts.rename(columns={regime_col: "regime_label_exogenous"})
    counts["regime_lift"] = counts["primitive_share"] / counts["base_regime_share"]
    return counts.sort_values(["primitive_id", "regime_lift"], ascending=[True, False]).reset_index(drop=True)


def build_concept_alignment(assignments: pd.DataFrame) -> pd.DataFrame:
    frame = assignments.copy()
    concepts: dict[str, pd.Series] = {}
    for source, name in [
        ("vix_last", "high_vix"),
        ("current_drawdown_last", "drawdown_stress"),
        ("portfolio_hhi_last", "high_concentration"),
        ("portfolio_benchmark_deviation_l1_last", "high_benchmark_deviation"),
        ("turnover_last", "high_turnover"),
        ("action_jitter_l2_last", "high_action_jitter"),
    ]:
        if source in frame.columns:
            values = pd.to_numeric(frame[source], errors="coerce")
            cutoff = values.quantile(0.67)
            concepts[name] = values >= cutoff
    if "benchmark_return" in frame.columns:
        concepts["market_down"] = pd.to_numeric(frame["benchmark_return"], errors="coerce") < 0.0
    if {"vix_last", "benchmark_return"}.issubset(frame.columns):
        concepts["risk_off"] = concepts.get("high_vix", pd.Series(False, index=frame.index)) & (pd.to_numeric(frame["benchmark_return"], errors="coerce") < 0.0)
    rows: list[dict[str, Any]] = []
    for primitive, group in frame.groupby("primitive_id", sort=True):
        idx = group.index
        for concept, mask in concepts.items():
            mask = mask.fillna(False)
            base_rate = float(mask.mean())
            prim_rate = float(mask.loc[idx].mean()) if len(idx) else np.nan
            rows.append(
                {
                    "primitive_id": primitive,
                    "concept": concept,
                    "primitive_concept_rate": prim_rate,
                    "base_concept_rate": base_rate,
                    "concept_lift": prim_rate / base_rate if base_rate > 0 else np.nan,
                    "windows": int(len(idx)),
                }
            )
    return pd.DataFrame(rows).sort_values(["primitive_id", "concept_lift"], ascending=[True, False]).reset_index(drop=True)


def build_style_alignment(assignments: pd.DataFrame) -> pd.DataFrame:
    synthetic_cols = [col for col in assignments.columns if col.startswith("synthetic_") and col.endswith("_return")]
    rows: list[dict[str, Any]] = []
    for primitive, group in assignments.groupby("primitive_id", sort=True):
        activation = assignments["primitive_id"].eq(primitive).astype(float)
        for col in synthetic_cols:
            strategy = pd.to_numeric(assignments[col], errors="coerce")
            if strategy.notna().sum() < 10 or strategy.std(ddof=0) <= 1e-12:
                corr = np.nan
            else:
                corr = float(np.corrcoef(activation[strategy.notna()], strategy[strategy.notna()])[0, 1])
            active_mean = float(pd.to_numeric(group[col], errors="coerce").mean())
            base_mean = float(strategy.mean())
            rows.append(
                {
                    "primitive_id": primitive,
                    "synthetic_strategy": col.replace("synthetic_", "").replace("_return", ""),
                    "activation_correlation": corr,
                    "active_strategy_return_mean": active_mean,
                    "base_strategy_return_mean": base_mean,
                    "active_minus_base_strategy_return": active_mean - base_mean,
                }
            )
    return pd.DataFrame(rows).sort_values(["primitive_id", "activation_correlation"], ascending=[True, False]).reset_index(drop=True)


def write_reports(output: Path, summary: pd.DataFrame, metadata: Mapping[str, Any]) -> None:
    top = summary.head(3)
    bad = summary.sort_values("reward_excess_return_mean").head(3)
    lines = [
        "# Behavior Interpretability Audit",
        "",
        "Status: completed offline interpretability audit for the `base_macro` PPO teacher.",
        "",
        "## Decision Context",
        "",
        "G1 and G2 failed, so this audit does not claim a new trading improvement. It identifies recurring behavior primitives that can generate targeted future hypotheses.",
        "",
        "## Primitive Leaderboard",
        "",
        _markdown_table(top, ["primitive_id", "primitive_type", "windows", "share", "reward_excess_return_mean", "reward_excess_sharpe", "reward_excess_hit_rate", "test_turnover_mean", "dominant_regime", "primitive_reliability_score"]),
        "",
        "## Worst Failure Candidates",
        "",
        _markdown_table(bad, ["primitive_id", "primitive_type", "windows", "reward_excess_return_mean", "reward_excess_sharpe", "worst_fold_excess_return_mean", "action_change_l1_mean", "test_turnover_mean", "dominant_regime"]),
        "",
        "## Interpretation",
        "",
        "- Use `behavior_primitive_assignments.csv` for per-date primitive labels.",
        "- Use `behavior_primitive_summary.csv` to identify profitable, noisy, concentrated, benchmark-tracking, and regime-specific primitives.",
        "- Use `primitive_regime_alignment.csv`, `primitive_style_alignment.csv`, and `primitive_concept_alignment.csv` to convert primitives into finance-readable hypotheses.",
        "- TCAV-style hidden-state probes remain deferred until model hidden activations are exported; this audit provides the primitive labels needed for those probes.",
        "",
        "## Joseph Hand-Off",
        "",
        "Joseph should consume the primitive assignment table keyed by `run_key/fold_id/seed/date` and return concept/probe scores against these stable primitive IDs, not against ad hoc action clusters.",
    ]
    (output / "BEHAVIOR_INTERPRETABILITY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    request = """# Joseph Interpretability Request

Please use the generated behavior primitive assignments as the canonical labels:

- `behavior_primitive_assignments.csv`: one row per behavior window, keyed by `run_key/fold_id/seed/date`.
- `behavior_primitive_summary.csv`: primitive-level performance and action diagnostics.
- `primitive_regime_alignment.csv`: primitive vs market regime activation.
- `primitive_style_alignment.csv`: primitive vs synthetic strategy/style alignment.
- `primitive_concept_alignment.csv`: direct concept lifts.

Needed from Joseph:

1. TCAV-style or linear-probe scores for the same primitive IDs once hidden states are exportable.
2. A short interpretation for each bad primitive: what concept/state/action feature makes it activate?
3. A falsifiable intervention proposal for each bad primitive.
4. A warning if any primitive is not semantically stable across folds/seeds.

Do not relabel primitives without preserving the original `primitive_id`.
"""
    (output / "JOSEPH_INTERPRETABILITY_REQUEST.md").write_text(request, encoding="utf-8")

    objective = {
        "ReliabilityScore": {
            "test_sharpe_median": 0.25,
            "benchmark_excess_sharpe_median": 0.20,
            "benchmark_excess_return_median": 0.15,
            "sortino_or_calmar": 0.10,
            "worst_regime_excess_return": 0.10,
            "winner_fold_count": 0.08,
            "median_regret": -0.05,
            "seed_fold_instability": -0.04,
            "excessive_turnover_without_excess_return": -0.03,
        },
        "kill_rules": [
            "Lower turnover is not success without Sharpe and benchmark-relative improvement.",
            "Higher return is not success if Sharpe, drawdown, or worst-regime performance worsens.",
            "One seed/fold success is screening only.",
            "Benchmark-relative Sharpe/return must not degrade.",
            "Median wins do not count if worst-regime performance collapses.",
        ],
    }
    (output / "reliability_objective.json").write_text(json.dumps(_json_safe(objective), indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "status": "completed",
        "rows": metadata.get("decision_rows"),
        "behavior_windows": metadata.get("behavior_windows"),
        "n_primitives": metadata.get("n_primitives_actual"),
        "kmeans_inertia": metadata.get("kmeans_inertia"),
        "outputs": metadata.get("outputs"),
        "next_step": "Review bad/noisy primitives before designing any new PPO intervention.",
    }
    (output / "audit_report.json").write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, cols: Sequence[str]) -> str:
    existing = [col for col in cols if col in frame.columns]
    if not existing:
        return ""
    header = "| " + " | ".join(existing) + " |"
    sep = "| " + " | ".join("---" for _ in existing) + " |"
    body_rows = []
    for row in frame[existing].to_dict(orient="records"):
        values = []
        for col in existing:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{float(value):.4f}")
            else:
                values.append(str(value))
        body_rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body_rows])


def write_figures(output: Path, summary: pd.DataFrame, assignments: pd.DataFrame, regime: pd.DataFrame, style: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    ordered = summary.sort_values("primitive_reliability_score", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(ordered["primitive_id"], ordered["primitive_reliability_score"], color="#176b87")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_title("Behavior primitive reliability leaderboard", fontsize=13, weight="bold")
    ax.set_ylabel("Primitive reliability score")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "01_primitive_leaderboard.png", dpi=220)
    fig.savefig(figures / "01_primitive_leaderboard.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.scatter(summary["test_turnover_mean"], summary["reward_excess_return_mean"], s=90, c=summary["primitive_reliability_score"], cmap="RdYlGn", edgecolor="#222222", linewidth=0.5)
    for _, row in summary.iterrows():
        ax.annotate(row["primitive_id"], (row["test_turnover_mean"], row["reward_excess_return_mean"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Mean turnover in primitive windows")
    ax.set_ylabel("Next excess return vs benchmark")
    ax.set_title("Primitive action intensity vs benchmark-relative outcome", fontsize=13, weight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "02_action_risk_scatter.png", dpi=220)
    fig.savefig(figures / "02_action_risk_scatter.svg")
    plt.close(fig)

    if not regime.empty:
        matrix = regime.pivot(index="primitive_id", columns="regime_label_exogenous", values="regime_lift").fillna(0.0)
        fig, ax = plt.subplots(figsize=(8.8, 5.6))
        im = ax.imshow(matrix.to_numpy(), cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8)
        ax.set_title("Primitive regime activation lift", fontsize=13, weight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Lift vs base regime frequency")
        fig.tight_layout()
        fig.savefig(figures / "03_regime_lift_heatmap.png", dpi=220)
        fig.savefig(figures / "03_regime_lift_heatmap.svg")
        plt.close(fig)

    if not style.empty:
        matrix = style.pivot(index="primitive_id", columns="synthetic_strategy", values="activation_correlation").fillna(0.0)
        vmax = np.nanmax(np.abs(matrix.to_numpy())) or 1.0
        fig, ax = plt.subplots(figsize=(10, 5.6))
        im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8)
        ax.set_title("Primitive vs synthetic strategy alignment", fontsize=13, weight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Activation correlation")
        fig.tight_layout()
        fig.savefig(figures / "04_style_alignment_heatmap.png", dpi=220)
        fig.savefig(figures / "04_style_alignment_heatmap.svg")
        plt.close(fig)

    timeline = assignments.copy()
    if {"fold_id", "date", "primitive_id"}.issubset(timeline.columns):
        timeline["fold_num"] = timeline["fold_id"].astype(str).str.extract(r"(\d+)").astype(float)
        primitive_order = {pid: idx for idx, pid in enumerate(sorted(timeline["primitive_id"].unique()))}
        timeline["primitive_num"] = timeline["primitive_id"].map(primitive_order)
        fig, ax = plt.subplots(figsize=(11, 5.0))
        sample = timeline.sort_values(["fold_num", "date"]).copy()
        ax.scatter(sample["date"], sample["fold_num"], c=sample["primitive_num"], cmap="tab10", s=8)
        ax.set_title("Primitive activation timeline by fold", fontsize=13, weight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Fold")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / "05_primitive_timeline_by_fold.png", dpi=220)
        fig.savefig(figures / "05_primitive_timeline_by_fold.svg")
        plt.close(fig)


def run_behavior_interpretability_audit(
    *,
    actions_path: str | Path,
    observations_path: str | Path,
    daily_returns_path: str | Path,
    processed_dataset_path: str | Path,
    output_dir: str | Path,
    n_primitives: int = 6,
    window: int = 5,
    min_periods: int = 3,
    seed: int = 42,
    make_figures: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows, tickers, _action_cols = build_decision_rows(
        actions_path=actions_path,
        observations_path=observations_path,
        daily_returns_path=daily_returns_path,
        processed_dataset_path=processed_dataset_path,
    )
    windows, feature_cols = build_behavior_windows(rows, tickers, window=window, min_periods=min_periods)
    if windows.empty:
        raise ValueError("No behavior windows were produced.")
    assignments, cluster_metadata = assign_primitives(windows, feature_cols, n_primitives=n_primitives, seed=seed)
    summary = build_primitive_summary(assignments)
    regime = build_regime_alignment(assignments)
    concepts = build_concept_alignment(assignments)
    style = build_style_alignment(assignments)

    outputs = {
        "decision_state_action_rows": str(output / "decision_state_action_rows.csv"),
        "behavior_window_features": str(output / "behavior_window_features.csv"),
        "behavior_primitive_assignments": str(output / "behavior_primitive_assignments.csv"),
        "behavior_primitive_summary": str(output / "behavior_primitive_summary.csv"),
        "primitive_regime_alignment": str(output / "primitive_regime_alignment.csv"),
        "primitive_concept_alignment": str(output / "primitive_concept_alignment.csv"),
        "primitive_style_alignment": str(output / "primitive_style_alignment.csv"),
        "audit_report": str(output / "audit_report.json"),
        "markdown_report": str(output / "BEHAVIOR_INTERPRETABILITY_AUDIT.md"),
        "joseph_request": str(output / "JOSEPH_INTERPRETABILITY_REQUEST.md"),
        "reliability_objective": str(output / "reliability_objective.json"),
    }
    rows.to_csv(output / "decision_state_action_rows.csv", index=False)
    windows.to_csv(output / "behavior_window_features.csv", index=False)
    assignments.to_csv(output / "behavior_primitive_assignments.csv", index=False)
    summary.to_csv(output / "behavior_primitive_summary.csv", index=False)
    regime.to_csv(output / "primitive_regime_alignment.csv", index=False)
    concepts.to_csv(output / "primitive_concept_alignment.csv", index=False)
    style.to_csv(output / "primitive_style_alignment.csv", index=False)

    metadata = {
        **cluster_metadata,
        "decision_rows": int(len(rows)),
        "behavior_windows": int(len(assignments)),
        "tickers": list(tickers),
        "window": int(window),
        "min_periods": int(min_periods),
        "outputs": outputs,
    }
    (output / "clustering_metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    write_reports(output, summary, metadata)
    if make_figures:
        write_figures(output, summary, assignments, regime, style)
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run behavior-level interpretability audit for PPO/FinRL teacher outputs.")
    parser.add_argument("--teacher-dir", default=str(DEFAULT_TEACHER_DIR))
    parser.add_argument("--actions")
    parser.add_argument("--observations")
    parser.add_argument("--daily-returns")
    parser.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-primitives", type=int, default=6)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    teacher_dir = Path(args.teacher_dir)
    metadata = run_behavior_interpretability_audit(
        actions_path=args.actions or teacher_dir / "walk_forward_test_actions.csv",
        observations_path=args.observations or teacher_dir / "walk_forward_test_observations.csv",
        daily_returns_path=args.daily_returns or teacher_dir / "walk_forward_daily_test_returns.csv",
        processed_dataset_path=args.processed_dataset,
        output_dir=args.output_dir,
        n_primitives=args.n_primitives,
        window=args.window,
        min_periods=args.min_periods,
        seed=args.seed,
        make_figures=not args.no_figures,
    )
    print(json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
