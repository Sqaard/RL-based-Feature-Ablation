from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from dow30_horizon_a import (
    SelectionRuleSpec,
    PRIMARY_BENCHMARK_ID,
    build_benchmark_suite_frame,
    build_selection_rule_registry,
    compute_selection_score_from_frame,
    infer_feature_metadata,
)


REGIME_EXPANDED_COLUMNS = {
    "regime",
    "regime_mean_return",
    "regime_sharpe",
    "n_days",
    "mean_return",
    "observations",
    "regime_label_exogenous",
    "daily_return",
    "portfolio_value",
    "turnover",
    "benchmark_return",
    "excess_return_vs_benchmark",
    "hit_rate",
}


def _iqr(values: pd.Series) -> float:
    clean = pd.Series(values).dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(0.75) - clean.quantile(0.25))


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
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Series,)):
        return {str(k): _json_safe(v) for k, v in value.to_dict().items()}
    if isinstance(value, (pd.DataFrame,)):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def ensure_feature_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "feature_set" not in out.columns:
        return out

    metadata_df = pd.DataFrame([infer_feature_metadata(name) for name in out["feature_set"]])
    for col in ("feature_family", "is_negative_control", "feature_set_description"):
        if col not in out.columns:
            out[col] = metadata_df[col].values
        else:
            missing_mask = out[col].isna()
            out.loc[missing_mask, col] = metadata_df.loc[missing_mask, col].values
    return out


def deduplicate_run_level_results(
    raw_results_df: pd.DataFrame,
    *,
    key_col: str = "run_key",
    allowed_expanded_cols: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if raw_results_df.empty:
        diagnostics = {
            "raw_row_count": 0,
            "unique_run_key_count": 0,
            "regime_expanded_row_count": 0,
            "duplicate_run_key_count": 0,
            "expansion_detected": False,
            "notes": ["Input results frame is empty."],
        }
        return raw_results_df.copy(), diagnostics

    data = raw_results_df.copy()
    if key_col not in data.columns:
        raise KeyError(f"Expected `{key_col}` in results columns.")

    allowed_cols = set(allowed_expanded_cols or REGIME_EXPANDED_COLUMNS)
    allowed_cols.add(key_col)
    invariant_cols = [col for col in data.columns if col not in allowed_cols]

    inconsistent_keys: list[str] = []
    duplicate_sizes = data.groupby(key_col, dropna=False).size()
    duplicated_keys = duplicate_sizes[duplicate_sizes > 1]
    for run_key, frame in data.groupby(key_col, dropna=False):
        if len(frame) <= 1:
            continue
        for col in invariant_cols:
            normalized = frame[col].astype(str).replace({"nan": "<NA>", "NaT": "<NA>"})
            if normalized.nunique(dropna=False) > 1:
                inconsistent_keys.append(str(run_key))
                break

    if inconsistent_keys:
        sample = ", ".join(sorted(dict.fromkeys(inconsistent_keys))[:5])
        raise ValueError(
            "run_key duplication is not a pure regime expansion. Inconsistent invariant values found for: "
            f"{sample}"
        )

    unique_df = data.drop_duplicates(subset=[key_col], keep="first").reset_index(drop=True)
    unique_df = ensure_feature_metadata(unique_df)
    if "selection_rule" not in unique_df.columns:
        unique_df["selection_rule"] = "checkpoint_robust_score"

    diagnostics = {
        "raw_row_count": int(len(data)),
        "unique_run_key_count": int(unique_df[key_col].nunique()),
        "regime_expanded_row_count": int(len(data) - len(unique_df)),
        "duplicate_run_key_count": int(len(duplicated_keys)),
        "expansion_detected": bool((duplicate_sizes > 1).any()),
        "notes": [],
    }
    if diagnostics["raw_row_count"] > diagnostics["unique_run_key_count"]:
        diagnostics["notes"].append(
            "Raw rows exceeded unique run keys, so reporting was rebuilt on deduplicated run-level rows."
        )
    else:
        diagnostics["notes"].append("Input results were already unique at run_key level.")

    return unique_df, diagnostics


def build_corrected_walk_forward_summary(
    unique_results_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("feature_set", "feature_family", "is_negative_control"),
) -> pd.DataFrame:
    if unique_results_df.empty:
        return pd.DataFrame()

    data = ensure_feature_metadata(unique_results_df)
    if "robust_selection_score" not in data.columns:
        data["robust_selection_score"] = np.nan
    grouped = (
        data.groupby(list(group_cols), dropna=False)
        .agg(
            runs=("run_key", "nunique"),
            folds=("fold_id", "nunique"),
            seeds=("seed", "nunique"),
            n_features=("n_features", "max"),
            validation_sharpe_median=("validation_sharpe", "median"),
            validation_sharpe_iqr=("validation_sharpe", _iqr),
            test_sharpe_median=("test_sharpe", "median"),
            test_sharpe_iqr=("test_sharpe", _iqr),
            validation_return_pct_median=("validation_return_pct", "median"),
            test_return_pct_median=("test_return_pct", "median"),
            test_max_drawdown_median=("test_max_drawdown", "median"),
            test_max_drawdown_iqr=("test_max_drawdown", _iqr),
            test_turnover_median=("test_turnover", "median"),
            test_turnover_iqr=("test_turnover", _iqr),
            robust_selection_score_median=("robust_selection_score", "median"),
            generalization_ratio_median=("generalization_ratio", "median"),
            retention_ratio_median=("retention_ratio", "median"),
        )
        .reset_index()
        .sort_values(["test_sharpe_median", "retention_ratio_median"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return grouped


def _build_value_series_from_returns(
    returns: pd.Series,
    *,
    initial_value: float = 1_000_000.0,
) -> pd.Series:
    clean_returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    return pd.Series(initial_value, index=clean_returns.index, dtype=float) * (1.0 + clean_returns).cumprod()


def _compute_path_metrics(
    returns: pd.Series,
    portfolio_values: Optional[pd.Series] = None,
) -> dict[str, float]:
    clean_returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if portfolio_values is None:
        values = _build_value_series_from_returns(clean_returns)
    else:
        values = pd.to_numeric(portfolio_values, errors="coerce")
        if values.dropna().empty:
            values = _build_value_series_from_returns(clean_returns)
        else:
            values = values.ffill().bfill()

    daily_vol = float(clean_returns.std(ddof=0)) if len(clean_returns) > 0 else np.nan
    sharpe = (
        float(np.sqrt(252.0) * clean_returns.mean() / daily_vol)
        if len(clean_returns) > 0 and daily_vol > 1e-12
        else np.nan
    )
    start_value = float(values.iloc[0]) if len(values) > 0 else np.nan
    end_value = float(values.iloc[-1]) if len(values) > 0 else np.nan
    return_pct = (
        float((end_value / start_value - 1.0) * 100.0)
        if not np.isnan(start_value) and abs(start_value) > 1e-12 and not np.isnan(end_value)
        else np.nan
    )
    return {
        "return_pct": return_pct,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown_from_values(values),
    }


def _aggregate_selection_inputs(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "validation_sharpe_median": float(pd.to_numeric(frame["validation_sharpe"], errors="coerce").median()),
        "test_sharpe_median": float(pd.to_numeric(frame["test_sharpe"], errors="coerce").median()),
        "test_return_pct_median": float(pd.to_numeric(frame["test_return_pct"], errors="coerce").median()),
        "test_max_drawdown_median": float(pd.to_numeric(frame["test_max_drawdown"], errors="coerce").median()),
        "test_turnover_median": float(pd.to_numeric(frame["test_turnover"], errors="coerce").median()),
        "seed_count": int(frame["seed"].nunique()) if "seed" in frame.columns else int(len(frame)),
        "run_count": int(frame["run_key"].nunique()) if "run_key" in frame.columns else int(len(frame)),
    }


def build_selection_rule_comparison(
    unique_results_df: pd.DataFrame,
    *,
    rules: Optional[Mapping[str, SelectionRuleSpec]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if unique_results_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    data = ensure_feature_metadata(unique_results_df)
    selection_rules = OrderedLike(rules or build_selection_rule_registry())

    comparison_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []

    for fold_id, fold_frame in data.groupby("fold_id", dropna=False):
        actual_test_grouped = (
            fold_frame.groupby("feature_set", dropna=False)["test_sharpe"]
            .median()
            .sort_values(ascending=False)
        )
        actual_test_winner = actual_test_grouped.index[0] if not actual_test_grouped.empty else np.nan
        actual_test_winner_sharpe = (
            float(actual_test_grouped.iloc[0]) if not actual_test_grouped.empty else np.nan
        )

        for rule in selection_rules.values():
            fold_rule_rows: list[dict[str, Any]] = []
            for feature_set, frame in fold_frame.groupby("feature_set", dropna=False):
                metadata = infer_feature_metadata(feature_set)
                score_payload = compute_selection_score_from_frame(frame, rule)
                score_payload.update(_aggregate_selection_inputs(frame))
                score_payload.update(metadata)
                score_payload["fold_id"] = fold_id
                fold_rule_rows.append(score_payload)
            if not fold_rule_rows:
                continue

            fold_rule_df = pd.DataFrame(fold_rule_rows).sort_values(
                ["score", "test_sharpe_median"],
                ascending=[False, False],
            )
            selected_feature_set = fold_rule_df.iloc[0]["feature_set"]
            selected_score = float(fold_rule_df.iloc[0]["score"])
            selected_test_sharpe = float(fold_rule_df.iloc[0]["test_sharpe_median"])
            fold_rule_df["selected_by_rule"] = fold_rule_df["feature_set"] == selected_feature_set
            comparison_rows.extend(fold_rule_df.to_dict(orient="records"))

            winner_rows.append(
                {
                    "fold_id": fold_id,
                    "selection_rule": rule.name,
                    "selected_feature_set": selected_feature_set,
                    "selected_score": selected_score,
                    "selected_test_sharpe_median": selected_test_sharpe,
                    "actual_test_winner_feature_set": actual_test_winner,
                    "actual_test_winner_sharpe_median": actual_test_winner_sharpe,
                    "selection_matches_test_winner": bool(selected_feature_set == actual_test_winner),
                    "test_winner_regret": float(actual_test_winner_sharpe - selected_test_sharpe)
                    if not np.isnan(actual_test_winner_sharpe)
                    else np.nan,
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    winners_df = pd.DataFrame(winner_rows)
    if winners_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = (
            winners_df.groupby("selection_rule", dropna=False)
            .agg(
                folds=("fold_id", "nunique"),
                selected_feature_sets=("selected_feature_set", lambda s: ", ".join(sorted(set(map(str, s))))),
                selected_test_sharpe_median=("selected_test_sharpe_median", "median"),
                actual_test_winner_sharpe_median=("actual_test_winner_sharpe_median", "median"),
                selection_matches_test_winner_rate=("selection_matches_test_winner", "mean"),
                median_test_winner_regret=("test_winner_regret", "median"),
            )
            .reset_index()
            .sort_values(
                ["selected_test_sharpe_median", "selection_matches_test_winner_rate"],
                ascending=[False, False],
            )
            .reset_index(drop=True)
        )

    return comparison_df, summary_df, winners_df


def _paired_permutation_test(
    results_df: pd.DataFrame,
    left_label: str,
    right_label: str,
    *,
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


def recompute_pairwise_permutation_tests(
    unique_results_df: pd.DataFrame,
    *,
    strategy_col: str = "feature_set",
    value_col: str = "test_sharpe",
    pair_cols: Sequence[str] = ("fold_id", "seed"),
    n_permutations: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    labels = list(dict.fromkeys(unique_results_df[strategy_col].dropna().tolist()))
    rows = []
    for idx, left in enumerate(labels):
        for right in labels[idx + 1 :]:
            rows.append(
                _paired_permutation_test(
                    unique_results_df,
                    left,
                    right,
                    strategy_col=strategy_col,
                    value_col=value_col,
                    pair_cols=pair_cols,
                    n_permutations=n_permutations,
                    random_state=random_state,
                )
    )
    return pd.DataFrame(rows)


def build_benchmark_comparison_reports(
    daily_df: pd.DataFrame,
    benchmark_suite_df: pd.DataFrame,
    *,
    primary_benchmark_id: str = PRIMARY_BENCHMARK_ID,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if daily_df.empty or benchmark_suite_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    required_daily_cols = {
        "date",
        "run_key",
        "feature_set",
        "fold_id",
        "seed",
        "daily_return",
        "portfolio_value",
    }
    required_benchmark_cols = {
        "date",
        "fold_id",
        "benchmark_id",
        "benchmark_name",
        "benchmark_family",
        "benchmark_return",
        "benchmark_portfolio_value",
    }
    missing_daily = sorted(required_daily_cols - set(daily_df.columns))
    missing_benchmark = sorted(required_benchmark_cols - set(benchmark_suite_df.columns))
    if missing_daily:
        raise KeyError(
            "Daily test frame is missing required columns for benchmark comparisons: "
            + ", ".join(missing_daily)
        )
    if missing_benchmark:
        raise KeyError(
            "Benchmark suite frame is missing required columns for benchmark comparisons: "
            + ", ".join(missing_benchmark)
        )

    daily = ensure_feature_metadata(daily_df.copy())
    daily["date"] = pd.to_datetime(daily["date"])
    daily["daily_return"] = pd.to_numeric(daily["daily_return"], errors="coerce").fillna(0.0)
    daily["portfolio_value"] = pd.to_numeric(daily["portfolio_value"], errors="coerce")
    if "turnover" in daily.columns:
        daily["turnover"] = pd.to_numeric(daily["turnover"], errors="coerce")

    benchmarks = benchmark_suite_df.copy()
    benchmarks["date"] = pd.to_datetime(benchmarks["date"])
    benchmarks = benchmarks.rename(
        columns={
            "benchmark_id": "suite_benchmark_id",
            "benchmark_name": "suite_benchmark_name",
            "benchmark_family": "suite_benchmark_family",
            "is_primary_benchmark": "suite_is_primary_benchmark",
            "benchmark_return": "suite_benchmark_return",
            "benchmark_portfolio_value": "suite_benchmark_portfolio_value",
            "benchmark_turnover": "suite_benchmark_turnover",
            "benchmark_transaction_cost": "suite_benchmark_transaction_cost",
        }
    )
    for col in (
        "suite_benchmark_return",
        "suite_benchmark_portfolio_value",
        "suite_benchmark_turnover",
        "suite_benchmark_transaction_cost",
    ):
        if col in benchmarks.columns:
            benchmarks[col] = pd.to_numeric(benchmarks[col], errors="coerce")

    merged = daily.merge(benchmarks, on=["date", "fold_id"], how="inner")
    if merged.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    run_rows: list[dict[str, Any]] = []
    for (run_key, benchmark_id), frame in merged.groupby(["run_key", "suite_benchmark_id"], dropna=False):
        ordered = frame.sort_values("date").reset_index(drop=True)
        agent_metrics = _compute_path_metrics(ordered["daily_return"], ordered["portfolio_value"])
        benchmark_metrics = _compute_path_metrics(
            ordered["suite_benchmark_return"],
            ordered["suite_benchmark_portfolio_value"],
        )
        excess_daily = ordered["daily_return"] - ordered["suite_benchmark_return"]
        excess_metrics = _compute_path_metrics(excess_daily)
        template = ordered.iloc[0].to_dict()

        run_rows.append(
            {
                "run_key": run_key,
                "feature_set": template.get("feature_set"),
                "feature_family": template.get("feature_family"),
                "is_negative_control": template.get("is_negative_control"),
                "fold_id": template.get("fold_id"),
                "seed": template.get("seed"),
                "selection_rule": template.get("selection_rule"),
                "selected_model_type": template.get("selected_model_type"),
                "benchmark_id": benchmark_id,
                "benchmark_name": template.get("suite_benchmark_name"),
                "benchmark_family": template.get("suite_benchmark_family"),
                "is_primary_benchmark": bool(
                    template.get("suite_is_primary_benchmark", benchmark_id == primary_benchmark_id)
                ),
                "n_days": int(len(ordered)),
                "agent_return_pct": agent_metrics["return_pct"],
                "agent_sharpe": agent_metrics["sharpe"],
                "agent_max_drawdown": agent_metrics["max_drawdown"],
                "agent_turnover": float(ordered["turnover"].mean())
                if "turnover" in ordered.columns and ordered["turnover"].notna().any()
                else np.nan,
                "benchmark_return_pct": benchmark_metrics["return_pct"],
                "benchmark_sharpe": benchmark_metrics["sharpe"],
                "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
                "benchmark_turnover": float(ordered["suite_benchmark_turnover"].mean())
                if "suite_benchmark_turnover" in ordered.columns and ordered["suite_benchmark_turnover"].notna().any()
                else np.nan,
                "benchmark_transaction_cost_total": float(ordered["suite_benchmark_transaction_cost"].sum())
                if "suite_benchmark_transaction_cost" in ordered.columns
                else np.nan,
                "daily_excess_return_mean": float(excess_daily.mean()),
                "daily_excess_return_sharpe": excess_metrics["sharpe"],
                "excess_return_pct": agent_metrics["return_pct"] - benchmark_metrics["return_pct"],
                "excess_sharpe": agent_metrics["sharpe"] - benchmark_metrics["sharpe"],
                "benchmark_relative_regret_return_pct": benchmark_metrics["return_pct"] - agent_metrics["return_pct"],
                "benchmark_relative_regret_sharpe": benchmark_metrics["sharpe"] - agent_metrics["sharpe"],
                "outperformed_benchmark_on_return": bool(agent_metrics["return_pct"] > benchmark_metrics["return_pct"])
                if not np.isnan(agent_metrics["return_pct"]) and not np.isnan(benchmark_metrics["return_pct"])
                else False,
                "outperformed_benchmark_on_sharpe": bool(agent_metrics["sharpe"] > benchmark_metrics["sharpe"])
                if not np.isnan(agent_metrics["sharpe"]) and not np.isnan(benchmark_metrics["sharpe"])
                else False,
                "hit_rate_vs_benchmark": float((ordered["daily_return"] > ordered["suite_benchmark_return"]).mean()),
            }
        )

    run_level_df = pd.DataFrame(run_rows)
    if run_level_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    summary_by_feature = (
        run_level_df.groupby(
            [
                "feature_set",
                "feature_family",
                "is_negative_control",
                "benchmark_id",
                "benchmark_name",
                "benchmark_family",
                "is_primary_benchmark",
            ],
            dropna=False,
        )
        .agg(
            runs=("run_key", "nunique"),
            folds=("fold_id", "nunique"),
            seeds=("seed", "nunique"),
            agent_return_pct_median=("agent_return_pct", "median"),
            benchmark_return_pct_median=("benchmark_return_pct", "median"),
            excess_return_pct_median=("excess_return_pct", "median"),
            agent_sharpe_median=("agent_sharpe", "median"),
            benchmark_sharpe_median=("benchmark_sharpe", "median"),
            excess_sharpe_median=("excess_sharpe", "median"),
            agent_max_drawdown_median=("agent_max_drawdown", "median"),
            benchmark_max_drawdown_median=("benchmark_max_drawdown", "median"),
            benchmark_relative_regret_return_pct_median=("benchmark_relative_regret_return_pct", "median"),
            benchmark_relative_regret_sharpe_median=("benchmark_relative_regret_sharpe", "median"),
            agent_turnover_median=("agent_turnover", "median"),
            benchmark_turnover_median=("benchmark_turnover", "median"),
            daily_excess_return_mean_median=("daily_excess_return_mean", "median"),
            hit_rate_vs_benchmark_median=("hit_rate_vs_benchmark", "median"),
            outperformed_benchmark_on_return_rate=("outperformed_benchmark_on_return", "mean"),
            outperformed_benchmark_on_sharpe_rate=("outperformed_benchmark_on_sharpe", "mean"),
        )
        .reset_index()
        .sort_values(
            ["is_primary_benchmark", "excess_sharpe_median", "excess_return_pct_median"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    summary_by_fold = (
        run_level_df.groupby(["fold_id", "feature_set", "benchmark_id"], dropna=False)
        .agg(
            runs=("run_key", "nunique"),
            excess_return_pct_median=("excess_return_pct", "median"),
            excess_sharpe_median=("excess_sharpe", "median"),
            benchmark_win_rate_return=("outperformed_benchmark_on_return", "mean"),
            benchmark_win_rate_sharpe=("outperformed_benchmark_on_sharpe", "mean"),
            benchmark_relative_regret_return_pct_median=("benchmark_relative_regret_return_pct", "median"),
        )
        .reset_index()
    )

    return run_level_df, summary_by_feature, summary_by_fold


def build_primary_benchmark_enriched_summary(
    corrected_summary_df: pd.DataFrame,
    benchmark_summary_by_feature_df: pd.DataFrame,
    *,
    primary_benchmark_id: str = PRIMARY_BENCHMARK_ID,
) -> pd.DataFrame:
    if corrected_summary_df.empty or benchmark_summary_by_feature_df.empty:
        return corrected_summary_df.copy()

    corrected = ensure_feature_metadata(corrected_summary_df.copy())
    primary = benchmark_summary_by_feature_df[
        benchmark_summary_by_feature_df["benchmark_id"] == primary_benchmark_id
    ].copy()
    if primary.empty:
        return corrected
    primary = ensure_feature_metadata(primary)

    primary = primary.rename(
        columns={
            "benchmark_name": "primary_benchmark_name",
            "benchmark_return_pct_median": "primary_benchmark_return_pct_median",
            "benchmark_sharpe_median": "primary_benchmark_sharpe_median",
            "benchmark_max_drawdown_median": "primary_benchmark_max_drawdown_median",
            "excess_return_pct_median": "primary_benchmark_excess_return_pct_median",
            "excess_sharpe_median": "primary_benchmark_excess_sharpe_median",
            "benchmark_relative_regret_return_pct_median": "primary_benchmark_regret_return_pct_median",
            "benchmark_relative_regret_sharpe_median": "primary_benchmark_regret_sharpe_median",
            "outperformed_benchmark_on_return_rate": "primary_benchmark_outperform_return_rate",
            "outperformed_benchmark_on_sharpe_rate": "primary_benchmark_outperform_sharpe_rate",
            "hit_rate_vs_benchmark_median": "primary_benchmark_hit_rate_median",
        }
    )
    keep_cols = [
        "feature_set",
        "feature_family",
        "is_negative_control",
        "primary_benchmark_name",
        "primary_benchmark_return_pct_median",
        "primary_benchmark_sharpe_median",
        "primary_benchmark_max_drawdown_median",
        "primary_benchmark_excess_return_pct_median",
        "primary_benchmark_excess_sharpe_median",
        "primary_benchmark_regret_return_pct_median",
        "primary_benchmark_regret_sharpe_median",
        "primary_benchmark_outperform_return_rate",
        "primary_benchmark_outperform_sharpe_rate",
        "primary_benchmark_hit_rate_median",
    ]
    merge_cols = ["feature_set", "feature_family", "is_negative_control"]
    if any(col not in corrected.columns for col in merge_cols):
        return corrected
    if any(col not in primary.columns for col in merge_cols):
        return corrected
    return corrected.merge(
        primary[keep_cols],
        on=merge_cols,
        how="left",
    )


def build_statistical_credibility_report(
    unique_results_df: pd.DataFrame,
    *,
    selection_summary_df: Optional[pd.DataFrame] = None,
    benchmark_summary_by_feature_df: Optional[pd.DataFrame] = None,
    primary_benchmark_id: str = PRIMARY_BENCHMARK_ID,
) -> dict[str, Any]:
    selection_summary = selection_summary_df if selection_summary_df is not None else pd.DataFrame()
    benchmark_summary = benchmark_summary_by_feature_df if benchmark_summary_by_feature_df is not None else pd.DataFrame()

    return {
        "status": "partial_implemented_guardrails",
        "implemented_metrics": {
            "retention_ratio": "implemented",
            "generalization_ratio": "implemented",
            "selection_rule_regret": "implemented",
            "winner_match_rate": "implemented",
            "benchmark_relative_metrics": "implemented" if not benchmark_summary.empty else "not_available",
        },
        "multiple_trials_context": {
            "run_key_count": int(unique_results_df["run_key"].nunique()) if not unique_results_df.empty else 0,
            "feature_set_count": int(unique_results_df["feature_set"].nunique()) if "feature_set" in unique_results_df.columns else 0,
            "fold_count": int(unique_results_df["fold_id"].nunique()) if "fold_id" in unique_results_df.columns else 0,
            "seed_count": int(unique_results_df["seed"].nunique()) if "seed" in unique_results_df.columns else 0,
            "selection_rule_count": int(selection_summary["selection_rule"].nunique()) if not selection_summary.empty else 0,
            "benchmark_count": int(benchmark_summary["benchmark_id"].nunique()) if not benchmark_summary.empty else 0,
            "primary_benchmark_id": primary_benchmark_id,
            "interpretation_note": (
                "Configuration selection and benchmark comparison still span multiple trials. "
                "Raw winner frequencies and median gaps are not a multiple-testing adjustment."
            ),
        },
        "selection_bias_guardrails": {
            "status": "implemented_partial",
            "notes": [
                "Configuration-level retention and generalization metrics are reported.",
                "Winner-match rate and selection regret are reported by selection rule.",
                "Benchmark-relative reporting is available when benchmark suite exports are present.",
            ],
        },
        "advanced_statistics_todo": {
            "deflated_sharpe": {
                "status": "todo_not_implemented",
                "next_step": "Add Deflated-Sharpe-style significance reporting for selected strategies.",
            },
            "probability_of_backtest_overfitting": {
                "status": "todo_not_implemented",
                "next_step": "Add PBO-style workflow once multiple comparable trial batches are accumulated.",
            },
        },
    }


def compute_turnover_series_from_actions(
    df_actions: Optional[pd.DataFrame],
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    if df_actions is None or df_actions.empty:
        return pd.DataFrame(columns=[date_col, "turnover"])

    actions = df_actions.copy()
    if date_col not in actions.columns:
        actions = actions.reset_index()
        if "index" in actions.columns and date_col not in actions.columns:
            actions = actions.rename(columns={"index": date_col})
    actions[date_col] = pd.to_datetime(actions[date_col])

    if {"tic", "weight"}.issubset(actions.columns):
        wide = actions.pivot_table(index=date_col, columns="tic", values="weight", aggfunc="last")
    elif {"tic", "action"}.issubset(actions.columns):
        wide = actions.pivot_table(index=date_col, columns="tic", values="action", aggfunc="last")
    else:
        numeric_cols = [
            col for col in actions.columns if col != date_col and pd.api.types.is_numeric_dtype(actions[col])
        ]
        wide = actions[[date_col] + numeric_cols].set_index(date_col)

    if wide.empty:
        return pd.DataFrame(columns=[date_col, "turnover"])

    wide = wide.sort_index().fillna(0.0)
    turnover = wide.diff().abs().sum(axis=1).fillna(0.0) / 2.0
    return turnover.rename("turnover").reset_index()


def _max_drawdown_from_values(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    return float(drawdown.min())


def build_regime_reports_from_daily(
    daily_df: pd.DataFrame,
    *,
    min_days_per_regime: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if daily_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    required_cols = {
        "date",
        "run_key",
        "feature_set",
        "fold_id",
        "seed",
        "daily_return",
        "portfolio_value",
        "regime_label_exogenous",
    }
    missing = sorted(required_cols - set(daily_df.columns))
    if missing:
        raise KeyError(
            "Daily test frame is missing required columns for regime diagnostics: "
            + ", ".join(missing)
        )

    data = ensure_feature_metadata(daily_df.copy())
    data["date"] = pd.to_datetime(data["date"])
    data["daily_return"] = pd.to_numeric(data["daily_return"], errors="coerce")
    if "benchmark_return" in data.columns:
        data["benchmark_return"] = pd.to_numeric(data["benchmark_return"], errors="coerce")
    if "turnover" in data.columns:
        data["turnover"] = pd.to_numeric(data["turnover"], errors="coerce")

    run_rows: list[dict[str, Any]] = []
    for keys, frame in data.groupby(["run_key", "regime_label_exogenous"], dropna=False):
        run_key, regime_label = keys
        ordered = frame.sort_values("date").reset_index(drop=True)
        n_days = int(len(ordered))
        mean_return = float(ordered["daily_return"].mean())
        daily_vol = float(ordered["daily_return"].std(ddof=0)) if n_days > 0 else np.nan
        sharpe = (
            float(np.sqrt(252.0) * mean_return / daily_vol)
            if n_days >= min_days_per_regime and daily_vol > 1e-12
            else np.nan
        )
        turnover_mean = (
            float(ordered["turnover"].mean())
            if "turnover" in ordered.columns and ordered["turnover"].notna().any()
            else np.nan
        )
        benchmark_excess = (
            float((ordered["daily_return"] - ordered["benchmark_return"]).mean())
            if "benchmark_return" in ordered.columns and ordered["benchmark_return"].notna().any()
            else np.nan
        )
        template = ordered.iloc[0].to_dict()
        run_rows.append(
            {
                "run_key": run_key,
                "regime_label_exogenous": regime_label,
                "fold_id": template.get("fold_id"),
                "seed": template.get("seed"),
                "feature_set": template.get("feature_set"),
                "feature_family": template.get("feature_family"),
                "is_negative_control": template.get("is_negative_control"),
                "selection_rule": template.get("selection_rule"),
                "selected_model_type": template.get("selected_model_type"),
                "benchmark_id": template.get("benchmark_id"),
                "n_days": n_days,
                "mean_daily_return": mean_return,
                "daily_volatility": daily_vol,
                "volatility": float(daily_vol * np.sqrt(252.0)) if not np.isnan(daily_vol) else np.nan,
                "sharpe": sharpe,
                "max_drawdown": _max_drawdown_from_values(ordered["portfolio_value"]),
                "turnover": turnover_mean,
                "hit_rate": float((ordered["daily_return"] > 0).mean()),
                "excess_return_vs_benchmark": benchmark_excess,
                "insufficient_days_for_sharpe": bool(n_days < min_days_per_regime),
            }
        )

    run_level_df = pd.DataFrame(run_rows)
    if run_level_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    summary_by_feature = (
        run_level_df.groupby(
            ["feature_set", "feature_family", "is_negative_control", "regime_label_exogenous"],
            dropna=False,
        )
        .agg(
            runs=("run_key", "nunique"),
            folds=("fold_id", "nunique"),
            seeds=("seed", "nunique"),
            n_days_median=("n_days", "median"),
            mean_daily_return_median=("mean_daily_return", "median"),
            sharpe_median=("sharpe", "median"),
            max_drawdown_median=("max_drawdown", "median"),
            turnover_median=("turnover", "median"),
            hit_rate_median=("hit_rate", "median"),
            excess_return_vs_benchmark_median=("excess_return_vs_benchmark", "median"),
        )
        .reset_index()
    )
    summary_by_fold = (
        run_level_df.groupby(["fold_id", "regime_label_exogenous", "feature_set"], dropna=False)
        .agg(
            runs=("run_key", "nunique"),
            sharpe_median=("sharpe", "median"),
            max_drawdown_median=("max_drawdown", "median"),
            turnover_median=("turnover", "median"),
            hit_rate_median=("hit_rate", "median"),
        )
        .reset_index()
    )

    return run_level_df, summary_by_feature, summary_by_fold


def merge_csv_files(
    inputs: Sequence[str | Path],
    *,
    key_col: str,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    return merge_csv_files_by_keys(
        inputs,
        key_cols=(key_col,),
        output_path=output_path,
        allow_identical_duplicates=False,
    )


def merge_csv_files_by_keys(
    inputs: Sequence[str | Path],
    *,
    key_cols: Sequence[str],
    output_path: Optional[str | Path] = None,
    allow_identical_duplicates: bool = False,
    allow_schema_union: bool = False,
) -> pd.DataFrame:
    if not inputs:
        raise ValueError("At least one input CSV is required.")

    frames = []
    schema: Optional[tuple[str, ...]] = None
    union_schema: list[str] = []
    resolved_key_cols = tuple(str(col) for col in key_cols)
    if not resolved_key_cols:
        raise ValueError("At least one key column is required for merging.")

    for input_path in inputs:
        frame = pd.read_csv(input_path)
        frame_cols = tuple(frame.columns.tolist())
        if schema is None:
            schema = frame_cols
            union_schema = list(frame_cols)
        elif frame_cols != schema and not allow_schema_union:
            raise ValueError(
                f"Incompatible schema for `{input_path}`. Expected columns: {schema}. Got: {frame_cols}."
            )
        if allow_schema_union:
            for col in frame.columns:
                if col not in union_schema:
                    union_schema.append(col)
        missing_cols = [col for col in resolved_key_cols if col not in frame.columns]
        if missing_cols:
            raise KeyError(
                f"`{input_path}` is missing required key columns: {', '.join(missing_cols)}."
            )
        frames.append(frame)

    if allow_schema_union and union_schema:
        frames = [frame.reindex(columns=union_schema) for frame in frames]

    merged = pd.concat(frames, ignore_index=True)
    duplicate_mask = merged.duplicated(subset=list(resolved_key_cols), keep=False)
    if duplicate_mask.any():
        duplicate_rows = merged.loc[duplicate_mask].copy()
        duplicate_keys_df = duplicate_rows[list(resolved_key_cols)].drop_duplicates()
        sample_keys = duplicate_keys_df.head(5).to_dict(orient="records")
        if not allow_identical_duplicates:
            raise ValueError(
                "Conflicting duplicate keys encountered while merging. "
                f"Sample duplicates: {sample_keys}"
            )

        invariant_cols = [col for col in merged.columns if col not in resolved_key_cols]
        inconsistent_keys: list[dict[str, Any]] = []
        for key_values, frame in duplicate_rows.groupby(list(resolved_key_cols), dropna=False):
            if len(frame) <= 1:
                continue
            for col in invariant_cols:
                normalized = frame[col].astype(str).replace({"nan": "<NA>", "NaT": "<NA>"})
                if normalized.nunique(dropna=False) > 1:
                    if not isinstance(key_values, tuple):
                        key_values = (key_values,)
                    inconsistent_keys.append(
                        {
                            col_name: key_value
                            for col_name, key_value in zip(resolved_key_cols, key_values)
                        }
                    )
                    break
        if inconsistent_keys:
            raise ValueError(
                "Overlapping keys were not identical while merging. "
                f"Sample inconsistent keys: {inconsistent_keys[:5]}"
            )
        merged = merged.drop_duplicates(subset=list(resolved_key_cols), keep="first").reset_index(drop=True)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output, index=False)
    return merged


def merge_research_output_dirs(
    inputs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    dataset_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("At least one research output directory is required.")

    resolved_inputs = [Path(path) for path in inputs]
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    results_inputs: list[Path] = []
    daily_inputs: list[Path] = []
    action_inputs: list[Path] = []
    observation_inputs: list[Path] = []
    folds_inputs: list[Path] = []
    benchmark_inputs: list[Path] = []
    missing_daily_dirs: list[str] = []
    missing_action_dirs: list[str] = []
    missing_observation_dirs: list[str] = []
    missing_benchmark_dirs: list[str] = []
    missing_folds_dirs: list[str] = []

    for source_dir in resolved_inputs:
        results_path = source_dir / "walk_forward_results.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing required walk_forward_results.csv in `{source_dir}`.")
        results_inputs.append(results_path)

        daily_path = source_dir / "walk_forward_daily_test_returns.csv"
        if daily_path.exists():
            daily_inputs.append(daily_path)
        else:
            missing_daily_dirs.append(str(source_dir))

        action_path = source_dir / "walk_forward_test_actions.csv"
        if action_path.exists():
            action_inputs.append(action_path)
        else:
            missing_action_dirs.append(str(source_dir))

        observation_path = source_dir / "walk_forward_test_observations.csv"
        if observation_path.exists():
            observation_inputs.append(observation_path)
        else:
            missing_observation_dirs.append(str(source_dir))

        folds_path = source_dir / "walk_forward_folds.csv"
        if folds_path.exists():
            folds_inputs.append(folds_path)
        else:
            missing_folds_dirs.append(str(source_dir))

        benchmark_path = source_dir / "benchmark_suite_daily.csv"
        if benchmark_path.exists():
            benchmark_inputs.append(benchmark_path)
        else:
            missing_benchmark_dirs.append(str(source_dir))

    merged_results = merge_csv_files_by_keys(
        results_inputs,
        key_cols=("run_key",),
        output_path=target_dir / "walk_forward_results_merged.csv",
    )

    merged_daily = pd.DataFrame()
    if daily_inputs:
        merged_daily = merge_csv_files_by_keys(
            daily_inputs,
            key_cols=("run_key", "date"),
            output_path=target_dir / "walk_forward_daily_test_returns_merged.csv",
            allow_schema_union=True,
        )

    merged_actions = pd.DataFrame()
    if action_inputs:
        merged_actions = merge_csv_files_by_keys(
            action_inputs,
            key_cols=("run_key", "action_row_id"),
            output_path=target_dir / "walk_forward_test_actions_merged.csv",
            allow_schema_union=True,
        )

    merged_observations = pd.DataFrame()
    if observation_inputs:
        merged_observations = merge_csv_files_by_keys(
            observation_inputs,
            key_cols=("run_key", "observation_row_id"),
            output_path=target_dir / "walk_forward_test_observations_merged.csv",
            allow_schema_union=True,
        )

    merged_folds = pd.DataFrame()
    if folds_inputs:
        merged_folds = merge_csv_files_by_keys(
            folds_inputs,
            key_cols=("fold_id",),
            output_path=target_dir / "walk_forward_folds_merged.csv",
            allow_identical_duplicates=True,
        )

    benchmark_source = "none"
    merged_benchmark = pd.DataFrame()
    if benchmark_inputs:
        merged_benchmark = merge_csv_files_by_keys(
            benchmark_inputs,
            key_cols=("fold_id", "date", "benchmark_id"),
            output_path=target_dir / "benchmark_suite_daily_merged.csv",
            allow_identical_duplicates=True,
        )
        benchmark_source = "merged_existing_outputs"
    elif dataset_path is not None and not merged_folds.empty:
        dataset_df = pd.read_csv(dataset_path)
        merged_benchmark = build_benchmark_suite_from_dataset(
            dataset_df,
            merged_folds,
            output_path=target_dir / "benchmark_suite_daily_merged.csv",
        )
        benchmark_source = "rebuilt_from_dataset"

    analysis_dir = target_dir / "analysis"
    rebuild_bundle = rebuild_walk_forward_report(
        merged_results,
        outdir=analysis_dir,
        daily_test_df=merged_daily if not merged_daily.empty else None,
        test_actions_df=merged_actions if not merged_actions.empty else None,
        test_observations_df=merged_observations if not merged_observations.empty else None,
        benchmark_suite_df=merged_benchmark if not merged_benchmark.empty else None,
    )

    manifest = {
        "input_dirs": [str(path.resolve()) for path in resolved_inputs],
        "merged_results_path": str((target_dir / "walk_forward_results_merged.csv").resolve()),
        "merged_daily_path": (
            str((target_dir / "walk_forward_daily_test_returns_merged.csv").resolve())
            if not merged_daily.empty
            else None
        ),
        "merged_folds_path": (
            str((target_dir / "walk_forward_folds_merged.csv").resolve())
            if not merged_folds.empty
            else None
        ),
        "merged_actions_path": (
            str((target_dir / "walk_forward_test_actions_merged.csv").resolve())
            if not merged_actions.empty
            else None
        ),
        "merged_observations_path": (
            str((target_dir / "walk_forward_test_observations_merged.csv").resolve())
            if not merged_observations.empty
            else None
        ),
        "merged_benchmark_path": (
            str((target_dir / "benchmark_suite_daily_merged.csv").resolve())
            if not merged_benchmark.empty
            else None
        ),
        "benchmark_source": benchmark_source,
        "missing_daily_dirs": missing_daily_dirs,
        "missing_action_dirs": missing_action_dirs,
        "missing_observation_dirs": missing_observation_dirs,
        "missing_benchmark_dirs": missing_benchmark_dirs,
        "missing_folds_dirs": missing_folds_dirs,
        "analysis_dir": str(analysis_dir.resolve()),
    }
    (target_dir / "merge_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "merged_results": merged_results,
        "merged_daily": merged_daily,
        "merged_actions": merged_actions,
        "merged_observations": merged_observations,
        "merged_folds": merged_folds,
        "merged_benchmark": merged_benchmark,
        "rebuild_bundle": rebuild_bundle,
        "manifest": manifest,
    }


def rebuild_walk_forward_report(
    raw_results_df: pd.DataFrame,
    *,
    outdir: str | Path,
    daily_test_df: Optional[pd.DataFrame] = None,
    test_actions_df: Optional[pd.DataFrame] = None,
    test_observations_df: Optional[pd.DataFrame] = None,
    benchmark_suite_df: Optional[pd.DataFrame] = None,
    legacy_regime_df: Optional[pd.DataFrame] = None,
    selection_rules: Optional[Mapping[str, SelectionRuleSpec]] = None,
    min_days_per_regime: int = 10,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    unique_results_df, diagnostics = deduplicate_run_level_results(raw_results_df)
    corrected_summary = build_corrected_walk_forward_summary(unique_results_df)
    selection_comparison, selection_summary, winner_df = build_selection_rule_comparison(
        unique_results_df,
        rules=selection_rules,
    )
    pairwise_df = recompute_pairwise_permutation_tests(unique_results_df)

    regime_run_df = pd.DataFrame()
    regime_feature_df = pd.DataFrame()
    regime_fold_df = pd.DataFrame()
    benchmark_run_df = pd.DataFrame()
    benchmark_feature_df = pd.DataFrame()
    benchmark_fold_df = pd.DataFrame()
    enriched_summary = corrected_summary.copy()
    warnings: list[str] = []
    if daily_test_df is not None and not daily_test_df.empty:
        regime_run_df, regime_feature_df, regime_fold_df = build_regime_reports_from_daily(
            daily_test_df,
            min_days_per_regime=min_days_per_regime,
        )
        daily_test_df.to_csv(outdir / "walk_forward_daily_test_returns.csv", index=False)
    else:
        warnings.append(
            "Exogenous regime diagnostics were not rebuilt because `walk_forward_daily_test_returns.csv` was not provided."
        )
        if legacy_regime_df is not None and not legacy_regime_df.empty:
            legacy_regime_df.to_csv(outdir / "legacy_walk_forward_regime_breakdown.csv", index=False)

    if test_actions_df is not None and not test_actions_df.empty:
        test_actions_df.to_csv(outdir / "walk_forward_test_actions.csv", index=False)

    if test_observations_df is not None and not test_observations_df.empty:
        test_observations_df.to_csv(outdir / "walk_forward_test_observations.csv", index=False)

    if benchmark_suite_df is not None and not benchmark_suite_df.empty and daily_test_df is not None and not daily_test_df.empty:
        benchmark_run_df, benchmark_feature_df, benchmark_fold_df = build_benchmark_comparison_reports(
            daily_test_df,
            benchmark_suite_df,
        )
        enriched_summary = build_primary_benchmark_enriched_summary(
            corrected_summary,
            benchmark_feature_df,
        )
        benchmark_suite_df.to_csv(outdir / "benchmark_suite_daily.csv", index=False)
    else:
        warnings.append(
            "Benchmark-suite comparisons were not rebuilt because `benchmark_suite_daily.csv` was not provided."
        )

    statistical_credibility = build_statistical_credibility_report(
        unique_results_df,
        selection_summary_df=selection_summary,
        benchmark_summary_by_feature_df=benchmark_feature_df,
    )

    unique_results_df.to_csv(outdir / "unique_run_level_results.csv", index=False)
    corrected_summary.to_csv(outdir / "corrected_walk_forward_summary.csv", index=False)
    enriched_summary.to_csv(outdir / "corrected_walk_forward_summary_with_primary_benchmark.csv", index=False)
    selection_comparison.to_csv(outdir / "selection_rule_comparison.csv", index=False)
    selection_summary.to_csv(outdir / "selection_rule_summary.csv", index=False)
    winner_df.to_csv(outdir / "validation_vs_test_winner_by_fold.csv", index=False)
    pairwise_df.to_csv(outdir / "pairwise_permutation_tests_recomputed.csv", index=False)
    if not regime_run_df.empty:
        regime_run_df.to_csv(outdir / "regime_run_level_metrics.csv", index=False)
        regime_feature_df.to_csv(outdir / "regime_summary_by_feature_set.csv", index=False)
        regime_fold_df.to_csv(outdir / "regime_summary_by_fold.csv", index=False)
    if not benchmark_run_df.empty:
        benchmark_run_df.to_csv(outdir / "benchmark_run_level_metrics.csv", index=False)
        benchmark_feature_df.to_csv(outdir / "benchmark_summary_by_feature_set.csv", index=False)
        benchmark_fold_df.to_csv(outdir / "benchmark_summary_by_fold.csv", index=False)
    (outdir / "statistical_credibility_report.json").write_text(
        json.dumps(_json_safe(statistical_credibility), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    artifact_index = {
        "raw_row_count": diagnostics["raw_row_count"],
        "unique_run_key_count": diagnostics["unique_run_key_count"],
        "regime_expanded_row_count": diagnostics["regime_expanded_row_count"],
        "expansion_detected": diagnostics["expansion_detected"],
        "test_action_rows": int(len(test_actions_df)) if test_actions_df is not None else 0,
        "test_observation_rows": int(len(test_observations_df)) if test_observations_df is not None else 0,
        "warnings": warnings,
        "notes": diagnostics["notes"],
        "outputs": {
            "unique_run_level_results": str(outdir / "unique_run_level_results.csv"),
            "corrected_walk_forward_summary": str(outdir / "corrected_walk_forward_summary.csv"),
            "corrected_walk_forward_summary_with_primary_benchmark": str(
                outdir / "corrected_walk_forward_summary_with_primary_benchmark.csv"
            ),
            "selection_rule_comparison": str(outdir / "selection_rule_comparison.csv"),
            "selection_rule_summary": str(outdir / "selection_rule_summary.csv"),
            "validation_vs_test_winner_by_fold": str(outdir / "validation_vs_test_winner_by_fold.csv"),
            "pairwise_permutation_tests_recomputed": str(
                outdir / "pairwise_permutation_tests_recomputed.csv"
            ),
            "statistical_credibility_report": str(outdir / "statistical_credibility_report.json"),
        },
    }
    if not regime_run_df.empty:
        artifact_index["outputs"].update(
            {
                "walk_forward_daily_test_returns": str(outdir / "walk_forward_daily_test_returns.csv"),
                "regime_run_level_metrics": str(outdir / "regime_run_level_metrics.csv"),
                "regime_summary_by_feature_set": str(outdir / "regime_summary_by_feature_set.csv"),
                "regime_summary_by_fold": str(outdir / "regime_summary_by_fold.csv"),
            }
        )
    if test_actions_df is not None and not test_actions_df.empty:
        artifact_index["outputs"]["walk_forward_test_actions"] = str(
            outdir / "walk_forward_test_actions.csv"
        )
    if test_observations_df is not None and not test_observations_df.empty:
        artifact_index["outputs"]["walk_forward_test_observations"] = str(
            outdir / "walk_forward_test_observations.csv"
        )
    if not benchmark_run_df.empty:
        artifact_index["outputs"].update(
            {
                "benchmark_suite_daily": str(outdir / "benchmark_suite_daily.csv"),
                "benchmark_run_level_metrics": str(outdir / "benchmark_run_level_metrics.csv"),
                "benchmark_summary_by_feature_set": str(outdir / "benchmark_summary_by_feature_set.csv"),
                "benchmark_summary_by_fold": str(outdir / "benchmark_summary_by_fold.csv"),
            }
        )

    (outdir / "artifact_index.json").write_text(
        json.dumps(_json_safe(artifact_index), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "unique_run_level_results": unique_results_df,
        "corrected_walk_forward_summary": corrected_summary,
        "corrected_walk_forward_summary_with_primary_benchmark": enriched_summary,
        "selection_rule_comparison": selection_comparison,
        "selection_rule_summary": selection_summary,
        "validation_vs_test_winner_by_fold": winner_df,
        "pairwise_permutation_tests_recomputed": pairwise_df,
        "regime_run_level_metrics": regime_run_df,
        "regime_summary_by_feature_set": regime_feature_df,
        "regime_summary_by_fold": regime_fold_df,
        "benchmark_run_level_metrics": benchmark_run_df,
        "benchmark_summary_by_feature_set": benchmark_feature_df,
        "benchmark_summary_by_fold": benchmark_fold_df,
        "test_actions_df": test_actions_df if test_actions_df is not None else pd.DataFrame(),
        "test_observations_df": test_observations_df if test_observations_df is not None else pd.DataFrame(),
        "statistical_credibility_report": statistical_credibility,
        "artifact_index": artifact_index,
    }


def build_benchmark_suite_from_dataset(
    dataset_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    *,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    data = dataset_df.copy()
    data["date"] = pd.to_datetime(data["date"])
    suite_rows: list[pd.DataFrame] = []
    for _, row in folds_df.iterrows():
        fold_id = str(row["fold_id"])
        test_start = pd.Timestamp(row["test_start"])
        test_end = pd.Timestamp(row["test_end"])
        benchmark_source = data[data["date"] <= test_end].copy()
        suite_rows.append(
            build_benchmark_suite_frame(
                benchmark_source,
                fold_id=fold_id,
                test_start=test_start,
                test_end=test_end,
            )
        )

    combined = pd.concat(suite_rows, ignore_index=True) if suite_rows else pd.DataFrame()
    if output_path is not None and not combined.empty:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output, index=False)
    return combined


class OrderedLike(dict):
    def values(self):
        return [self[key] for key in self.keys()]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dow30 walk-forward report utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser(
        "merge-walkforward-results",
        help="Merge multiple raw walk-forward result CSV files with schema and run_key validation.",
    )
    merge_parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV paths.")
    merge_parser.add_argument("--output", required=True, help="Merged CSV path.")

    rebuild_parser = subparsers.add_parser(
        "rebuild-walkforward-report",
        help="Rebuild corrected run-level, selection, pairwise, and regime reports.",
    )
    rebuild_parser.add_argument("--input", required=True, help="Raw or merged walk-forward results CSV.")
    rebuild_parser.add_argument("--outdir", required=True, help="Output directory for rebuilt artifacts.")
    rebuild_parser.add_argument(
        "--daily-input",
        default=None,
        help="Optional walk_forward_daily_test_returns.csv for exogenous regime diagnostics.",
    )
    rebuild_parser.add_argument(
        "--test-actions-input",
        default=None,
        help="Optional walk_forward_test_actions.csv for latent-action teacher diagnostics.",
    )
    rebuild_parser.add_argument(
        "--test-observations-input",
        default=None,
        help="Optional walk_forward_test_observations.csv with exact policy observation traces.",
    )
    rebuild_parser.add_argument(
        "--benchmark-suite-input",
        default=None,
        help="Optional benchmark_suite_daily.csv for multi-benchmark reporting.",
    )
    rebuild_parser.add_argument(
        "--legacy-regime-input",
        default=None,
        help="Optional legacy walk_forward_regime_breakdown.csv for reference only.",
    )
    rebuild_parser.add_argument(
        "--min-days-per-regime",
        type=int,
        default=10,
        help="Minimum number of days required before Sharpe is computed inside a regime.",
    )

    benchmark_parser = subparsers.add_parser(
        "build-benchmark-suite",
        help="Build benchmark_suite_daily.csv from a processed dataset and walk-forward folds CSV.",
    )
    benchmark_parser.add_argument("--dataset", required=True, help="Processed dataset CSV path.")
    benchmark_parser.add_argument("--folds-input", required=True, help="walk_forward_folds.csv path.")
    benchmark_parser.add_argument("--output", required=True, help="Benchmark suite CSV path.")

    merge_outputs_parser = subparsers.add_parser(
        "merge-research-outputs",
        help="Merge multiple research output directories and rebuild a unified analysis folder.",
    )
    merge_outputs_parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Research output directories to merge.",
    )
    merge_outputs_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where merged CSVs and rebuilt analysis should be written.",
    )
    merge_outputs_parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Optional processed dataset CSV path used to rebuild benchmark_suite_daily.csv "
            "when no merged benchmark suite is available from the input directories."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.command == "merge-walkforward-results":
        merge_csv_files(args.inputs, key_col="run_key", output_path=args.output)
        return 0

    if args.command == "rebuild-walkforward-report":
        raw_results_df = pd.read_csv(args.input)
        daily_test_df = pd.read_csv(args.daily_input) if args.daily_input else None
        test_actions_df = pd.read_csv(args.test_actions_input) if args.test_actions_input else None
        test_observations_df = (
            pd.read_csv(args.test_observations_input) if args.test_observations_input else None
        )
        benchmark_suite_df = pd.read_csv(args.benchmark_suite_input) if args.benchmark_suite_input else None
        legacy_regime_df = pd.read_csv(args.legacy_regime_input) if args.legacy_regime_input else None
        rebuild_walk_forward_report(
            raw_results_df,
            outdir=args.outdir,
            daily_test_df=daily_test_df,
            test_actions_df=test_actions_df,
            test_observations_df=test_observations_df,
            benchmark_suite_df=benchmark_suite_df,
            legacy_regime_df=legacy_regime_df,
            min_days_per_regime=args.min_days_per_regime,
        )
        return 0

    if args.command == "build-benchmark-suite":
        dataset_df = pd.read_csv(args.dataset)
        folds_df = pd.read_csv(args.folds_input)
        build_benchmark_suite_from_dataset(
            dataset_df,
            folds_df,
            output_path=args.output,
        )
        return 0

    if args.command == "merge-research-outputs":
        merge_research_output_dirs(
            args.inputs,
            output_dir=args.output_dir,
            dataset_path=args.dataset,
        )
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
