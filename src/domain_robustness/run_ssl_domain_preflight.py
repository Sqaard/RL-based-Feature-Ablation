from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, *, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _feature_columns(dataset: pd.DataFrame, prefixes: Sequence[str]) -> list[str]:
    cols: list[str] = []
    for prefix in prefixes:
        cols.extend([col for col in dataset.columns if col.startswith(prefix)])
    cols = sorted(dict.fromkeys(cols))
    if not cols:
        raise ValueError(f"No feature columns found for prefixes: {list(prefixes)}")
    return cols


def _make_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ]
    )


def _make_random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=120,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )


def _majority(y_train: pd.Series, n: int) -> np.ndarray:
    return np.repeat(str(y_train.value_counts().index[0]), n)


def _prepare_dataset(path: str | Path, feature_prefixes: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    date_col = "action_date" if "action_date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(["fold_id", date_col, "seed"] if "seed" in df.columns else ["fold_id", date_col]).reset_index(
        drop=True
    )
    df["calendar_year"] = df[date_col].dt.year.astype("Int64").astype(str)
    df["era_bucket"] = np.select(
        [
            df[date_col].dt.year.le(2017),
            df[date_col].dt.year.between(2018, 2019),
            df[date_col].dt.year.eq(2020),
            df[date_col].dt.year.ge(2021),
        ],
        ["2016_2017", "2018_2019", "2020_covid", "2021_2022"],
        default="unknown",
    )

    bench_by_date = (
        df[[date_col, "reward_benchmark_return"]]
        .drop_duplicates(date_col)
        .sort_values(date_col)
        .copy()
    )
    bench_by_date["benchmark_abs_return"] = bench_by_date["reward_benchmark_return"].abs()
    bench_by_date["benchmark_rolling_21d_vol"] = (
        bench_by_date["reward_benchmark_return"].rolling(21, min_periods=5).std().bfill().ffill()
    )
    if bench_by_date["benchmark_rolling_21d_vol"].nunique(dropna=True) >= 3:
        bench_by_date["benchmark_vol_tercile"] = pd.qcut(
            bench_by_date["benchmark_rolling_21d_vol"].rank(method="first"),
            q=3,
            labels=["low_vol", "mid_vol", "high_vol"],
        ).astype(str)
    else:
        bench_by_date["benchmark_vol_tercile"] = "unknown"
    if bench_by_date["benchmark_abs_return"].nunique(dropna=True) >= 3:
        bench_by_date["benchmark_abs_return_tercile"] = pd.qcut(
            bench_by_date["benchmark_abs_return"].rank(method="first"),
            q=3,
            labels=["small_abs_return", "mid_abs_return", "large_abs_return"],
        ).astype(str)
    else:
        bench_by_date["benchmark_abs_return_tercile"] = "unknown"

    df = df.merge(
        bench_by_date[
            [
                date_col,
                "benchmark_rolling_21d_vol",
                "benchmark_vol_tercile",
                "benchmark_abs_return_tercile",
            ]
        ],
        on=date_col,
        how="left",
    )
    df["benchmark_return_sign"] = np.where(df["reward_benchmark_return"].ge(0), "up", "down")
    df["reward_return_sign"] = np.where(df["reward_daily_return"].ge(0), "up", "down")
    df["excess_return_sign"] = np.where(df["reward_excess_return_vs_benchmark"].ge(0), "up", "down")

    feature_cols = _feature_columns(df, feature_prefixes)
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    return df, x, feature_cols


def _domain_distribution(dataset: pd.DataFrame, domain_cols: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for col in domain_cols:
        if col not in dataset.columns:
            continue
        counts = dataset[col].astype(str).value_counts(dropna=False).rename_axis("label").reset_index(name="rows")
        counts.insert(0, "domain", col)
        counts["share"] = counts["rows"] / counts["rows"].sum()
        frames.append(counts)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _domain_predictability(
    dataset: pd.DataFrame,
    x: pd.DataFrame,
    *,
    domain_cols: Sequence[str],
    group_col: str,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    groups = dataset[group_col].astype(str)
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=1729)
    for domain_col in domain_cols:
        if domain_col not in dataset.columns:
            continue
        y = dataset[domain_col].astype(str)
        if y.nunique(dropna=True) < 2:
            continue
        for split_id, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=groups), start=1):
            x_train = x.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue
            majority_pred = _majority(y_train, len(y_test))
            logistic = _make_logistic()
            logistic.fit(x_train, y_train)
            logistic_pred = logistic.predict(x_test)
            rf = _make_random_forest(20_000 + split_id)
            rf.fit(x_train, y_train)
            rf_pred = rf.predict(x_test)
            row = {
                "target": domain_col,
                "split_id": split_id,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_label_count": int(y_train.nunique()),
                "test_label_count": int(y_test.nunique()),
            }
            row.update(_classification_metrics(y_test, majority_pred, prefix="majority"))
            row.update(_classification_metrics(y_test, logistic_pred, prefix="obs_logistic"))
            row.update(_classification_metrics(y_test, rf_pred, prefix="obs_random_forest"))
            rows.append(row)
    by_split = pd.DataFrame(rows)
    metric_cols = [col for col in by_split.columns if col.endswith(("accuracy", "f1"))]
    summary = (
        by_split.groupby("target", dropna=False)[metric_cols]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return by_split, summary


def _leave_fold_reward_predictability(dataset: pd.DataFrame, x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = ["reward_return_sign", "benchmark_return_sign", "excess_return_sign"]
    rows: list[dict[str, Any]] = []
    for target_col in targets:
        y = dataset[target_col].astype(str)
        for fold_id in sorted(dataset["fold_id"].dropna().unique()):
            test_mask = dataset["fold_id"].eq(fold_id)
            train_mask = ~test_mask
            x_train = x.loc[train_mask]
            x_test = x.loc[test_mask]
            y_train = y.loc[train_mask]
            y_test = y.loc[test_mask]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue
            majority_pred = _majority(y_train, len(y_test))
            logistic = _make_logistic()
            logistic.fit(x_train, y_train)
            logistic_pred = logistic.predict(x_test)
            rf = _make_random_forest(30_000 + int(str(fold_id).split("_")[-1]))
            rf.fit(x_train, y_train)
            rf_pred = rf.predict(x_test)
            row = {
                "target": target_col,
                "fold_id": fold_id,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_positive_rate": float((y_train == "up").mean()),
                "test_positive_rate": float((y_test == "up").mean()),
            }
            row.update(_classification_metrics(y_test, majority_pred, prefix="majority"))
            row.update(_classification_metrics(y_test, logistic_pred, prefix="obs_logistic"))
            row.update(_classification_metrics(y_test, rf_pred, prefix="obs_random_forest"))
            rows.append(row)
    by_fold = pd.DataFrame(rows)
    metric_cols = [col for col in by_fold.columns if col.endswith(("accuracy", "f1"))]
    summary = (
        by_fold.groupby("target", dropna=False)[metric_cols]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return by_fold, summary


def _fold_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    def _sharpe(series: pd.Series) -> float:
        std = float(series.std(ddof=1))
        return float(np.sqrt(252.0) * series.mean() / std) if std > 0 else np.nan

    grouped = dataset.groupby("fold_id", dropna=False)
    summary = grouped.agg(
        rows=("fold_id", "size"),
        start=("action_date", "min"),
        end=("action_date", "max"),
        reward_mean=("reward_daily_return", "mean"),
        reward_vol=("reward_daily_return", "std"),
        benchmark_mean=("reward_benchmark_return", "mean"),
        benchmark_vol=("reward_benchmark_return", "std"),
        excess_mean=("reward_excess_return_vs_benchmark", "mean"),
        excess_vol=("reward_excess_return_vs_benchmark", "std"),
        reward_hit_rate=("reward_daily_return", lambda s: float((s >= 0).mean())),
        excess_hit_rate=("reward_excess_return_vs_benchmark", lambda s: float((s >= 0).mean())),
        benchmark_hit_rate=("reward_benchmark_return", lambda s: float((s >= 0).mean())),
        dominant_vol_bucket=("benchmark_vol_tercile", lambda s: s.astype(str).mode().iloc[0]),
    ).reset_index()
    summary["reward_sharpe"] = grouped["reward_daily_return"].apply(_sharpe).to_numpy()
    summary["benchmark_sharpe"] = grouped["reward_benchmark_return"].apply(_sharpe).to_numpy()
    summary["excess_sharpe"] = grouped["reward_excess_return_vs_benchmark"].apply(_sharpe).to_numpy()
    if "simple_action_code" in dataset.columns:
        summary["flat_action_rate"] = grouped["simple_action_code"].apply(lambda s: float((s == "flat__flat__flat").mean())).to_numpy()
    return summary


def _fold_observation_distances(dataset: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)
    fold_means = x_scaled.groupby(dataset["fold_id"]).mean()
    rows: list[dict[str, Any]] = []
    for left in fold_means.index:
        for right in fold_means.index:
            rows.append(
                {
                    "left_fold": left,
                    "right_fold": right,
                    "mean_obs_l2_distance": float(np.linalg.norm(fold_means.loc[left] - fold_means.loc[right])),
                }
            )
    return pd.DataFrame(rows)


def run_ssl_domain_preflight(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    feature_prefixes: Optional[Sequence[str]] = None,
    n_splits: int = 10,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(feature_prefixes or ("obs_",))
    dataset, x, feature_cols = _prepare_dataset(dataset_path, prefixes)
    domain_cols = [
        "fold_id",
        "calendar_year",
        "era_bucket",
        "benchmark_vol_tercile",
        "benchmark_abs_return_tercile",
        "benchmark_return_sign",
    ]

    distribution = _domain_distribution(dataset, domain_cols)
    domain_by_split, domain_summary = _domain_predictability(
        dataset,
        x,
        domain_cols=domain_cols,
        group_col="action_date",
        n_splits=n_splits,
    )
    reward_by_fold, reward_summary = _leave_fold_reward_predictability(dataset, x)
    fold_summary = _fold_summary(dataset)
    fold_distances = _fold_observation_distances(dataset, x)

    dataset_path_out = output / "ssl_domain_generalization_dataset.csv"
    distribution_path = output / "domain_label_distribution.csv"
    domain_by_split_path = output / "domain_predictability_by_split.csv"
    domain_summary_path = output / "domain_predictability_summary.csv"
    reward_by_fold_path = output / "leave_fold_reward_sign_predictability.csv"
    reward_summary_path = output / "leave_fold_reward_sign_predictability_summary.csv"
    fold_summary_path = output / "fold_domain_reward_summary.csv"
    fold_distances_path = output / "fold_observation_mean_distances.csv"
    report_path = output / "ssl_domain_preflight_report.json"

    dataset.to_csv(dataset_path_out, index=False)
    distribution.to_csv(distribution_path, index=False)
    domain_by_split.to_csv(domain_by_split_path, index=False)
    domain_summary.to_csv(domain_summary_path, index=False)
    reward_by_fold.to_csv(reward_by_fold_path, index=False)
    reward_summary.to_csv(reward_summary_path, index=False)
    fold_summary.to_csv(fold_summary_path, index=False)
    fold_distances.to_csv(fold_distances_path, index=False)

    report = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "rows": int(len(dataset)),
        "feature_prefixes": list(prefixes),
        "feature_columns": int(len(feature_cols)),
        "folds": int(dataset["fold_id"].nunique()),
        "date_min": str(dataset["action_date"].min().date()),
        "date_max": str(dataset["action_date"].max().date()),
        "domain_targets": domain_cols,
        "grouped_domain_predictability_splits": int(n_splits),
        "outputs": {
            "ssl_domain_generalization_dataset": str(dataset_path_out),
            "domain_label_distribution": str(distribution_path),
            "domain_predictability_by_split": str(domain_by_split_path),
            "domain_predictability_summary": str(domain_summary_path),
            "leave_fold_reward_sign_predictability": str(reward_by_fold_path),
            "leave_fold_reward_sign_predictability_summary": str(reward_summary_path),
            "fold_domain_reward_summary": str(fold_summary_path),
            "fold_observation_mean_distances": str(fold_distances_path),
        },
    }
    report_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "dataset": dataset,
        "domain_predictability_summary": domain_summary,
        "reward_predictability_summary": reward_summary,
        "fold_summary": fold_summary,
        "fold_distances": fold_distances,
        "report": report,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run SSL/domain-generalization preflight diagnostics.")
    parser.add_argument("--dataset", required=True, help="Exact observation/action/reward dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for domain-generalization preflight outputs.")
    parser.add_argument("--feature-prefix", action="append", dest="feature_prefixes", help="Defaults to obs_.")
    parser.add_argument("--n-splits", type=int, default=10)
    args = parser.parse_args(argv)
    run_ssl_domain_preflight(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        feature_prefixes=args.feature_prefixes,
        n_splits=args.n_splits,
    )


if __name__ == "__main__":
    main()
