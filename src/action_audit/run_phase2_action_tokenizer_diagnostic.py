from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_state_action_code_predictability import (
    _classification_metrics,
    _feature_columns,
    _majority_predict,
    _markov_label_predict,
    _previous_label_predict,
    _sort_columns,
    _train_markov_map,
)
from latent_action_phase2_tools import METADATA_COLS


RAW_POLICY_ACTION_RE = re.compile(r"^raw_policy_action_\d+$")


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


def _read_dataset(path: str | Path, feature_prefixes: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(path)
    if "action_date" in dataset.columns:
        dataset["action_date"] = pd.to_datetime(dataset["action_date"], errors="coerce")
    sort_cols = _sort_columns(dataset)
    if sort_cols:
        dataset = dataset.sort_values(sort_cols).reset_index(drop=True)
    feature_cols = _feature_columns(dataset, feature_prefixes)
    x_obs = dataset[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_obs = x_obs.fillna(x_obs.median(numeric_only=True)).fillna(0.0)
    return dataset, x_obs


def _executed_action_columns(dataset: pd.DataFrame, matrix_path: Optional[str | Path]) -> list[str]:
    if matrix_path:
        matrix = pd.read_csv(matrix_path, nrows=1)
        cols = [col for col in matrix.columns if col not in METADATA_COLS and pd.api.types.is_numeric_dtype(matrix[col])]
        cols = [col for col in cols if col in dataset.columns]
        if cols:
            return cols

    blocked_prefixes = (
        "obs_",
        "raw_policy_action_",
        "state_",
        "reward_",
        "primary_benchmark_",
    )
    blocked_cols = set(METADATA_COLS) | {
        "action_date",
        "reward_date",
        "action_l1",
        "action_l2",
        "action_max_abs",
        "active_action_dims",
        "positive_action_dims",
        "negative_action_dims",
        "raw_policy_action_l1",
        "raw_policy_active_action_dims",
        "is_first_action_in_run",
        "action_code_changed",
        "executed_raw_direction_match",
    }
    label_like = (
        "_code",
        "_flag",
        "_status",
        "_label",
        "_description",
        "_family",
        "_type",
        "_rule",
    )
    cols: list[str] = []
    for col in dataset.columns:
        if col in blocked_cols or col.startswith(blocked_prefixes) or col.endswith(label_like):
            continue
        if pd.api.types.is_numeric_dtype(dataset[col]):
            cols.append(col)
    return cols


def _action_matrix(dataset: pd.DataFrame, source: str, executed_cols: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    if source == "executed":
        cols = list(executed_cols)
    elif source == "raw_policy":
        cols = [col for col in dataset.columns if RAW_POLICY_ACTION_RE.match(col)]
    else:
        raise ValueError(f"Unknown action source: {source}")
    if not cols:
        raise ValueError(f"No action columns found for source: {source}")
    matrix = dataset[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return matrix, cols


def _make_kmeans(n_clusters: int, random_state: int) -> KMeans:
    return KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )


def _entropy(labels: pd.Series | np.ndarray) -> tuple[float, float, float]:
    counts = pd.Series(labels).astype(str).value_counts()
    shares = counts / counts.sum()
    entropy = float(-(shares * np.log2(shares)).sum()) if not shares.empty else 0.0
    effective = float(2**entropy)
    dominant = float(shares.max()) if not shares.empty else 0.0
    return entropy, effective, dominant


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


def _fit_predict_model(model: Any, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, fallback: str) -> np.ndarray:
    if y_train.nunique() < 2:
        return np.repeat(fallback, len(x_test))
    try:
        model.fit(x_train, y_train)
        return model.predict(x_test)
    except Exception:
        return np.repeat(fallback, len(x_test))


def _reconstruction_metrics(values: np.ndarray, centers: np.ndarray, labels: np.ndarray, train_values: np.ndarray) -> dict[str, float]:
    recon = centers[labels]
    mean_recon = np.repeat(train_values.mean(axis=0, keepdims=True), len(values), axis=0)
    mse = float(mean_squared_error(values, recon))
    mean_mse = float(mean_squared_error(values, mean_recon))
    return {
        "reconstruction_mse": mse,
        "reconstruction_mae": float(mean_absolute_error(values, recon)),
        "mean_action_reconstruction_mse": mean_mse,
        "reconstruction_mse_ratio_vs_mean": float(mse / mean_mse) if mean_mse > 0 else np.nan,
    }


def _fold_tokenizer_diagnostics(
    dataset: pd.DataFrame,
    x_obs: pd.DataFrame,
    action_values: pd.DataFrame,
    *,
    source: str,
    n_clusters: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold_number, fold_id in enumerate(sorted(dataset["fold_id"].dropna().unique()), start=1):
        test_mask = dataset["fold_id"].eq(fold_id)
        train_mask = ~test_mask
        train_values = action_values.loc[train_mask].to_numpy(dtype=float)
        test_values = action_values.loc[test_mask].to_numpy(dtype=float)
        model = _make_kmeans(n_clusters, random_state=10_000 + n_clusters * 100 + fold_number)
        train_labels = pd.Series(model.fit_predict(train_values), index=action_values.index[train_mask]).astype(str)
        test_labels = pd.Series(model.predict(test_values), index=action_values.index[test_mask]).astype(str)
        fallback = str(train_labels.value_counts().index[0])

        majority_pred = _majority_predict(train_labels, len(test_labels))
        previous_pred = _previous_label_predict(test_labels, dataset.loc[test_mask], fallback=fallback)
        markov_pred = _markov_label_predict(
            test_labels,
            dataset.loc[test_mask],
            _train_markov_map(train_labels, dataset.loc[train_mask]),
            fallback=fallback,
        )
        logistic_pred = _fit_predict_model(
            _make_logistic(),
            x_obs.loc[train_mask],
            train_labels,
            x_obs.loc[test_mask],
            fallback,
        )
        forest_pred = _fit_predict_model(
            _make_random_forest(20_000 + n_clusters * 100 + fold_number),
            x_obs.loc[train_mask],
            train_labels,
            x_obs.loc[test_mask],
            fallback,
        )

        train_entropy, train_effective, train_dominant = _entropy(train_labels)
        test_entropy, test_effective, test_dominant = _entropy(test_labels)
        row = {
            "source": source,
            "n_clusters": int(n_clusters),
            "fold_id": fold_id,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_token_count": int(train_labels.nunique()),
            "test_token_count": int(test_labels.nunique()),
            "train_token_entropy_bits": train_entropy,
            "train_effective_tokens": train_effective,
            "train_dominant_token_share": train_dominant,
            "test_token_entropy_bits": test_entropy,
            "test_effective_tokens": test_effective,
            "test_dominant_token_share": test_dominant,
        }
        row.update(_reconstruction_metrics(test_values, model.cluster_centers_, test_labels.astype(int).to_numpy(), train_values))
        for name, pred in [
            ("majority", majority_pred),
            ("previous_token", previous_pred),
            ("markov_token", markov_pred),
            ("obs_logistic", logistic_pred),
            ("obs_random_forest", forest_pred),
        ]:
            row.update(_classification_metrics(test_labels, pred, prefix=name))
        rows.append(row)
    return rows


def _global_tokenizer_counts(
    dataset: pd.DataFrame,
    action_values: pd.DataFrame,
    *,
    source: str,
    n_clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = action_values.to_numpy(dtype=float)
    model = _make_kmeans(n_clusters, random_state=50_000 + n_clusters)
    labels = model.fit_predict(values)
    counts = pd.Series(labels).astype(str).value_counts().rename_axis("token").reset_index(name="count")
    counts.insert(0, "n_clusters", int(n_clusters))
    counts.insert(0, "source", source)
    counts["share"] = counts["count"] / counts["count"].sum()
    assignments = dataset[[col for col in ["run_key", "fold_id", "seed", "action_date", "action_step"] if col in dataset.columns]].copy()
    assignments["source"] = source
    assignments["n_clusters"] = int(n_clusters)
    assignments["token"] = labels.astype(str)
    assignments["reconstruction_l2"] = np.sqrt(((values - model.cluster_centers_[labels]) ** 2).sum(axis=1))
    return counts, assignments


def run_action_tokenizer_diagnostic(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    executed_action_matrix_path: Optional[str | Path] = None,
    feature_prefixes: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
    cluster_counts: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(feature_prefixes or ("obs_",))
    selected_sources = tuple(sources or ("executed", "raw_policy"))
    selected_clusters = tuple(cluster_counts or (4, 8, 16))

    dataset, x_obs = _read_dataset(dataset_path, prefixes)
    executed_cols = _executed_action_columns(dataset, executed_action_matrix_path)

    rows: list[dict[str, Any]] = []
    count_frames: list[pd.DataFrame] = []
    assignment_frames: list[pd.DataFrame] = []
    source_dims: dict[str, int] = {}
    for source in selected_sources:
        action_values, cols = _action_matrix(dataset, source, executed_cols)
        source_dims[source] = len(cols)
        for n_clusters in selected_clusters:
            rows.extend(
                _fold_tokenizer_diagnostics(
                    dataset,
                    x_obs,
                    action_values,
                    source=source,
                    n_clusters=int(n_clusters),
                )
            )
            counts, assignments = _global_tokenizer_counts(
                dataset,
                action_values,
                source=source,
                n_clusters=int(n_clusters),
            )
            count_frames.append(counts)
            assignment_frames.append(assignments)

    by_fold = pd.DataFrame(rows)
    metric_cols = [col for col in by_fold.columns if col.endswith(("accuracy", "f1"))]
    value_cols = [
        "train_token_count",
        "test_token_count",
        "train_token_entropy_bits",
        "train_effective_tokens",
        "train_dominant_token_share",
        "test_token_entropy_bits",
        "test_effective_tokens",
        "test_dominant_token_share",
        "reconstruction_mse",
        "reconstruction_mae",
        "mean_action_reconstruction_mse",
        "reconstruction_mse_ratio_vs_mean",
    ]
    summary = (
        by_fold.groupby(["source", "n_clusters"], dropna=False)[value_cols + metric_cols]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]

    counts_df = pd.concat(count_frames, ignore_index=True) if count_frames else pd.DataFrame()
    assignments_df = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()

    by_fold_path = output / "action_tokenizer_diagnostic_by_fold.csv"
    summary_path = output / "action_tokenizer_diagnostic_summary.csv"
    counts_path = output / "action_tokenizer_global_code_counts.csv"
    assignments_path = output / "action_tokenizer_global_code_assignments.csv"
    report_path = output / "action_tokenizer_diagnostic_report.json"
    by_fold.to_csv(by_fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    counts_df.to_csv(counts_path, index=False)
    assignments_df.to_csv(assignments_path, index=False)

    report = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "rows": int(len(dataset)),
        "feature_prefixes": list(prefixes),
        "feature_columns": int(len(_feature_columns(dataset, prefixes))),
        "sources": list(selected_sources),
        "action_dimensions": source_dims,
        "cluster_counts": [int(k) for k in selected_clusters],
        "folds": int(dataset["fold_id"].nunique()),
        "outputs": {
            "action_tokenizer_diagnostic_by_fold": str(by_fold_path),
            "action_tokenizer_diagnostic_summary": str(summary_path),
            "action_tokenizer_global_code_counts": str(counts_path),
            "action_tokenizer_global_code_assignments": str(assignments_path),
        },
    }
    report_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "by_fold": by_fold,
        "summary": summary,
        "global_counts": counts_df,
        "global_assignments": assignments_df,
        "report": report,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run Phase-2 direct action-vector tokenizer diagnostics.")
    parser.add_argument("--dataset", required=True, help="Path to phase2_action_label_variants_dataset.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for tokenizer diagnostic outputs.")
    parser.add_argument("--executed-action-matrix", help="Optional latent_action_teacher_matrix.csv path for action cols.")
    parser.add_argument("--feature-prefix", action="append", dest="feature_prefixes", help="Defaults to obs_.")
    parser.add_argument("--source", action="append", dest="sources", choices=["executed", "raw_policy"])
    parser.add_argument("--n-clusters", action="append", type=int, dest="cluster_counts")
    args = parser.parse_args(argv)
    run_action_tokenizer_diagnostic(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        executed_action_matrix_path=args.executed_action_matrix,
        feature_prefixes=args.feature_prefixes,
        sources=args.sources,
        cluster_counts=args.cluster_counts,
    )


if __name__ == "__main__":
    main()
