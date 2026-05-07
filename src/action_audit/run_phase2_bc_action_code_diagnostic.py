from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
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


def _make_bc_model(*, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=64,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    early_stopping=True,
                    n_iter_no_change=20,
                    validation_fraction=0.15,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _balanced_resample(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    target_per_class: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(random_state)
    pieces: list[np.ndarray] = []
    for _, idx in y.groupby(y).groups.items():
        idx_array = np.array(list(idx))
        replace = len(idx_array) < target_per_class
        take = rng.choice(idx_array, size=target_per_class, replace=replace)
        pieces.append(take)
    sampled = np.concatenate(pieces)
    rng.shuffle(sampled)
    return x.loc[sampled].reset_index(drop=True), y.loc[sampled].reset_index(drop=True)


def _binary_from_multiclass(labels: Sequence[str]) -> np.ndarray:
    labels_arr = np.asarray(labels).astype(str)
    return np.where(labels_arr == "flat__flat__flat", "flat", "nonflat")


def run_bc_action_code_diagnostic(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    feature_prefixes: Optional[Sequence[str]] = None,
    balanced_target_per_class: int = 250,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(dataset_path)
    if "action_date" in dataset.columns:
        dataset["action_date"] = pd.to_datetime(dataset["action_date"], errors="coerce")
    sort_cols = _sort_columns(dataset)
    if sort_cols:
        dataset = dataset.sort_values(sort_cols).reset_index(drop=True)

    prefixes = tuple(feature_prefixes or ("obs_",))
    feature_cols = _feature_columns(dataset, prefixes)
    x_all = dataset[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_all = x_all.fillna(x_all.median(numeric_only=True)).fillna(0.0)
    y_all = dataset["simple_action_code"].astype(str)

    rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold_number, fold_id in enumerate(sorted(dataset["fold_id"].dropna().unique()), start=1):
        test_mask = dataset["fold_id"].eq(fold_id)
        train_mask = ~test_mask
        x_train = x_all.loc[train_mask]
        x_test = x_all.loc[test_mask]
        y_train = y_all.loc[train_mask]
        y_test = y_all.loc[test_mask]
        train_meta = dataset.loc[train_mask]
        test_meta = dataset.loc[test_mask]

        majority_pred = _majority_predict(y_train, len(y_test))
        majority_label = str(y_train.value_counts().index[0])
        previous_pred = _previous_label_predict(y_test, test_meta, fallback=majority_label)
        markov_map = _train_markov_map(y_train, train_meta)
        markov_pred = _markov_label_predict(y_test, test_meta, markov_map, fallback=majority_label)

        natural_model = _make_bc_model(random_state=10_000 + fold_number)
        balanced_model = _make_bc_model(random_state=20_000 + fold_number)
        x_balanced, y_balanced = _balanced_resample(
            x_train,
            y_train,
            target_per_class=balanced_target_per_class,
            random_state=30_000 + fold_number,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            natural_model.fit(x_train, y_train)
            balanced_model.fit(x_balanced, y_balanced)

        natural_pred = natural_model.predict(x_test)
        balanced_pred = balanced_model.predict(x_test)
        y_binary = pd.Series(_binary_from_multiclass(y_test), index=y_test.index)

        row = {
            "fold_id": fold_id,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_label_count": int(y_train.nunique()),
            "test_label_count": int(y_test.nunique()),
            "balanced_train_rows": int(len(y_balanced)),
            "balanced_target_per_class": int(balanced_target_per_class),
        }
        for name, pred in [
            ("majority", majority_pred),
            ("previous", previous_pred),
            ("markov", markov_pred),
            ("bc_mlp_natural", natural_pred),
            ("bc_mlp_balanced", balanced_pred),
        ]:
            row.update(_classification_metrics(y_test, pred, prefix=name))
            row.update(
                _classification_metrics(
                    y_binary,
                    _binary_from_multiclass(pred),
                    prefix=f"binary_{name}",
                )
            )
        rows.append(row)

        pred_meta_cols = [
            col
            for col in ["run_key", "feature_set", "fold_id", "seed", "action_date", "action_step", "observation_row_id"]
            if col in test_meta.columns
        ]
        pred_frame = test_meta[pred_meta_cols].copy()
        pred_frame["row_index"] = y_test.index
        pred_frame["true_simple_action_code"] = y_test.to_numpy()
        pred_frame["majority_pred_simple_action_code"] = majority_pred
        pred_frame["previous_pred_simple_action_code"] = previous_pred
        pred_frame["markov_pred_simple_action_code"] = markov_pred
        pred_frame["bc_mlp_natural_pred_simple_action_code"] = natural_pred
        pred_frame["bc_mlp_balanced_pred_simple_action_code"] = balanced_pred
        prediction_rows.append(pred_frame)

    by_fold = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    metric_cols = [col for col in by_fold.columns if col.endswith(("accuracy", "f1"))]
    summary = by_fold[metric_cols].agg(["mean", "median", "min", "max"]).reset_index().rename(
        columns={"index": "statistic"}
    )

    by_fold_path = output / "bc_action_code_diagnostic_by_fold.csv"
    summary_path = output / "bc_action_code_diagnostic_summary.csv"
    predictions_path = output / "bc_action_code_diagnostic_predictions.csv"
    report_path = output / "bc_action_code_diagnostic_report.json"
    by_fold.to_csv(by_fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    report = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "rows": int(len(dataset)),
        "feature_prefixes": list(prefixes),
        "feature_columns": int(len(feature_cols)),
        "folds": int(by_fold["fold_id"].nunique()),
        "label_count": int(y_all.nunique()),
        "balanced_target_per_class": int(balanced_target_per_class),
        "model": "StandardScaler + MLPClassifier(128,64), cross-entropy-style BC diagnostic",
        "outputs": {
            "bc_action_code_diagnostic_by_fold": str(by_fold_path),
            "bc_action_code_diagnostic_summary": str(summary_path),
            "bc_action_code_diagnostic_predictions": str(predictions_path),
        },
    }
    report_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "by_fold": by_fold,
        "summary": summary,
        "predictions": predictions,
        "report": report,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run Phase-2 offline BC diagnostic for simple action codes.")
    parser.add_argument("--dataset", required=True, help="Path to exact observation/action/reward dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for BC diagnostic outputs.")
    parser.add_argument(
        "--feature-prefix",
        action="append",
        dest="feature_prefixes",
        help="Feature-column prefix to use. Defaults to obs_.",
    )
    parser.add_argument(
        "--balanced-target-per-class",
        type=int,
        default=250,
        help="Rows per class for the balanced oversampled BC variant.",
    )
    args = parser.parse_args(argv)
    run_bc_action_code_diagnostic(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        feature_prefixes=args.feature_prefixes,
        balanced_target_per_class=args.balanced_target_per_class,
    )


if __name__ == "__main__":
    main()
