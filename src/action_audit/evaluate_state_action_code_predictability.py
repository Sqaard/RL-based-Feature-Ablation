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


def _majority_predict(y_train: pd.Series, n: int) -> np.ndarray:
    majority = y_train.value_counts().index[0]
    return np.repeat(majority, n)


def _make_logistic_model(*, class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _make_random_forest_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=1729,
        n_jobs=1,
    )


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


def _sort_columns(dataset: pd.DataFrame) -> list[str]:
    return [col for col in ["run_key", "action_date", "date", "action_step", "observation_row_id"] if col in dataset.columns]


def _previous_label_predict(
    y_test: pd.Series,
    metadata: pd.DataFrame,
    *,
    fallback: str,
) -> np.ndarray:
    meta = metadata.copy()
    meta["_label"] = y_test.to_numpy()
    meta["_original_order"] = np.arange(len(meta))
    sort_cols = [col for col in ["run_key", "action_date", "date", "action_step", "observation_row_id"] if col in meta.columns]
    if sort_cols:
        meta = meta.sort_values(sort_cols)
    preds = []
    if "run_key" in meta.columns:
        for _, group in meta.groupby("run_key", sort=False):
            previous = group["_label"].shift(1).fillna(fallback)
            tmp = pd.DataFrame({"_original_order": group["_original_order"], "_pred": previous.to_numpy()})
            preds.append(tmp)
        pred_frame = pd.concat(preds, ignore_index=True)
    else:
        previous = meta["_label"].shift(1).fillna(fallback)
        pred_frame = pd.DataFrame({"_original_order": meta["_original_order"], "_pred": previous.to_numpy()})
    return pred_frame.sort_values("_original_order")["_pred"].to_numpy()


def _train_markov_map(y_train: pd.Series, metadata: pd.DataFrame) -> dict[str, str]:
    meta = metadata.copy()
    meta["_label"] = y_train.to_numpy()
    sort_cols = [col for col in ["run_key", "action_date", "date", "action_step", "observation_row_id"] if col in meta.columns]
    if sort_cols:
        meta = meta.sort_values(sort_cols)

    pairs = []
    if "run_key" in meta.columns:
        grouped = meta.groupby("run_key", sort=False)
    else:
        grouped = [("_all", meta)]
    for _, group in grouped:
        tmp = pd.DataFrame(
            {
                "previous_label": group["_label"].shift(1),
                "next_label": group["_label"],
            }
        ).dropna()
        pairs.append(tmp)
    if not pairs:
        return {}
    transitions = pd.concat(pairs, ignore_index=True)
    if transitions.empty:
        return {}
    counts = transitions.groupby(["previous_label", "next_label"]).size().rename("count").reset_index()
    counts = counts.sort_values(["previous_label", "count", "next_label"], ascending=[True, False, True])
    return counts.drop_duplicates("previous_label").set_index("previous_label")["next_label"].to_dict()


def _markov_label_predict(
    y_test: pd.Series,
    metadata: pd.DataFrame,
    transition_map: dict[str, str],
    *,
    fallback: str,
) -> np.ndarray:
    previous = _previous_label_predict(y_test, metadata, fallback=fallback)
    return np.array([transition_map.get(str(label), fallback) for label in previous])


def evaluate_predictability(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    feature_prefixes: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(dataset_path)
    if "action_date" in dataset.columns:
        dataset["action_date"] = pd.to_datetime(dataset["action_date"], errors="coerce")
    sort_cols = _sort_columns(dataset)
    if sort_cols:
        dataset = dataset.sort_values(sort_cols).reset_index(drop=True)

    prefixes = tuple(feature_prefixes or ("state_",))
    state_cols = _feature_columns(dataset, prefixes)
    required = {"fold_id", "simple_action_code"}
    missing = required - set(dataset.columns)
    if missing:
        raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")

    x_all = dataset[state_cols].apply(pd.to_numeric, errors="coerce")
    x_all = x_all.fillna(x_all.median(numeric_only=True)).fillna(0.0)
    y_all = dataset["simple_action_code"].astype(str)
    y_binary = np.where(y_all.eq("flat__flat__flat"), "flat", "nonflat")

    rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold_id in sorted(dataset["fold_id"].dropna().unique()):
        test_mask = dataset["fold_id"].eq(fold_id)
        train_mask = ~test_mask
        x_train = x_all.loc[train_mask]
        x_test = x_all.loc[test_mask]
        y_train = y_all.loc[train_mask]
        y_test = y_all.loc[test_mask]
        yb_train = pd.Series(y_binary[train_mask.to_numpy()], index=y_train.index)
        yb_test = pd.Series(y_binary[test_mask.to_numpy()], index=y_test.index)
        train_meta = dataset.loc[train_mask]
        test_meta = dataset.loc[test_mask]

        majority_pred = _majority_predict(y_train, len(y_test))
        majority_label = str(y_train.value_counts().index[0])
        previous_pred = _previous_label_predict(y_test, test_meta, fallback=majority_label)
        markov_map = _train_markov_map(y_train, train_meta)
        markov_pred = _markov_label_predict(y_test, test_meta, markov_map, fallback=majority_label)

        logistic_model = _make_logistic_model()
        logistic_model.fit(x_train, y_train)
        logistic_pred = logistic_model.predict(x_test)

        logistic_unweighted_model = _make_logistic_model(class_weight=None)
        logistic_unweighted_model.fit(x_train, y_train)
        logistic_unweighted_pred = logistic_unweighted_model.predict(x_test)

        forest_model = _make_random_forest_model()
        forest_model.fit(x_train, y_train)
        forest_pred = forest_model.predict(x_test)

        binary_majority_pred = _majority_predict(yb_train, len(yb_test))
        binary_majority_label = str(yb_train.value_counts().index[0])
        binary_previous_pred = _previous_label_predict(yb_test, test_meta, fallback=binary_majority_label)
        binary_markov_map = _train_markov_map(yb_train, train_meta)
        binary_markov_pred = _markov_label_predict(
            yb_test,
            test_meta,
            binary_markov_map,
            fallback=binary_majority_label,
        )

        binary_logistic_model = _make_logistic_model()
        binary_logistic_model.fit(x_train, yb_train)
        binary_logistic_pred = binary_logistic_model.predict(x_test)

        binary_logistic_unweighted_model = _make_logistic_model(class_weight=None)
        binary_logistic_unweighted_model.fit(x_train, yb_train)
        binary_logistic_unweighted_pred = binary_logistic_unweighted_model.predict(x_test)

        binary_forest_model = _make_random_forest_model()
        binary_forest_model.fit(x_train, yb_train)
        binary_forest_pred = binary_forest_model.predict(x_test)

        row = {
            "fold_id": fold_id,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_label_count": int(y_train.nunique()),
            "test_label_count": int(y_test.nunique()),
        }
        row.update(_classification_metrics(y_test, majority_pred, prefix="majority"))
        row.update(_classification_metrics(y_test, previous_pred, prefix="previous"))
        row.update(_classification_metrics(y_test, markov_pred, prefix="markov"))
        row.update(_classification_metrics(y_test, logistic_pred, prefix="logistic"))
        row.update(_classification_metrics(y_test, logistic_unweighted_pred, prefix="logistic_unweighted"))
        row.update(_classification_metrics(y_test, forest_pred, prefix="random_forest"))
        row.update(_classification_metrics(yb_test, binary_majority_pred, prefix="binary_majority"))
        row.update(_classification_metrics(yb_test, binary_previous_pred, prefix="binary_previous"))
        row.update(_classification_metrics(yb_test, binary_markov_pred, prefix="binary_markov"))
        row.update(_classification_metrics(yb_test, binary_logistic_pred, prefix="binary_logistic"))
        row.update(
            _classification_metrics(
                yb_test,
                binary_logistic_unweighted_pred,
                prefix="binary_logistic_unweighted",
            )
        )
        row.update(_classification_metrics(yb_test, binary_forest_pred, prefix="binary_random_forest"))
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
        pred_frame["logistic_pred_simple_action_code"] = logistic_pred
        pred_frame["logistic_unweighted_pred_simple_action_code"] = logistic_unweighted_pred
        pred_frame["random_forest_pred_simple_action_code"] = forest_pred
        pred_frame["true_binary_action_code"] = yb_test.to_numpy()
        pred_frame["binary_majority_pred"] = binary_majority_pred
        pred_frame["binary_previous_pred"] = binary_previous_pred
        pred_frame["binary_markov_pred"] = binary_markov_pred
        pred_frame["binary_logistic_pred"] = binary_logistic_pred
        pred_frame["binary_logistic_unweighted_pred"] = binary_logistic_unweighted_pred
        pred_frame["binary_random_forest_pred"] = binary_forest_pred
        prediction_rows.append(
            pred_frame
        )

    by_fold = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    metric_cols = [col for col in by_fold.columns if col.endswith(("accuracy", "f1"))]
    summary = by_fold[metric_cols].agg(["mean", "median", "min", "max"]).reset_index().rename(
        columns={"index": "statistic"}
    )

    by_fold_path = output / "state_action_code_predictability_by_fold.csv"
    summary_path = output / "state_action_code_predictability_summary.csv"
    predictions_path = output / "state_action_code_predictability_predictions.csv"
    report_path = output / "predictability_report.json"
    by_fold.to_csv(by_fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    report = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "rows": int(len(dataset)),
        "feature_prefixes": list(prefixes),
        "state_feature_columns": int(len(state_cols)),
        "folds": int(by_fold["fold_id"].nunique()),
        "label_count": int(y_all.nunique()),
        "binary_positive_rate": float((pd.Series(y_binary) == "nonflat").mean()),
        "outputs": {
            "state_action_code_predictability_by_fold": str(by_fold_path),
            "state_action_code_predictability_summary": str(summary_path),
            "state_action_code_predictability_predictions": str(predictions_path),
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
    parser = argparse.ArgumentParser(description="Evaluate fold-held-out action-code predictability.")
    parser.add_argument("--dataset", required=True, help="Path to teacher_state_action_reward_dataset.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for predictability outputs.")
    parser.add_argument(
        "--feature-prefix",
        action="append",
        dest="feature_prefixes",
        help="Feature-column prefix to use. Defaults to state_. Use obs_ for exact observation datasets.",
    )
    args = parser.parse_args(argv)
    evaluate_predictability(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        feature_prefixes=args.feature_prefixes,
    )


if __name__ == "__main__":
    main()
