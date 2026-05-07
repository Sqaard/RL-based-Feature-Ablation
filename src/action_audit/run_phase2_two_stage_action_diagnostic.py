from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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


ModelFactory = Callable[[int], Any]


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


def _make_logistic_balanced(_: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    )


def _make_logistic_unweighted(_: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight=None, solver="lbfgs")),
        ]
    )


def _make_random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )


def _make_mlp(seed: int) -> Pipeline:
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
                    random_state=seed,
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


def _fit_predict(
    factory: ModelFactory,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    fallback: str,
    random_state: int,
) -> np.ndarray:
    if y_train.nunique(dropna=True) < 2:
        return np.repeat(fallback, len(x_test))
    try:
        model = factory(random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(x_train, y_train)
        return model.predict(x_test)
    except Exception:
        return np.repeat(fallback, len(x_test))


def _fit_predict_balanced(
    factory: ModelFactory,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    fallback: str,
    target_per_class: int,
    random_state: int,
) -> np.ndarray:
    if y_train.nunique(dropna=True) < 2:
        return np.repeat(fallback, len(x_test))
    x_balanced, y_balanced = _balanced_resample(
        x_train,
        y_train,
        target_per_class=target_per_class,
        random_state=random_state,
    )
    return _fit_predict(
        factory,
        x_balanced,
        y_balanced,
        x_test,
        fallback=fallback,
        random_state=random_state,
    )


def _conditional_key_majority_map(
    train_keys: pd.Series,
    y_train: pd.Series,
) -> dict[str, str]:
    tmp = pd.DataFrame({"key": train_keys.astype(str), "target": y_train.astype(str)})
    tmp = tmp[tmp["key"].notna() & tmp["target"].notna()]
    if tmp.empty:
        return {}
    counts = tmp.groupby(["key", "target"]).size().rename("count").reset_index()
    counts = counts.sort_values(["key", "count", "target"], ascending=[True, False, True])
    return counts.drop_duplicates("key").set_index("key")["target"].to_dict()


def _conditional_key_predict(
    train_keys: pd.Series,
    y_train: pd.Series,
    test_keys: pd.Series,
    *,
    fallback: str,
) -> np.ndarray:
    mapping = _conditional_key_majority_map(train_keys, y_train)
    return np.array([mapping.get(str(key), fallback) for key in test_keys.astype(str)])


def _binary_flat_nonflat(labels: Sequence[str]) -> np.ndarray:
    labels_arr = np.asarray(labels).astype(str)
    return np.where(labels_arr == "flat__flat__flat", "flat", "nonflat")


def _summarize(by_fold: pd.DataFrame, group_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    metric_cols = [col for col in by_fold.columns if col.endswith(("accuracy", "f1"))]
    if not group_cols:
        return by_fold[metric_cols].agg(["mean", "median", "min", "max"]).reset_index().rename(
            columns={"index": "statistic"}
        )
    rows: list[pd.DataFrame] = []
    for group_values, group in by_fold.groupby(list(group_cols), dropna=False):
        summary = group[metric_cols].agg(["mean", "median", "min", "max"]).reset_index().rename(
            columns={"index": "statistic"}
        )
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        for col, value in zip(group_cols, group_values):
            summary.insert(0, col, value)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _prepare_dataset(dataset_path: str | Path, feature_prefixes: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(dataset_path)
    if "action_date" in dataset.columns:
        dataset["action_date"] = pd.to_datetime(dataset["action_date"], errors="coerce")
    if "is_first_action_in_run" in dataset.columns:
        dataset["is_first_action_in_run"] = dataset["is_first_action_in_run"].astype(bool)
    sort_cols = _sort_columns(dataset)
    if sort_cols:
        dataset = dataset.sort_values(sort_cols).reset_index(drop=True)
    feature_cols = _feature_columns(dataset, feature_prefixes)
    x_all = dataset[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_all = x_all.fillna(x_all.median(numeric_only=True)).fillna(0.0)
    return dataset, x_all


def _stage1_hold_change(
    dataset: pd.DataFrame,
    x_all: pd.DataFrame,
    *,
    stage1_target_per_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = dataset[~dataset["is_first_action_in_run"]].copy()
    y_all = frame["action_change_flag"].astype(str)
    rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []

    for fold_number, fold_id in enumerate(sorted(frame["fold_id"].dropna().unique()), start=1):
        test_mask = frame["fold_id"].eq(fold_id)
        train_mask = ~test_mask
        train_idx = frame.index[train_mask]
        test_idx = frame.index[test_mask]
        x_train = x_all.loc[train_idx]
        x_test = x_all.loc[test_idx]
        y_train = y_all.loc[train_idx]
        y_test = y_all.loc[test_idx]
        train_meta = frame.loc[train_idx]
        test_meta = frame.loc[test_idx]
        fallback = str(y_train.value_counts().index[0])

        preds = {
            "majority": _majority_predict(y_train, len(y_test)),
            "previous_flag": _previous_label_predict(y_test, test_meta, fallback=fallback),
            "markov_flag": _markov_label_predict(
                y_test,
                test_meta,
                _train_markov_map(y_train, train_meta),
                fallback=fallback,
            ),
            "logistic_balanced": _fit_predict(
                _make_logistic_balanced,
                x_train,
                y_train,
                x_test,
                fallback=fallback,
                random_state=10_000 + fold_number,
            ),
            "logistic_unweighted": _fit_predict(
                _make_logistic_unweighted,
                x_train,
                y_train,
                x_test,
                fallback=fallback,
                random_state=20_000 + fold_number,
            ),
            "random_forest": _fit_predict(
                _make_random_forest,
                x_train,
                y_train,
                x_test,
                fallback=fallback,
                random_state=30_000 + fold_number,
            ),
            "bc_mlp_natural": _fit_predict(
                _make_mlp,
                x_train,
                y_train,
                x_test,
                fallback=fallback,
                random_state=40_000 + fold_number,
            ),
            "bc_mlp_balanced": _fit_predict_balanced(
                _make_mlp,
                x_train,
                y_train,
                x_test,
                fallback=fallback,
                target_per_class=stage1_target_per_class,
                random_state=50_000 + fold_number,
            ),
        }

        row = {
            "fold_id": fold_id,
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "train_change_rate": float((y_train == "change").mean()),
            "test_change_rate": float((y_test == "change").mean()),
        }
        for name, pred in preds.items():
            row.update(_classification_metrics(y_test, pred, prefix=name))
        rows.append(row)

        pred_cols = [
            col
            for col in ["run_key", "fold_id", "seed", "action_date", "action_step", "previous_simple_action_code"]
            if col in test_meta.columns
        ]
        pred_frame = test_meta[pred_cols].copy()
        pred_frame["true_action_change_flag"] = y_test.to_numpy()
        for name, pred in preds.items():
            pred_frame[f"{name}_pred_action_change_flag"] = pred
        pred_frames.append(pred_frame)

    by_fold = pd.DataFrame(rows)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return by_fold, predictions


def _stage2_change_targets(
    dataset: pd.DataFrame,
    x_all: pd.DataFrame,
    *,
    target_columns: Sequence[str],
    stage2_target_per_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    change_frame = dataset[dataset["action_change_flag"].astype(str).eq("change")].copy()

    for target_col in target_columns:
        if target_col not in change_frame.columns:
            continue
        target_frame = change_frame[change_frame[target_col].notna()].copy()
        y_all = target_frame[target_col].astype(str)
        if target_frame.empty:
            continue

        for fold_number, fold_id in enumerate(sorted(target_frame["fold_id"].dropna().unique()), start=1):
            test_mask = target_frame["fold_id"].eq(fold_id)
            train_mask = ~test_mask
            train_idx = target_frame.index[train_mask]
            test_idx = target_frame.index[test_mask]
            x_train = x_all.loc[train_idx]
            x_test = x_all.loc[test_idx]
            y_train = y_all.loc[train_idx]
            y_test = y_all.loc[test_idx]
            train_meta = target_frame.loc[train_idx]
            test_meta = target_frame.loc[test_idx]
            fallback = str(y_train.value_counts().index[0])

            preds = {
                "majority_change": _majority_predict(y_train, len(y_test)),
                "previous_code_conditional": _conditional_key_predict(
                    train_meta["previous_simple_action_code"],
                    y_train,
                    test_meta["previous_simple_action_code"],
                    fallback=fallback,
                ),
                "logistic_balanced": _fit_predict(
                    _make_logistic_balanced,
                    x_train,
                    y_train,
                    x_test,
                    fallback=fallback,
                    random_state=60_000 + fold_number,
                ),
                "random_forest": _fit_predict(
                    _make_random_forest,
                    x_train,
                    y_train,
                    x_test,
                    fallback=fallback,
                    random_state=70_000 + fold_number,
                ),
                "bc_mlp_balanced": _fit_predict_balanced(
                    _make_mlp,
                    x_train,
                    y_train,
                    x_test,
                    fallback=fallback,
                    target_per_class=stage2_target_per_class,
                    random_state=80_000 + fold_number,
                ),
            }

            row = {
                "stage2_target": target_col,
                "fold_id": fold_id,
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "train_label_count": int(y_train.nunique()),
                "test_label_count": int(y_test.nunique()),
            }
            for name, pred in preds.items():
                row.update(_classification_metrics(y_test, pred, prefix=name))
            rows.append(row)

            pred_cols = [
                col
                for col in ["run_key", "fold_id", "seed", "action_date", "action_step", "previous_simple_action_code"]
                if col in test_meta.columns
            ]
            pred_frame = test_meta[pred_cols].copy()
            pred_frame["stage2_target"] = target_col
            pred_frame[f"true_{target_col}"] = y_test.to_numpy()
            for name, pred in preds.items():
                pred_frame[f"{name}_pred_{target_col}"] = pred
            pred_frames.append(pred_frame)

    by_fold = pd.DataFrame(rows)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return by_fold, predictions


def _end_to_end_simple_code(
    dataset: pd.DataFrame,
    x_all: pd.DataFrame,
    *,
    stage1_target_per_class: int,
    stage2_target_per_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = dataset[~dataset["is_first_action_in_run"]].copy()
    change_train_frame = dataset[dataset["action_change_flag"].astype(str).eq("change")].copy()
    rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []

    for fold_number, fold_id in enumerate(sorted(frame["fold_id"].dropna().unique()), start=1):
        test_mask = frame["fold_id"].eq(fold_id)
        train_mask = ~test_mask
        train_idx = frame.index[train_mask]
        test_idx = frame.index[test_mask]
        x_stage1_train = x_all.loc[train_idx]
        x_test = x_all.loc[test_idx]
        y_stage1_train = frame.loc[train_idx, "action_change_flag"].astype(str)
        y_test_code = frame.loc[test_idx, "simple_action_code"].astype(str)
        y_test_binary = pd.Series(_binary_flat_nonflat(y_test_code), index=y_test_code.index)
        test_meta = frame.loc[test_idx]
        previous_code = test_meta["previous_simple_action_code"].astype(str).to_numpy()
        stage1_fallback = str(y_stage1_train.value_counts().index[0])

        stage1_preds = {
            "majority": _majority_predict(y_stage1_train, len(x_test)),
            "logistic_balanced": _fit_predict(
                _make_logistic_balanced,
                x_stage1_train,
                y_stage1_train,
                x_test,
                fallback=stage1_fallback,
                random_state=90_000 + fold_number,
            ),
            "random_forest": _fit_predict(
                _make_random_forest,
                x_stage1_train,
                y_stage1_train,
                x_test,
                fallback=stage1_fallback,
                random_state=100_000 + fold_number,
            ),
            "bc_mlp_balanced": _fit_predict_balanced(
                _make_mlp,
                x_stage1_train,
                y_stage1_train,
                x_test,
                fallback=stage1_fallback,
                target_per_class=stage1_target_per_class,
                random_state=110_000 + fold_number,
            ),
        }

        change_train = change_train_frame[~change_train_frame["fold_id"].eq(fold_id)].copy()
        x_stage2_train = x_all.loc[change_train.index]
        y_stage2_train = change_train["simple_action_code"].astype(str)
        stage2_fallback = str(y_stage2_train.value_counts().index[0])
        stage2_preds = {
            "previous_code_conditional": _conditional_key_predict(
                change_train["previous_simple_action_code"],
                y_stage2_train,
                test_meta["previous_simple_action_code"],
                fallback=stage2_fallback,
            ),
            "logistic_balanced": _fit_predict(
                _make_logistic_balanced,
                x_stage2_train,
                y_stage2_train,
                x_test,
                fallback=stage2_fallback,
                random_state=120_000 + fold_number,
            ),
            "random_forest": _fit_predict(
                _make_random_forest,
                x_stage2_train,
                y_stage2_train,
                x_test,
                fallback=stage2_fallback,
                random_state=130_000 + fold_number,
            ),
            "bc_mlp_balanced": _fit_predict_balanced(
                _make_mlp,
                x_stage2_train,
                y_stage2_train,
                x_test,
                fallback=stage2_fallback,
                target_per_class=stage2_target_per_class,
                random_state=140_000 + fold_number,
            ),
        }

        baseline_previous = previous_code
        row = {
            "fold_id": fold_id,
            "test_rows": int(len(y_test_code)),
            "test_change_rate": float(test_meta["action_change_flag"].astype(str).eq("change").mean()),
        }
        row.update(_classification_metrics(y_test_code, baseline_previous, prefix="previous_code"))
        row.update(
            _classification_metrics(
                y_test_binary,
                _binary_flat_nonflat(baseline_previous),
                prefix="binary_previous_code",
            )
        )

        pred_cols = [
            col
            for col in ["run_key", "fold_id", "seed", "action_date", "action_step", "previous_simple_action_code"]
            if col in test_meta.columns
        ]
        pred_frame = test_meta[pred_cols].copy()
        pred_frame["true_simple_action_code"] = y_test_code.to_numpy()
        pred_frame["previous_code_pred_simple_action_code"] = baseline_previous

        for stage1_name, stage1_pred in stage1_preds.items():
            for stage2_name, stage2_pred in stage2_preds.items():
                model_name = f"{stage1_name}__{stage2_name}"
                combined = np.where(np.asarray(stage1_pred).astype(str) == "hold", previous_code, stage2_pred)
                row.update(_classification_metrics(y_test_code, combined, prefix=model_name))
                row.update(
                    _classification_metrics(
                        y_test_binary,
                        _binary_flat_nonflat(combined),
                        prefix=f"binary_{model_name}",
                    )
                )
                pred_frame[f"{model_name}_pred_simple_action_code"] = combined
        rows.append(row)
        pred_frames.append(pred_frame)

    by_fold = pd.DataFrame(rows)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return by_fold, predictions


def run_two_stage_action_diagnostic(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    feature_prefixes: Optional[Sequence[str]] = None,
    stage2_targets: Optional[Sequence[str]] = None,
    stage1_target_per_class: int = 750,
    stage2_target_per_class: int = 80,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(feature_prefixes or ("obs_",))
    targets = tuple(stage2_targets or ("change_or_current_code", "changed_transition_code"))
    dataset, x_all = _prepare_dataset(dataset_path, prefixes)

    stage1_by_fold, stage1_predictions = _stage1_hold_change(
        dataset,
        x_all,
        stage1_target_per_class=stage1_target_per_class,
    )
    stage2_by_fold, stage2_predictions = _stage2_change_targets(
        dataset,
        x_all,
        target_columns=targets,
        stage2_target_per_class=stage2_target_per_class,
    )
    e2e_by_fold, e2e_predictions = _end_to_end_simple_code(
        dataset,
        x_all,
        stage1_target_per_class=stage1_target_per_class,
        stage2_target_per_class=stage2_target_per_class,
    )

    outputs = {
        "stage1_by_fold": output / "two_stage_stage1_hold_change_by_fold.csv",
        "stage1_summary": output / "two_stage_stage1_hold_change_summary.csv",
        "stage1_predictions": output / "two_stage_stage1_hold_change_predictions.csv",
        "stage2_by_fold": output / "two_stage_stage2_change_targets_by_fold.csv",
        "stage2_summary": output / "two_stage_stage2_change_targets_summary.csv",
        "stage2_predictions": output / "two_stage_stage2_change_targets_predictions.csv",
        "end_to_end_by_fold": output / "two_stage_end_to_end_simple_code_by_fold.csv",
        "end_to_end_summary": output / "two_stage_end_to_end_simple_code_summary.csv",
        "end_to_end_predictions": output / "two_stage_end_to_end_simple_code_predictions.csv",
        "report": output / "two_stage_action_diagnostic_report.json",
    }

    stage1_summary = _summarize(stage1_by_fold)
    stage2_summary = _summarize(stage2_by_fold, group_cols=["stage2_target"])
    e2e_summary = _summarize(e2e_by_fold)

    stage1_by_fold.to_csv(outputs["stage1_by_fold"], index=False)
    stage1_summary.to_csv(outputs["stage1_summary"], index=False)
    stage1_predictions.to_csv(outputs["stage1_predictions"], index=False)
    stage2_by_fold.to_csv(outputs["stage2_by_fold"], index=False)
    stage2_summary.to_csv(outputs["stage2_summary"], index=False)
    stage2_predictions.to_csv(outputs["stage2_predictions"], index=False)
    e2e_by_fold.to_csv(outputs["end_to_end_by_fold"], index=False)
    e2e_summary.to_csv(outputs["end_to_end_summary"], index=False)
    e2e_predictions.to_csv(outputs["end_to_end_predictions"], index=False)

    report = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "rows": int(len(dataset)),
        "feature_prefixes": list(prefixes),
        "feature_columns": int(len(_feature_columns(dataset, prefixes))),
        "folds": int(dataset["fold_id"].nunique()),
        "stage1_rows_excluding_starts": int((~dataset["is_first_action_in_run"]).sum()),
        "stage1_change_rate_excluding_starts": float(
            dataset.loc[~dataset["is_first_action_in_run"], "action_change_flag"].astype(str).eq("change").mean()
        ),
        "stage2_change_rows": int(dataset["action_change_flag"].astype(str).eq("change").sum()),
        "stage2_targets": list(targets),
        "stage1_target_per_class": int(stage1_target_per_class),
        "stage2_target_per_class": int(stage2_target_per_class),
        "outputs": {key: str(value) for key, value in outputs.items() if key != "report"},
    }
    outputs["report"].write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "stage1_by_fold": stage1_by_fold,
        "stage1_summary": stage1_summary,
        "stage2_by_fold": stage2_by_fold,
        "stage2_summary": stage2_summary,
        "end_to_end_by_fold": e2e_by_fold,
        "end_to_end_summary": e2e_summary,
        "report": report,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run Phase-2 two-stage hold/change action diagnostic.")
    parser.add_argument("--dataset", required=True, help="Path to phase2_action_label_variants_dataset.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for two-stage diagnostic outputs.")
    parser.add_argument(
        "--feature-prefix",
        action="append",
        dest="feature_prefixes",
        help="Feature-column prefix to use. Defaults to obs_.",
    )
    parser.add_argument(
        "--stage2-target",
        action="append",
        dest="stage2_targets",
        help="Stage-2 target column. May be repeated. Defaults to change_or_current_code and changed_transition_code.",
    )
    parser.add_argument("--stage1-target-per-class", type=int, default=750)
    parser.add_argument("--stage2-target-per-class", type=int, default=80)
    args = parser.parse_args(argv)
    run_two_stage_action_diagnostic(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        feature_prefixes=args.feature_prefixes,
        stage2_targets=args.stage2_targets,
        stage1_target_per_class=args.stage1_target_per_class,
        stage2_target_per_class=args.stage2_target_per_class,
    )


if __name__ == "__main__":
    main()
