from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import pandas as pd

from dow30_horizon_a import (
    build_controlled_feature_registry,
    build_reference_experiment_config,
    ensure_event_calendar_features,
)
from dow30_research_support import (
    DEFAULT_FEATURE_GROUPS,
    TrainOnlyPreprocessor,
    audit_dataset_integrity,
    build_data_card,
    evaluate_equity_curve,
    generate_walk_forward_folds,
    run_feature_ablation_ladder,
    select_best_artifact,
    _serialize_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "processed_final.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "research_outputs"
REFERENCE_EXPERIMENT_CONFIG = build_reference_experiment_config("custom_custom")

PROJECT_PROTOCOL = {
    "start_date": "2010-01-01",
    "first_test_start": "2016-01-04",
    "end_date": "2023-03-01",
    "min_train_months": 60,
    "inner_validation_months": 3,
    "test_window_months": 3,
    "step_months": 6,
    "embargo_days": 5,
}

PROJECT_SELECTION_CONFIG = {
    "objective_col": "validation_sharpe",
    "train_metric_col": "train_sharpe",
    "drawdown_col": "validation_max_drawdown",
    "turnover_col": "validation_turnover",
    "generalization_weight": 0.35,
    "drawdown_weight": 0.10,
    "turnover_weight": 0.05,
}


def load_processed_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    df["date"] = pd.to_datetime(df["date"])
    if "date_available" in df.columns:
        df["date_available"] = pd.to_datetime(df["date_available"], errors="coerce")
    return ensure_event_calendar_features(df)


def run_research_gate(
    df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    constituent_history: Optional[pd.DataFrame] = None,
    reference_config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_df = ensure_event_calendar_features(df)
    feature_registry = build_controlled_feature_registry(DEFAULT_FEATURE_GROUPS)
    feature_cols = sorted(
        {
            col
            for spec in feature_registry.values()
            for col in spec.columns
        }
    )
    resolved_reference_config = dict(reference_config or REFERENCE_EXPERIMENT_CONFIG.to_dict())

    audit = audit_dataset_integrity(
        df=prepared_df,
        feature_cols=feature_cols,
        constituent_history=constituent_history,
    )
    _serialize_json(audit, output_dir / "audit_report.json")
    data_card = build_data_card(
        df=prepared_df,
        audit_report=audit,
        feature_ladder=DEFAULT_FEATURE_GROUPS,
        dataset_name="dow30_processed_final",
        output_path=output_dir / "data_card.json",
    )
    _serialize_json(resolved_reference_config, output_dir / "reference_experiment_config.json")
    folds = generate_walk_forward_folds(prepared_df, **PROJECT_PROTOCOL)
    folds_df = pd.DataFrame([fold.to_dict() for fold in folds])
    folds_df.to_csv(output_dir / "walk_forward_folds.csv", index=False)

    return {
        "audit": audit,
        "data_card": data_card,
        "folds": folds,
        "folds_df": folds_df,
    }


def build_train_only_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    preprocessor = TrainOnlyPreprocessor(
        feature_cols=feature_cols,
        date_col="date",
        ticker_col="tic",
        fill_strategy="train_median",
        scale_strategy="zscore",
    )
    return preprocessor.fit_transform_splits(train_df, validation_df, test_df)


def build_callback_based_runner(
    train_and_select_fn: Callable[..., Mapping[str, Any]],
    selection_config: Optional[Mapping[str, Any]] = None,
    checkpoint_selection_rule: str = "checkpoint_robust_score",
) -> Callable[..., Mapping[str, Any]]:
    score_config = dict(selection_config or PROJECT_SELECTION_CONFIG)

    def run_fold(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: Sequence[str],
        fold: Any,
        seed: int,
        feature_set_name: str,
    ) -> Mapping[str, Any]:
        train_scaled, validation_scaled, test_scaled, preprocessing_summary = build_train_only_splits(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            feature_cols=feature_cols,
        )

        raw_result = dict(
            train_and_select_fn(
                train_df=train_scaled,
                validation_df=validation_scaled,
                test_df=test_scaled,
                feature_cols=list(feature_cols),
                fold=fold,
                seed=seed,
                feature_set_name=feature_set_name,
            )
        )

        candidates = raw_result.get("candidate_df")
        selected_artifact = raw_result.get("selected_artifact")
        if isinstance(candidates, pd.DataFrame) and not candidates.empty:
            scored_candidates, robust_selected = select_best_artifact(
                candidates,
                selection_rule=checkpoint_selection_rule,
                **score_config,
            )
            raw_result["candidate_df"] = scored_candidates
            raw_result["selected_artifact"] = robust_selected
            raw_result["selected_artifact_type"] = robust_selected.get("artifact_type")
            raw_result["checkpoint_selection_rule"] = checkpoint_selection_rule
        elif isinstance(selected_artifact, Mapping):
            raw_result["selected_artifact_type"] = selected_artifact.get("artifact_type")
            raw_result["checkpoint_selection_rule"] = checkpoint_selection_rule

        raw_result["preprocessing_summary"] = preprocessing_summary
        return raw_result

    return run_fold


def summarize_candidate_from_curves(
    train_curve: pd.DataFrame,
    validation_curve: pd.DataFrame,
    test_curve: pd.DataFrame,
    validation_actions: Optional[pd.DataFrame] = None,
    test_actions: Optional[pd.DataFrame] = None,
    regime_frame: Optional[pd.DataFrame] = None,
    artifact_type: str = "selected_model",
) -> dict[str, Any]:
    train_eval = evaluate_equity_curve(train_curve)
    validation_eval = evaluate_equity_curve(
        validation_curve,
        df_actions=validation_actions,
        regime_frame=regime_frame,
    )
    test_eval = evaluate_equity_curve(
        test_curve,
        df_actions=test_actions,
        regime_frame=regime_frame,
    )

    return {
        "artifact_type": artifact_type,
        "train_metrics": train_eval["metrics"],
        "validation_metrics": validation_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "regime_breakdown": test_eval["regime_breakdown"],
    }


def run_project_ablation(
    df: pd.DataFrame,
    train_and_select_fn: Callable[..., Mapping[str, Any]],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seeds: Sequence[int] = (42, 123, 999, 2024, 2025),
) -> dict[str, Any]:
    prepared_df = ensure_event_calendar_features(df)
    gate = run_research_gate(prepared_df, output_dir=output_dir)
    fold_runner = build_callback_based_runner(
        train_and_select_fn,
        PROJECT_SELECTION_CONFIG,
        checkpoint_selection_rule=REFERENCE_EXPERIMENT_CONFIG.checkpoint_selection_rule,
    )
    results = run_feature_ablation_ladder(
        df=prepared_df,
        folds=gate["folds"],
        run_fold_fn=fold_runner,
        feature_ladder=DEFAULT_FEATURE_GROUPS,
        seeds=seeds,
        output_dir=output_dir,
        selection_config=PROJECT_SELECTION_CONFIG,
        model_name="dow30_ppo",
    )
    return {"gate": gate, "results": results}
