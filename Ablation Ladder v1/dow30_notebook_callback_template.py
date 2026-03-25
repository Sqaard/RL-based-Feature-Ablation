from __future__ import annotations

from typing import Any, Callable, Mapping

import pandas as pd

from dow30_project_research import summarize_candidate_from_curves


def build_curve_payload_callback(
    period_runner_fn: Callable[..., Mapping[str, Any]],
) -> Callable[..., Mapping[str, Any]]:
    """
    Adapter for the new walk-forward harness.

    Expected `period_runner_fn(...)` payload:
    {
        "train_curve": <account value DataFrame>,
        "validation_curve": <account value DataFrame>,
        "test_curve": <account value DataFrame>,
        "validation_actions": <optional actions DataFrame>,
        "test_actions": <optional actions DataFrame>,
        "regime_frame": <optional long market frame with Market_Regime>,
        "candidate_df": <optional DataFrame with checkpoint/model metrics>,
        "selected_artifact": <optional pre-selected row/dict>,
        "selected_artifact_type": <optional string>,
        "training_config": <optional dict>,
    }
    """

    def callback(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        fold: Any,
        seed: int,
        feature_set_name: str,
    ) -> Mapping[str, Any]:
        payload = dict(
            period_runner_fn(
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
                feature_cols=feature_cols,
                fold=fold,
                seed=seed,
                feature_set_name=feature_set_name,
            )
        )

        summary = summarize_candidate_from_curves(
            train_curve=payload["train_curve"],
            validation_curve=payload["validation_curve"],
            test_curve=payload["test_curve"],
            validation_actions=payload.get("validation_actions"),
            test_actions=payload.get("test_actions"),
            regime_frame=payload.get("regime_frame"),
            artifact_type=payload.get("selected_artifact_type", "selected_model"),
        )

        return {
            "train_metrics": summary["train_metrics"],
            "validation_metrics": summary["validation_metrics"],
            "test_metrics": summary["test_metrics"],
            "regime_breakdown": summary["regime_breakdown"],
            "candidate_df": payload.get("candidate_df"),
            "selected_artifact": payload.get("selected_artifact"),
            "selected_artifact_type": payload.get("selected_artifact_type", "selected_model"),
            "training_config": payload.get("training_config", {}),
        }

    return callback
