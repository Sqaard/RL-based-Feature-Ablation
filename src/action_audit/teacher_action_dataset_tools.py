from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from latent_action_phase2_tools import action_trace_to_matrix, build_simple_action_codes


BASE_MACRO_FEATURES = (
    "daily_return",
    "atr_rel",
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
    "volume_ratio",
    "obv_pct_change",
    "turbulence",
    "10Y_Yield",
    "VIX",
    "SP500_Trend",
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


def _read_csv(path: str | Path, *, parse_date: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if parse_date and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def build_reward_alignment(daily_returns: pd.DataFrame) -> pd.DataFrame:
    daily = daily_returns.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.sort_values(["run_key", "date"]).reset_index(drop=True)
    daily["action_date"] = daily.groupby("run_key")["date"].shift(1)
    reward = daily[daily["action_date"].notna()].copy()
    reward = reward.rename(
        columns={
            "date": "reward_date",
            "daily_return": "reward_daily_return",
            "benchmark_return": "reward_benchmark_return",
            "excess_return_vs_benchmark": "reward_excess_return_vs_benchmark",
            "portfolio_value": "reward_portfolio_value",
            "regime_label_exogenous": "reward_regime_label_exogenous",
        }
    )
    keep_cols = [
        "run_key",
        "action_date",
        "reward_date",
        "reward_daily_return",
        "reward_benchmark_return",
        "reward_excess_return_vs_benchmark",
        "reward_portfolio_value",
        "reward_regime_label_exogenous",
    ]
    return reward[[col for col in keep_cols if col in reward.columns]].copy()


def build_state_feature_table(
    processed_dataset: pd.DataFrame,
    *,
    feature_cols: Sequence[str] = BASE_MACRO_FEATURES,
) -> pd.DataFrame:
    panel = processed_dataset.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    available = [col for col in feature_cols if col in panel.columns]
    if not available:
        raise ValueError("No requested state feature columns were found in the processed dataset.")

    frames: list[pd.DataFrame] = []
    grouped = panel.groupby("date", dropna=False)
    for feature in available:
        numeric = pd.to_numeric(panel[feature], errors="coerce")
        tmp = pd.DataFrame({"date": panel["date"], feature: numeric})
        agg = (
            tmp.groupby("date", dropna=False)[feature]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"state_{feature}_xsec_mean",
                    "std": f"state_{feature}_xsec_std",
                    "min": f"state_{feature}_xsec_min",
                    "max": f"state_{feature}_xsec_max",
                }
            )
        )
        frames.append(agg)

    state = frames[0]
    for frame in frames[1:]:
        state = state.merge(frame, on="date", how="outer")

    if "tic" in panel.columns:
        coverage = grouped["tic"].nunique().rename("state_ticker_count").reset_index()
        state = state.merge(coverage, on="date", how="left")
    return state.sort_values("date").reset_index(drop=True)


def build_teacher_state_action_reward_dataset(
    *,
    actions_path: str | Path,
    daily_returns_path: str | Path,
    processed_dataset_path: str | Path,
    output_dir: str | Path,
    teacher_feature_sets: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    actions = _read_csv(actions_path)
    daily = _read_csv(daily_returns_path)
    processed = _read_csv(processed_dataset_path)

    teachers = tuple(teacher_feature_sets or ("base_macro",))
    if "feature_set" in actions.columns and teachers:
        actions = actions[actions["feature_set"].isin(teachers)].copy()
    if actions.empty:
        raise ValueError("No action rows remained after teacher feature-set filtering.")

    matrix, action_cols = action_trace_to_matrix(actions)
    coded = build_simple_action_codes(matrix, action_cols)
    if coded.empty:
        raise ValueError("Action coding produced no rows.")
    coded["date"] = pd.to_datetime(coded["date"], errors="coerce")

    reward = build_reward_alignment(daily)
    state = build_state_feature_table(processed)

    dataset = coded.merge(
        reward,
        left_on=["run_key", "date"],
        right_on=["run_key", "action_date"],
        how="left",
        indicator="reward_merge_status",
    )
    dataset = dataset.merge(
        state,
        on="date",
        how="left",
        indicator="state_merge_status",
    )
    dataset = dataset.rename(columns={"date": "action_date"})

    dataset_path = output / "teacher_state_action_reward_dataset.csv"
    quality_path = output / "teacher_dataset_quality_report.json"
    label_path = output / "teacher_action_label_distribution.csv"
    fold_label_path = output / "teacher_action_label_distribution_by_fold.csv"

    label_distribution = (
        dataset.groupby("simple_action_code", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    label_distribution["share"] = label_distribution["count"] / label_distribution["count"].sum()
    fold_label_distribution = (
        dataset.groupby(["fold_id", "simple_action_code"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["fold_id", "count"], ascending=[True, False])
    )
    fold_label_distribution["fold_share"] = fold_label_distribution["count"] / fold_label_distribution.groupby(
        "fold_id"
    )["count"].transform("sum")

    state_cols = [col for col in dataset.columns if col.startswith("state_")]
    quality = {
        "actions_path": str(Path(actions_path).resolve()),
        "daily_returns_path": str(Path(daily_returns_path).resolve()),
        "processed_dataset_path": str(Path(processed_dataset_path).resolve()),
        "teacher_feature_sets": list(teachers),
        "raw_action_rows": int(len(actions)),
        "teacher_dataset_rows": int(len(dataset)),
        "action_dim": int(len(action_cols)),
        "state_feature_columns": int(len(state_cols)),
        "simple_action_code_count": int(dataset["simple_action_code"].nunique(dropna=True)),
        "missing_reward_rows": int(dataset["reward_daily_return"].isna().sum()),
        "missing_state_rows": int(dataset[state_cols].isna().all(axis=1).sum()) if state_cols else int(len(dataset)),
        "reward_alignment": "action_date_t_to_reward_date_t_plus_1",
        "outputs": {
            "teacher_state_action_reward_dataset": str(dataset_path),
            "teacher_action_label_distribution": str(label_path),
            "teacher_action_label_distribution_by_fold": str(fold_label_path),
            "teacher_dataset_quality_report": str(quality_path),
        },
    }

    dataset.to_csv(dataset_path, index=False)
    label_distribution.to_csv(label_path, index=False)
    fold_label_distribution.to_csv(fold_label_path, index=False)
    quality_path.write_text(json.dumps(_json_safe(quality), indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "teacher_dataset": dataset,
        "label_distribution": label_distribution,
        "fold_label_distribution": fold_label_distribution,
        "quality_report": quality,
    }


def _observation_feature_columns(observations: pd.DataFrame) -> tuple[list[str], list[str]]:
    obs_cols = [col for col in observations.columns if col.startswith("obs_")]
    raw_action_cols = [col for col in observations.columns if col.startswith("raw_policy_action_")]
    if not obs_cols:
        raise ValueError("No exact observation columns found. Expected columns starting with `obs_`.")
    return obs_cols, raw_action_cols


def _merge_key_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    preferred = [
        "run_key",
        "feature_set",
        "fold_id",
        "seed",
        "split_name",
        "date",
        "action_step",
    ]
    keys = [col for col in preferred if col in left.columns and col in right.columns]
    required = {"run_key", "date"}
    if not required.issubset(keys):
        raise KeyError(f"Cannot align actions and observations without keys: {sorted(required)}")
    return keys


def build_teacher_exact_observation_action_reward_dataset(
    *,
    actions_path: str | Path,
    observations_path: str | Path,
    daily_returns_path: str | Path,
    output_dir: str | Path,
    teacher_feature_sets: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    actions = _read_csv(actions_path)
    observations = _read_csv(observations_path)
    daily = _read_csv(daily_returns_path)

    teachers = tuple(teacher_feature_sets or ("base_macro",))
    if "feature_set" in actions.columns and teachers:
        actions = actions[actions["feature_set"].isin(teachers)].copy()
    if "feature_set" in observations.columns and teachers:
        observations = observations[observations["feature_set"].isin(teachers)].copy()
    if actions.empty:
        raise ValueError("No action rows remained after teacher feature-set filtering.")
    if observations.empty:
        raise ValueError("No observation rows remained after teacher feature-set filtering.")

    matrix, action_cols = action_trace_to_matrix(actions)
    coded = build_simple_action_codes(matrix, action_cols)
    if coded.empty:
        raise ValueError("Action coding produced no rows.")
    coded["date"] = pd.to_datetime(coded["date"], errors="coerce")

    observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
    obs_cols, raw_action_cols = _observation_feature_columns(observations)
    merge_keys = _merge_key_columns(coded, observations)
    obs_keep_cols = merge_keys + [
        col
        for col in ["observation_row_id", "selected_model_type", "selection_rule"]
        if col in observations.columns and col not in merge_keys
    ]
    obs_keep_cols += obs_cols + raw_action_cols
    obs_frame = observations[obs_keep_cols].copy()

    dataset = coded.merge(
        obs_frame,
        on=merge_keys,
        how="left",
        indicator="observation_merge_status",
        suffixes=("", "_observation"),
    )
    reward = build_reward_alignment(daily)
    dataset = dataset.merge(
        reward,
        left_on=["run_key", "date"],
        right_on=["run_key", "action_date"],
        how="left",
        indicator="reward_merge_status",
    )
    dataset = dataset.rename(columns={"date": "action_date"})

    dataset_path = output / "teacher_exact_observation_action_reward_dataset.csv"
    quality_path = output / "teacher_exact_observation_dataset_quality_report.json"
    label_path = output / "teacher_exact_observation_action_label_distribution.csv"
    fold_label_path = output / "teacher_exact_observation_action_label_distribution_by_fold.csv"

    label_distribution = (
        dataset.groupby("simple_action_code", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    label_distribution["share"] = label_distribution["count"] / label_distribution["count"].sum()
    fold_label_distribution = (
        dataset.groupby(["fold_id", "simple_action_code"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["fold_id", "count"], ascending=[True, False])
    )
    fold_label_distribution["fold_share"] = fold_label_distribution["count"] / fold_label_distribution.groupby(
        "fold_id"
    )["count"].transform("sum")

    action_dupes = int(coded.duplicated(subset=merge_keys).sum())
    obs_dupes = int(observations.duplicated(subset=merge_keys).sum())
    quality = {
        "actions_path": str(Path(actions_path).resolve()),
        "observations_path": str(Path(observations_path).resolve()),
        "daily_returns_path": str(Path(daily_returns_path).resolve()),
        "teacher_feature_sets": list(teachers),
        "merge_keys": merge_keys,
        "raw_action_rows": int(len(actions)),
        "raw_observation_rows": int(len(observations)),
        "teacher_dataset_rows": int(len(dataset)),
        "action_dim": int(len(action_cols)),
        "exact_observation_columns": int(len(obs_cols)),
        "raw_policy_action_columns": int(len(raw_action_cols)),
        "simple_action_code_count": int(dataset["simple_action_code"].nunique(dropna=True)),
        "duplicate_action_key_rows": action_dupes,
        "duplicate_observation_key_rows": obs_dupes,
        "missing_observation_rows": int(dataset[obs_cols].isna().all(axis=1).sum()),
        "missing_reward_rows": int(dataset["reward_daily_return"].isna().sum()),
        "reward_alignment": "action_date_t_to_reward_date_t_plus_1",
        "outputs": {
            "teacher_exact_observation_action_reward_dataset": str(dataset_path),
            "teacher_exact_observation_action_label_distribution": str(label_path),
            "teacher_exact_observation_action_label_distribution_by_fold": str(fold_label_path),
            "teacher_exact_observation_dataset_quality_report": str(quality_path),
        },
    }

    dataset.to_csv(dataset_path, index=False)
    label_distribution.to_csv(label_path, index=False)
    fold_label_distribution.to_csv(fold_label_path, index=False)
    quality_path.write_text(json.dumps(_json_safe(quality), indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "teacher_dataset": dataset,
        "label_distribution": label_distribution,
        "fold_label_distribution": fold_label_distribution,
        "quality_report": quality,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build aligned latent-action teacher dataset.")
    parser.add_argument("--actions", required=True, help="Path to walk_forward_test_actions.csv.")
    parser.add_argument("--daily", required=True, help="Path to walk_forward_daily_test_returns.csv.")
    parser.add_argument("--processed-dataset", help="Path to processed panel CSV for aggregate-state dataset.")
    parser.add_argument("--observations", help="Path to walk_forward_test_observations.csv for exact-observation dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for teacher dataset outputs.")
    parser.add_argument(
        "--teacher-feature-set",
        action="append",
        dest="teacher_feature_sets",
        help="Teacher feature set to keep. May be repeated. Defaults to base_macro.",
    )
    args = parser.parse_args(argv)
    if args.observations:
        build_teacher_exact_observation_action_reward_dataset(
            actions_path=args.actions,
            observations_path=args.observations,
            daily_returns_path=args.daily,
            output_dir=args.output_dir,
            teacher_feature_sets=args.teacher_feature_sets,
        )
    else:
        if not args.processed_dataset:
            parser.error("--processed-dataset is required when --observations is not provided.")
        build_teacher_state_action_reward_dataset(
            actions_path=args.actions,
            daily_returns_path=args.daily,
            processed_dataset_path=args.processed_dataset,
            output_dir=args.output_dir,
            teacher_feature_sets=args.teacher_feature_sets,
        )


if __name__ == "__main__":
    main()
