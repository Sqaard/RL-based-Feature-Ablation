from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_TEACHER_FEATURE_SETS = ("base_macro",)
METADATA_COLS = {
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
    "tic",
    "weight",
    "action",
}


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


def _read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _index_cols(actions: pd.DataFrame, *, include_action_row_id: bool) -> list[str]:
    preferred = [
        "run_key",
        "feature_set",
        "feature_family",
        "fold_id",
        "seed",
        "split_name",
        "date",
        "action_step",
    ]
    if include_action_row_id:
        preferred.append("action_row_id")
    return [col for col in preferred if col in actions.columns]


def action_trace_to_matrix(actions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return one row per policy decision and one numeric column per action dimension."""

    if actions.empty:
        return pd.DataFrame(), []

    frame = actions.copy()
    if {"tic", "weight"}.issubset(frame.columns):
        value_col = "weight"
    elif {"tic", "action"}.issubset(frame.columns):
        value_col = "action"
    else:
        value_col = ""

    if value_col:
        index_cols = _index_cols(frame, include_action_row_id=False)
        matrix = (
            frame.pivot_table(
                index=index_cols,
                columns="tic",
                values=value_col,
                aggfunc="last",
            )
            .sort_index()
            .reset_index()
        )
        matrix.columns = [str(col) for col in matrix.columns]
        action_cols = [col for col in matrix.columns if col not in index_cols]
        return matrix, action_cols

    numeric_cols = [
        col
        for col in frame.columns
        if col not in METADATA_COLS and pd.api.types.is_numeric_dtype(frame[col])
    ]
    index_cols = _index_cols(frame, include_action_row_id=True)
    keep_cols = index_cols + numeric_cols
    matrix = frame[keep_cols].copy().sort_values(index_cols).reset_index(drop=True)
    return matrix, numeric_cols


def build_simple_action_codes(
    action_matrix: pd.DataFrame,
    action_cols: Sequence[str],
    *,
    tol: float = 1e-8,
) -> pd.DataFrame:
    if action_matrix.empty or not action_cols:
        return pd.DataFrame()

    coded = action_matrix.copy()
    values = coded[list(action_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    abs_values = values.abs()
    active = abs_values.gt(tol)
    positive = values.gt(tol)
    negative = values.lt(-tol)

    coded["action_l1"] = abs_values.sum(axis=1)
    coded["action_l2"] = np.sqrt((values**2).sum(axis=1))
    coded["action_max_abs"] = abs_values.max(axis=1)
    coded["active_action_dims"] = active.sum(axis=1)
    coded["positive_action_dims"] = positive.sum(axis=1)
    coded["negative_action_dims"] = negative.sum(axis=1)

    active_mask = coded["active_action_dims"].gt(0)
    coded["direction_code"] = "flat"
    coded.loc[
        active_mask & coded["negative_action_dims"].eq(0),
        "direction_code",
    ] = "buy_only"
    coded.loc[
        active_mask & coded["positive_action_dims"].eq(0),
        "direction_code",
    ] = "sell_only"
    coded.loc[
        active_mask & coded["positive_action_dims"].gt(0) & coded["negative_action_dims"].gt(0),
        "direction_code",
    ] = "mixed"

    nonflat_l1 = coded.loc[active_mask, "action_l1"]
    coded["magnitude_code"] = "flat"
    if nonflat_l1.nunique(dropna=True) >= 3:
        coded.loc[active_mask, "magnitude_code"] = pd.qcut(
            nonflat_l1.rank(method="first"),
            q=3,
            labels=["low", "medium", "high"],
        ).astype(str)
    elif not nonflat_l1.empty:
        coded.loc[active_mask, "magnitude_code"] = "nonflat"

    concentrated_cutoff = max(1, min(3, len(action_cols) // 4 or 1))
    coded["concentration_code"] = np.where(
        coded["active_action_dims"].le(concentrated_cutoff),
        "concentrated",
        "diversified",
    )
    coded.loc[~active_mask, "concentration_code"] = "flat"
    coded["simple_action_code"] = (
        coded["direction_code"]
        + "__"
        + coded["magnitude_code"]
        + "__"
        + coded["concentration_code"]
    )
    return coded


def summarize_action_codes(coded_actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if coded_actions.empty:
        return pd.DataFrame(), pd.DataFrame()

    group_cols = [col for col in ["feature_set", "fold_id", "seed"] if col in coded_actions.columns]
    if not group_cols:
        group_cols = ["simple_action_code"]

    summary = (
        coded_actions.groupby(group_cols, dropna=False)
        .agg(
            action_rows=("simple_action_code", "size"),
            action_code_count=("simple_action_code", "nunique"),
            action_l1_mean=("action_l1", "mean"),
            action_l1_median=("action_l1", "median"),
            action_max_abs_median=("action_max_abs", "median"),
            active_action_dims_median=("active_action_dims", "median"),
            flat_action_rate=("direction_code", lambda s: float((s == "flat").mean())),
            mixed_action_rate=("direction_code", lambda s: float((s == "mixed").mean())),
        )
        .reset_index()
    )

    count_group_cols = [col for col in ["feature_set", "simple_action_code"] if col in coded_actions.columns]
    code_counts = (
        coded_actions.groupby(count_group_cols, dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(count_group_cols)
    )
    if "feature_set" in code_counts.columns:
        denominator = code_counts.groupby("feature_set")["count"].transform("sum")
    else:
        denominator = code_counts["count"].sum()
    code_counts["share"] = code_counts["count"] / denominator
    return summary, code_counts


def summarize_code_dynamics(coded_actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if coded_actions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sort_cols = [col for col in ["run_key", "date", "action_step", "action_row_id"] if col in coded_actions.columns]
    frame = coded_actions.copy()
    if sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(drop=True)
    if "run_key" not in frame.columns:
        frame["run_key"] = "_all"

    transition_frames: list[pd.DataFrame] = []
    dwell_rows: list[dict[str, Any]] = []
    for run_key, group in frame.groupby("run_key", sort=False):
        labels = group["simple_action_code"].astype(str).reset_index(drop=True)
        if len(labels) > 1:
            transition_frames.append(
                pd.DataFrame(
                    {
                        "run_key": run_key,
                        "previous_simple_action_code": labels.shift(1).iloc[1:].to_numpy(),
                        "next_simple_action_code": labels.iloc[1:].to_numpy(),
                    }
                )
            )

        starts = labels.ne(labels.shift(1)).cumsum()
        meta = group.reset_index(drop=True)
        for _, dwell in meta.groupby(starts, sort=False):
            row = {
                "run_key": run_key,
                "simple_action_code": str(dwell["simple_action_code"].iloc[0]),
                "dwell_length": int(len(dwell)),
            }
            for col in ["feature_set", "fold_id", "seed"]:
                if col in dwell.columns:
                    row[col] = dwell[col].iloc[0]
            dwell_rows.append(row)

    transitions = pd.concat(transition_frames, ignore_index=True) if transition_frames else pd.DataFrame()
    if not transitions.empty:
        transition_counts = (
            transitions.groupby(["previous_simple_action_code", "next_simple_action_code"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["previous_simple_action_code", "count"], ascending=[True, False])
        )
        transition_counts["share_from_previous"] = transition_counts["count"] / transition_counts.groupby(
            "previous_simple_action_code"
        )["count"].transform("sum")
    else:
        transition_counts = pd.DataFrame(
            columns=["previous_simple_action_code", "next_simple_action_code", "count", "share_from_previous"]
        )

    dwell_lengths = pd.DataFrame(dwell_rows)
    counts = frame["simple_action_code"].astype(str).value_counts()
    shares = counts / counts.sum()
    entropy = float(-(shares * np.log2(shares)).sum()) if not shares.empty else 0.0
    self_transition_rate = (
        float(
            (
                transitions["previous_simple_action_code"].astype(str)
                == transitions["next_simple_action_code"].astype(str)
            ).mean()
        )
        if not transitions.empty
        else 0.0
    )
    dynamics_summary = pd.DataFrame(
        [
            {
                "action_rows": int(len(frame)),
                "transition_rows": int(len(transitions)),
                "action_code_count": int(counts.size),
                "action_code_entropy_bits": entropy,
                "effective_action_codes": float(2**entropy),
                "dominant_code_share": float(shares.max()) if not shares.empty else 0.0,
                "self_transition_rate": self_transition_rate,
                "dwell_segment_count": int(len(dwell_lengths)),
                "dwell_length_median": float(dwell_lengths["dwell_length"].median()) if not dwell_lengths.empty else 0.0,
                "dwell_length_p90": float(dwell_lengths["dwell_length"].quantile(0.9)) if not dwell_lengths.empty else 0.0,
                "dwell_length_max": int(dwell_lengths["dwell_length"].max()) if not dwell_lengths.empty else 0,
            }
        ]
    )
    return transition_counts, dwell_lengths, dynamics_summary


def run_teacher_action_audit(
    *,
    actions_path: str | Path,
    output_dir: str | Path,
    teacher_feature_sets: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    actions = _read_csv(actions_path)
    teachers = tuple(teacher_feature_sets or DEFAULT_TEACHER_FEATURE_SETS)
    if "feature_set" in actions.columns and teachers:
        actions = actions[actions["feature_set"].isin(teachers)].copy()
    if actions.empty:
        raise ValueError("No action rows remained after teacher feature-set filtering.")

    matrix, action_cols = action_trace_to_matrix(actions)
    coded = build_simple_action_codes(matrix, action_cols)
    summary, code_counts = summarize_action_codes(coded)
    transition_counts, dwell_lengths, dynamics_summary = summarize_code_dynamics(coded)

    matrix_path = output / "latent_action_teacher_matrix.csv"
    coded_path = output / "latent_action_teacher_simple_codes.csv"
    summary_path = output / "latent_action_teacher_action_summary.csv"
    counts_path = output / "latent_action_teacher_code_counts.csv"
    transitions_path = output / "latent_action_teacher_code_transitions.csv"
    dwell_path = output / "latent_action_teacher_code_dwell_lengths.csv"
    dynamics_path = output / "latent_action_teacher_code_dynamics_summary.csv"
    matrix.to_csv(matrix_path, index=False)
    coded.to_csv(coded_path, index=False)
    summary.to_csv(summary_path, index=False)
    code_counts.to_csv(counts_path, index=False)
    transition_counts.to_csv(transitions_path, index=False)
    dwell_lengths.to_csv(dwell_path, index=False)
    dynamics_summary.to_csv(dynamics_path, index=False)

    artifact_index = {
        "actions_path": str(Path(actions_path).resolve()),
        "teacher_feature_sets": list(teachers),
        "action_rows": int(len(actions)),
        "matrix_rows": int(len(matrix)),
        "action_dim": int(len(action_cols)),
        "outputs": {
            "latent_action_teacher_matrix": str(matrix_path),
            "latent_action_teacher_simple_codes": str(coded_path),
            "latent_action_teacher_action_summary": str(summary_path),
            "latent_action_teacher_code_counts": str(counts_path),
            "latent_action_teacher_code_transitions": str(transitions_path),
            "latent_action_teacher_code_dwell_lengths": str(dwell_path),
            "latent_action_teacher_code_dynamics_summary": str(dynamics_path),
        },
    }
    (output / "artifact_index.json").write_text(
        json.dumps(_json_safe(artifact_index), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "teacher_matrix": matrix,
        "coded_actions": coded,
        "summary": summary,
        "code_counts": code_counts,
        "transition_counts": transition_counts,
        "dwell_lengths": dwell_lengths,
        "dynamics_summary": dynamics_summary,
        "artifact_index": artifact_index,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build Phase-2 latent-action teacher diagnostics.")
    parser.add_argument("--actions", required=True, help="Path to walk_forward_test_actions.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for latent-action audit outputs.")
    parser.add_argument(
        "--teacher-feature-set",
        action="append",
        dest="teacher_feature_sets",
        help="Teacher feature set to keep. May be repeated. Defaults to base_macro.",
    )
    args = parser.parse_args(argv)
    run_teacher_action_audit(
        actions_path=args.actions,
        output_dir=args.output_dir,
        teacher_feature_sets=args.teacher_feature_sets,
    )


if __name__ == "__main__":
    main()
