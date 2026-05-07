from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from latent_action_phase2_tools import build_simple_action_codes


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


def _sort_columns(dataset: pd.DataFrame) -> list[str]:
    return [col for col in ["run_key", "action_date", "date", "action_step", "observation_row_id"] if col in dataset.columns]


def _add_previous_and_change_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    sort_cols = _sort_columns(frame)
    if sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(drop=True)
    if "run_key" not in frame.columns:
        frame["run_key"] = "_all"

    frame["previous_simple_action_code"] = frame.groupby("run_key")["simple_action_code"].shift(1)
    frame["is_first_action_in_run"] = frame["previous_simple_action_code"].isna()
    frame["action_code_changed"] = (
        frame["simple_action_code"].astype(str) != frame["previous_simple_action_code"].astype(str)
    )
    frame.loc[frame["is_first_action_in_run"], "action_code_changed"] = True
    frame["action_change_flag"] = np.where(frame["action_code_changed"], "change", "hold")
    frame.loc[frame["is_first_action_in_run"], "action_change_flag"] = "start"
    frame["change_or_current_code"] = np.where(
        frame["action_change_flag"].eq("hold"),
        "hold_previous_code",
        frame["simple_action_code"].astype(str),
    )
    frame["transition_code"] = np.where(
        frame["is_first_action_in_run"],
        "start->" + frame["simple_action_code"].astype(str),
        frame["previous_simple_action_code"].astype(str) + "->" + frame["simple_action_code"].astype(str),
    )
    frame["changed_transition_code"] = np.where(
        frame["action_change_flag"].eq("change"),
        frame["transition_code"],
        frame["action_change_flag"],
    )
    return frame


def _add_raw_policy_action_codes(dataset: pd.DataFrame) -> pd.DataFrame:
    raw_cols = [col for col in dataset.columns if col.startswith("raw_policy_action_")]
    if not raw_cols:
        return dataset
    index_cols = [
        col
        for col in [
            "run_key",
            "feature_set",
            "feature_family",
            "fold_id",
            "seed",
            "split_name",
            "action_date",
            "observation_row_id",
        ]
        if col in dataset.columns
    ]
    raw_matrix = dataset[index_cols + raw_cols].copy()
    raw_coded = build_simple_action_codes(raw_matrix, raw_cols)
    raw_keep = index_cols + [
        "direction_code",
        "magnitude_code",
        "concentration_code",
        "simple_action_code",
        "action_l1",
        "active_action_dims",
    ]
    raw_keep = [col for col in raw_keep if col in raw_coded.columns]
    raw_coded = raw_coded[raw_keep].rename(
        columns={
            "direction_code": "raw_policy_direction_code",
            "magnitude_code": "raw_policy_magnitude_code",
            "concentration_code": "raw_policy_concentration_code",
            "simple_action_code": "raw_policy_simple_action_code",
            "action_l1": "raw_policy_action_l1",
            "active_action_dims": "raw_policy_active_action_dims",
        }
    )
    merge_keys = [col for col in index_cols if col in raw_coded.columns and col in dataset.columns]
    enriched = dataset.merge(raw_coded, on=merge_keys, how="left", suffixes=("", "_raw_policy"))
    if "direction_code" in enriched.columns and "raw_policy_direction_code" in enriched.columns:
        enriched["executed_raw_direction_match"] = (
            enriched["direction_code"].astype(str) == enriched["raw_policy_direction_code"].astype(str)
        )
    return enriched


def _label_count_table(dataset: pd.DataFrame, label_cols: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label_col in label_cols:
        if label_col not in dataset.columns:
            continue
        counts = dataset[label_col].astype(str).value_counts(dropna=False).rename_axis("label").reset_index(name="count")
        counts["share"] = counts["count"] / counts["count"].sum()
        counts.insert(0, "label_column", label_col)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_action_label_variants(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(dataset_path)
    if "action_date" in dataset.columns:
        dataset["action_date"] = pd.to_datetime(dataset["action_date"], errors="coerce")
    enriched = _add_previous_and_change_labels(dataset)
    enriched = _add_raw_policy_action_codes(enriched)

    label_cols = [
        "simple_action_code",
        "action_change_flag",
        "change_or_current_code",
        "changed_transition_code",
        "raw_policy_simple_action_code",
    ]
    label_counts = _label_count_table(enriched, label_cols)

    variant_path = output / "phase2_action_label_variants_dataset.csv"
    counts_path = output / "phase2_action_label_variant_counts.csv"
    report_path = output / "phase2_action_label_variant_report.json"
    enriched.to_csv(variant_path, index=False)
    label_counts.to_csv(counts_path, index=False)

    report = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "rows": int(len(enriched)),
        "label_columns": [col for col in label_cols if col in enriched.columns],
        "action_change_rate_including_starts": float(enriched["action_code_changed"].mean()),
        "hold_rate_excluding_starts": float(
            enriched.loc[~enriched["is_first_action_in_run"], "action_change_flag"].eq("hold").mean()
        ),
        "raw_policy_action_code_count": int(enriched["raw_policy_simple_action_code"].nunique(dropna=True))
        if "raw_policy_simple_action_code" in enriched.columns
        else 0,
        "executed_raw_direction_match_rate": float(enriched["executed_raw_direction_match"].mean())
        if "executed_raw_direction_match" in enriched.columns
        else None,
        "outputs": {
            "phase2_action_label_variants_dataset": str(variant_path),
            "phase2_action_label_variant_counts": str(counts_path),
            "phase2_action_label_variant_report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "dataset": enriched,
        "label_counts": label_counts,
        "report": report,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build Phase-2 action label variants for tokenizer/BC redesign.")
    parser.add_argument("--dataset", required=True, help="Path to exact observation/action/reward dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for action-label variant outputs.")
    args = parser.parse_args(argv)
    build_action_label_variants(dataset_path=args.dataset, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
