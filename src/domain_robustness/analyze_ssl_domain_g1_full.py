from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SSL_ROOT = PROJECT_ROOT / "SSL Domain Generalization"
DEFAULT_MANIFEST = SSL_ROOT / "g1_parallel_batch" / "g1_parallel_batch_manifest.csv"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "Latent Actions" / "research_outputs_phase2_base_macro_teacher"
DEFAULT_G0_DIR = SSL_ROOT / "research_outputs_ssl_domain_g0_base_macro_temporal_robust_selection"
DEFAULT_BATCH_ANALYSIS_DIR = SSL_ROOT / "research_outputs_ssl_domain_g1_batch_analysis"

SUMMARY_FILE = "corrected_walk_forward_summary_with_primary_benchmark.csv"
RUN_FILE = "walk_forward_results.csv"
BENCHMARK_FILE = "benchmark_run_level_metrics.csv"
NEGATIVE_CONTROLS = {
    "g1c_abs_reverse_negative_control",
    "g1f_vol_reverse_negative_control",
}


def _read_summary(output_dir: Path) -> pd.Series:
    frame = pd.read_csv(output_dir / SUMMARY_FILE)
    if frame.empty:
        raise ValueError(f"Empty summary file: {output_dir / SUMMARY_FILE}")
    return frame.iloc[0]


def _read_runs(output_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(output_dir / RUN_FILE)
    return frame[frame["feature_set"].eq("base_macro")].copy()


def _read_primary_benchmark_runs(output_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(output_dir / BENCHMARK_FILE)
    return frame[frame["is_primary_benchmark"].astype(bool)].copy()


def _delta(left: Any, right: Any) -> float:
    left_value = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_value = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    return float(left_value - right_value)


def _branch_label(variant_id: str) -> str:
    if "abs_" in variant_id:
        return "absolute_return_stress"
    if "vol_" in variant_id:
        return "volatility_stress"
    if "downside_return" in variant_id:
        return "downside_return"
    if "drawdown" in variant_id:
        return "drawdown_stress"
    if "market_regime" in variant_id:
        return "market_regime_balance"
    return "unknown"


def _strength_label(variant_id: str) -> str:
    if "reverse_negative_control" in variant_id:
        return "reverse_negative_control"
    if "strong" in variant_id:
        return "strong"
    if "mild" in variant_id:
        return "mild"
    if "inverse_frequency" in variant_id:
        return "inverse_frequency"
    return "unknown"


def _paired_run_stats(candidate: pd.DataFrame, reference: pd.DataFrame, metric: str) -> dict[str, float]:
    keys = ["fold_id", "seed"]
    merged = candidate[keys + [metric]].merge(
        reference[keys + [metric]],
        on=keys,
        suffixes=("_candidate", "_reference"),
    )
    deltas = merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
    return {
        f"paired_delta_{metric}_median": float(deltas.median()),
        f"paired_delta_{metric}_mean": float(deltas.mean()),
        f"paired_win_rate_{metric}": float((deltas > 0.0).mean()),
    }


def _paired_benchmark_stats(candidate: pd.DataFrame, reference: pd.DataFrame, metric: str) -> dict[str, float]:
    keys = ["fold_id", "seed"]
    merged = candidate[keys + [metric]].merge(
        reference[keys + [metric]],
        on=keys,
        suffixes=("_candidate", "_reference"),
    )
    deltas = merged[f"{metric}_candidate"].astype(float) - merged[f"{metric}_reference"].astype(float)
    return {
        f"paired_delta_primary_{metric}_median": float(deltas.median()),
        f"paired_delta_primary_{metric}_mean": float(deltas.mean()),
        f"paired_win_rate_primary_{metric}": float((deltas > 0.0).mean()),
    }


def _fold_win_count(candidate: pd.DataFrame, reference: pd.DataFrame, metric: str) -> int:
    keys = ["fold_id", "seed"]
    merged = candidate[keys + [metric]].merge(
        reference[keys + [metric]],
        on=keys,
        suffixes=("_candidate", "_reference"),
    )
    merged[f"delta_{metric}"] = merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
    fold_medians = merged.groupby("fold_id")[f"delta_{metric}"].median()
    return int((fold_medians > 0.0).sum())


def _verdict(row: dict[str, Any]) -> str:
    if row["is_negative_control_variant"]:
        return "negative_control_not_promotable"
    if bool(row["provisional_gate_pass"]):
        return "pass"
    if row["delta_vs_reference_test_turnover_median"] < 0 and row["delta_vs_reference_test_sharpe_median"] < 0:
        return "lower_turnover_but_lower_sharpe"
    return "fail"


def build_full_table(
    *,
    manifest_path: Path,
    reference_dir: Path,
    g0_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    reference_summary = _read_summary(reference_dir)
    g0_summary = _read_summary(g0_dir)
    reference_runs = _read_runs(reference_dir)
    reference_bench = _read_primary_benchmark_runs(reference_dir)

    rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        variant_id = str(item["variant_id"])
        variant_dir = PROJECT_ROOT / str(item["output_dir"])
        summary = _read_summary(variant_dir)
        runs = _read_runs(variant_dir)
        bench = _read_primary_benchmark_runs(variant_dir)

        row: dict[str, Any] = {
            "variant_id": variant_id,
            "experiment_name": str(item["experiment_name"]),
            "branch": _branch_label(variant_id),
            "strength": _strength_label(variant_id),
            "derived_domain": str(item["derived_domain"]),
            "mode": str(item["mode"]),
            "hypothesis": str(item["hypothesis"]),
            "is_negative_control_variant": variant_id in NEGATIVE_CONTROLS,
        }

        summary_metrics = [
            "test_sharpe_median",
            "test_return_pct_median",
            "test_max_drawdown_median",
            "test_turnover_median",
            "primary_benchmark_excess_return_pct_median",
            "primary_benchmark_excess_sharpe_median",
            "primary_benchmark_outperform_return_rate",
            "primary_benchmark_outperform_sharpe_rate",
            "primary_benchmark_hit_rate_median",
            "generalization_ratio_median",
            "retention_ratio_median",
        ]
        for metric in summary_metrics:
            row[metric] = float(summary[metric])
            row[f"delta_vs_reference_{metric}"] = _delta(summary[metric], reference_summary[metric])
            row[f"delta_vs_g0_{metric}"] = _delta(summary[metric], g0_summary[metric])

        for metric in [
            "validation_sharpe",
            "validation_return_pct",
            "test_sharpe",
            "test_return_pct",
            "test_max_drawdown",
            "test_turnover",
            "generalization_ratio",
            "retention_ratio",
        ]:
            row.update(_paired_run_stats(runs, reference_runs, metric))

        for metric in [
            "excess_return_pct",
            "excess_sharpe",
            "outperformed_benchmark_on_return",
            "outperformed_benchmark_on_sharpe",
            "hit_rate_vs_benchmark",
        ]:
            row.update(_paired_benchmark_stats(bench, reference_bench, metric))

        row["folds_with_positive_test_sharpe_delta"] = _fold_win_count(runs, reference_runs, "test_sharpe")
        row["folds_with_positive_test_return_delta"] = _fold_win_count(runs, reference_runs, "test_return_pct")
        row["folds_with_lower_turnover"] = int(14 - _fold_win_count(runs, reference_runs, "test_turnover"))
        row["gate_score"] = (
            1.00 * row["delta_vs_reference_primary_benchmark_excess_sharpe_median"]
            + 0.35 * row["delta_vs_reference_primary_benchmark_excess_return_pct_median"]
            + 0.25 * row["delta_vs_reference_test_sharpe_median"]
            - 0.05 * max(0.0, row["delta_vs_reference_test_turnover_median"])
        )
        row["provisional_gate_pass"] = bool(
            not row["is_negative_control_variant"]
            and row["delta_vs_reference_test_sharpe_median"] > 0.0
            and row["delta_vs_reference_primary_benchmark_excess_sharpe_median"] > 0.0
            and row["delta_vs_reference_primary_benchmark_excess_return_pct_median"] > 0.0
            and row["gate_score"] > 0.0
        )
        row["verdict"] = _verdict(row)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("gate_score", ascending=False).reset_index(drop=True)
    table.insert(0, "rank_by_gate_score", range(1, len(table) + 1))
    table.to_csv(output_dir / "g1_full_analysis_table.csv", index=False)
    _write_markdown_table(output_dir / "G1_FULL_ANALYSIS_TABLE.md", table)
    return table


def _write_markdown_table(path: Path, table: pd.DataFrame) -> None:
    cols = [
        "rank_by_gate_score",
        "variant_id",
        "branch",
        "strength",
        "is_negative_control_variant",
        "verdict",
        "gate_score",
        "delta_vs_reference_test_sharpe_median",
        "delta_vs_reference_test_return_pct_median",
        "delta_vs_reference_primary_benchmark_excess_return_pct_median",
        "delta_vs_reference_primary_benchmark_excess_sharpe_median",
        "delta_vs_reference_test_turnover_median",
        "paired_delta_test_sharpe_median",
        "paired_win_rate_test_sharpe",
        "paired_delta_primary_excess_return_pct_median",
        "paired_delta_primary_excess_sharpe_median",
        "folds_with_positive_test_sharpe_delta",
        "folds_with_positive_test_return_delta",
    ]
    out = table[cols].copy()
    for col in out.select_dtypes(include="number").columns:
        if col == "rank_by_gate_score":
            continue
        out[col] = out[col].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(out.columns) + " |"
    separator = "| " + " | ".join("---" for _ in out.columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(row[col]) for col in out.columns) + " |"
        for row in out.to_dict(orient="records")
    )
    lines = [
        "# G1 Full Analysis Table",
        "",
        "Reference: `Latent Actions/research_outputs_phase2_base_macro_teacher`.",
        "",
        header,
        separator,
        body,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full SSL/domain G1 analysis table.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--g0-dir", default=str(DEFAULT_G0_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_BATCH_ANALYSIS_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    table = build_full_table(
        manifest_path=Path(args.manifest),
        reference_dir=Path(args.reference_dir),
        g0_dir=Path(args.g0_dir),
        output_dir=Path(args.output_dir),
    )
    print(table[["rank_by_gate_score", "variant_id", "verdict", "gate_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
