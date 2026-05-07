from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRIMARY_BENCHMARK_ID = "dow30_equal_weight_rebalance_matched"


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
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _summary_delta(g0_dir: Path, reference_dir: Path) -> pd.DataFrame:
    g0 = _read_csv(g0_dir / "corrected_walk_forward_summary_with_primary_benchmark.csv")
    ref = _read_csv(reference_dir / "corrected_walk_forward_summary_with_primary_benchmark.csv")
    merged = g0.merge(ref, on="feature_set", suffixes=("_g0", "_reference"))

    metrics = [
        "validation_sharpe_median",
        "test_sharpe_median",
        "test_sharpe_iqr",
        "test_return_pct_median",
        "test_max_drawdown_median",
        "test_turnover_median",
        "primary_benchmark_excess_return_pct_median",
        "primary_benchmark_excess_sharpe_median",
        "primary_benchmark_outperform_return_rate",
        "primary_benchmark_outperform_sharpe_rate",
    ]
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        for metric in metrics:
            g0_value = row.get(f"{metric}_g0")
            ref_value = row.get(f"{metric}_reference")
            rows.append(
                {
                    "feature_set": row["feature_set"],
                    "metric": metric,
                    "g0": g0_value,
                    "reference": ref_value,
                    "delta": g0_value - ref_value,
                }
            )
    return pd.DataFrame(rows)


def _run_level_delta(g0_dir: Path, reference_dir: Path) -> pd.DataFrame:
    keys = ["fold_id", "seed", "feature_set"]
    metrics = [
        "selected_artifact_type",
        "validation_sharpe",
        "test_sharpe",
        "test_return_pct",
        "test_max_drawdown",
        "test_turnover",
        "robust_selection_score",
        "selected_artifact_score",
    ]
    g0 = _read_csv(g0_dir / "walk_forward_results.csv")
    ref = _read_csv(reference_dir / "walk_forward_results.csv")
    merged = g0[keys + metrics].merge(ref[keys + metrics], on=keys, suffixes=("_g0", "_reference"))
    for metric in metrics:
        if metric == "selected_artifact_type":
            merged[f"{metric}_changed"] = merged[f"{metric}_g0"] != merged[f"{metric}_reference"]
        else:
            merged[f"delta_{metric}"] = merged[f"{metric}_g0"] - merged[f"{metric}_reference"]
    return merged


def _fold_median_delta(run_delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_id, frame in run_delta.groupby("fold_id", dropna=False):
        row: dict[str, Any] = {"fold_id": fold_id, "runs": int(len(frame))}
        for metric in ("test_sharpe", "test_return_pct", "test_max_drawdown", "test_turnover"):
            row[f"{metric}_g0_median"] = float(frame[f"{metric}_g0"].median())
            row[f"{metric}_reference_median"] = float(frame[f"{metric}_reference"].median())
            row[f"delta_{metric}_median"] = float(row[f"{metric}_g0_median"] - row[f"{metric}_reference_median"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)


def _primary_benchmark_delta(g0_dir: Path, reference_dir: Path) -> pd.DataFrame:
    keys = ["run_key", "fold_id", "seed", "feature_set", "benchmark_id"]
    metrics = [
        "agent_sharpe",
        "benchmark_sharpe",
        "excess_sharpe",
        "agent_return_pct",
        "benchmark_return_pct",
        "excess_return_pct",
        "agent_max_drawdown",
        "agent_turnover",
        "outperformed_benchmark_on_return",
        "outperformed_benchmark_on_sharpe",
        "hit_rate_vs_benchmark",
    ]
    g0 = _read_csv(g0_dir / "benchmark_run_level_metrics.csv")
    ref = _read_csv(reference_dir / "benchmark_run_level_metrics.csv")
    g0 = g0[g0["benchmark_id"].eq(PRIMARY_BENCHMARK_ID)].copy()
    ref = ref[ref["benchmark_id"].eq(PRIMARY_BENCHMARK_ID)].copy()
    merged = g0[keys + metrics].merge(ref[keys + metrics], on=keys, suffixes=("_g0", "_reference"))
    for metric in metrics:
        left = merged[f"{metric}_g0"]
        right = merged[f"{metric}_reference"]
        if left.dtype == bool:
            left = left.astype(float)
        if right.dtype == bool:
            right = right.astype(float)
        merged[f"delta_{metric}"] = (
            pd.to_numeric(left, errors="coerce")
            - pd.to_numeric(right, errors="coerce")
        )
    return merged


def _decision_payload(
    summary_delta: pd.DataFrame,
    run_delta: pd.DataFrame,
    benchmark_delta: pd.DataFrame,
) -> dict[str, Any]:
    def metric_delta(name: str) -> float:
        rows = summary_delta[summary_delta["metric"].eq(name)]
        return float(rows["delta"].iloc[0]) if not rows.empty else float("nan")

    changed_runs = int(
        (
            pd.to_numeric(run_delta["delta_test_sharpe"], errors="coerce").abs().fillna(0.0)
            > 1e-9
        ).sum()
    )
    worse_primary_excess_runs = int(
        (pd.to_numeric(benchmark_delta["delta_excess_sharpe"], errors="coerce") < -1e-9).sum()
    )
    better_primary_excess_runs = int(
        (pd.to_numeric(benchmark_delta["delta_excess_sharpe"], errors="coerce") > 1e-9).sum()
    )

    return {
        "decision": "neutral_fail_to_advance_as_ssl_win",
        "rationale": (
            "G0 changed only a small number of run-level outcomes and did not improve median "
            "test Sharpe, benchmark-excess Sharpe, or benchmark outperformance rates."
        ),
        "run_count": int(len(run_delta)),
        "changed_test_sharpe_runs": changed_runs,
        "better_primary_excess_sharpe_runs": better_primary_excess_runs,
        "worse_primary_excess_sharpe_runs": worse_primary_excess_runs,
        "summary_deltas": {
            "test_sharpe_median": metric_delta("test_sharpe_median"),
            "test_return_pct_median": metric_delta("test_return_pct_median"),
            "test_turnover_median": metric_delta("test_turnover_median"),
            "primary_benchmark_excess_return_pct_median": metric_delta(
                "primary_benchmark_excess_return_pct_median"
            ),
            "primary_benchmark_excess_sharpe_median": metric_delta(
                "primary_benchmark_excess_sharpe_median"
            ),
            "primary_benchmark_outperform_sharpe_rate": metric_delta(
                "primary_benchmark_outperform_sharpe_rate"
            ),
        },
    }


def _write_markdown(
    path: Path,
    *,
    g0_dir: Path,
    reference_dir: Path,
    decision: dict[str, Any],
    summary_delta: pd.DataFrame,
    run_delta: pd.DataFrame,
    fold_delta: pd.DataFrame,
) -> None:
    def d(metric: str) -> float:
        row = summary_delta[summary_delta["metric"].eq(metric)]
        return float(row["delta"].iloc[0]) if not row.empty else float("nan")

    changed = run_delta[
        pd.to_numeric(run_delta["delta_test_sharpe"], errors="coerce").abs().fillna(0.0) > 1e-9
    ].copy()
    changed_rows = "\n".join(
        "| {fold_id} | {seed} | {test_sharpe_g0:.3f} | {test_sharpe_reference:.3f} | {delta_test_sharpe:.3f} | {test_turnover_g0:.3f} | {test_turnover_reference:.3f} |".format(
            **row
        )
        for row in changed.to_dict(orient="records")
    )
    if not changed_rows:
        changed_rows = "| none | | | | | | |"

    worst_fold_rows = "\n".join(
        "| {fold_id} | {test_sharpe_g0_median:.3f} | {test_sharpe_reference_median:.3f} | {delta_test_sharpe_median:.3f} | {delta_test_turnover_median:.3f} |".format(
            **row
        )
        for row in fold_delta.sort_values("delta_test_sharpe_median").head(5).to_dict(orient="records")
    )

    text = f"""# SSL Domain G0 Result Analysis

Status: completed.

G0 output:

`{g0_dir}`

Reference output:

`{reference_dir}`

## Decision

`{decision["decision"]}`

G0 is not a sufficient SSL/domain-generalization win. It is effectively neutral versus the frozen `base_macro` teacher/reference.

## Summary Deltas

| Metric | Delta |
|---|---:|
| Median test Sharpe | {d("test_sharpe_median"):.6f} |
| Median test return pct | {d("test_return_pct_median"):.6f} |
| Median test turnover | {d("test_turnover_median"):.6f} |
| Median primary benchmark excess return pct | {d("primary_benchmark_excess_return_pct_median"):.6f} |
| Median primary benchmark excess Sharpe | {d("primary_benchmark_excess_sharpe_median"):.6f} |
| Primary benchmark outperform Sharpe rate | {d("primary_benchmark_outperform_sharpe_rate"):.6f} |

## Changed Runs

Only `{decision["changed_test_sharpe_runs"]}` of `{decision["run_count"]}` fold/seed runs changed on test Sharpe.

| Fold | Seed | G0 test Sharpe | Reference test Sharpe | Delta | G0 turnover | Reference turnover |
|---|---:|---:|---:|---:|---:|---:|
{changed_rows}

## Worst Fold-Level Median Deltas

| Fold | G0 median test Sharpe | Reference median test Sharpe | Delta | Delta turnover |
|---|---:|---:|---:|---:|
{worst_fold_rows}

## Interpretation

The temporal-robust checkpoint rule mostly selected the same behavioral policy as the original robust rule. It improved test Sharpe in two changed runs, but one of those changes also raised turnover and reduced excess return pct. The central OOS and benchmark-relative metrics were unchanged.

This closes G0 as a useful control, not as a new winning method. A more serious domain-generalization branch would need to affect training itself, for example domain-balanced sampling or a REx-style reward-dispersion penalty. Generic state compression remains deferred.
"""
    path.write_text(text, encoding="utf-8")


def _write_figures(
    output_dir: Path,
    *,
    summary_delta: pd.DataFrame,
    run_delta: pd.DataFrame,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_metrics = [
        "test_sharpe_median",
        "test_return_pct_median",
        "test_turnover_median",
        "primary_benchmark_excess_return_pct_median",
        "primary_benchmark_excess_sharpe_median",
        "primary_benchmark_outperform_sharpe_rate",
    ]
    summary_plot = (
        summary_delta[summary_delta["metric"].isin(summary_metrics)]
        .set_index("metric")
        .loc[summary_metrics]
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#4c78a8" if value >= 0 else "#d62728" for value in summary_plot["delta"]]
    ax.barh(summary_plot["metric"], summary_plot["delta"], color=colors)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("G0 minus reference summary deltas")
    ax.set_xlabel("Delta")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures_dir / "01_g0_summary_deltas.png", dpi=180)
    fig.savefig(figures_dir / "01_g0_summary_deltas.svg")
    plt.close(fig)

    changed = run_delta[
        pd.to_numeric(run_delta["delta_test_sharpe"], errors="coerce").abs().fillna(0.0) > 1e-9
    ].copy()
    if not changed.empty:
        changed["run_label"] = changed["fold_id"].astype(str) + " seed " + changed["seed"].astype(str)
        x = np.arange(len(changed))
        fig, ax1 = plt.subplots(figsize=(8, 4.2))
        ax1.bar(x - 0.18, changed["delta_test_sharpe"], width=0.36, label="test Sharpe", color="#4c78a8")
        ax1.set_ylabel("Delta test Sharpe")
        ax1.axhline(0.0, color="#333333", linewidth=0.8)
        ax2 = ax1.twinx()
        ax2.bar(x + 0.18, changed["delta_test_turnover"], width=0.36, label="turnover", color="#f58518")
        ax2.set_ylabel("Delta turnover")
        ax1.set_xticks(x, changed["run_label"], rotation=0)
        ax1.set_title("G0 changed runs versus reference")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
        fig.tight_layout()
        fig.savefig(figures_dir / "02_g0_changed_runs.png", dpi=180)
        fig.savefig(figures_dir / "02_g0_changed_runs.svg")
        plt.close(fig)


def run_analysis(g0_dir: Path, reference_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_delta = _summary_delta(g0_dir, reference_dir)
    run_delta = _run_level_delta(g0_dir, reference_dir)
    fold_delta = _fold_median_delta(run_delta)
    benchmark_delta = _primary_benchmark_delta(g0_dir, reference_dir)
    decision = _decision_payload(summary_delta, run_delta, benchmark_delta)

    summary_delta.to_csv(output_dir / "g0_vs_reference_summary_delta.csv", index=False)
    run_delta.to_csv(output_dir / "g0_vs_reference_run_level_delta.csv", index=False)
    fold_delta.to_csv(output_dir / "g0_vs_reference_fold_median_delta.csv", index=False)
    benchmark_delta.to_csv(output_dir / "g0_vs_reference_primary_benchmark_delta.csv", index=False)
    (output_dir / "g0_decision_report.json").write_text(
        json.dumps(_json_safe(decision), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(
        output_dir / "SSL_DOMAIN_G0_RESULT_ANALYSIS.md",
        g0_dir=g0_dir,
        reference_dir=reference_dir,
        decision=decision,
        summary_delta=summary_delta,
        run_delta=run_delta,
        fold_delta=fold_delta,
    )
    _write_figures(output_dir, summary_delta=summary_delta, run_delta=run_delta)
    return decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SSL/domain-generalization G0 results.")
    parser.add_argument("--g0-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    decision = run_analysis(Path(args.g0_dir), Path(args.reference_dir), Path(args.output_dir))
    print(json.dumps(_json_safe(decision), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
