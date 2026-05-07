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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "SSL Domain Generalization" / "g1_parallel_batch" / "g1_parallel_batch_manifest.csv"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "Latent Actions" / "research_outputs_phase2_base_macro_teacher"
DEFAULT_G0_DIR = PROJECT_ROOT / "SSL Domain Generalization" / "research_outputs_ssl_domain_g0_base_macro_temporal_robust_selection"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "SSL Domain Generalization" / "research_outputs_ssl_domain_g1_batch_analysis"


SUMMARY_FILE = "corrected_walk_forward_summary_with_primary_benchmark.csv"
RUN_FILE = "walk_forward_results.csv"
NEGATIVE_CONTROL_VARIANTS = {
    "g1c_abs_reverse_negative_control",
    "g1f_vol_reverse_negative_control",
}


def _read_summary(output_dir: Path) -> pd.Series:
    path = output_dir / SUMMARY_FILE
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty summary: {path}")
    return df.iloc[0]


def _read_runs(output_dir: Path) -> pd.DataFrame:
    path = output_dir / RUN_FILE
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _delta(g1_value: Any, ref_value: Any) -> float:
    return float(pd.to_numeric(pd.Series([g1_value]), errors="coerce").iloc[0]) - float(
        pd.to_numeric(pd.Series([ref_value]), errors="coerce").iloc[0]
    )


def _variant_status(output_dir: Path) -> str:
    required = [
        SUMMARY_FILE,
        "walk_forward_daily_test_returns.csv",
        "benchmark_run_level_metrics.csv",
        RUN_FILE,
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    return "complete" if not missing else "missing:" + ",".join(missing)


def analyze_batch(
    *,
    manifest_path: Path,
    reference_dir: Path,
    g0_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    reference_summary = _read_summary(reference_dir)
    g0_summary = _read_summary(g0_dir)

    rows: list[dict[str, Any]] = []
    run_rows: list[pd.DataFrame] = []
    for _, item in manifest.iterrows():
        variant_id = str(item["variant_id"])
        outdir = PROJECT_ROOT / str(item["output_dir"])
        status = _variant_status(outdir)
        base_row = item.to_dict()
        base_row["status"] = status
        base_row["is_negative_control_variant"] = variant_id in NEGATIVE_CONTROL_VARIANTS
        if status != "complete":
            base_row["provisional_gate_pass"] = False
            rows.append(base_row)
            continue

        summary = _read_summary(outdir)
        metrics = {
            "test_sharpe_median": summary.get("test_sharpe_median"),
            "test_return_pct_median": summary.get("test_return_pct_median"),
            "test_max_drawdown_median": summary.get("test_max_drawdown_median"),
            "test_turnover_median": summary.get("test_turnover_median"),
            "primary_benchmark_excess_return_pct_median": summary.get(
                "primary_benchmark_excess_return_pct_median"
            ),
            "primary_benchmark_excess_sharpe_median": summary.get(
                "primary_benchmark_excess_sharpe_median"
            ),
            "primary_benchmark_outperform_return_rate": summary.get(
                "primary_benchmark_outperform_return_rate"
            ),
            "primary_benchmark_outperform_sharpe_rate": summary.get(
                "primary_benchmark_outperform_sharpe_rate"
            ),
        }
        for metric, value in metrics.items():
            base_row[metric] = value
            base_row[f"delta_vs_reference_{metric}"] = _delta(value, reference_summary.get(metric))
            base_row[f"delta_vs_g0_{metric}"] = _delta(value, g0_summary.get(metric))
        base_row["gate_score"] = (
            1.00 * base_row["delta_vs_reference_primary_benchmark_excess_sharpe_median"]
            + 0.35 * base_row["delta_vs_reference_primary_benchmark_excess_return_pct_median"]
            + 0.25 * base_row["delta_vs_reference_test_sharpe_median"]
            - 0.05 * max(0.0, base_row["delta_vs_reference_test_turnover_median"])
        )
        base_row["provisional_gate_pass"] = bool(
            not base_row["is_negative_control_variant"]
            and base_row["delta_vs_reference_test_sharpe_median"] > 0.0
            and base_row["delta_vs_reference_primary_benchmark_excess_sharpe_median"] > 0.0
            and base_row["delta_vs_reference_primary_benchmark_excess_return_pct_median"] > 0.0
            and base_row["gate_score"] > 0.0
        )
        rows.append(base_row)

        runs = _read_runs(outdir)
        runs["variant_id"] = variant_id
        runs["experiment_name"] = item["experiment_name"]
        run_rows.append(runs)

    ranking = pd.DataFrame(rows)
    if "gate_score" not in ranking.columns:
        ranking["gate_score"] = np.nan
    ranking = ranking.sort_values(["status", "gate_score"], ascending=[True, False], na_position="last")
    ranking.to_csv(output_dir / "g1_batch_ranking.csv", index=False)
    if run_rows:
        pd.concat(run_rows, ignore_index=True).to_csv(output_dir / "g1_batch_run_level_results.csv", index=False)

    complete = ranking[ranking["status"].eq("complete")].copy()
    eligible = complete[~complete["is_negative_control_variant"]].copy()
    pass_variants = complete[complete["provisional_gate_pass"]].copy()
    best_complete = None if complete.empty else str(complete.iloc[0]["variant_id"])
    best_eligible = None if eligible.empty else str(eligible.iloc[0]["variant_id"])
    decision = {
        "complete_variants": int(len(complete)),
        "missing_variants": int((~ranking["status"].eq("complete")).sum()),
        "best_complete_variant_by_gate_score": best_complete,
        "best_non_negative_control_by_gate_score": best_eligible,
        "provisional_gate_pass_variants": pass_variants["variant_id"].astype(str).tolist(),
        "negative_controls": sorted(NEGATIVE_CONTROL_VARIANTS),
        "negative_control_warning": (
            "Best complete variant is a negative control; treat this as a falsification/noise warning, "
            "not as evidence for the G1 hypothesis."
            if best_complete in NEGATIVE_CONTROL_VARIANTS
            else ""
        ),
        "gate_rule": (
            "A variant needs positive benchmark-excess Sharpe and return deltas versus reference, "
            "positive test Sharpe delta, stable turnover/drawdown, and separation from reverse negative controls."
        ),
    }
    (output_dir / "g1_batch_decision_stub.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "G1_BATCH_ANALYSIS.md", ranking, decision)
    _write_figures(output_dir, ranking)
    return decision


def _write_markdown(path: Path, ranking: pd.DataFrame, decision: dict[str, Any]) -> None:
    if ranking.empty:
        table = "No variants found."
    else:
        cols = [
            "variant_id",
            "status",
            "is_negative_control_variant",
            "provisional_gate_pass",
            "derived_domain",
            "mode",
            "delta_vs_reference_test_sharpe_median",
            "delta_vs_reference_primary_benchmark_excess_sharpe_median",
            "delta_vs_reference_primary_benchmark_excess_return_pct_median",
            "delta_vs_reference_test_turnover_median",
            "gate_score",
        ]
        existing = [col for col in cols if col in ranking.columns]
        table_frame = ranking[existing].copy()
        header = "| " + " | ".join(existing) + " |"
        separator = "| " + " | ".join("---" for _ in existing) + " |"
        body = "\n".join(
            "| " + " | ".join(str(row[col]) for col in existing) + " |"
            for row in table_frame.to_dict(orient="records")
        )
        table = "\n".join([header, separator, body])
    text = f"""# SSL Domain G1 Batch Analysis

Status: generated by `analyze_ssl_domain_g1_batch.py`.

Complete variants: `{decision['complete_variants']}`
Missing variants: `{decision['missing_variants']}`

Best complete variant by provisional gate score: `{decision['best_complete_variant_by_gate_score']}`
Best non-negative-control variant by provisional gate score: `{decision['best_non_negative_control_by_gate_score']}`
Provisional gate-pass variants: `{decision['provisional_gate_pass_variants']}`

Negative-control warning: {decision['negative_control_warning'] or 'none'}

Gate rule:

{decision['gate_rule']}

## Ranking

{table}
"""
    path.write_text(text, encoding="utf-8")


def _write_figures(output_dir: Path, ranking: pd.DataFrame) -> None:
    complete = ranking[ranking["status"].eq("complete")].copy()
    if complete.empty:
        return
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metric = "delta_vs_reference_primary_benchmark_excess_sharpe_median"
    complete = complete.sort_values(metric)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4c78a8" if value >= 0 else "#d62728" for value in complete[metric]]
    ax.barh(complete["variant_id"], complete[metric], color=colors)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("G1 variants: benchmark-excess Sharpe delta vs reference")
    ax.set_xlabel("Delta")
    fig.tight_layout()
    fig.savefig(figures_dir / "01_g1_excess_sharpe_delta.png", dpi=180)
    fig.savefig(figures_dir / "01_g1_excess_sharpe_delta.svg")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed SSL/domain G1 parallel batch outputs.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--g0-dir", default=str(DEFAULT_G0_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    decision = analyze_batch(
        manifest_path=Path(args.manifest),
        reference_dir=Path(args.reference_dir),
        g0_dir=Path(args.g0_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
