from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SSL_ROOT = PROJECT_ROOT / "SSL Domain Generalization"
DEFAULT_MANIFEST = SSL_ROOT / "g2_parallel_batch" / "g2_parallel_batch_manifest.csv"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "Latent Actions" / "research_outputs_phase2_base_macro_teacher"
DEFAULT_OUTPUT_DIR = SSL_ROOT / "research_outputs_ssl_domain_g2_batch_analysis"

SUMMARY_FILE = "corrected_walk_forward_summary_with_primary_benchmark.csv"
RUN_FILE = "walk_forward_results.csv"
BENCHMARK_FILE = "benchmark_run_level_metrics.csv"
DAILY_FILE = "walk_forward_daily_test_returns.csv"
REGIME_FILE = "regime_summary_by_feature_set.csv"
NEGATIVE_CONTROL_VARIANT = "g2g_overregularized_negative_control"

SUMMARY_METRICS = [
    "test_sharpe_median",
    "test_return_pct_median",
    "test_max_drawdown_median",
    "test_turnover_median",
    "primary_benchmark_excess_return_pct_median",
    "primary_benchmark_excess_sharpe_median",
    "primary_benchmark_regret_return_pct_median",
    "primary_benchmark_regret_sharpe_median",
    "primary_benchmark_outperform_return_rate",
    "primary_benchmark_outperform_sharpe_rate",
    "primary_benchmark_hit_rate_median",
]

RUN_METRICS = [
    "test_sharpe",
    "test_return_pct",
    "test_max_drawdown",
    "test_turnover",
    "validation_sharpe",
    "validation_return_pct",
    "generalization_ratio",
    "retention_ratio",
]

BENCHMARK_RUN_METRICS = [
    "excess_return_pct",
    "excess_sharpe",
    "outperformed_benchmark_on_return",
    "outperformed_benchmark_on_sharpe",
    "hit_rate_vs_benchmark",
]

REGIME_DELTA_METRICS = [
    "mean_daily_return_median",
    "max_drawdown_median",
    "turnover_median",
    "hit_rate_median",
    "excess_return_vs_benchmark_median",
]


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return default if pd.isna(out) else float(out)
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if pd.isna(value):
        return None
    return value


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    number = _to_float(value)
    if math.isfinite(number):
        return f"{number:.4f}"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _markdown_table(frame: pd.DataFrame, cols: Sequence[str]) -> str:
    existing = [col for col in cols if col in frame.columns]
    if not existing:
        return ""
    out = frame[existing].copy()
    header = "| " + " | ".join(existing) + " |"
    separator = "| " + " | ".join("---" for _ in existing) + " |"
    body = "\n".join(
        "| " + " | ".join(_format_value(row[col]) for col in existing) + " |"
        for row in out.to_dict(orient="records")
    )
    return "\n".join([header, separator, body])


def _variant_status(output_dir: Path) -> str:
    required = [SUMMARY_FILE, RUN_FILE, BENCHMARK_FILE, DAILY_FILE]
    missing = [name for name in required if not (output_dir / name).exists()]
    return "complete" if not missing else "missing:" + ",".join(missing)


def _read_summary(output_dir: Path) -> pd.Series:
    frame = pd.read_csv(output_dir / SUMMARY_FILE)
    if frame.empty:
        raise ValueError(f"Empty summary: {output_dir / SUMMARY_FILE}")
    return frame.iloc[0]


def _read_runs(output_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(output_dir / RUN_FILE)
    return frame[frame["feature_set"].eq("base_macro")].copy()


def _read_primary_benchmark_runs(output_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(output_dir / BENCHMARK_FILE)
    return frame[frame["is_primary_benchmark"].astype(bool)].copy()


def _read_regime(output_dir: Path) -> pd.DataFrame:
    path = output_dir / REGIME_FILE
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "feature_set" in frame.columns:
        frame = frame[frame["feature_set"].eq("base_macro")].copy()
    return frame


def _delta(value: Any, reference: Any) -> float:
    return _to_float(value) - _to_float(reference)


def _paired_delta(candidate: pd.DataFrame, reference: pd.DataFrame, metric: str) -> dict[str, float]:
    keys = ["fold_id", "seed"]
    if metric not in candidate.columns or metric not in reference.columns:
        return {
            f"paired_delta_{metric}_median": float("nan"),
            f"paired_delta_{metric}_mean": float("nan"),
            f"paired_win_rate_{metric}": float("nan"),
        }
    merged = candidate[keys + [metric]].merge(
        reference[keys + [metric]],
        on=keys,
        suffixes=("_candidate", "_reference"),
    )
    if merged.empty:
        return {
            f"paired_delta_{metric}_median": float("nan"),
            f"paired_delta_{metric}_mean": float("nan"),
            f"paired_win_rate_{metric}": float("nan"),
        }
    deltas = merged[f"{metric}_candidate"].astype(float) - merged[f"{metric}_reference"].astype(float)
    return {
        f"paired_delta_{metric}_median": float(deltas.median()),
        f"paired_delta_{metric}_mean": float(deltas.mean()),
        f"paired_win_rate_{metric}": float((deltas > 0.0).mean()),
    }


def _fold_positive_count(candidate: pd.DataFrame, reference: pd.DataFrame, metric: str) -> int:
    keys = ["fold_id", "seed"]
    if metric not in candidate.columns or metric not in reference.columns:
        return 0
    merged = candidate[keys + [metric]].merge(
        reference[keys + [metric]],
        on=keys,
        suffixes=("_candidate", "_reference"),
    )
    if merged.empty:
        return 0
    merged["delta"] = merged[f"{metric}_candidate"].astype(float) - merged[f"{metric}_reference"].astype(float)
    return int((merged.groupby("fold_id")["delta"].median() > 0.0).sum())


def _seed_matched_reference_delta(candidate: pd.DataFrame, reference: pd.DataFrame, metric: str) -> dict[str, float]:
    if metric not in candidate.columns or metric not in reference.columns:
        return {
            f"seed_matched_reference_{metric}_median": float("nan"),
            f"delta_vs_seed_matched_reference_{metric}_median": float("nan"),
        }
    seeds = sorted(candidate["seed"].dropna().unique().tolist())
    matched = reference[reference["seed"].isin(seeds)].copy()
    if matched.empty:
        return {
            f"seed_matched_reference_{metric}_median": float("nan"),
            f"delta_vs_seed_matched_reference_{metric}_median": float("nan"),
        }
    candidate_median = float(candidate[metric].median())
    reference_median = float(matched[metric].median())
    return {
        f"seed_matched_reference_{metric}_median": reference_median,
        f"delta_vs_seed_matched_reference_{metric}_median": candidate_median - reference_median,
    }


def _gate_pass(row: dict[str, Any]) -> bool:
    if row["is_negative_control_variant"]:
        return False
    if row.get("turnover_only_warning"):
        return False
    if row.get("near_inactive_warning"):
        return False
    if row["delta_vs_reference_test_sharpe_median"] <= 0.0:
        return False
    if row["delta_vs_reference_primary_benchmark_excess_sharpe_median"] <= 0.0:
        return False
    if row["delta_vs_reference_primary_benchmark_excess_return_pct_median"] <= 0.0:
        return False
    if row["delta_vs_reference_test_max_drawdown_median"] < -0.01:
        return False
    if row["paired_win_rate_test_sharpe"] < 0.55:
        return False
    return True


def _regime_delta_frame(variant_id: str, candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    if candidate.empty or reference.empty or "regime_label_exogenous" not in candidate.columns:
        return pd.DataFrame()
    merge_cols = ["regime_label_exogenous"]
    keep = merge_cols + [col for col in REGIME_DELTA_METRICS if col in candidate.columns and col in reference.columns]
    if len(keep) <= 1:
        return pd.DataFrame()
    merged = candidate[keep].merge(
        reference[keep],
        on=merge_cols,
        suffixes=("_candidate", "_reference"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged.insert(0, "variant_id", variant_id)
    for metric in [col for col in keep if col not in merge_cols]:
        merged[f"delta_{metric}"] = merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
    return merged


def _regime_summary(regime_delta: pd.DataFrame) -> dict[str, Any]:
    if regime_delta.empty:
        return {
            "regime_count": 0,
            "regimes_with_positive_excess_return_delta": 0,
            "mean_regime_excess_return_delta": float("nan"),
            "worst_regime_excess_return_delta": float("nan"),
            "worst_regime_by_excess_return": "",
            "mean_regime_hit_rate_delta": float("nan"),
            "worst_regime_hit_rate_delta": float("nan"),
            "worst_regime_max_drawdown_delta": float("nan"),
            "mean_regime_turnover_delta": float("nan"),
        }
    excess_col = "delta_excess_return_vs_benchmark_median"
    hit_col = "delta_hit_rate_median"
    drawdown_col = "delta_max_drawdown_median"
    turnover_col = "delta_turnover_median"
    out: dict[str, Any] = {"regime_count": int(len(regime_delta))}
    if excess_col in regime_delta.columns:
        excess = pd.to_numeric(regime_delta[excess_col], errors="coerce")
        worst_idx = excess.idxmin()
        out["regimes_with_positive_excess_return_delta"] = int((excess > 0.0).sum())
        out["mean_regime_excess_return_delta"] = float(excess.mean())
        out["worst_regime_excess_return_delta"] = float(excess.min())
        out["worst_regime_by_excess_return"] = str(regime_delta.loc[worst_idx, "regime_label_exogenous"])
    if hit_col in regime_delta.columns:
        hit = pd.to_numeric(regime_delta[hit_col], errors="coerce")
        out["mean_regime_hit_rate_delta"] = float(hit.mean())
        out["worst_regime_hit_rate_delta"] = float(hit.min())
    if drawdown_col in regime_delta.columns:
        drawdown = pd.to_numeric(regime_delta[drawdown_col], errors="coerce")
        out["worst_regime_max_drawdown_delta"] = float(drawdown.min())
    if turnover_col in regime_delta.columns:
        turnover = pd.to_numeric(regime_delta[turnover_col], errors="coerce")
        out["mean_regime_turnover_delta"] = float(turnover.mean())
    return out


def _decision_label(row: pd.Series) -> str:
    if bool(row.get("is_negative_control_variant", False)):
        return "negative_control_not_promotable"
    if bool(row.get("screening_pass_candidate", False)):
        return "screening_pass_candidate"
    positive_core = sum(
        _to_float(row.get(metric)) > 0.0
        for metric in [
            "delta_vs_reference_test_sharpe_median",
            "delta_vs_reference_test_return_pct_median",
            "delta_vs_reference_primary_benchmark_excess_return_pct_median",
            "delta_vs_reference_primary_benchmark_excess_sharpe_median",
        ]
    )
    if positive_core >= 2 or _to_float(row.get("gate_score")) > 0.0:
        return "mixed_informative"
    return "screening_fail"


def _variant_title(variant_id: str) -> str:
    match = re.search(r"g2([a-z])_", variant_id)
    letter = match.group(1).upper() if match else "?"
    return f"G2{letter} {variant_id}"


def _variant_note_text(row: pd.Series) -> str:
    label = str(row["decision_label"])
    status_sentence = {
        "screening_pass_candidate": "вариант проходит one-seed screening gate, но требует rerun на 3 seeds.",
        "mixed_informative": "вариант информативен, но не проходит screening gate.",
        "screening_fail": "вариант не проходит screening gate.",
        "negative_control_not_promotable": "это negative control; он не может быть promoted даже при сильных headline metrics.",
    }.get(label, "вариант требует ручной проверки.")
    rows = pd.DataFrame(
        [
            ["Gate score", row.get("gate_score")],
            ["Test Sharpe delta", row.get("delta_vs_reference_test_sharpe_median")],
            ["Test return delta", row.get("delta_vs_reference_test_return_pct_median")],
            ["Benchmark-excess Sharpe delta", row.get("delta_vs_reference_primary_benchmark_excess_sharpe_median")],
            ["Benchmark-excess return delta", row.get("delta_vs_reference_primary_benchmark_excess_return_pct_median")],
            ["Turnover delta", row.get("delta_vs_reference_test_turnover_median")],
            ["Paired Sharpe win rate", row.get("paired_win_rate_test_sharpe")],
            ["Folds with positive Sharpe delta", row.get("folds_with_positive_test_sharpe_delta")],
            ["Worst regime excess-return delta", row.get("worst_regime_excess_return_delta")],
        ],
        columns=["Metric", "Value"],
    )
    warnings = []
    if bool(row.get("turnover_only_warning", False)):
        warnings.append("turnover-only warning")
    if bool(row.get("near_inactive_warning", False)):
        warnings.append("near-inactive warning")
    if not bool(row.get("beats_negative_control_gate", True)):
        warnings.append("negative-control comparison warning")
    warning_text = ", ".join(warnings) if warnings else "none"
    return f"""# {_variant_title(str(row['variant_id']))} Critical Analysis

Decision label: `{label}`.

Короткий вывод: {status_sentence}

Mechanism: `{row.get('mechanism')}`.

Hypothesis:

{row.get('hypothesis')}

## Key Metrics

{_markdown_table(rows, ['Metric', 'Value'])}

## Interpretation

- Warning flags: `{warning_text}`.
- Turnover interpretation: median test turnover delta vs reference is `{_format_value(row.get('delta_vs_reference_test_turnover_median'))}`. Lower turnover counts as useful only if Sharpe and benchmark-relative metrics also improve.
- Regime interpretation: `{row.get('regimes_with_positive_excess_return_delta')}` regimes have positive benchmark-excess return delta; worst regime is `{row.get('worst_regime_by_excess_return')}`.
- Rerun rule: rerun на 3 seeds нужен только для `screening_pass_candidate`.
"""


def _write_variant_notes(output_dir: Path, ranking: pd.DataFrame) -> None:
    for _, row in ranking.iterrows():
        variant_id = str(row["variant_id"])
        match = re.search(r"g2([a-z])_", variant_id)
        suffix = match.group(1).upper() if match else variant_id.upper()
        (output_dir / f"G2{suffix}_CRITICAL_ANALYSIS.md").write_text(
            _variant_note_text(row),
            encoding="utf-8",
        )


def _write_batch_analysis(path: Path, ranking: pd.DataFrame, decision: dict[str, Any]) -> None:
    compact_cols = [
        "rank_by_gate_score",
        "variant_id",
        "mechanism",
        "decision_label",
        "gate_score",
        "delta_vs_reference_test_sharpe_median",
        "delta_vs_reference_test_return_pct_median",
        "delta_vs_reference_primary_benchmark_excess_return_pct_median",
        "delta_vs_reference_primary_benchmark_excess_sharpe_median",
        "delta_vs_reference_test_turnover_median",
        "paired_win_rate_test_sharpe",
        "folds_with_positive_test_sharpe_delta",
        "turnover_only_warning",
        "near_inactive_warning",
    ]
    text = f"""# SSL Domain G2 Batch Analysis

Status: generated by `analyze_ssl_domain_g2_batch.py`.

Complete variants: `{decision['complete_variants']}`
Missing variants: `{decision['missing_variants']}`
Screening decision: `{decision['screening_decision']}`
Best variant by gate score: `{decision['best_variant_by_gate_score']}`
Screening pass candidates: `{decision['screening_pass_candidates']}`

Important: this is one-seed screening, not a final pass/fail claim for production.

## Gate Rule

{decision['gate_rule']}

## Compact Ranking

{_markdown_table(ranking, compact_cols)}
"""
    path.write_text(text, encoding="utf-8")


def _write_root_reports(ranking: pd.DataFrame, decision: dict[str, Any]) -> None:
    best = ranking.iloc[0] if not ranking.empty else pd.Series(dtype=object)
    pass_candidates = decision["screening_pass_candidates"]
    result_text = f"""# SSL Domain G2 Batch Result Analysis

Статус: one-seed G2 screening batch completed.

## Короткий Вывод

Screening decision: `{decision['screening_decision']}`.

Лучший вариант по gate score: `{decision['best_variant_by_gate_score']}`.

Screening pass candidates: `{pass_candidates}`.

Важно: G2 был запущен с одним seed, поэтому даже лучший вариант не может быть
финальным pass без rerun на 3 seeds.

## Что Проверялось

G2 проверял ортогональную к G1 гипотезу: может ли conservative action
regularization улучшить OOS trading quality PPO, а не просто снизить turnover.

Проверенные механизмы:

- mild/strong turnover penalty;
- smoothness penalty;
- turnover + smoothness;
- concentration/max-weight penalty;
- combined conservative penalty;
- overregularized negative control.

## Compact Ranking

{_markdown_table(ranking, [
    'rank_by_gate_score',
    'variant_id',
    'mechanism',
    'decision_label',
    'gate_score',
    'delta_vs_reference_test_sharpe_median',
    'delta_vs_reference_primary_benchmark_excess_return_pct_median',
    'delta_vs_reference_primary_benchmark_excess_sharpe_median',
    'delta_vs_reference_test_turnover_median',
    'paired_win_rate_test_sharpe',
])}

## Main Interpretation

Best headline:

- variant: `{best.get('variant_id', '')}`;
- decision label: `{best.get('decision_label', '')}`;
- test Sharpe delta: `{_format_value(best.get('delta_vs_reference_test_sharpe_median'))}`;
- benchmark-excess return delta: `{_format_value(best.get('delta_vs_reference_primary_benchmark_excess_return_pct_median'))}`;
- benchmark-excess Sharpe delta: `{_format_value(best.get('delta_vs_reference_primary_benchmark_excess_sharpe_median'))}`;
- turnover delta: `{_format_value(best.get('delta_vs_reference_test_turnover_median'))}`.

Lower turnover is not counted as success unless benchmark-relative Sharpe/return
and frozen-test Sharpe improve together.

## Figures

- `figures/08_g2_gate_and_core_deltas.png`
- `figures/09_g2_turnover_vs_sharpe_delta.png`
- `figures/10_g2_regularization_dose_response.png`
- `figures/11_g2_benchmark_excess_deltas.png`
- `figures/12_g2_regime_robustness.png`
"""
    (SSL_ROOT / "SSL_DOMAIN_G2_BATCH_RESULT_ANALYSIS.md").write_text(result_text, encoding="utf-8")

    closeout_text = f"""# SSL Domain G2 Batch Closeout

Финальный статус G2 screening: `{decision['screening_decision']}`.

## Decision Rule

G2 candidate должен улучшить real OOS trading quality:

- median frozen-test Sharpe;
- benchmark-excess Sharpe;
- benchmark-excess return;
- drawdown без material deterioration;
- paired/fold support;
- no turnover-only success;
- no near-inactive policy;
- no promotion of overregularized negative control.

## Final Decision

Screening pass candidates: `{pass_candidates}`.

Если список пуст, G2 не дает основания для G3. Если список не пуст, следующий
шаг - rerun кандидатов на 3 seeds, и только потом решение о G3.

## Negative Control

Negative control: `{NEGATIVE_CONTROL_VARIANT}`.

Negative control top warning: `{decision['negative_control_top_warning']}`.

Если negative control оказывается самым сильным вариантом по gate score, это
означает, что apparent improvement может быть артефактом over-regularization.

## G3 Consequence

G3 не готовится автоматически. G3 допустим только после 3-seed подтверждения
хотя бы одного non-negative-control G2 candidate.

Artifacts:

- `research_outputs_ssl_domain_g2_batch_analysis/g2_summary.csv`
- `research_outputs_ssl_domain_g2_batch_analysis/g2_decision.json`
- `research_outputs_ssl_domain_g2_batch_analysis/G2_BATCH_ANALYSIS.md`
- `research_outputs_ssl_domain_g2_batch_analysis/G2A_CRITICAL_ANALYSIS.md` ... `G2G_CRITICAL_ANALYSIS.md`
"""
    (SSL_ROOT / "SSL_DOMAIN_G2_BATCH_CLOSEOUT.md").write_text(closeout_text, encoding="utf-8")


def analyze_batch(
    *,
    manifest_path: Path,
    reference_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    reference_summary = _read_summary(reference_dir)
    reference_runs = _read_runs(reference_dir)
    reference_bench = _read_primary_benchmark_runs(reference_dir)
    reference_regime = _read_regime(reference_dir)

    rows: list[dict[str, Any]] = []
    complete_run_frames: list[pd.DataFrame] = []
    regime_frames: list[pd.DataFrame] = []

    for _, item in manifest.iterrows():
        variant_id = str(item["variant_id"])
        outdir = PROJECT_ROOT / str(item["output_dir"])
        status = _variant_status(outdir)
        row = item.to_dict()
        row["status"] = status
        row["is_negative_control_variant"] = variant_id == NEGATIVE_CONTROL_VARIANT
        if status != "complete":
            row["screening_gate_pass_before_control"] = False
            rows.append(row)
            continue

        summary = _read_summary(outdir)
        runs = _read_runs(outdir)
        bench = _read_primary_benchmark_runs(outdir)
        regime = _read_regime(outdir)

        for metric in SUMMARY_METRICS:
            row[metric] = _to_float(summary.get(metric))
            row[f"delta_vs_reference_{metric}"] = _delta(summary.get(metric), reference_summary.get(metric))

        for metric in RUN_METRICS:
            if metric in runs.columns:
                row[f"{metric}_mean"] = float(runs[metric].mean())
                row[f"{metric}_median_from_runs"] = float(runs[metric].median())
                row[f"delta_vs_reference_{metric}_mean"] = float(runs[metric].mean() - reference_runs[metric].mean())
                row.update(_seed_matched_reference_delta(runs, reference_runs, metric))
            row.update(_paired_delta(runs, reference_runs, metric))
            row[f"folds_with_positive_{metric}_delta"] = _fold_positive_count(runs, reference_runs, metric)

        for metric in BENCHMARK_RUN_METRICS:
            row.update({f"primary_{key}": value for key, value in _paired_delta(bench, reference_bench, metric).items()})

        regime_delta = _regime_delta_frame(variant_id, regime, reference_regime)
        if not regime_delta.empty:
            regime_frames.append(regime_delta)
        row.update(_regime_summary(regime_delta))

        row["gate_score"] = (
            1.00 * row["delta_vs_reference_primary_benchmark_excess_sharpe_median"]
            + 0.35 * row["delta_vs_reference_primary_benchmark_excess_return_pct_median"]
            + 0.25 * row["delta_vs_reference_test_sharpe_median"]
            - 0.05 * max(0.0, row["delta_vs_reference_test_turnover_median"])
        )
        row["turnover_only_warning"] = bool(
            row["delta_vs_reference_test_turnover_median"] < 0.0
            and row["delta_vs_reference_test_sharpe_median"] <= 0.0
            and row["delta_vs_reference_primary_benchmark_excess_sharpe_median"] <= 0.0
        )
        row["near_inactive_warning"] = bool(
            row["test_turnover_median"] < 0.25 * _to_float(reference_summary.get("test_turnover_median"))
        )
        row["screening_gate_pass_before_control"] = _gate_pass(row)
        rows.append(row)

        runs = runs.copy()
        runs["variant_id"] = variant_id
        complete_run_frames.append(runs)

    ranking = pd.DataFrame(rows)
    if "gate_score" not in ranking.columns:
        ranking["gate_score"] = np.nan
    ranking = ranking.sort_values(["status", "gate_score"], ascending=[True, False], na_position="last").reset_index(drop=True)

    complete = ranking[ranking["status"].eq("complete")].copy()
    negative_control_gate = float("nan")
    if not complete.empty:
        negative_rows = complete[complete["is_negative_control_variant"].astype(bool)]
        if not negative_rows.empty:
            negative_control_gate = float(negative_rows["gate_score"].max())
    ranking["beats_negative_control_gate"] = ranking.apply(
        lambda row: bool(
            row["is_negative_control_variant"]
            or not math.isfinite(negative_control_gate)
            or _to_float(row.get("gate_score")) > negative_control_gate
        ),
        axis=1,
    )
    ranking["screening_pass_candidate"] = ranking.apply(
        lambda row: bool(row.get("screening_gate_pass_before_control", False) and row.get("beats_negative_control_gate", True)),
        axis=1,
    )
    ranking["decision_label"] = ranking.apply(_decision_label, axis=1)
    ranking = ranking.sort_values(["status", "gate_score"], ascending=[True, False], na_position="last").reset_index(drop=True)
    ranking.insert(0, "rank_by_gate_score", range(1, len(ranking) + 1))

    ranking.to_csv(output_dir / "g2_batch_ranking.csv", index=False)
    ranking.to_csv(output_dir / "g2_summary.csv", index=False)
    if complete_run_frames:
        pd.concat(complete_run_frames, ignore_index=True).to_csv(
            output_dir / "g2_batch_run_level_results.csv",
            index=False,
        )
    if regime_frames:
        pd.concat(regime_frames, ignore_index=True).to_csv(
            output_dir / "g2_regime_delta_table.csv",
            index=False,
        )

    complete = ranking[ranking["status"].eq("complete")].copy()
    pass_variants = complete[complete["screening_pass_candidate"]].copy()
    mixed = complete[complete["decision_label"].eq("mixed_informative")].copy()
    best_variant = None if complete.empty else str(complete.iloc[0]["variant_id"])
    best_non_negative = complete[~complete["is_negative_control_variant"].astype(bool)].head(1)
    negative_control_top_warning = bool(best_variant == NEGATIVE_CONTROL_VARIANT)
    if not pass_variants.empty:
        screening_decision = "screening_pass_candidates_require_3_seed_rerun"
    elif not mixed.empty:
        screening_decision = "mixed_informative_no_gate_pass"
    else:
        screening_decision = "screening_fail"

    decision = {
        "branch": "SSL Domain Generalization G2",
        "status": "completed",
        "seed_policy": "one_seed_screening_only",
        "screening_decision": screening_decision,
        "complete_variants": int(len(complete)),
        "missing_variants": int((~ranking["status"].eq("complete")).sum()),
        "best_variant_by_gate_score": best_variant,
        "best_non_negative_control_variant_by_gate_score": None
        if best_non_negative.empty
        else str(best_non_negative.iloc[0]["variant_id"]),
        "negative_control_variant": NEGATIVE_CONTROL_VARIANT,
        "negative_control_gate_score": negative_control_gate,
        "negative_control_top_warning": negative_control_top_warning,
        "screening_pass_candidates": pass_variants["variant_id"].astype(str).tolist(),
        "mixed_informative_variants": mixed["variant_id"].astype(str).tolist(),
        "gate_rule": (
            "One-seed screening candidate requires higher test Sharpe, higher benchmark-excess Sharpe "
            "and return, no material drawdown deterioration, paired/fold support, no inactivity, "
            "no turnover-only warning, and a better gate score than the overregularized negative control."
        ),
        "g3_implication": (
            "Do not prepare G3 unless at least one non-negative-control G2 candidate is confirmed "
            "with a 3-seed rerun."
        ),
    }
    (output_dir / "g2_decision.json").write_text(
        json.dumps(_json_safe(decision), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_batch_analysis(output_dir / "G2_BATCH_ANALYSIS.md", ranking, decision)
    _write_variant_notes(output_dir, ranking)
    _write_root_reports(ranking, decision)
    return decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed SSL/domain G2 action-regularization outputs.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    decision = analyze_batch(
        manifest_path=Path(args.manifest),
        reference_dir=Path(args.reference_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(_json_safe(decision), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
