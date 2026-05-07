from __future__ import annotations

import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "Behavior Interpretability Audit"
DEFAULT_OUTPUT = AUDIT_ROOT / "research_outputs_behavior_interpretability_base_macro"
ZIP_NAME = "joseph_behavior_interpretability_handoff.zip"


JOSEPH_MESSAGE = """Hey Joseph,

Quick update on my main PPO/FinRL project.

The main repo is here:
https://github.com/Sqaard/RL_for_financial_time_series_forecasting

Important note: the newest SSL / Domain Generalization results are not pushed there yet. I will update the repo next, but for now I am sending you the local audit outputs directly.

Setup:
The main project is a Dow 30 portfolio trading pipeline using PPO / FinRL. The strongest current feature set is still base_macro. I tested several robustness branches after the feature-family ablations:

1. G1: training-time domain/stress reward reweighting.
   I tested abs-return stress, volatility stress, downside return stress, drawdown stress, inverse-frequency regimes, plus reverse negative controls.
   Result: fail. Stress-day weighting changed behavior, but did not reliably improve frozen-test risk-adjusted OOS performance.

2. G2: conservative action regularization.
   I tested turnover penalty, smoothness penalty, turnover + smoothness, concentration penalty, full conservative regularization, and an overregularized negative control.
   Result: screening fail. Lower turnover / smoother actions alone did not improve Sharpe, benchmark-relative Sharpe/return, and drawdown together.

So I did not continue to G3. Instead, I moved to an interpretability-first audit: before adding more PPO interventions, I want to understand which recurring behavior primitives explain OOS success and failure.

What I did:
I took the frozen base_macro PPO teacher outputs:
- test actions,
- test observations,
- daily test returns,
- benchmark returns,
- regime/context features.

Then I built rolling state-action-return windows and clustered behavior windows into 6 behavior primitives. This is intentionally different from simple one-day action tokenization: the primitive is a short behavior segment, not just one action code.

The methodology is:
1. Discover behavior primitives from rolling state-action-return windows.
2. Compute direct portfolio/action diagnostics for each primitive.
3. Align primitives with regimes, concepts, and synthetic finance-style strategies.
4. Identify good primitives and failure primitives.
5. Use those primitive IDs later for hidden-state probes / TCAV / RBSA-style labeling.

Main files:
- BEHAVIOR_INTERPRETABILITY_AUDIT.md
  Short summary of the audit and the current primitive leaderboard.

- behavior_primitive_summary.csv
  One row per primitive. Contains primitive type, activation share, excess return, excess Sharpe, hit rate, drawdown, turnover, concentration, benchmark deviation, dominant regime, and reliability score.

- behavior_primitive_assignments.csv
  Full assignment table. One row per behavior window, keyed by run_key / fold_id / seed / date. This is the canonical primitive_id label table.

- primitive_daily_labels_compact.csv
  Smaller version of the assignment table for easier reading.

- primitive_lookup_for_joseph.csv
  Compact cheat sheet for each primitive: performance, action behavior, regime/concept/style summary.

- primitive_examples_for_joseph.csv
  Best and worst example windows for each primitive.

- primitive_regime_alignment.csv
  Shows which regimes each primitive activates in.

- primitive_concept_alignment.csv
  Concept lifts: high VIX, drawdown stress, high turnover, high concentration, benchmark deviation, risk-off, etc.

- primitive_style_alignment.csv
  Synthetic-strategy matching: equal-weight, momentum tilt, low-vol defensive, defensive-minus-cyclical, sector dispersion.

- figures/
  PNG plots for primitive leaderboard, action-risk scatter, regime heatmap, style heatmap, and primitive timeline.

What I need from you now:

1. Use primitive_id as the stable label. Please do not relabel primitives without preserving the original primitive_id.

2. For each primitive, write a short interpretation:
   - What behavior does it represent?
   - Is it economically meaningful?
   - Is it closer to momentum, defensive rotation, risk-off, benchmark tracking, concentration, overtrading, or something else?

3. Focus especially on bad primitives:
   - primitive_02: large activation share, negative excess Sharpe.
   - primitive_04: very high action change / turnover, negative excess Sharpe.
   - primitive_05: drawdown-stress aligned, poor excess performance.
   Please check whether these look like real failure modes or just clustering artifacts.

4. If possible, propose which hidden-state probes / TCAV concepts we should run next using these primitive_id labels.

5. Give me falsifiable intervention ideas.
   Example: "if primitive_04 is an overtrading failure primitive, test a targeted penalty only when this primitive-like state/action pattern appears," not another global turnover penalty.

6. Tell me if the current CSV files are enough for your part.
   If you need hidden states / penultimate policy activations for TCAV-style probes, I will export them separately.

My goal is not just to name the primitives. The goal is to use interpretability to design the next robust PPO experiment after G1/G2 failed.
"""


def _read_csv(output: Path, name: str) -> pd.DataFrame:
    path = output / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required audit file: {path}")
    return pd.read_csv(path)


def _top_join(frame: pd.DataFrame, group_col: str, value_col: str, label_cols: list[str], *, n: int = 3) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if frame.empty:
        return pd.DataFrame(columns=[group_col, value_col])
    scored = frame.copy()
    scored[value_col] = pd.to_numeric(scored[value_col], errors="coerce")
    scored = scored.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
    for key, group in scored.groupby(group_col, sort=True):
        group = group.reindex(group[value_col].abs().sort_values(ascending=False).index).head(n)
        chunks = []
        for _, row in group.iterrows():
            label = row[label_cols[0]]
            detail = ", ".join(f"{col}={row[col]:.4g}" for col in label_cols[1:] if pd.notna(row[col]))
            chunks.append(f"{label} ({detail})" if detail else str(label))
        rows.append({group_col: key, f"top_{label_cols[0]}s": "; ".join(chunks)})
    return pd.DataFrame(rows)


def build_lookup(output: Path) -> pd.DataFrame:
    summary = _read_csv(output, "behavior_primitive_summary.csv")
    concepts = _read_csv(output, "primitive_concept_alignment.csv")
    styles = _read_csv(output, "primitive_style_alignment.csv")

    top_concepts = _top_join(
        concepts,
        "primitive_id",
        "concept_lift",
        ["concept", "concept_lift", "primitive_concept_rate", "base_concept_rate"],
    )
    top_styles = _top_join(
        styles,
        "primitive_id",
        "activation_correlation",
        ["synthetic_strategy", "activation_correlation", "active_minus_base_strategy_return"],
    )

    keep = [
        "primitive_id",
        "primitive_type",
        "windows",
        "share",
        "folds",
        "seeds",
        "dominant_regime",
        "dominant_regime_lift",
        "reward_excess_return_mean",
        "reward_excess_return_median",
        "reward_excess_sharpe",
        "reward_excess_hit_rate",
        "reward_excess_max_drawdown",
        "worst_fold_excess_return_mean",
        "action_change_l1_mean",
        "action_jitter_l2_mean",
        "test_turnover_mean",
        "portfolio_hhi_mean",
        "portfolio_max_weight_mean",
        "portfolio_benchmark_deviation_l1_mean",
        "current_drawdown_worst",
        "vix_mean",
        "primitive_reliability_score",
    ]
    lookup = summary[[col for col in keep if col in summary.columns]].copy()
    lookup = lookup.merge(top_concepts, on="primitive_id", how="left")
    lookup = lookup.merge(top_styles, on="primitive_id", how="left")
    return lookup


def build_compact_labels(output: Path) -> pd.DataFrame:
    assignments = _read_csv(output, "behavior_primitive_assignments.csv")
    summary = _read_csv(output, "behavior_primitive_summary.csv")
    type_map = summary[["primitive_id", "primitive_type"]].drop_duplicates()
    labels = assignments.merge(type_map, on="primitive_id", how="left")
    keep = [
        "behavior_window_id",
        "run_key",
        "feature_set",
        "feature_family",
        "fold_id",
        "seed",
        "date",
        "window_start_date",
        "window_end_date",
        "window_length",
        "primitive_id",
        "primitive_type",
        "reward_daily_return",
        "reward_benchmark_return",
        "reward_excess_return_vs_benchmark",
        "same_day_excess_return_vs_benchmark",
        "current_drawdown",
        "regime_label_exogenous",
        "market_regime",
        "analysis_regime",
        "action_l1_wmean",
        "action_change_l1_wmean",
        "action_jitter_l2_wmean",
        "turnover_wmean",
        "portfolio_hhi_wmean",
        "portfolio_max_weight_wmean",
        "portfolio_benchmark_deviation_l1_wmean",
        "portfolio_sector_max_weight_wmean",
        "vix_wmean",
        "sp500_trend_wmean",
    ]
    return labels[[col for col in keep if col in labels.columns]].copy()


def build_examples(compact: pd.DataFrame, *, n: int = 3) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    score = "reward_excess_return_vs_benchmark"
    compact = compact.copy()
    compact[score] = pd.to_numeric(compact[score], errors="coerce")
    for primitive, group in compact.dropna(subset=[score]).groupby("primitive_id", sort=True):
        best = group.nlargest(n, score).copy()
        best["example_type"] = "best_excess_return_window"
        best["example_rank"] = range(1, len(best) + 1)
        worst = group.nsmallest(n, score).copy()
        worst["example_type"] = "worst_excess_return_window"
        worst["example_rank"] = range(1, len(worst) + 1)
        rows.extend([best, worst])
    if not rows:
        return pd.DataFrame()
    examples = pd.concat(rows, ignore_index=True)
    front = ["primitive_id", "primitive_type", "example_type", "example_rank"]
    return examples[front + [col for col in examples.columns if col not in front]]


def write_manifest(output: Path, zip_name: str) -> None:
    manifest = f"""# Joseph Handoff File Manifest

Zip file: `{zip_name}`

## Recommended Reading Order

1. `JOSEPH_HANDOFF_MESSAGE.md`
2. `BEHAVIOR_INTERPRETABILITY_AUDIT.md`
3. `primitive_lookup_for_joseph.csv`
4. `primitive_examples_for_joseph.csv`
5. `primitive_daily_labels_compact.csv`
6. Alignment tables and figures.

## Files

- `JOSEPH_HANDOFF_MESSAGE.md`: message explaining the project setup, G1/G2 failures, audit method, and requested Joseph tasks.
- `BEHAVIOR_INTERPRETABILITY_AUDIT.md`: short audit closeout and primitive leaderboard.
- `JOSEPH_INTERPRETABILITY_REQUEST.md`: concise task list for probe/interpretability work.
- `reliability_objective.json`: composite reliability score and kill rules.
- `audit_report.json`: machine-readable audit status and output paths.
- `behavior_primitive_summary.csv`: one row per primitive with performance and action diagnostics.
- `behavior_primitive_assignments.csv`: full canonical primitive labels per behavior window.
- `primitive_lookup_for_joseph.csv`: compact primitive cheat sheet with top concept/style evidence.
- `primitive_daily_labels_compact.csv`: compact primitive labels with main outcome/action/regime fields.
- `primitive_examples_for_joseph.csv`: best and worst example windows per primitive.
- `primitive_regime_alignment.csv`: primitive activation by regime.
- `primitive_concept_alignment.csv`: concept lift table.
- `primitive_style_alignment.csv`: synthetic strategy/style matching table.
- `figures/*.png`: quick visual summaries.
- `behavior_interpretability_audit.py`: script used to create the audit.

## Deliberately Excluded

- `decision_state_action_rows.csv`
- `behavior_window_features.csv`
- `clustering_metadata.json`
- SVG figures

These are useful for full reproduction, but they are not needed for Joseph's immediate interpretation task.

## Current Limitation

These files are enough for primitive-level finance/action/regime explanations. They are not enough for true TCAV-style hidden-state probes because policy hidden activations are not exported yet.
"""
    (output / "JOSEPH_HANDOFF_FILE_MANIFEST.md").write_text(manifest, encoding="utf-8")


def write_zip(output: Path) -> Path:
    zip_path = output / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()
    include_names = [
        "JOSEPH_HANDOFF_MESSAGE.md",
        "JOSEPH_HANDOFF_FILE_MANIFEST.md",
        "BEHAVIOR_INTERPRETABILITY_AUDIT.md",
        "JOSEPH_INTERPRETABILITY_REQUEST.md",
        "reliability_objective.json",
        "audit_report.json",
        "behavior_primitive_summary.csv",
        "behavior_primitive_assignments.csv",
        "primitive_lookup_for_joseph.csv",
        "primitive_daily_labels_compact.csv",
        "primitive_examples_for_joseph.csv",
        "primitive_regime_alignment.csv",
        "primitive_concept_alignment.csv",
        "primitive_style_alignment.csv",
    ]
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for name in include_names:
            archive.write(output / name, arcname=name)
        for figure in sorted((output / "figures").glob("*.png")):
            archive.write(figure, arcname=f"figures/{figure.name}")
        archive.write(AUDIT_ROOT / "behavior_interpretability_audit.py", arcname="behavior_interpretability_audit.py")
    return zip_path


def main() -> None:
    output = DEFAULT_OUTPUT
    if not output.exists():
        raise FileNotFoundError(f"Audit output directory not found: {output}")

    (output / "JOSEPH_HANDOFF_MESSAGE.md").write_text(JOSEPH_MESSAGE, encoding="utf-8")
    lookup = build_lookup(output)
    compact = build_compact_labels(output)
    examples = build_examples(compact)
    lookup.to_csv(output / "primitive_lookup_for_joseph.csv", index=False)
    compact.to_csv(output / "primitive_daily_labels_compact.csv", index=False)
    examples.to_csv(output / "primitive_examples_for_joseph.csv", index=False)
    write_manifest(output, ZIP_NAME)
    zip_path = write_zip(output)

    summary = {
        "status": "completed",
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "primitive_lookup_rows": int(len(lookup)),
        "compact_label_rows": int(len(compact)),
        "example_rows": int(len(examples)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
