"""Generate final next-cycle Ablation Ladder v2 figures.

The script reads the final merged Horizon A analysis bundle and writes
publication-ready figures plus the source plotting tables used by each figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = (
    ROOT
    / "merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol"
    / "analysis"
)
FIGURE_DIR = ROOT / "paper_figures"
TABLE_DIR = FIGURE_DIR / "plotting_tables"

SUMMARY_PATH = ANALYSIS_DIR / "corrected_walk_forward_summary.csv"
BENCHMARK_PATH = ANALYSIS_DIR / "corrected_walk_forward_summary_with_primary_benchmark.csv"
UNIQUE_RUNS_PATH = ANALYSIS_DIR / "unique_run_level_results.csv"
PAIRWISE_PATH = ANALYSIS_DIR / "pairwise_permutation_tests_recomputed.csv"
SELECTION_PATH = ANALYSIS_DIR / "selection_rule_summary.csv"
WINNERS_PATH = ANALYSIS_DIR / "validation_vs_test_winner_by_fold.csv"
REGIME_PATH = ANALYSIS_DIR / "regime_summary_by_feature_set.csv"
DAILY_PATH = ANALYSIS_DIR / "walk_forward_daily_test_returns.csv"


SHORT_LABELS = {
    "base": "Base",
    "base_macro": "Base+Macro",
    "base_macro_exogenous_plus": "Exog+",
    "base_macro_hmm": "HMM",
    "base_macro_gru": "GRU",
    "base_macro_xsec_dispersion_correlation_regime": "XSec",
    "base_macro_breadth_internal_structure": "Breadth",
    "base_macro_sector_relative_context": "Sector",
    "base_macro_xsec_sector_gated_context": "XSec/Sector Gate",
    "base_macro_credit_stress_proxies": "Credit",
    "base_macro_rates_term_structure_lsc": "Rates",
    "base_macro_analyst_or_fund_revision_features": "Analyst",
    "base_macro_vol_term_or_implied_vol_proxy": "Vol Proxy",
}

FAMILY_GROUPS = {
    "base": "historical",
    "base_macro": "historical",
    "base_macro_exogenous_plus": "historical",
    "base_macro_hmm": "negative_control",
    "base_macro_gru": "negative_control",
    "base_macro_xsec_dispersion_correlation_regime": "candidate_keep",
    "base_macro_breadth_internal_structure": "candidate_low_priority",
    "base_macro_sector_relative_context": "candidate_keep",
    "base_macro_xsec_sector_gated_context": "candidate_reject",
    "base_macro_credit_stress_proxies": "candidate_top_tier",
    "base_macro_rates_term_structure_lsc": "candidate_top_tier",
    "base_macro_analyst_or_fund_revision_features": "candidate_reject",
    "base_macro_vol_term_or_implied_vol_proxy": "candidate_top_tier",
}

PALETTE = {
    "historical": "#39434f",
    "negative_control": "#8a7a55",
    "candidate_top_tier": "#176b87",
    "candidate_keep": "#4f7f52",
    "candidate_low_priority": "#b47c30",
    "candidate_reject": "#9a4d4d",
}

GROUP_LABELS = {
    "historical": "Historical/reference",
    "negative_control": "Negative control",
    "candidate_top_tier": "Top-tier next-cycle candidate",
    "candidate_keep": "Retain for diagnostics",
    "candidate_low_priority": "Low-priority diagnostic",
    "candidate_reject": "Do not promote",
}


def _ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / name, index=False)


def _save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = FIGURE_DIR / f"{stem}.png"
    svg_path = FIGURE_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _label(feature_set: str) -> str:
    return SHORT_LABELS.get(feature_set, feature_set.replace("base_macro_", ""))


def _colors(feature_sets: Iterable[str]) -> list[str]:
    return [PALETTE[FAMILY_GROUPS.get(fs, "historical")] for fs in feature_sets]


def _load_data() -> dict[str, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_PATH)
    benchmark = pd.read_csv(BENCHMARK_PATH)
    unique = pd.read_csv(UNIQUE_RUNS_PATH)
    pairwise = pd.read_csv(PAIRWISE_PATH)
    selection = pd.read_csv(SELECTION_PATH)
    winners = pd.read_csv(WINNERS_PATH)
    regime = pd.read_csv(REGIME_PATH)
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])

    means = (
        unique.groupby("feature_set")
        .agg(
            test_sharpe_mean=("test_sharpe", "mean"),
            test_return_pct_mean=("test_return_pct", "mean"),
            positive_test_sharpe_rate=("test_sharpe", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    summary = summary.merge(means, on="feature_set", how="left")
    summary["short_label"] = summary["feature_set"].map(_label)
    summary["family_group"] = summary["feature_set"].map(FAMILY_GROUPS).fillna("historical")

    benchmark["short_label"] = benchmark["feature_set"].map(_label)
    benchmark["family_group"] = benchmark["feature_set"].map(FAMILY_GROUPS).fillna("historical")
    unique["short_label"] = unique["feature_set"].map(_label)
    unique["family_group"] = unique["feature_set"].map(FAMILY_GROUPS).fillna("historical")
    daily["short_label"] = daily["feature_set"].map(_label)
    daily["family_group"] = daily["feature_set"].map(FAMILY_GROUPS).fillna("historical")
    regime["short_label"] = regime["feature_set"].map(_label)

    return {
        "summary": summary,
        "benchmark": benchmark,
        "unique": unique,
        "pairwise": pairwise,
        "selection": selection,
        "winners": winners,
        "regime": regime,
        "daily": daily,
    }


def _add_group_legend(ax: plt.Axes) -> None:
    handles = [
        Patch(facecolor=color, label=GROUP_LABELS[group])
        for group, color in PALETTE.items()
        if group in GROUP_LABELS
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)


def figure_21_final_scoreboard(data: dict[str, pd.DataFrame]) -> None:
    summary = data["summary"].sort_values("test_sharpe_median", ascending=True).copy()
    _save_table(
        summary[
            [
                "feature_set",
                "short_label",
                "family_group",
                "test_sharpe_median",
                "test_sharpe_mean",
                "test_return_pct_median",
                "test_return_pct_mean",
                "positive_test_sharpe_rate",
                "retention_ratio_median",
                "generalization_ratio_median",
            ]
        ],
        "21_next_cycle_final_scoreboard.csv",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    metrics = [
        ("test_sharpe_median", "Median test Sharpe"),
        ("test_return_pct_median", "Median test return (%)"),
        ("primary_benchmark_excess_sharpe_median", "Median excess Sharpe vs benchmark"),
    ]
    merged = summary.merge(
        data["benchmark"][
            ["feature_set", "primary_benchmark_excess_sharpe_median"]
        ],
        on="feature_set",
        how="left",
    )
    for ax, (metric, title) in zip(axes, metrics):
        ax.barh(merged["short_label"], merged[metric], color=_colors(merged["feature_set"]))
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=9)
    axes[0].set_ylabel("Feature set")
    _add_group_legend(axes[-1])
    fig.suptitle("Final Horizon A next-cycle ranking", fontsize=15, weight="bold")
    fig.tight_layout()
    _save_figure(fig, "21_next_cycle_final_scoreboard")


def figure_22_candidate_decision_heatmap(data: dict[str, pd.DataFrame]) -> None:
    summary = data["summary"]
    benchmark = data["benchmark"]
    candidates = [
        "base_macro_vol_term_or_implied_vol_proxy",
        "base_macro_rates_term_structure_lsc",
        "base_macro_credit_stress_proxies",
        "base_macro_xsec_dispersion_correlation_regime",
        "base_macro_sector_relative_context",
        "base_macro_breadth_internal_structure",
        "base_macro_analyst_or_fund_revision_features",
        "base_macro_xsec_sector_gated_context",
    ]
    df = (
        summary[summary["feature_set"].isin(candidates)]
        .merge(
            benchmark[
                [
                    "feature_set",
                    "primary_benchmark_excess_sharpe_median",
                    "primary_benchmark_outperform_sharpe_rate",
                ]
            ],
            on="feature_set",
            how="left",
        )
        .copy()
    )
    metrics = [
        "test_sharpe_median",
        "test_sharpe_mean",
        "test_return_pct_median",
        "positive_test_sharpe_rate",
        "retention_ratio_median",
        "primary_benchmark_excess_sharpe_median",
        "primary_benchmark_outperform_sharpe_rate",
    ]
    df["short_label"] = df["feature_set"].map(_label)
    df = df.set_index("short_label").loc[[SHORT_LABELS[c] for c in candidates]]
    z = df[metrics].copy()
    normalized = (z - z.mean()) / z.std(ddof=0).replace(0, np.nan)
    normalized = normalized.fillna(0.0)
    _save_table(df.reset_index(), "22_next_cycle_candidate_decision_heatmap.csv")

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(normalized.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-1.5, vmax=1.5)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(
        [
            "Median\nSharpe",
            "Mean\nSharpe",
            "Median\nReturn",
            "Positive\nSharpe Rate",
            "Retention",
            "Bench Excess\nSharpe",
            "Bench Sharpe\nOutperform",
        ],
        rotation=0,
        fontsize=9,
    )
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, f"{normalized.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Cross-candidate z-score")
    ax.set_title("Candidate decision heatmap: strengths are complementary, not decisive", fontsize=13, weight="bold")
    fig.tight_layout()
    _save_figure(fig, "22_next_cycle_candidate_decision_heatmap")


def figure_23_pairwise_vs_base_macro(data: dict[str, pd.DataFrame]) -> None:
    pairwise = data["pairwise"].copy()
    base = "base_macro"
    rows = []
    for _, row in pairwise.iterrows():
        left = row["left"]
        right = row["right"]
        if left == base and right != base:
            rows.append(
                {
                    "feature_set": right,
                    "mean_sharpe_delta_vs_base_macro": -row["observed_diff"],
                    "p_value": row["p_value"],
                }
            )
        elif right == base and left != base:
            rows.append(
                {
                    "feature_set": left,
                    "mean_sharpe_delta_vs_base_macro": row["observed_diff"],
                    "p_value": row["p_value"],
                }
            )
    df = pd.DataFrame(rows)
    df["short_label"] = df["feature_set"].map(_label)
    df["family_group"] = df["feature_set"].map(FAMILY_GROUPS).fillna("historical")
    df = df.sort_values("mean_sharpe_delta_vs_base_macro", ascending=True)
    _save_table(df, "23_next_cycle_pairwise_delta_vs_base_macro.csv")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["short_label"], df["mean_sharpe_delta_vs_base_macro"], color=_colors(df["feature_set"]))
    ax.axvline(0, color="#111111", linewidth=0.9)
    for y, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["mean_sharpe_delta_vs_base_macro"],
            y,
            f" p={row['p_value']:.3f}",
            va="center",
            ha="left" if row["mean_sharpe_delta_vs_base_macro"] >= 0 else "right",
            fontsize=8,
        )
    ax.set_title("Pairwise mean test Sharpe delta versus Base+Macro", fontsize=13, weight="bold")
    ax.set_xlabel("Mean Sharpe delta, feature minus Base+Macro")
    ax.grid(axis="x", alpha=0.25)
    _add_group_legend(ax)
    fig.tight_layout()
    _save_figure(fig, "23_next_cycle_pairwise_delta_vs_base_macro")


def figure_24_fold_winner_map(data: dict[str, pd.DataFrame]) -> None:
    winners = data["winners"].copy()
    rows = (
        winners[["fold_id", "actual_test_winner_feature_set", "actual_test_winner_sharpe_median"]]
        .drop_duplicates()
        .sort_values("fold_id")
    )
    rows["winner_label"] = rows["actual_test_winner_feature_set"].map(_label)
    rows["fold_num"] = rows["fold_id"].str.extract(r"(\d+)").astype(int)
    order = rows.sort_values("fold_num")
    _save_table(order, "24_next_cycle_fold_winner_map.csv")

    unique_winners = order["winner_label"].unique().tolist()
    label_to_int = {label: i for i, label in enumerate(unique_winners)}
    values = order["winner_label"].map(label_to_int).to_numpy()[None, :]
    cmap = plt.get_cmap("tab20", len(unique_winners))
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order["fold_id"], rotation=45, ha="right", fontsize=8)
    ax.set_yticks([0])
    ax.set_yticklabels(["Actual winner"])
    for x, (_, row) in enumerate(order.iterrows()):
        ax.text(x, 0, row["winner_label"], ha="center", va="center", fontsize=7, color="black")
    handles = [Patch(facecolor=cmap(i), label=label) for label, i in label_to_int.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=4, frameon=False, fontsize=8)
    ax.set_title("Fold-level actual test winners after all next-cycle candidates", fontsize=13, weight="bold")
    fig.tight_layout()
    _save_figure(fig, "24_next_cycle_fold_winner_map")


def figure_25_selection_rule_degradation(data: dict[str, pd.DataFrame]) -> None:
    final_selection = data["selection"].copy()
    historical_path = ROOT / "comparison_outputs" / "selection_rule_summary.csv"
    historical = pd.read_csv(historical_path)
    final_selection["panel"] = "Final all-candidate panel"
    historical["panel"] = "Historical v2 panel"
    combined = pd.concat([historical, final_selection], ignore_index=True, sort=False)
    _save_table(combined, "25_next_cycle_selection_rule_comparison.csv")

    metrics = [
        ("selection_matches_test_winner_rate", "Winner match rate"),
        ("median_test_winner_regret", "Median test-winner regret"),
        ("selected_test_sharpe_median", "Selected median test Sharpe"),
    ]
    rules = ["robust_q25_retention", "robust_q25", "sharpe_only"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False)
    x = np.arange(len(rules))
    width = 0.36
    panels = ["Historical v2 panel", "Final all-candidate panel"]
    colors = ["#9aa5b1", "#176b87"]
    for ax, (metric, title) in zip(axes, metrics):
        for idx, panel in enumerate(panels):
            vals = (
                combined[combined["panel"].eq(panel)]
                .set_index("selection_rule")
                .reindex(rules)[metric]
                .to_numpy()
            )
            ax.bar(x + (idx - 0.5) * width, vals, width=width, label=panel, color=colors[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(["Robust+\nretention", "Robust\nq25", "Sharpe\nonly"], fontsize=9)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Selection rules became less reliable as the candidate panel expanded", fontsize=14, weight="bold")
    fig.tight_layout()
    _save_figure(fig, "25_next_cycle_selection_rule_degradation")


def figure_26_benchmark_scatter(data: dict[str, pd.DataFrame]) -> None:
    benchmark = data["benchmark"].copy()
    benchmark["short_label"] = benchmark["feature_set"].map(_label)
    _save_table(
        benchmark[
            [
                "feature_set",
                "short_label",
                "primary_benchmark_excess_return_pct_median",
                "primary_benchmark_excess_sharpe_median",
                "primary_benchmark_outperform_return_rate",
                "primary_benchmark_outperform_sharpe_rate",
            ]
        ],
        "26_next_cycle_benchmark_relative_scatter.csv",
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sizes = 90 + benchmark["primary_benchmark_outperform_sharpe_rate"] * 280
    ax.scatter(
        benchmark["primary_benchmark_excess_return_pct_median"],
        benchmark["primary_benchmark_excess_sharpe_median"],
        s=sizes,
        c=_colors(benchmark["feature_set"]),
        alpha=0.82,
        edgecolor="#222222",
        linewidth=0.5,
    )
    for _, row in benchmark.iterrows():
        ax.text(
            row["primary_benchmark_excess_return_pct_median"],
            row["primary_benchmark_excess_sharpe_median"],
            f" {row['short_label']}",
            fontsize=8,
            va="center",
        )
    ax.axhline(0, color="#111111", linewidth=0.8)
    ax.axvline(0, color="#111111", linewidth=0.8)
    ax.set_xlabel("Median excess return vs primary benchmark (%)")
    ax.set_ylabel("Median excess Sharpe vs primary benchmark")
    ax.set_title("Benchmark-relative view: no family clears the passive bar", fontsize=13, weight="bold")
    ax.grid(alpha=0.25)
    _add_group_legend(ax)
    fig.tight_layout()
    _save_figure(fig, "26_next_cycle_benchmark_relative_scatter")


def figure_27_regime_excess_heatmap(data: dict[str, pd.DataFrame]) -> None:
    regime = data["regime"].copy()
    keep = [
        "base_macro",
        "base_macro_vol_term_or_implied_vol_proxy",
        "base_macro_rates_term_structure_lsc",
        "base_macro_credit_stress_proxies",
        "base_macro_xsec_dispersion_correlation_regime",
        "base_macro_sector_relative_context",
        "base_macro_analyst_or_fund_revision_features",
        "base_macro_xsec_sector_gated_context",
    ]
    regime = regime[regime["feature_set"].isin(keep)].copy()
    pivot = regime.pivot_table(
        index="short_label",
        columns="regime_label_exogenous",
        values="excess_return_vs_benchmark_median",
        aggfunc="median",
    )
    pivot = pivot.reindex([SHORT_LABELS[k] for k in keep])
    _save_table(pivot.reset_index(), "27_next_cycle_regime_excess_return_heatmap.csv")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    arr = pivot.to_numpy()
    vmax = np.nanmax(np.abs(arr))
    vmax = max(vmax, 0.001)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            text = "" if pd.isna(value) else f"{value:.4f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Median daily excess return")
    ax.set_title("Regime diagnostics: candidate edges are state-dependent and thin-sample", fontsize=13, weight="bold")
    fig.tight_layout()
    _save_figure(fig, "27_next_cycle_regime_excess_return_heatmap")


def figure_28_cumulative_paths(data: dict[str, pd.DataFrame]) -> None:
    daily = data["daily"].copy()
    keep = [
        "base_macro",
        "base_macro_vol_term_or_implied_vol_proxy",
        "base_macro_rates_term_structure_lsc",
        "base_macro_credit_stress_proxies",
        "base_macro_xsec_dispersion_correlation_regime",
        "base_macro_sector_relative_context",
    ]
    daily = daily[daily["feature_set"].isin(keep)].copy()
    daily = daily.sort_values(["feature_set", "date"])
    mean_daily = (
        daily.groupby(["feature_set", "short_label", "date"], as_index=False)["daily_return"]
        .mean()
        .sort_values(["feature_set", "date"])
    )
    mean_daily["cumulative_return"] = mean_daily.groupby("feature_set")["daily_return"].transform(
        lambda s: (1.0 + s).cumprod() - 1.0
    )
    _save_table(mean_daily, "28_next_cycle_main_candidate_cumulative_returns.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    for feature_set, group in mean_daily.groupby("feature_set"):
        ax.plot(
            group["date"],
            group["cumulative_return"] * 100.0,
            label=_label(feature_set),
            linewidth=2.0 if feature_set == "base_macro" else 1.6,
            color=PALETTE[FAMILY_GROUPS.get(feature_set, "historical")],
            alpha=0.95,
        )
    ax.axhline(0, color="#111111", linewidth=0.8)
    ax.set_title("Mean daily test cumulative return paths: main retained families", fontsize=13, weight="bold")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_figure(fig, "28_next_cycle_main_candidate_cumulative_returns")


def main() -> None:
    _ensure_dirs()
    data = _load_data()
    figure_21_final_scoreboard(data)
    figure_22_candidate_decision_heatmap(data)
    figure_23_pairwise_vs_base_macro(data)
    figure_24_fold_winner_map(data)
    figure_25_selection_rule_degradation(data)
    figure_26_benchmark_scatter(data)
    figure_27_regime_excess_heatmap(data)
    figure_28_cumulative_paths(data)


if __name__ == "__main__":
    main()
