from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PREFLIGHT_DIR = ROOT / "research_outputs_ssl_domain_preflight"
G1_ANALYSIS_DIR = ROOT / "research_outputs_ssl_domain_g1_batch_analysis"
G2_ANALYSIS_DIR = ROOT / "research_outputs_ssl_domain_g2_batch_analysis"
FIGURE_DIR = ROOT / "figures"
TABLE_DIR = FIGURE_DIR / "plotting_tables"


def _ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def figure_01_domain_predictability() -> None:
    path = PREFLIGHT_DIR / "domain_predictability_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    keep = [
        "target",
        "majority_balanced_accuracy_mean",
        "obs_logistic_balanced_accuracy_mean",
        "obs_random_forest_balanced_accuracy_mean",
    ]
    plot = df[keep].copy()
    plot.to_csv(TABLE_DIR / "01_domain_predictability.csv", index=False)

    x = np.arange(len(plot))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.bar(x - width, plot["majority_balanced_accuracy_mean"], width, label="Majority", color="#9aa5b1")
    ax.bar(x, plot["obs_logistic_balanced_accuracy_mean"], width, label="Obs logistic", color="#6a5aa3")
    ax.bar(x + width, plot["obs_random_forest_balanced_accuracy_mean"], width, label="Obs RF", color="#176b87")
    ax.axhline(0.5, color="#222222", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["target"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_title("Exact PPO observations strongly encode time/domain identity", fontsize=13, weight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "01_domain_predictability")


def figure_02_leave_fold_reward_predictability() -> None:
    path = PREFLIGHT_DIR / "leave_fold_reward_sign_predictability_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    keep = [
        "target",
        "majority_balanced_accuracy_mean",
        "obs_logistic_balanced_accuracy_mean",
        "obs_random_forest_balanced_accuracy_mean",
    ]
    plot = df[keep].copy()
    plot.to_csv(TABLE_DIR / "02_leave_fold_reward_predictability.csv", index=False)

    x = np.arange(len(plot))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width, plot["majority_balanced_accuracy_mean"], width, label="Majority", color="#9aa5b1")
    ax.bar(x, plot["obs_logistic_balanced_accuracy_mean"], width, label="Obs logistic", color="#6a5aa3")
    ax.bar(x + width, plot["obs_random_forest_balanced_accuracy_mean"], width, label="Obs RF", color="#176b87")
    ax.axhline(0.5, color="#222222", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["target"], rotation=20, ha="right")
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_title("Leave-fold reward-sign predictability is weak", fontsize=13, weight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "02_leave_fold_reward_predictability")


def figure_03_fold_reward_and_actions() -> None:
    path = PREFLIGHT_DIR / "fold_domain_reward_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["fold_num"] = df["fold_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("fold_num")
    df.to_csv(TABLE_DIR / "03_fold_reward_and_actions.csv", index=False)

    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(12, 5.2))
    ax1.bar(x, df["excess_sharpe"], color="#176b87", alpha=0.86)
    ax1.axhline(0, color="#222222", linewidth=0.8)
    ax1.set_ylabel("Fold excess Sharpe")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["fold_id"], rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, df["flat_action_rate"] * 100.0, color="#8a7a55", marker="o", linewidth=2)
    ax2.set_ylabel("Flat action rate (%)")
    ax2.set_ylim(0, 100)
    ax1.set_title("Fold outcomes and action sparsity are regime-dependent", fontsize=13, weight="bold")
    fig.tight_layout()
    _save(fig, "03_fold_reward_and_actions")


def figure_04_fold_observation_distance_heatmap() -> None:
    path = PREFLIGHT_DIR / "fold_observation_mean_distances.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    matrix = df.pivot(index="left_fold", columns="right_fold", values="mean_obs_l2_distance")
    order = sorted(matrix.index, key=lambda value: int(str(value).split("_")[-1]))
    matrix = matrix.loc[order, order]
    matrix.to_csv(TABLE_DIR / "04_fold_observation_distance_heatmap.csv")

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(matrix.to_numpy(), cmap="viridis")
    ax.set_xticks(np.arange(len(order)))
    ax.set_yticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_title("Standardized fold-mean observation distances", fontsize=13, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("L2 distance")
    fig.tight_layout()
    _save(fig, "04_fold_observation_distance_heatmap")


def _load_g1_summary() -> pd.DataFrame | None:
    path = G1_ANALYSIS_DIR / "g1_summary.csv"
    if not path.exists():
        path = G1_ANALYSIS_DIR / "g1_full_analysis_table.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.sort_values("rank_by_gate_score").reset_index(drop=True)


def _short_variant_label(variant_id: str) -> str:
    return (
        variant_id.replace("g1", "")
        .replace("_negative_control", "_control")
        .replace("_", "\n")
    )


def figure_05_g1_gate_and_core_deltas() -> None:
    df = _load_g1_summary()
    if df is None:
        return
    keep = [
        "variant_id",
        "rank_by_gate_score",
        "gate_score",
        "delta_vs_reference_test_sharpe_median",
        "delta_vs_reference_primary_benchmark_excess_sharpe_median",
        "delta_vs_reference_primary_benchmark_excess_return_pct_median",
        "is_negative_control_variant",
        "verdict",
    ]
    plot = df[keep].copy()
    plot.to_csv(TABLE_DIR / "05_g1_gate_and_core_deltas.csv", index=False)

    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    colors = np.where(plot["is_negative_control_variant"], "#9aa5b1", "#176b87")
    ax.bar(x, plot["gate_score"], color=colors, alpha=0.88, label="Gate score")
    ax.plot(
        x,
        plot["delta_vs_reference_test_sharpe_median"],
        color="#8b2f2f",
        marker="o",
        linewidth=2,
        label="Test Sharpe delta",
    )
    ax.plot(
        x,
        plot["delta_vs_reference_primary_benchmark_excess_sharpe_median"],
        color="#6a5aa3",
        marker="s",
        linewidth=2,
        label="Benchmark-excess Sharpe delta",
    )
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_variant_label(v) for v in plot["variant_id"]], fontsize=8)
    ax.set_ylabel("Delta / gate score")
    ax.set_title("G1 variants do not clear the benchmark-relative gate", fontsize=13, weight="bold")
    ax.legend(frameon=False, fontsize=8, ncols=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "05_g1_gate_and_core_deltas")


def figure_06_g1_dose_response() -> None:
    df = _load_g1_summary()
    if df is None:
        return
    dose = df[
        df["branch"].isin(["absolute_return_stress", "volatility_stress", "downside_return"])
        & df["strength"].isin(["mild", "strong"])
    ].copy()
    if dose.empty:
        return
    dose["branch_strength"] = dose["branch"] + "_" + dose["strength"]
    dose = dose.sort_values(["branch", "strength"])
    keep = [
        "variant_id",
        "branch",
        "strength",
        "gate_score",
        "delta_vs_reference_test_sharpe_median",
        "delta_vs_reference_primary_benchmark_excess_return_pct_median",
        "delta_vs_reference_primary_benchmark_excess_sharpe_median",
        "delta_vs_reference_test_turnover_median",
    ]
    dose[keep].to_csv(TABLE_DIR / "06_g1_dose_response.csv", index=False)

    branches = ["absolute_return_stress", "volatility_stress", "downside_return"]
    x = np.arange(len(branches))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    mild = dose[dose["strength"].eq("mild")].set_index("branch").reindex(branches)
    strong = dose[dose["strength"].eq("strong")].set_index("branch").reindex(branches)
    ax.bar(x - width / 2, mild["gate_score"], width, color="#176b87", label="Mild")
    ax.bar(x + width / 2, strong["gate_score"], width, color="#8b2f2f", label="Strong")
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Abs return", "Volatility", "Downside return"])
    ax.set_ylabel("Gate score")
    ax.set_title("G1 dose-response is not strong enough to promote a variant", fontsize=13, weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "06_g1_dose_response")


def figure_07_g1_turnover_vs_sharpe_delta() -> None:
    df = _load_g1_summary()
    if df is None:
        return
    keep = [
        "variant_id",
        "branch",
        "is_negative_control_variant",
        "delta_vs_reference_test_turnover_median",
        "delta_vs_reference_test_sharpe_median",
        "delta_vs_reference_primary_benchmark_excess_sharpe_median",
        "gate_score",
        "verdict",
    ]
    plot = df[keep].copy()
    plot.to_csv(TABLE_DIR / "07_g1_turnover_vs_sharpe_delta.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    colors = np.where(plot["is_negative_control_variant"], "#9aa5b1", "#176b87")
    ax.scatter(
        plot["delta_vs_reference_test_turnover_median"],
        plot["delta_vs_reference_test_sharpe_median"],
        s=80,
        c=colors,
        alpha=0.9,
        edgecolor="#222222",
        linewidth=0.4,
    )
    for _, row in plot.iterrows():
        label = str(row["variant_id"]).replace("g1", "").split("_")[0:3]
        ax.annotate("_".join(label), (row["delta_vs_reference_test_turnover_median"], row["delta_vs_reference_test_sharpe_median"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Median test turnover delta vs base_macro")
    ax.set_ylabel("Median test Sharpe delta vs base_macro")
    ax.set_title("G1 changes turnover, but not enough real OOS quality", fontsize=13, weight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "07_g1_turnover_vs_sharpe_delta")


def _load_g2_summary() -> pd.DataFrame | None:
    path = G2_ANALYSIS_DIR / "g2_summary.csv"
    if not path.exists():
        path = G2_ANALYSIS_DIR / "g2_batch_ranking.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.sort_values("rank_by_gate_score").reset_index(drop=True)


def figure_08_g2_gate_and_core_deltas() -> None:
    df = _load_g2_summary()
    if df is None:
        return
    keep = [
        "variant_id",
        "rank_by_gate_score",
        "gate_score",
        "delta_vs_reference_test_sharpe_median",
        "delta_vs_reference_primary_benchmark_excess_sharpe_median",
        "delta_vs_reference_primary_benchmark_excess_return_pct_median",
        "is_negative_control_variant",
        "decision_label",
    ]
    plot = df[[col for col in keep if col in df.columns]].copy()
    plot.to_csv(TABLE_DIR / "08_g2_gate_and_core_deltas.csv", index=False)

    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    colors = np.where(plot["is_negative_control_variant"].astype(bool), "#9aa5b1", "#176b87")
    ax.bar(x, plot["gate_score"], color=colors, alpha=0.88, label="Gate score")
    ax.plot(
        x,
        plot["delta_vs_reference_test_sharpe_median"],
        color="#8b2f2f",
        marker="o",
        linewidth=2,
        label="Test Sharpe delta",
    )
    ax.plot(
        x,
        plot["delta_vs_reference_primary_benchmark_excess_sharpe_median"],
        color="#6a5aa3",
        marker="s",
        linewidth=2,
        label="Benchmark-excess Sharpe delta",
    )
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_variant_label(v) for v in plot["variant_id"]], fontsize=8)
    ax.set_ylabel("Delta / gate score")
    ax.set_title("G2 one-seed action regularization screening", fontsize=13, weight="bold")
    ax.legend(frameon=False, fontsize=8, ncols=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "08_g2_gate_and_core_deltas")


def figure_09_g2_turnover_vs_sharpe_delta() -> None:
    df = _load_g2_summary()
    if df is None:
        return
    keep = [
        "variant_id",
        "mechanism",
        "is_negative_control_variant",
        "decision_label",
        "delta_vs_reference_test_turnover_median",
        "delta_vs_reference_test_sharpe_median",
        "delta_vs_reference_primary_benchmark_excess_sharpe_median",
        "gate_score",
    ]
    plot = df[[col for col in keep if col in df.columns]].copy()
    plot.to_csv(TABLE_DIR / "09_g2_turnover_vs_sharpe_delta.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    colors = np.where(plot["is_negative_control_variant"].astype(bool), "#9aa5b1", "#176b87")
    ax.scatter(
        plot["delta_vs_reference_test_turnover_median"],
        plot["delta_vs_reference_test_sharpe_median"],
        s=90,
        c=colors,
        alpha=0.9,
        edgecolor="#222222",
        linewidth=0.4,
    )
    for _, row in plot.iterrows():
        label = str(row["variant_id"]).split("_")[0].upper()
        ax.annotate(
            label,
            (row["delta_vs_reference_test_turnover_median"], row["delta_vs_reference_test_sharpe_median"]),
            fontsize=8,
            xytext=(5, 4),
            textcoords="offset points",
        )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Median test turnover delta vs base_macro")
    ax.set_ylabel("Median test Sharpe delta vs base_macro")
    ax.set_title("G2 success cannot be turnover-only", fontsize=13, weight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "09_g2_turnover_vs_sharpe_delta")


def figure_10_g2_regularization_dose_response() -> None:
    df = _load_g2_summary()
    if df is None:
        return
    keep = [
        "variant_id",
        "turnover_penalty",
        "smoothness_penalty",
        "concentration_penalty",
        "max_weight_penalty",
        "gate_score",
        "decision_label",
    ]
    plot = df[[col for col in keep if col in df.columns]].copy()
    plot.to_csv(TABLE_DIR / "10_g2_regularization_dose_response.csv", index=False)

    x = np.arange(len(plot))
    penalty_cols = [
        "turnover_penalty",
        "smoothness_penalty",
        "concentration_penalty",
        "max_weight_penalty",
    ]
    colors = ["#176b87", "#8a7a55", "#6a5aa3", "#8b2f2f"]
    fig, ax1 = plt.subplots(figsize=(12, 5.4))
    bottom = np.zeros(len(plot))
    for col, color in zip(penalty_cols, colors):
        values = pd.to_numeric(plot[col], errors="coerce").fillna(0.0).to_numpy()
        ax1.bar(x, values, bottom=bottom, color=color, alpha=0.82, label=col.replace("_penalty", ""))
        bottom += values
    ax1.set_ylabel("Regularization coefficient sum")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v).split("_")[0].upper() for v in plot["variant_id"]], fontsize=8)
    ax1.grid(axis="y", alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(x, plot["gate_score"], color="#222222", marker="o", linewidth=2.2, label="Gate score")
    ax2.axhline(0.0, color="#222222", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Gate score")
    ax1.set_title("G2 regularization intensity vs screening gate score", fontsize=13, weight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, ncols=5, loc="upper left")
    fig.tight_layout()
    _save(fig, "10_g2_regularization_dose_response")


def figure_11_g2_benchmark_excess_deltas() -> None:
    df = _load_g2_summary()
    if df is None:
        return
    keep = [
        "variant_id",
        "delta_vs_reference_primary_benchmark_excess_return_pct_median",
        "delta_vs_reference_primary_benchmark_excess_sharpe_median",
        "delta_vs_reference_primary_benchmark_outperform_return_rate",
        "delta_vs_reference_primary_benchmark_outperform_sharpe_rate",
        "is_negative_control_variant",
    ]
    plot = df[[col for col in keep if col in df.columns]].copy()
    plot.to_csv(TABLE_DIR / "11_g2_benchmark_excess_deltas.csv", index=False)

    x = np.arange(len(plot))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(
        x - width / 2,
        plot["delta_vs_reference_primary_benchmark_excess_return_pct_median"],
        width,
        color="#176b87",
        label="Excess return delta",
    )
    ax.bar(
        x + width / 2,
        plot["delta_vs_reference_primary_benchmark_excess_sharpe_median"],
        width,
        color="#8b2f2f",
        label="Excess Sharpe delta",
    )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v).split("_")[0].upper() for v in plot["variant_id"]], fontsize=8)
    ax.set_ylabel("Delta vs base_macro reference")
    ax.set_title("Benchmark-relative OOS deltas determine G2 usefulness", fontsize=13, weight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "11_g2_benchmark_excess_deltas")


def figure_12_g2_regime_robustness() -> None:
    path = G2_ANALYSIS_DIR / "g2_regime_delta_table.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    value_col = "delta_excess_return_vs_benchmark_median"
    if df.empty or value_col not in df.columns:
        return
    matrix = df.pivot(index="variant_id", columns="regime_label_exogenous", values=value_col)
    order = _load_g2_summary()
    if order is not None and "variant_id" in order.columns:
        ordered_variants = [v for v in order["variant_id"].tolist() if v in matrix.index]
        matrix = matrix.loc[ordered_variants]
    matrix.to_csv(TABLE_DIR / "12_g2_regime_robustness.csv")

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    values = matrix.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    im = ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels([str(v).split("_")[0].upper() for v in matrix.index], fontsize=8)
    ax.set_title("G2 regime excess-return deltas vs base_macro", fontsize=13, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta excess return vs benchmark")
    fig.tight_layout()
    _save(fig, "12_g2_regime_robustness")


def main() -> None:
    _ensure_dirs()
    figure_01_domain_predictability()
    figure_02_leave_fold_reward_predictability()
    figure_03_fold_reward_and_actions()
    figure_04_fold_observation_distance_heatmap()
    figure_05_g1_gate_and_core_deltas()
    figure_06_g1_dose_response()
    figure_07_g1_turnover_vs_sharpe_delta()
    figure_08_g2_gate_and_core_deltas()
    figure_09_g2_turnover_vs_sharpe_delta()
    figure_10_g2_regularization_dose_response()
    figure_11_g2_benchmark_excess_deltas()
    figure_12_g2_regime_robustness()


if __name__ == "__main__":
    main()
