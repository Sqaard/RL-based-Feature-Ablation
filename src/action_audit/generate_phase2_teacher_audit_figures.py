from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TEACHER_DIR = ROOT / "research_outputs_phase2_base_macro_teacher"
AUDIT_DIR = ROOT / "research_outputs_phase2_teacher_action_audit"
PREDICTABILITY_DIR = ROOT / "research_outputs_phase2_state_action_predictability"
EXACT_PREDICTABILITY_DIR = ROOT / "research_outputs_phase2_exact_observation_predictability"
BC_DIAGNOSTIC_DIR = ROOT / "research_outputs_phase2_bc_action_code_diagnostic"
TWO_STAGE_DIR = ROOT / "research_outputs_phase2_two_stage_action_diagnostic"
TOKENIZER_DIR = ROOT / "research_outputs_phase2_action_tokenizer_diagnostic"
FIGURE_DIR = ROOT / "phase2_figures"
TABLE_DIR = FIGURE_DIR / "plotting_tables"


def _ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def figure_01_action_code_distribution() -> None:
    counts = pd.read_csv(AUDIT_DIR / "latent_action_teacher_code_counts.csv")
    counts = counts.sort_values("share", ascending=True)
    counts.to_csv(TABLE_DIR / "01_action_code_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(counts["simple_action_code"], counts["share"] * 100.0, color="#176b87")
    ax.set_xlabel("Share of teacher decisions (%)")
    ax.set_title("Base+Macro teacher action-code distribution", fontsize=13, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    for y, (_, row) in enumerate(counts.iterrows()):
        ax.text(row["share"] * 100.0, y, f" {row['share'] * 100.0:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, "01_action_code_distribution")


def figure_02_fold_action_sparsity() -> None:
    coded = pd.read_csv(AUDIT_DIR / "latent_action_teacher_simple_codes.csv")
    unique = pd.read_csv(TEACHER_DIR / "unique_run_level_results.csv")
    fold_action = (
        coded.groupby("fold_id")
        .agg(
            action_rows=("simple_action_code", "size"),
            flat_action_rate=("direction_code", lambda s: float((s == "flat").mean())),
            action_code_count=("simple_action_code", "nunique"),
            action_l1_mean=("action_l1", "mean"),
            active_action_dims_mean=("active_action_dims", "mean"),
        )
        .reset_index()
    )
    fold_perf = (
        unique.groupby("fold_id")
        .agg(
            test_sharpe_median=("test_sharpe", "median"),
            test_return_pct_median=("test_return_pct", "median"),
            test_turnover_median=("test_turnover", "median"),
        )
        .reset_index()
    )
    df = fold_action.merge(fold_perf, on="fold_id", how="left")
    df["fold_num"] = df["fold_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("fold_num")
    df.to_csv(TABLE_DIR / "02_fold_action_sparsity.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    ax1.bar(x, df["flat_action_rate"] * 100.0, color="#8a7a55", alpha=0.86)
    ax1.set_ylabel("Flat action rate (%)")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["fold_id"], rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, df["test_sharpe_median"], color="#176b87", marker="o", linewidth=2)
    ax2.axhline(0, color="#222222", linewidth=0.8)
    ax2.set_ylabel("Median test Sharpe")
    ax1.set_title("Action sparsity by fold versus teacher test Sharpe", fontsize=13, weight="bold")
    fig.tight_layout()
    _save_figure(fig, "02_fold_action_sparsity_vs_sharpe")


def figure_03_action_complexity_vs_performance() -> None:
    summary = pd.read_csv(AUDIT_DIR / "latent_action_teacher_action_summary.csv")
    unique = pd.read_csv(TEACHER_DIR / "unique_run_level_results.csv")
    df = summary.merge(
        unique[
            [
                "fold_id",
                "seed",
                "test_sharpe",
                "test_return_pct",
                "test_turnover",
                "validation_sharpe",
            ]
        ],
        on=["fold_id", "seed"],
        how="left",
    )
    df.to_csv(TABLE_DIR / "03_action_complexity_vs_performance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter_specs = [
        ("action_code_count", "Action-code count"),
        ("flat_action_rate", "Flat action rate"),
    ]
    for ax, (x_col, x_label) in zip(axes, scatter_specs):
        sizes = 45 + df["test_turnover"].fillna(0.0) * 5
        ax.scatter(
            df[x_col],
            df["test_sharpe"],
            s=sizes,
            c=df["seed"],
            cmap="viridis",
            alpha=0.82,
            edgecolor="#222222",
            linewidth=0.4,
        )
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Test Sharpe")
        ax.grid(alpha=0.25)
    axes[0].set_title("More codes were not better", fontsize=11, weight="bold")
    axes[1].set_title("Flatness did not explain OOS wins alone", fontsize=11, weight="bold")
    fig.suptitle("Teacher action diagnostics: complexity is a constraint, not a target", fontsize=13, weight="bold")
    fig.tight_layout()
    _save_figure(fig, "03_action_complexity_vs_performance")


def figure_04_predictability_baseline() -> None:
    by_fold_path = PREDICTABILITY_DIR / "state_action_code_predictability_by_fold.csv"
    if not by_fold_path.exists():
        return
    df = pd.read_csv(by_fold_path)
    df["fold_num"] = df["fold_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("fold_num")
    df.to_csv(TABLE_DIR / "04_predictability_baseline.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    x = np.arange(len(df))
    width = 0.38
    axes[0].bar(
        x - width / 2,
        df["majority_accuracy"],
        width=width,
        label="Majority",
        color="#9aa5b1",
    )
    axes[0].bar(
        x + width / 2,
        df["logistic_accuracy"],
        width=width,
        label="State logistic",
        color="#176b87",
    )
    axes[0].set_title("Multiclass simple-action-code accuracy", fontsize=11, weight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(
        x - width / 2,
        df["binary_majority_balanced_accuracy"],
        width=width,
        label="Majority",
        color="#9aa5b1",
    )
    axes[1].bar(
        x + width / 2,
        df["binary_logistic_balanced_accuracy"],
        width=width,
        label="State logistic",
        color="#6a5aa3",
    )
    axes[1].axhline(0.5, color="#222222", linewidth=0.8, linestyle="--")
    axes[1].set_title("Flat vs nonflat balanced accuracy", fontsize=11, weight="bold")
    axes[1].set_ylabel("Balanced accuracy")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df["fold_id"], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "State-to-action-code predictability: aggregate state features are insufficient",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout()
    _save_figure(fig, "04_state_action_code_predictability_baseline")


def figure_05_exact_observation_predictability() -> None:
    by_fold_path = EXACT_PREDICTABILITY_DIR / "state_action_code_predictability_by_fold.csv"
    if not by_fold_path.exists():
        return
    df = pd.read_csv(by_fold_path)
    df["fold_num"] = df["fold_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("fold_num")
    df.to_csv(TABLE_DIR / "05_exact_observation_predictability.csv", index=False)

    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    multiclass_specs = [
        ("majority_accuracy", "Majority", "#9aa5b1"),
        ("previous_accuracy", "Previous code", "#8a7a55"),
        ("random_forest_accuracy", "Exact obs RF", "#176b87"),
        ("logistic_unweighted_accuracy", "Exact obs logistic", "#6a5aa3"),
    ]
    for col, label, color in multiclass_specs:
        axes[0].plot(x, df[col], marker="o", linewidth=1.8, label=label, color=color)
    axes[0].set_title("Multiclass simple-action-code accuracy", fontsize=11, weight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=8)

    binary_specs = [
        ("binary_majority_balanced_accuracy", "Majority", "#9aa5b1"),
        ("binary_previous_balanced_accuracy", "Previous code", "#8a7a55"),
        ("binary_random_forest_balanced_accuracy", "Exact obs RF", "#176b87"),
        ("binary_logistic_unweighted_balanced_accuracy", "Exact obs logistic", "#6a5aa3"),
    ]
    for col, label, color in binary_specs:
        axes[1].plot(x, df[col], marker="o", linewidth=1.8, label=label, color=color)
    axes[1].axhline(0.5, color="#222222", linewidth=0.8, linestyle="--")
    axes[1].set_title("Flat vs nonflat balanced accuracy", fontsize=11, weight="bold")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df["fold_id"], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Exact PPO observations add signal, but sequence persistence is the dominant baseline",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout()
    _save_figure(fig, "05_exact_observation_predictability")


def figure_06_bc_action_code_diagnostic() -> None:
    bc_summary_path = BC_DIAGNOSTIC_DIR / "bc_action_code_diagnostic_summary.csv"
    exact_summary_path = EXACT_PREDICTABILITY_DIR / "state_action_code_predictability_summary.csv"
    if not bc_summary_path.exists() or not exact_summary_path.exists():
        return

    bc_mean = pd.read_csv(bc_summary_path).query("statistic == 'mean'").iloc[0]
    exact_mean = pd.read_csv(exact_summary_path).query("statistic == 'mean'").iloc[0]
    rows = [
        {
            "model": "Majority",
            "multiclass_accuracy": bc_mean["majority_accuracy"],
            "multiclass_balanced_accuracy": bc_mean["majority_balanced_accuracy"],
            "binary_balanced_accuracy": bc_mean["binary_majority_balanced_accuracy"],
        },
        {
            "model": "Previous code",
            "multiclass_accuracy": bc_mean["previous_accuracy"],
            "multiclass_balanced_accuracy": bc_mean["previous_balanced_accuracy"],
            "binary_balanced_accuracy": bc_mean["binary_previous_balanced_accuracy"],
        },
        {
            "model": "Exact obs RF",
            "multiclass_accuracy": exact_mean["random_forest_accuracy"],
            "multiclass_balanced_accuracy": exact_mean["random_forest_balanced_accuracy"],
            "binary_balanced_accuracy": exact_mean["binary_random_forest_balanced_accuracy"],
        },
        {
            "model": "BC MLP natural",
            "multiclass_accuracy": bc_mean["bc_mlp_natural_accuracy"],
            "multiclass_balanced_accuracy": bc_mean["bc_mlp_natural_balanced_accuracy"],
            "binary_balanced_accuracy": bc_mean["binary_bc_mlp_natural_balanced_accuracy"],
        },
        {
            "model": "BC MLP balanced",
            "multiclass_accuracy": bc_mean["bc_mlp_balanced_accuracy"],
            "multiclass_balanced_accuracy": bc_mean["bc_mlp_balanced_balanced_accuracy"],
            "binary_balanced_accuracy": bc_mean["binary_bc_mlp_balanced_balanced_accuracy"],
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "06_bc_action_code_diagnostic.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    metrics = [
        ("multiclass_accuracy", "Multiclass accuracy"),
        ("multiclass_balanced_accuracy", "Multiclass balanced accuracy"),
        ("binary_balanced_accuracy", "Flat/nonflat balanced accuracy"),
    ]
    colors = ["#9aa5b1", "#8a7a55", "#176b87", "#6a5aa3", "#b65745"]
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(df["model"], df[metric], color=colors)
        ax.set_title(title, fontsize=10.5, weight="bold")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        if "balanced" in metric:
            ax.axhline(0.5, color="#222222", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Mean held-out score")
    fig.suptitle(
        "Offline behavior cloning does not yet clear the persistence baseline",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout()
    _save_figure(fig, "06_bc_action_code_diagnostic")


def figure_07_two_stage_action_diagnostic() -> None:
    stage1_path = TWO_STAGE_DIR / "two_stage_stage1_hold_change_summary.csv"
    stage2_path = TWO_STAGE_DIR / "two_stage_stage2_change_targets_summary.csv"
    e2e_path = TWO_STAGE_DIR / "two_stage_end_to_end_simple_code_summary.csv"
    if not stage1_path.exists() or not stage2_path.exists() or not e2e_path.exists():
        return

    stage1 = pd.read_csv(stage1_path).query("statistic == 'mean'").iloc[0]
    stage2 = pd.read_csv(stage2_path)
    stage2 = stage2[
        (stage2["stage2_target"] == "change_or_current_code")
        & (stage2["statistic"] == "mean")
    ].iloc[0]
    e2e = pd.read_csv(e2e_path).query("statistic == 'mean'").iloc[0]

    rows = [
        {"panel": "Stage 1 hold/change", "model": "Majority hold", "score": stage1["majority_balanced_accuracy"]},
        {"panel": "Stage 1 hold/change", "model": "Previous flag", "score": stage1["previous_flag_balanced_accuracy"]},
        {"panel": "Stage 1 hold/change", "model": "Obs logistic", "score": stage1["logistic_balanced_balanced_accuracy"]},
        {"panel": "Stage 1 hold/change", "model": "BC MLP balanced", "score": stage1["bc_mlp_balanced_balanced_accuracy"]},
        {"panel": "Stage 2 change code", "model": "Majority change", "score": stage2["majority_change_balanced_accuracy"]},
        {"panel": "Stage 2 change code", "model": "Prev-code map", "score": stage2["previous_code_conditional_balanced_accuracy"]},
        {"panel": "Stage 2 change code", "model": "Obs RF", "score": stage2["random_forest_balanced_accuracy"]},
        {"panel": "End-to-end code", "model": "Previous code", "score": e2e["previous_code_balanced_accuracy"]},
        {
            "panel": "End-to-end code",
            "model": "Best learned combo",
            "score": e2e[
                [
                    "logistic_balanced__previous_code_conditional_balanced_accuracy",
                    "logistic_balanced__logistic_balanced_balanced_accuracy",
                    "logistic_balanced__random_forest_balanced_accuracy",
                    "logistic_balanced__bc_mlp_balanced_balanced_accuracy",
                    "bc_mlp_balanced__previous_code_conditional_balanced_accuracy",
                    "bc_mlp_balanced__logistic_balanced_balanced_accuracy",
                    "bc_mlp_balanced__random_forest_balanced_accuracy",
                    "bc_mlp_balanced__bc_mlp_balanced_balanced_accuracy",
                ]
            ].max(),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "07_two_stage_action_diagnostic.csv", index=False)

    panels = list(df["panel"].drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    colors = ["#9aa5b1", "#8a7a55", "#176b87", "#6a5aa3"]
    for ax, panel in zip(axes, panels):
        sub = df[df["panel"] == panel]
        ax.bar(sub["model"], sub["score"], color=colors[: len(sub)])
        ax.axhline(0.5, color="#222222", linewidth=0.8, linestyle="--")
        ax.set_title(panel, fontsize=10.5, weight="bold")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].set_ylabel("Mean balanced accuracy")
    fig.suptitle("Two-stage labels remain dominated by persistence baselines", fontsize=13, weight="bold")
    fig.tight_layout()
    _save_figure(fig, "07_two_stage_action_diagnostic")


def figure_08_action_tokenizer_diagnostic() -> None:
    summary_path = TOKENIZER_DIR / "action_tokenizer_diagnostic_summary.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    keep_cols = [
        "source",
        "n_clusters",
        "test_effective_tokens_mean",
        "test_dominant_token_share_mean",
        "reconstruction_mse_ratio_vs_mean_mean",
        "previous_token_balanced_accuracy_mean",
        "obs_logistic_balanced_accuracy_mean",
        "obs_random_forest_balanced_accuracy_mean",
    ]
    plot_df = df[keep_cols].copy()
    plot_df["label"] = plot_df["source"] + " k=" + plot_df["n_clusters"].astype(str)
    plot_df.to_csv(TABLE_DIR / "08_action_tokenizer_diagnostic.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(plot_df))
    axes[0].bar(x, plot_df["test_dominant_token_share_mean"], color="#8a7a55")
    axes[0].set_title("Dominant token share", fontsize=10.5, weight="bold")
    axes[0].set_ylim(0, 1.02)

    axes[1].bar(x, plot_df["reconstruction_mse_ratio_vs_mean_mean"], color="#176b87")
    axes[1].axhline(1.0, color="#222222", linewidth=0.8, linestyle="--")
    axes[1].set_title("Held-out reconstruction vs mean action", fontsize=10.5, weight="bold")

    width = 0.26
    axes[2].bar(
        x - width,
        plot_df["previous_token_balanced_accuracy_mean"],
        width=width,
        label="Previous token",
        color="#8a7a55",
    )
    axes[2].bar(
        x,
        plot_df["obs_logistic_balanced_accuracy_mean"],
        width=width,
        label="Obs logistic",
        color="#6a5aa3",
    )
    axes[2].bar(
        x + width,
        plot_df["obs_random_forest_balanced_accuracy_mean"],
        width=width,
        label="Obs RF",
        color="#176b87",
    )
    axes[2].axhline(0.5, color="#222222", linewidth=0.8, linestyle="--")
    axes[2].set_title("Token predictability", fontsize=10.5, weight="bold")
    axes[2].set_ylim(0, 1.02)
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["label"], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean held-out value")
    fig.suptitle("Direct action-vector tokenizers do not clear persistence baselines", fontsize=13, weight="bold")
    fig.tight_layout()
    _save_figure(fig, "08_action_tokenizer_diagnostic")


def main() -> None:
    _ensure_dirs()
    figure_01_action_code_distribution()
    figure_02_fold_action_sparsity()
    figure_03_action_complexity_vs_performance()
    figure_04_predictability_baseline()
    figure_05_exact_observation_predictability()
    figure_06_bc_action_code_diagnostic()
    figure_07_two_stage_action_diagnostic()
    figure_08_action_tokenizer_diagnostic()


if __name__ == "__main__":
    main()
