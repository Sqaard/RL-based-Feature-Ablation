from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZON_A_ROOT = PROJECT_ROOT / "Ablation Ladder v2"
if str(HORIZON_A_ROOT) not in sys.path:
    sys.path.insert(0, str(HORIZON_A_ROOT))

from dow30_next_cycle_launch import run_launch_preflight  # noqa: E402


BATCH_ROOT = PROJECT_ROOT / "SSL Domain Generalization" / "g1_parallel_batch"
CONFIG_ROOT = PROJECT_ROOT / "configs"
OUTPUT_ROOT = PROJECT_ROOT / "SSL Domain Generalization"


VARIANTS: list[dict[str, Any]] = [
    {
        "id": "g1a_abs_stress_mild",
        "experiment_name": "ssl_domain_g1a_abs_stress_mild",
        "hypothesis": "Mildly upweight high absolute-return train days.",
        "domain_reward_scaling": {
            "enabled": True,
            "derived_domain": "benchmark_abs_return_tercile",
            "mode": "stress_upweight",
            "domain_weights": {
                "low_abs_return": 0.85,
                "mid_abs_return": 1.0,
                "high_abs_return": 1.20,
            },
            "min_multiplier": 0.75,
            "max_multiplier": 1.25,
        },
    },
    {
        "id": "g1b_abs_stress_strong",
        "experiment_name": "ssl_domain_g1b_abs_stress_strong",
        "hypothesis": "Test dose-response for absolute-return stress upweighting.",
        "domain_reward_scaling": {
            "enabled": True,
            "derived_domain": "benchmark_abs_return_tercile",
            "mode": "stress_upweight",
            "domain_weights": {
                "low_abs_return": 0.75,
                "mid_abs_return": 1.0,
                "high_abs_return": 1.35,
            },
            "min_multiplier": 0.65,
            "max_multiplier": 1.50,
        },
    },
    {
        "id": "g1c_abs_reverse_negative_control",
        "experiment_name": "ssl_domain_g1c_abs_reverse_negative_control",
        "hypothesis": "Negative control: downweight absolute-return stress days.",
        "domain_reward_scaling": {
            "enabled": True,
            "derived_domain": "benchmark_abs_return_tercile",
            "mode": "stress_upweight",
            "domain_weights": {
                "low_abs_return": 1.20,
                "mid_abs_return": 1.0,
                "high_abs_return": 0.85,
            },
            "min_multiplier": 0.75,
            "max_multiplier": 1.25,
        },
    },
    {
        "id": "g1d_vol_stress_mild",
        "experiment_name": "ssl_domain_g1d_vol_stress_mild",
        "hypothesis": "Mildly upweight high rolling-volatility train domains.",
        "domain_reward_scaling": {
            "enabled": True,
            "derived_domain": "benchmark_vol_tercile",
            "mode": "stress_upweight",
            "domain_weights": {
                "low_vol": 0.85,
                "mid_vol": 1.0,
                "high_vol": 1.20,
            },
            "min_multiplier": 0.75,
            "max_multiplier": 1.25,
        },
    },
    {
        "id": "g1e_vol_stress_strong",
        "experiment_name": "ssl_domain_g1e_vol_stress_strong",
        "hypothesis": "Test dose-response for volatility-domain upweighting.",
        "domain_reward_scaling": {
            "enabled": True,
            "derived_domain": "benchmark_vol_tercile",
            "mode": "stress_upweight",
            "domain_weights": {
                "low_vol": 0.75,
                "mid_vol": 1.0,
                "high_vol": 1.35,
            },
            "min_multiplier": 0.65,
            "max_multiplier": 1.50,
        },
    },
    {
        "id": "g1f_vol_reverse_negative_control",
        "experiment_name": "ssl_domain_g1f_vol_reverse_negative_control",
        "hypothesis": "Negative control: downweight high-volatility train domains.",
        "domain_reward_scaling": {
            "enabled": True,
            "derived_domain": "benchmark_vol_tercile",
            "mode": "stress_upweight",
            "domain_weights": {
                "low_vol": 1.20,
                "mid_vol": 1.0,
                "high_vol": 0.85,
            },
            "min_multiplier": 0.75,
            "max_multiplier": 1.25,
        },
    },
    {
        "id": "g1g_downside_return_mild",
        "experiment_name": "ssl_domain_g1g_downside_return_mild",
        "hypothesis": "Mildly upweight negative/low signed-return train domains.",
        "domain_reward_scaling": {
            "enabled": True,
            "derived_domain": "benchmark_signed_return_tercile",
            "mode": "stress_upweight",
            "domain_weights": {
                "low_return": 1.20,
                "mid_return": 1.0,
                "high_return": 0.85,
            },
            "min_multiplier": 0.75,
            "max_multiplier": 1.25,
        },
    },
    {
        "id": "g1h_downside_return_strong",
        "experiment_name": "ssl_domain_g1h_downside_return_strong",
        "hypothesis": "Test dose-response for downside-return domain upweighting.",
        "domain_reward_scaling": {
            "enabled": True,
            "derived_domain": "benchmark_signed_return_tercile",
            "mode": "stress_upweight",
            "domain_weights": {
                "low_return": 1.35,
                "mid_return": 1.0,
                "high_return": 0.75,
            },
            "min_multiplier": 0.65,
            "max_multiplier": 1.50,
        },
    },
    {
        "id": "g1i_drawdown_stress_mild",
        "experiment_name": "ssl_domain_g1i_drawdown_stress_mild",
        "hypothesis": "Mildly upweight deep train drawdown domains.",
        "domain_reward_scaling": {
            "enabled": True,
            "derived_domain": "benchmark_drawdown_tercile",
            "mode": "stress_upweight",
            "domain_weights": {
                "shallow_drawdown": 0.85,
                "mid_drawdown": 1.0,
                "deep_drawdown": 1.20,
            },
            "min_multiplier": 0.75,
            "max_multiplier": 1.25,
        },
    },
    {
        "id": "g1j_market_regime_inverse_frequency",
        "experiment_name": "ssl_domain_g1j_market_regime_inverse_frequency",
        "hypothesis": "Balance rare train Market_Regime domains by inverse frequency.",
        "domain_reward_scaling": {
            "enabled": True,
            "derived_domain": "market_regime",
            "mode": "inverse_frequency",
            "min_multiplier": 0.75,
            "max_multiplier": 1.25,
        },
    },
]


def _base_config(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_name": variant["experiment_name"],
        "status": "ready_to_run",
        "launch_selection": {
            "candidate_family": None,
            "panel_scope": None,
        },
        "reference_baseline": {
            "config_name": "custom_custom",
            "algorithm": "PPO",
            "reward_mode": "custom_reward_with_training_only_domain_reward_scaling",
            "policy_mode": "custom_mlp_policy",
            "checkpoint_selection_rule": "checkpoint_robust_score",
            "configuration_selection_rule": "robust_q25_retention",
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "action_constraint": "long_only_box_[-1,1]_scaled_by_hmax",
        },
        "notes": {
            "execution_mode": "Run only base_macro with the specified training-only domain reward scaling.",
            "dataset_requirement": "Use processed_final_fixed_external_lagclean_full.csv.",
            "baseline_reuse_expected": False,
            "phase": "SSL/domain-generalization G1 parallel batch.",
            "variant_id": variant["id"],
            "pre_registered_hypothesis": variant["hypothesis"],
            "isolation_rule": "Validation/test rewards are unscaled and checkpoint selection uses checkpoint_robust_score.",
            "kill_rules": "Reject if benchmark-relative OOS metrics do not improve versus the frozen base_macro teacher/reference.",
        },
        "runner_kwargs": {
            "base_config_name": "custom_custom",
            "candidate_feature_families": [],
            "feature_set_filter": ["base_macro"],
            "seeds": [42, 123, 999],
            "total_timesteps": 200000,
            "max_folds": None,
            "es_mode": "relaxed",
            "dropout_p": 0.1,
            "eval_freq": 8192,
            "checkpoint_freq": 4096,
            "checkpoint_selection_rule": "checkpoint_robust_score",
            "domain_reward_scaling": variant["domain_reward_scaling"],
            "verbose": 0,
        },
    }


def _portable_launch_cell(config_path: Path, output_dir: Path, dataset_path: Path) -> str:
    rel_config = config_path.relative_to(PROJECT_ROOT).as_posix()
    rel_dataset = dataset_path.relative_to(PROJECT_ROOT).as_posix()
    rel_output = output_dir.relative_to(PROJECT_ROOT).as_posix()
    return f'''import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
HORIZON_A_ROOT = PROJECT_ROOT / "Ablation Ladder v2"
if str(HORIZON_A_ROOT) not in sys.path:
    sys.path.insert(0, str(HORIZON_A_ROOT))

from dow30_next_cycle_launch import run_bootstrapped_notebook_launch_from_csv

launch_bundle = run_bootstrapped_notebook_launch_from_csv(
    config_path=PROJECT_ROOT / "{rel_config}",
    dataset_path=PROJECT_ROOT / "{rel_dataset}",
    output_dir=PROJECT_ROOT / "{rel_output}",
)

preflight = launch_bundle["preflight"]
research_bundle = launch_bundle["research_bundle"]
'''


def _markdown_manifest(rows: list[dict[str, Any]]) -> str:
    table_rows = "\n".join(
        (
            f"| `{row['variant_id']}` | `{row['experiment_name']}` | "
            f"`{row['derived_domain']}` | `{row['mode']}` | "
            f"[config]({row['config_path']}) | [launch]({row['portable_launch_cell']}) |"
        )
        for row in rows
    )
    return f"""# SSL Domain G1 Parallel Batch

Status: ready to launch.

This batch tests training-time domain reward scaling for `base_macro`. It is not state compression and not another feature-family search.

Run each variant in a separate Huawei worker/notebook from the repository root using the linked portable launch cell.

| Variant | Experiment | Domain | Mode | Config | Portable launch cell |
|---|---|---|---|---|---|
{table_rows}

## Evaluation

After runs finish, compare every output folder against:

- `Latent Actions/research_outputs_phase2_base_macro_teacher`
- `SSL Domain Generalization/research_outputs_ssl_domain_g0_base_macro_temporal_robust_selection`

Primary pass metrics:

- median test Sharpe;
- primary benchmark-excess Sharpe;
- primary benchmark-excess return pct;
- worst-fold behavior;
- drawdown and turnover discipline;
- negative-control separation.
"""


def prepare_batch(dataset_path: Path) -> list[dict[str, Any]]:
    dataset_path = dataset_path.resolve()
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        config_path = CONFIG_ROOT / f"{variant['experiment_name']}.yaml"
        output_dir = OUTPUT_ROOT / f"research_outputs_{variant['experiment_name']}"
        payload = _base_config(variant)
        config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        preflight = run_launch_preflight(
            config_path=config_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
        )
        portable_path = output_dir / "launch_notebook_cell_portable.py"
        portable_path.write_text(
            _portable_launch_cell(config_path, output_dir, dataset_path),
            encoding="utf-8",
        )

        scaling = variant["domain_reward_scaling"]
        rows.append(
            {
                "variant_id": variant["id"],
                "experiment_name": variant["experiment_name"],
                "hypothesis": variant["hypothesis"],
                "derived_domain": scaling.get("derived_domain"),
                "mode": scaling.get("mode"),
                "config_path": config_path.relative_to(PROJECT_ROOT).as_posix(),
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "portable_launch_cell": portable_path.relative_to(PROJECT_ROOT).as_posix(),
                "preflight_status": preflight["summary"]["status"],
                "fold_count": preflight["summary"]["fold_count"],
                "seed_count": preflight["summary"]["seed_count"],
                "audit_ok": preflight["summary"]["audit_ok"],
            }
        )

    manifest_json = BATCH_ROOT / "g1_parallel_batch_manifest.json"
    manifest_csv = BATCH_ROOT / "g1_parallel_batch_manifest.csv"
    manifest_md = BATCH_ROOT / "G1_PARALLEL_BATCH_MANIFEST.md"
    manifest_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(manifest_csv, index=False)
    manifest_md.write_text(_markdown_manifest(rows), encoding="utf-8")
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SSL/domain G1 parallel batch configs and preflights.")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "processed_final_fixed_external_lagclean_full.csv"),
        help="Processed dataset path.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = prepare_batch(Path(args.dataset))
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
