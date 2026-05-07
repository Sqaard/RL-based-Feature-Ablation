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


BATCH_ROOT = PROJECT_ROOT / "SSL Domain Generalization" / "g2_parallel_batch"
CONFIG_ROOT = PROJECT_ROOT / "configs"
OUTPUT_ROOT = PROJECT_ROOT / "SSL Domain Generalization"
G1A_NOTEBOOK_TEMPLATE = (
    OUTPUT_ROOT
    / "research_outputs_ssl_domain_g1a_abs_stress_mild"
    / "g1a.ipynb"
)


SCREENING_SEEDS = [42]


VARIANTS: list[dict[str, Any]] = [
    {
        "id": "g2a_turnover_mild",
        "experiment_name": "ssl_domain_g2a_turnover_mild",
        "short_notebook_name": "g2a.ipynb",
        "hypothesis": "Mild L1 turnover penalty reduces unnecessary rebalancing without suppressing useful adaptation.",
        "mechanism": "turnover_mild",
        "action_regularization": {
            "enabled": True,
            "turnover_penalty": 0.010,
            "smoothness_penalty": 0.0,
            "concentration_penalty": 0.0,
            "max_weight_penalty": 0.0,
            "kl_to_previous_penalty": 0.0,
            "normalize_penalties": True,
            "train_only": True,
        },
    },
    {
        "id": "g2b_turnover_strong",
        "experiment_name": "ssl_domain_g2b_turnover_strong",
        "short_notebook_name": "g2b.ipynb",
        "hypothesis": "Stronger turnover penalty tests dose-response and detects whether turnover reduction becomes performance damage.",
        "mechanism": "turnover_strong",
        "action_regularization": {
            "enabled": True,
            "turnover_penalty": 0.035,
            "smoothness_penalty": 0.0,
            "concentration_penalty": 0.0,
            "max_weight_penalty": 0.0,
            "kl_to_previous_penalty": 0.0,
            "normalize_penalties": True,
            "train_only": True,
        },
    },
    {
        "id": "g2c_smoothness_mild",
        "experiment_name": "ssl_domain_g2c_smoothness_mild",
        "short_notebook_name": "g2c.ipynb",
        "hypothesis": "Action-acceleration penalty reduces buy/sell oscillation without directly forcing low turnover.",
        "mechanism": "smoothness_mild",
        "action_regularization": {
            "enabled": True,
            "turnover_penalty": 0.0,
            "smoothness_penalty": 0.050,
            "concentration_penalty": 0.0,
            "max_weight_penalty": 0.0,
            "kl_to_previous_penalty": 0.0,
            "normalize_penalties": True,
            "train_only": True,
        },
    },
    {
        "id": "g2d_turnover_smoothness",
        "experiment_name": "ssl_domain_g2d_turnover_smoothness",
        "short_notebook_name": "g2d.ipynb",
        "hypothesis": "Mild turnover plus smoothness regularization improves action stability more reliably than either alone.",
        "mechanism": "turnover_plus_smoothness",
        "action_regularization": {
            "enabled": True,
            "turnover_penalty": 0.010,
            "smoothness_penalty": 0.030,
            "concentration_penalty": 0.0,
            "max_weight_penalty": 0.0,
            "kl_to_previous_penalty": 0.0,
            "normalize_penalties": True,
            "train_only": True,
        },
    },
    {
        "id": "g2e_concentration_mild",
        "experiment_name": "ssl_domain_g2e_concentration_mild",
        "short_notebook_name": "g2e.ipynb",
        "hypothesis": "Mild concentration penalty reduces overfit concentrated bets without eliminating active allocation.",
        "mechanism": "concentration_mild",
        "action_regularization": {
            "enabled": True,
            "turnover_penalty": 0.0,
            "smoothness_penalty": 0.0,
            "concentration_penalty": 0.015,
            "max_weight_penalty": 0.050,
            "max_weight_target": 0.20,
            "kl_to_previous_penalty": 0.0,
            "normalize_penalties": True,
            "train_only": True,
        },
    },
    {
        "id": "g2f_conservative_full",
        "experiment_name": "ssl_domain_g2f_conservative_full",
        "short_notebook_name": "g2f.ipynb",
        "hypothesis": "Combined conservative regularization tests whether stable, diversified action trajectories improve OOS robustness.",
        "mechanism": "turnover_smoothness_concentration",
        "action_regularization": {
            "enabled": True,
            "turnover_penalty": 0.010,
            "smoothness_penalty": 0.030,
            "concentration_penalty": 0.010,
            "max_weight_penalty": 0.030,
            "max_weight_target": 0.20,
            "kl_to_previous_penalty": 0.0,
            "normalize_penalties": True,
            "train_only": True,
        },
    },
    {
        "id": "g2g_overregularized_negative_control",
        "experiment_name": "ssl_domain_g2g_overregularized_negative_control",
        "short_notebook_name": "g2g.ipynb",
        "hypothesis": "Negative control: very strong penalties should expose inactive-policy false positives.",
        "mechanism": "overregularized_negative_control",
        "action_regularization": {
            "enabled": True,
            "turnover_penalty": 0.120,
            "smoothness_penalty": 0.100,
            "concentration_penalty": 0.050,
            "max_weight_penalty": 0.100,
            "max_weight_target": 0.12,
            "kl_to_previous_penalty": 0.0,
            "normalize_penalties": True,
            "train_only": True,
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
            "reward_mode": "custom_reward_with_training_only_action_regularization",
            "policy_mode": "custom_mlp_policy",
            "checkpoint_selection_rule": "checkpoint_robust_score",
            "configuration_selection_rule": "robust_q25_retention",
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "action_constraint": "long_only_box_[-1,1]_scaled_by_hmax",
        },
        "notes": {
            "execution_mode": "Run only base_macro with training-only action regularization.",
            "dataset_requirement": "Use processed_final_fixed_external_lagclean_full.csv.",
            "baseline_reuse_expected": False,
            "phase": "SSL/domain-generalization G2 action-regularization screening batch.",
            "variant_id": variant["id"],
            "pre_registered_hypothesis": variant["hypothesis"],
            "mechanism": variant["mechanism"],
            "seed_policy": "Screen with one seed first; rerun any positive variant with 3 seeds before claiming pass.",
            "isolation_rule": "Validation/test rewards are unregularized and checkpoint selection uses checkpoint_robust_score.",
            "kill_rules": "Reject if improvement is only lower turnover or if benchmark-relative OOS metrics do not improve.",
        },
        "runner_kwargs": {
            "base_config_name": "custom_custom",
            "candidate_feature_families": [],
            "feature_set_filter": ["base_macro"],
            "seeds": SCREENING_SEEDS,
            "total_timesteps": 200000,
            "max_folds": None,
            "es_mode": "relaxed",
            "dropout_p": 0.1,
            "eval_freq": 8192,
            "checkpoint_freq": 4096,
            "checkpoint_selection_rule": "checkpoint_robust_score",
            "domain_reward_scaling": {},
            "action_regularization": variant["action_regularization"],
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


def _write_notebook_from_template(template_path: Path, variant: dict[str, Any], config_path: Path, output_dir: Path) -> Path | None:
    if not template_path.exists():
        return None
    notebook = json.loads(template_path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0].get("source", []))
    source = source.replace(
        'CONFIG_PATH = PROJECT_ROOT / "configs" / "ssl_domain_g1a_abs_stress_mild.yaml"',
        f'CONFIG_PATH = PROJECT_ROOT / "configs" / "{config_path.name}"',
    )
    source = source.replace(
        'OUTPUT_DIR = PROJECT_ROOT / "SSL Domain Generalization" / "research_outputs_ssl_domain_g1a_abs_stress_mild"',
        f'OUTPUT_DIR = PROJECT_ROOT / "SSL Domain Generalization" / "{output_dir.name}"',
    )
    notebook["cells"][0]["source"] = source.splitlines(keepends=True)
    notebook_path = output_dir / str(variant["short_notebook_name"])
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    return notebook_path


def _markdown_manifest(rows: list[dict[str, Any]]) -> str:
    table_rows = "\n".join(
        (
            f"| `{row['variant_id']}` | `{row['mechanism']}` | "
            f"`{row['seed_count']}` | [config]({row['config_path']}) | "
            f"[cell]({row['portable_launch_cell']}) | "
            f"{'[notebook](' + row['notebook_path'] + ')' if row.get('notebook_path') else 'n/a'} |"
        )
        for row in rows
    )
    return f"""# SSL Domain G2 Parallel Batch

Status: ready to launch.

This is a one-seed screening batch for conservative action regularization on
`base_macro`. Validation/test accounting remains unregularized.

Any positive result must be rerun with three seeds before a pass claim.

| Variant | Mechanism | Seeds | Config | Portable launch cell | Huawei notebook |
|---|---|---:|---|---|---|
{table_rows}

## Gate

Do not count lower turnover alone as success. A G2 candidate needs better frozen
test Sharpe, primary benchmark-excess Sharpe, primary benchmark-excess return,
and no material drawdown deterioration. The overregularized negative control
must not be the best apparent result.
"""


def prepare_batch(dataset_path: Path) -> list[dict[str, Any]]:
    dataset_path = dataset_path.resolve()
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        config_path = CONFIG_ROOT / f"{variant['experiment_name']}.yaml"
        output_dir = OUTPUT_ROOT / f"research_outputs_{variant['experiment_name']}"
        output_dir.mkdir(parents=True, exist_ok=True)
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
        notebook_path = _write_notebook_from_template(
            G1A_NOTEBOOK_TEMPLATE,
            variant,
            config_path,
            output_dir,
        )

        action_reg = variant["action_regularization"]
        rows.append(
            {
                "variant_id": variant["id"],
                "experiment_name": variant["experiment_name"],
                "hypothesis": variant["hypothesis"],
                "mechanism": variant["mechanism"],
                "turnover_penalty": action_reg.get("turnover_penalty", 0.0),
                "smoothness_penalty": action_reg.get("smoothness_penalty", 0.0),
                "concentration_penalty": action_reg.get("concentration_penalty", 0.0),
                "max_weight_penalty": action_reg.get("max_weight_penalty", 0.0),
                "max_weight_target": action_reg.get("max_weight_target", ""),
                "config_path": config_path.relative_to(PROJECT_ROOT).as_posix(),
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "portable_launch_cell": portable_path.relative_to(PROJECT_ROOT).as_posix(),
                "notebook_path": notebook_path.relative_to(PROJECT_ROOT).as_posix()
                if notebook_path is not None
                else "",
                "preflight_status": preflight["summary"]["status"],
                "fold_count": preflight["summary"]["fold_count"],
                "seed_count": preflight["summary"]["seed_count"],
                "audit_ok": preflight["summary"]["audit_ok"],
            }
        )

    manifest_json = BATCH_ROOT / "g2_parallel_batch_manifest.json"
    manifest_csv = BATCH_ROOT / "g2_parallel_batch_manifest.csv"
    manifest_md = BATCH_ROOT / "G2_PARALLEL_BATCH_MANIFEST.md"
    root_manifest_md = OUTPUT_ROOT / "G2_PARALLEL_BATCH_MANIFEST.md"
    manifest_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(manifest_csv, index=False)
    manifest_text = _markdown_manifest(rows)
    manifest_md.write_text(manifest_text, encoding="utf-8")
    root_manifest_md.write_text(manifest_text, encoding="utf-8")
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SSL/domain G2 action-regularization batch.")
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
