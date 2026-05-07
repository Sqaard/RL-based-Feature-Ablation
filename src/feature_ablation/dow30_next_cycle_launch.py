from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import yaml

from dow30_horizon_a import (
    build_reference_experiment_config,
    build_next_cycle_feature_set_filter,
    ensure_candidate_feature_families,
    get_candidate_feature_set_name,
)
from dow30_notebook_research_runner import build_notebook_research_runner
from dow30_reference_notebook_runtime import (
    DEFAULT_REFERENCE_NOTEBOOK_PATH,
    load_reference_notebook_runtime,
)
from dow30_project_research import (
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_DIR,
    load_processed_dataset,
    run_research_gate,
)
from dow30_research_support import DEFAULT_FEATURE_GROUPS


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    if isinstance(value, tuple):
        return [_to_native(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_native(dict(payload)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_launch_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Launch config `{path}` must contain a mapping at the top level.")
    return payload


def _resolve_output_dir(
    *,
    config_path: str | Path,
    output_dir: Optional[str | Path],
) -> Path:
    launch_config = load_launch_config(config_path)
    return Path(output_dir or (DEFAULT_OUTPUT_DIR / launch_config["experiment_name"]))


def _snapshot_launch_config(
    *,
    launch_config: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir) / "launch_config_snapshot.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(_to_native(dict(launch_config)), sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def _prepare_dataframe_launch_input(
    df: pd.DataFrame,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    selected_candidate_family: Optional[str] = None,
    panel_scope: Optional[str] = None,
) -> tuple[pd.DataFrame, Path]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    candidate_feature_families = build_launch_kwargs(
        load_launch_config(config_path),
        selected_candidate_family=selected_candidate_family,
        panel_scope=panel_scope,
    )[
        "candidate_feature_families"
    ]
    processed = ensure_candidate_feature_families(
        df,
        candidate_feature_families=candidate_feature_families,
    )
    dataset_snapshot_path = resolved_output_dir / "processed_dataset_snapshot.csv"
    processed.to_csv(dataset_snapshot_path, index=False)
    return processed, dataset_snapshot_path


def _resolve_selected_candidate_family(
    launch_config: Mapping[str, Any],
    *,
    selected_candidate_family: Optional[str] = None,
) -> Optional[str]:
    if selected_candidate_family:
        return str(selected_candidate_family)

    launch_selection = dict(launch_config.get("launch_selection", {}))
    if launch_selection.get("candidate_family"):
        return str(launch_selection["candidate_family"])

    next_cycle_candidate = dict(launch_config.get("next_cycle_candidate", {}))
    if next_cycle_candidate.get("family_id"):
        return str(next_cycle_candidate["family_id"])

    runner_kwargs = dict(launch_config.get("runner_kwargs", {}))
    configured_families = tuple(runner_kwargs.get("candidate_feature_families", ()))
    if configured_families:
        return str(configured_families[0])
    return None


def _resolve_panel_scope(
    launch_config: Mapping[str, Any],
    *,
    panel_scope: Optional[str] = None,
) -> Optional[str]:
    if panel_scope:
        return str(panel_scope)
    launch_selection = dict(launch_config.get("launch_selection", {}))
    if launch_selection.get("panel_scope"):
        return str(launch_selection["panel_scope"])
    return None


def build_launch_kwargs(
    launch_config: Mapping[str, Any],
    *,
    selected_candidate_family: Optional[str] = None,
    panel_scope: Optional[str] = None,
) -> dict[str, Any]:
    runner_kwargs = dict(launch_config.get("runner_kwargs", {}))
    reference_baseline = dict(launch_config.get("reference_baseline", {}))
    resolved_candidate_family = _resolve_selected_candidate_family(
        launch_config,
        selected_candidate_family=selected_candidate_family,
    )
    resolved_panel_scope = _resolve_panel_scope(
        launch_config,
        panel_scope=panel_scope,
    )

    if resolved_candidate_family is not None:
        candidate_feature_families = (resolved_candidate_family,)
    else:
        candidate_feature_families = tuple(runner_kwargs.get("candidate_feature_families", ()))

    if resolved_candidate_family is not None and resolved_panel_scope is not None:
        feature_set_filter = tuple(
            build_next_cycle_feature_set_filter(
                resolved_candidate_family,
                panel_scope=resolved_panel_scope,
            )
        )
    else:
        feature_set_filter = tuple(runner_kwargs.get("feature_set_filter", ()))

    return {
        "base_config_name": runner_kwargs["base_config_name"],
        "feature_groups": "DEFAULT_FEATURE_GROUPS",
        "candidate_feature_families": candidate_feature_families,
        "feature_set_filter": feature_set_filter,
        "seeds": tuple(runner_kwargs.get("seeds", ())),
        "total_timesteps": int(runner_kwargs["total_timesteps"]),
        "max_folds": runner_kwargs.get("max_folds"),
        "es_mode": runner_kwargs["es_mode"],
        "dropout_p": float(runner_kwargs["dropout_p"]),
        "eval_freq": int(runner_kwargs["eval_freq"]),
        "checkpoint_freq": int(runner_kwargs["checkpoint_freq"]),
        "checkpoint_selection_rule": str(
            runner_kwargs.get(
                "checkpoint_selection_rule",
                reference_baseline.get("checkpoint_selection_rule", "checkpoint_robust_score"),
            )
        ),
        "domain_reward_scaling": dict(runner_kwargs.get("domain_reward_scaling", {})),
        "action_regularization": dict(runner_kwargs.get("action_regularization", {})),
        "verbose": int(runner_kwargs["verbose"]),
        "selected_candidate_family": resolved_candidate_family,
        "panel_scope": resolved_panel_scope,
    }


def format_notebook_launch_snippet(
    *,
    config_path: str | Path,
    processed_dataset_path: str | Path,
    output_dir: str | Path,
    selected_candidate_family: Optional[str],
    panel_scope: Optional[str],
) -> str:
    optional_lines: list[str] = []
    if selected_candidate_family is not None:
        optional_lines.append(f'    selected_candidate_family="{selected_candidate_family}",')
    if panel_scope is not None:
        optional_lines.append(f'    panel_scope="{panel_scope}",')
    optional_block = "\n".join(optional_lines)
    if optional_block:
        optional_block = "\n" + optional_block
    horizon_a_root = Path(config_path).resolve().parents[1] / "Ablation Ladder v2"
    return f"""import sys
from pathlib import Path

HORIZON_A_ROOT = Path(r"{horizon_a_root}")
if str(HORIZON_A_ROOT) not in sys.path:
    sys.path.insert(0, str(HORIZON_A_ROOT))

from dow30_next_cycle_launch import run_bootstrapped_notebook_launch_from_csv

launch_bundle = run_bootstrapped_notebook_launch_from_csv(
    config_path=r"{Path(config_path).resolve()}",
    dataset_path=r"{Path(processed_dataset_path).resolve()}",
    output_dir=r"{Path(output_dir).resolve()}",{optional_block}
)

preflight = launch_bundle["preflight"]
research_bundle = launch_bundle["research_bundle"]
"""


def format_post_run_rebuild_commands(
    *,
    output_dir: str | Path,
    processed_dataset_path: str | Path,
) -> list[str]:
    output = Path(output_dir).resolve()
    processed = Path(processed_dataset_path).resolve()
    return [
        (
            "python -m dow30_reporting build-benchmark-suite "
            f"--dataset \"{processed}\" "
            f"--folds-input \"{output / 'walk_forward_folds.csv'}\" "
            f"--output \"{output / 'benchmark_suite_daily.csv'}\""
        ),
        (
            "python -m dow30_reporting rebuild-walkforward-report "
            f"--input \"{output / 'walk_forward_results.csv'}\" "
            f"--daily-input \"{output / 'walk_forward_daily_test_returns.csv'}\" "
            f"--test-actions-input \"{output / 'walk_forward_test_actions.csv'}\" "
            f"--test-observations-input \"{output / 'walk_forward_test_observations.csv'}\" "
            f"--benchmark-suite-input \"{output / 'benchmark_suite_daily.csv'}\" "
            f"--outdir \"{output / 'analysis_rebuilt'}\""
        ),
    ]


def _run_launch_preflight_core(
    *,
    processed_df: pd.DataFrame,
    config_path: str | Path,
    dataset_label: str,
    output_dir: str | Path,
    selected_candidate_family: Optional[str] = None,
    panel_scope: Optional[str] = None,
) -> dict[str, Any]:
    launch_config = load_launch_config(config_path)
    launch_kwargs = build_launch_kwargs(
        launch_config,
        selected_candidate_family=selected_candidate_family,
        panel_scope=panel_scope,
    )
    reference_config = build_reference_experiment_config(
        launch_kwargs["base_config_name"],
        seed_list=launch_kwargs["seeds"],
    ).to_dict()
    reference_config["checkpoint_selection_rule"] = launch_kwargs["checkpoint_selection_rule"]

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    gate = run_research_gate(
        processed_df,
        output_dir=resolved_output_dir,
        reference_config=reference_config,
        candidate_feature_families=launch_kwargs["candidate_feature_families"],
        feature_set_filter=launch_kwargs["feature_set_filter"],
    )
    notebook_snippet = format_notebook_launch_snippet(
        config_path=config_path,
        processed_dataset_path=dataset_label,
        output_dir=resolved_output_dir,
        selected_candidate_family=launch_kwargs["selected_candidate_family"],
        panel_scope=launch_kwargs["panel_scope"],
    )
    rebuild_commands = format_post_run_rebuild_commands(
        output_dir=resolved_output_dir,
        processed_dataset_path=dataset_label,
    )
    config_snapshot_path = _snapshot_launch_config(
        launch_config=launch_config,
        output_dir=resolved_output_dir,
    )

    available_feature_sets = list(gate["data_card"]["feature_sets"].keys())
    requested_feature_sets = list(launch_kwargs["feature_set_filter"])
    missing_requested_feature_sets = [
        feature_set for feature_set in requested_feature_sets if feature_set not in available_feature_sets
    ]

    summary = {
        "status": "ready_to_launch" if not missing_requested_feature_sets else "blocked",
        "experiment_name": launch_config["experiment_name"],
        "config_path": str(Path(config_path).resolve()),
        "dataset_label": dataset_label,
        "resolved_output_dir": str(resolved_output_dir.resolve()),
        "candidate_feature_families": list(launch_kwargs["candidate_feature_families"]),
        "selected_candidate_family": launch_kwargs["selected_candidate_family"],
        "selected_candidate_feature_set": (
            get_candidate_feature_set_name(launch_kwargs["selected_candidate_family"])
            if launch_kwargs["selected_candidate_family"] is not None
            else None
        ),
        "panel_scope": launch_kwargs["panel_scope"],
        "feature_set_filter": requested_feature_sets,
        "available_feature_sets": available_feature_sets,
        "missing_requested_feature_sets": missing_requested_feature_sets,
        "seed_count": len(launch_kwargs["seeds"]),
        "fold_count": len(gate["folds"]),
        "audit_ok": bool(gate["audit"].get("ok")),
        "checkpoint_selection_rule": launch_kwargs["checkpoint_selection_rule"],
        "domain_reward_scaling": launch_kwargs["domain_reward_scaling"],
        "action_regularization": launch_kwargs["action_regularization"],
        "user_action_required_now": True,
        "launch_mode": "notebook_config_driven",
        "launch_config_snapshot_path": str(config_snapshot_path.resolve()),
        "next_user_action": "Launch the notebook experiment with the generated snippet." if not missing_requested_feature_sets else "Resolve missing requested feature sets before launch.",
        "post_run_rebuild_commands": rebuild_commands,
    }

    _write_json(summary, resolved_output_dir / "launch_preflight_report.json")
    (resolved_output_dir / "launch_notebook_cell.py").write_text(notebook_snippet, encoding="utf-8")
    _write_json({"runner_kwargs": _to_native(launch_kwargs)}, resolved_output_dir / "launch_kwargs.json")
    _write_json(
        {"commands": rebuild_commands},
        resolved_output_dir / "post_run_rebuild_commands.json",
    )

    return {
        "summary": summary,
        "gate": gate,
        "launch_kwargs": launch_kwargs,
        "processed_df": processed_df.copy(),
        "notebook_snippet": notebook_snippet,
        "post_run_rebuild_commands": rebuild_commands,
    }


def run_launch_preflight(
    *,
    config_path: str | Path,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: Optional[str | Path] = None,
    selected_candidate_family: Optional[str] = None,
    panel_scope: Optional[str] = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Processed dataset CSV was not found at "
            f"`{dataset_path}`. If the dataset currently exists only inside the notebook, "
            "run `run_launch_preflight_from_dataframe(processed.copy(), ...)` instead."
        )
    processed = load_processed_dataset(
        dataset_path,
        candidate_feature_families=build_launch_kwargs(
            load_launch_config(config_path),
            selected_candidate_family=selected_candidate_family,
            panel_scope=panel_scope,
        )["candidate_feature_families"],
    )
    resolved_output_dir = _resolve_output_dir(
        config_path=config_path,
        output_dir=output_dir,
    )
    return _run_launch_preflight_core(
        processed_df=processed,
        config_path=config_path,
        dataset_label=str(Path(dataset_path).resolve()),
        output_dir=resolved_output_dir,
        selected_candidate_family=selected_candidate_family,
        panel_scope=panel_scope,
    )


def run_launch_preflight_from_dataframe(
    df: pd.DataFrame,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    dataset_label: str = "processed_dataframe_in_memory",
    selected_candidate_family: Optional[str] = None,
    panel_scope: Optional[str] = None,
) -> dict[str, Any]:
    processed, dataset_snapshot_path = _prepare_dataframe_launch_input(
        df,
        config_path=config_path,
        output_dir=output_dir,
        selected_candidate_family=selected_candidate_family,
        panel_scope=panel_scope,
    )
    return _run_launch_preflight_core(
        processed_df=processed,
        config_path=config_path,
        dataset_label=str(dataset_snapshot_path.resolve()),
        output_dir=output_dir,
        selected_candidate_family=selected_candidate_family,
        panel_scope=panel_scope,
    )


def run_notebook_launch_from_dataframe(
    df: pd.DataFrame,
    *,
    notebook_ns: Mapping[str, Any],
    config_path: str | Path,
    output_dir: Optional[str | Path] = None,
    selected_candidate_family: Optional[str] = None,
    panel_scope: Optional[str] = None,
) -> dict[str, Any]:
    resolved_output_dir = _resolve_output_dir(
        config_path=config_path,
        output_dir=output_dir,
    )
    preflight = run_launch_preflight_from_dataframe(
        df,
        config_path=config_path,
        output_dir=resolved_output_dir,
        selected_candidate_family=selected_candidate_family,
        panel_scope=panel_scope,
    )
    if preflight["summary"]["status"] != "ready_to_launch":
        raise ValueError(
            "Launch preflight did not pass. Resolve the reported issues before starting the notebook run."
        )

    launch_kwargs = preflight["launch_kwargs"]
    research_runner = build_notebook_research_runner(notebook_ns)
    research_bundle = research_runner(
        df=preflight["processed_df"].copy(),
        output_dir=resolved_output_dir,
        base_config_name=launch_kwargs["base_config_name"],
        feature_groups=DEFAULT_FEATURE_GROUPS,
        candidate_feature_families=launch_kwargs["candidate_feature_families"],
        feature_set_filter=launch_kwargs["feature_set_filter"],
        seeds=launch_kwargs["seeds"],
        total_timesteps=launch_kwargs["total_timesteps"],
        max_folds=launch_kwargs["max_folds"],
        es_mode=launch_kwargs["es_mode"],
        dropout_p=launch_kwargs["dropout_p"],
        eval_freq=launch_kwargs["eval_freq"],
        checkpoint_freq=launch_kwargs["checkpoint_freq"],
        checkpoint_selection_rule=launch_kwargs["checkpoint_selection_rule"],
        domain_reward_scaling=launch_kwargs["domain_reward_scaling"],
        action_regularization=launch_kwargs["action_regularization"],
        verbose=launch_kwargs["verbose"],
    )

    execution_summary = {
        "status": "completed",
        "experiment_name": preflight["summary"]["experiment_name"],
        "config_path": preflight["summary"]["config_path"],
        "resolved_output_dir": str(resolved_output_dir.resolve()),
        "dataset_snapshot_path": preflight["summary"]["dataset_label"],
        "launch_mode": "notebook_config_driven",
        "selected_candidate_family": launch_kwargs["selected_candidate_family"],
        "selected_candidate_feature_set": (
            get_candidate_feature_set_name(launch_kwargs["selected_candidate_family"])
            if launch_kwargs["selected_candidate_family"] is not None
            else None
        ),
        "panel_scope": launch_kwargs["panel_scope"],
        "checkpoint_selection_rule": launch_kwargs["checkpoint_selection_rule"],
        "domain_reward_scaling": launch_kwargs["domain_reward_scaling"],
        "action_regularization": launch_kwargs["action_regularization"],
        "user_action_required_now": False,
        "next_user_action": "Inspect the generated reports in the launch output directory.",
    }
    _write_json(execution_summary, resolved_output_dir / "launch_execution_report.json")

    return {
        "preflight": preflight,
        "research_bundle": research_bundle,
        "execution_summary": execution_summary,
    }


def run_bootstrapped_notebook_launch_from_csv(
    *,
    config_path: str | Path,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: Optional[str | Path] = None,
    reference_notebook_path: str | Path = DEFAULT_REFERENCE_NOTEBOOK_PATH,
    selected_candidate_family: Optional[str] = None,
    panel_scope: Optional[str] = None,
) -> dict[str, Any]:
    candidate_feature_families = build_launch_kwargs(
        load_launch_config(config_path),
        selected_candidate_family=selected_candidate_family,
        panel_scope=panel_scope,
    )[
        "candidate_feature_families"
    ]
    processed = load_processed_dataset(
        dataset_path,
        candidate_feature_families=candidate_feature_families,
    )
    notebook_ns = load_reference_notebook_runtime(
        notebook_path=reference_notebook_path,
    )
    return run_notebook_launch_from_dataframe(
        processed,
        notebook_ns=notebook_ns,
        config_path=config_path,
        output_dir=output_dir,
        selected_candidate_family=selected_candidate_family,
        panel_scope=panel_scope,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Next-cycle launch utilities for Dow30 Horizon A.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight-launch",
        help="Validate a launch config, dataset, and exact feature panel before running the notebook experiment.",
    )
    preflight_parser.add_argument("--config", required=True, help="Launch YAML config path.")
    preflight_parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Processed dataset CSV path.",
    )
    preflight_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override for the generated preflight artifacts.",
    )
    preflight_parser.add_argument(
        "--selected-candidate-family",
        default=None,
        help="Optional implemented candidate family override for this launch.",
    )
    preflight_parser.add_argument(
        "--panel-scope",
        default=None,
        help=(
            "Optional panel scope override. Supported values include "
            "`candidate_only`, `candidate_plus_reference_panel`, and "
            "`candidate_plus_reference_with_base_anchor`."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.command == "preflight-launch":
        run_launch_preflight(
            config_path=args.config,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            selected_candidate_family=args.selected_candidate_family,
            panel_scope=args.panel_scope,
        )
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
