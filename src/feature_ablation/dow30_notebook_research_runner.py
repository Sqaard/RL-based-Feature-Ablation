from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from dow30_horizon_a import (
    PRIMARY_BENCHMARK_ID,
    build_reference_experiment_config,
    ensure_candidate_feature_families,
    ensure_event_calendar_features,
    infer_feature_metadata,
)
from dow30_project_research import (
    PROJECT_SELECTION_CONFIG,
    build_train_only_splits,
    run_research_gate,
)
from dow30_research_support import (
    DEFAULT_FEATURE_GROUPS,
    build_daily_test_export,
    build_fold_benchmark_suite_export,
    build_test_action_export,
    build_test_observation_export,
    build_domain_reward_scale_by_date,
    compute_temporal_subdomain_metrics,
    evaluate_equity_curve,
    make_action_regularized_env_cls,
    make_domain_reward_scaled_env_cls,
    normalize_action_regularization_config,
    run_feature_ablation_ladder,
    select_best_artifact,
)


REQUIRED_NOTEBOOK_NAMES = [
    "StockTradingEnv",
    "ZhangStockTradingEnv",
    "CustomStockTradingEnv",
    "DRLAgent",
    "PPO",
    "PPO_model_kwargs",
    "build_callbacks",
    "build_custom_policy_kwargs",
    "evaluate_model_on_env",
    "set_all_seeds",
]


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, type):
        return value.__name__
    if callable(value) and hasattr(value, "__name__"):
        return value.__name__
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _require_notebook_names(ns: Mapping[str, Any]) -> None:
    missing = [name for name in REQUIRED_NOTEBOOK_NAMES if name not in ns]
    if missing:
        raise KeyError(
            "Notebook is missing required names for walk-forward runner: "
            + ", ".join(missing)
        )


def get_eval_env_kwargs(fold_env_spec: Mapping[str, Any], split_name: str) -> Mapping[str, Any]:
    kwargs_key = "train_eval_kwargs" if split_name == "train" else f"{split_name}_kwargs"
    return fold_env_spec[kwargs_key]


def build_notebook_research_runner(ns: Mapping[str, Any]):
    _require_notebook_names(ns)

    StockTradingEnv = ns["StockTradingEnv"]
    ZhangStockTradingEnv = ns["ZhangStockTradingEnv"]
    CustomStockTradingEnv = ns["CustomStockTradingEnv"]
    DRLAgent = ns["DRLAgent"]
    PPO = ns["PPO"]
    PPO_model_kwargs = dict(ns["PPO_model_kwargs"])
    build_callbacks = ns["build_callbacks"]
    build_custom_policy_kwargs = ns["build_custom_policy_kwargs"]
    evaluate_model_on_env = ns["evaluate_model_on_env"]
    set_all_seeds = ns["set_all_seeds"]

    sigma_target = ns.get("sigma_target", 0.15 / (252 ** 0.5))
    mu_param = ns.get("mu_param", 1.0)
    lambda_param = ns.get("lambda_param", 0.01)

    def _evaluate_model_on_env_with_observation_trace(model, environment) -> dict[str, Any]:
        try:
            test_env, test_obs = environment.get_sb_env()
            test_env.reset()
            max_steps = len(environment.df.index.unique()) - 1
            observation_rows: list[dict[str, Any]] = []
            for observation_row_id in range(max_steps):
                obs_flat = np.asarray(test_obs, dtype=float).reshape(-1)
                action, _states = model.predict(test_obs, deterministic=True)
                action_flat = np.asarray(action, dtype=float).reshape(-1)
                row = {"observation_row_id": observation_row_id}
                row.update({f"obs_{idx:04d}": value for idx, value in enumerate(obs_flat)})
                row.update({f"raw_policy_action_{idx:03d}": value for idx, value in enumerate(action_flat)})
                observation_rows.append(row)
                test_obs, rewards, dones, info = test_env.step(action)
                if bool(np.asarray(dones).reshape(-1)[0]):
                    break

            account_memory = test_env.env_method(method_name="save_asset_memory")
            actions_memory = test_env.env_method(method_name="save_action_memory")
            df_account_value = account_memory[0]
            df_action = actions_memory[0]
            df_observation = pd.DataFrame(observation_rows)
            if not df_observation.empty:
                if "date" in df_action.columns:
                    action_dates = pd.to_datetime(df_action["date"], errors="coerce")
                else:
                    action_dates = pd.to_datetime(pd.Index(df_action.index), errors="coerce")
                df_observation = df_observation.iloc[: len(action_dates)].copy()
                df_observation.insert(1, "date", action_dates[: len(df_observation)].to_numpy())

            initial_val = df_account_value.iloc[0, 1]
            final_val = df_account_value.iloc[-1, 1]
            returns_pct = ((final_val / initial_val) - 1) * 100
            return {
                "df_account_value": df_account_value,
                "df_action": df_action,
                "df_observation": df_observation,
                "stats": {},
                "stats_df": pd.DataFrame(),
                "initial_val": initial_val,
                "final_val": final_val,
                "return_pct": returns_pct,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "annual_return": np.nan,
            }
        except Exception:
            raw_result = evaluate_model_on_env(model, environment)
            raw_result["df_observation"] = pd.DataFrame()
            return raw_result

    config_specs = {
        "finrl_finrl": {
            "env_cls": StockTradingEnv,
            "use_custom_policy": False,
            "extra_kwargs": {},
        },
        "finrl_custom": {
            "env_cls": StockTradingEnv,
            "use_custom_policy": True,
            "extra_kwargs": {},
        },
        "zhang_finrl": {
            "env_cls": ZhangStockTradingEnv,
            "use_custom_policy": False,
            "extra_kwargs": {"sigma_target": sigma_target, "mu": mu_param},
        },
        "zhang_custom": {
            "env_cls": ZhangStockTradingEnv,
            "use_custom_policy": True,
            "extra_kwargs": {"sigma_target": sigma_target, "mu": mu_param},
        },
        "custom_finrl": {
            "env_cls": CustomStockTradingEnv,
            "use_custom_policy": False,
            "extra_kwargs": {"sigma_target": sigma_target, "lambda_param": lambda_param},
        },
        "custom_custom": {
            "env_cls": CustomStockTradingEnv,
            "use_custom_policy": True,
            "extra_kwargs": {"sigma_target": sigma_target, "lambda_param": lambda_param},
        },
    }

    def _prepare_research_df(
        df: pd.DataFrame,
        *,
        candidate_feature_families: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        out = df.copy()
        unnamed_cols = [col for col in out.columns if str(col).startswith("Unnamed:")]
        if unnamed_cols:
            out = out.drop(columns=unnamed_cols)
        out["date"] = pd.to_datetime(out["date"])
        if "date_available" in out.columns:
            out["date_available"] = pd.to_datetime(out["date_available"], errors="coerce")
        out = ensure_candidate_feature_families(
            out,
            candidate_feature_families=candidate_feature_families,
            date_col="date",
        )
        return out.sort_values(["date", "tic"]).reset_index(drop=True)

    def _to_finrl_panel_frame(df: pd.DataFrame) -> pd.DataFrame:
        """
        FinRL StockTradingEnv expects the index to represent trading days, with
        the same integer index repeated for all tickers on the same date.
        """
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values(["date", "tic"]).reset_index(drop=True)
        out.index = out["date"].factorize()[0]
        return out

    def _build_fold_env_spec(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: Sequence[str],
        base_config_name: str,
        print_verbosity: int = 10_000,
        domain_reward_scaling: Optional[Mapping[str, Any]] = None,
        action_regularization: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        if base_config_name not in config_specs:
            raise ValueError(f"Unknown base_config_name: {base_config_name}")

        stock_dimension = int(train_df["tic"].nunique())
        state_space = 1 + (2 * stock_dimension) + (len(feature_cols) * stock_dimension)

        base_env_kwargs = {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "buy_cost_pct": [0.001] * stock_dimension,
            "sell_cost_pct": [0.001] * stock_dimension,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": list(feature_cols),
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
            "print_verbosity": max(1, int(print_verbosity)),
        }

        spec = config_specs[base_config_name]
        env_cls = spec["env_cls"]
        extra_kwargs = dict(spec["extra_kwargs"])
        reward_scaling_payload = build_domain_reward_scale_by_date(
            train_df,
            domain_reward_scaling,
        )
        train_env_cls = (
            make_domain_reward_scaled_env_cls(env_cls)
            if reward_scaling_payload.get("enabled")
            else env_cls
        )
        action_regularization_config = normalize_action_regularization_config(action_regularization)
        if action_regularization_config.get("enabled"):
            train_env_cls = make_action_regularized_env_cls(train_env_cls)
        domain_train_kwargs = {}
        if reward_scaling_payload.get("enabled"):
            domain_train_kwargs = {
                "domain_reward_scale_by_date": reward_scaling_payload["scale_by_date"],
                "domain_reward_scaling_report": reward_scaling_payload["report"],
            }
        action_train_kwargs = {}
        if action_regularization_config.get("enabled"):
            action_train_kwargs = {
                "action_regularization_config": action_regularization_config,
            }

        return {
            "base_config_name": base_config_name,
            "env_cls": env_cls,
            "train_env_cls": train_env_cls,
            "use_custom_policy": spec["use_custom_policy"],
            "feature_cols": list(feature_cols),
            "stock_dimension": stock_dimension,
            "state_space": state_space,
            "train_df": _to_finrl_panel_frame(train_df),
            "validation_df": _to_finrl_panel_frame(validation_df),
            "test_df": _to_finrl_panel_frame(test_df),
            "domain_reward_scaling_report": reward_scaling_payload["report"],
            "action_regularization_report": action_regularization_config,
            "train_kwargs": {
                "df": _to_finrl_panel_frame(train_df),
                "num_stock_shares": [0] * stock_dimension,
                **base_env_kwargs,
                **extra_kwargs,
                **domain_train_kwargs,
                **action_train_kwargs,
            },
            "train_eval_kwargs": {
                "df": _to_finrl_panel_frame(train_df),
                "num_stock_shares": [0] * stock_dimension,
                **base_env_kwargs,
                **extra_kwargs,
            },
            "validation_kwargs": {
                "df": _to_finrl_panel_frame(validation_df),
                "num_stock_shares": [0] * stock_dimension,
                "turbulence_threshold": 70,
                "risk_indicator_col": "turbulence",
                **base_env_kwargs,
                **extra_kwargs,
            },
            "test_kwargs": {
                "df": _to_finrl_panel_frame(test_df),
                "num_stock_shares": [0] * stock_dimension,
                "turbulence_threshold": 70,
                "risk_indicator_col": "turbulence",
                **base_env_kwargs,
                **extra_kwargs,
            },
        }

    def _make_train_env(fold_env_spec: Mapping[str, Any]):
        train_env_cls = fold_env_spec.get("train_env_cls", fold_env_spec["env_cls"])
        train_env = train_env_cls(**fold_env_spec["train_kwargs"])
        env_train, _ = train_env.get_sb_env()
        return env_train, train_env

    def _make_eval_env(fold_env_spec: Mapping[str, Any], split_name: str):
        kwargs = get_eval_env_kwargs(fold_env_spec, split_name)
        return fold_env_spec["env_cls"](**kwargs)

    def _evaluate_model(model, fold_env_spec: Mapping[str, Any], split_name: str) -> dict[str, Any]:
        environment = _make_eval_env(fold_env_spec, split_name)
        raw_result = _evaluate_model_on_env_with_observation_trace(model, environment)
        regime_frame = None
        split_df = fold_env_spec.get(f"{split_name}_df")
        if isinstance(split_df, pd.DataFrame) and "Market_Regime" in split_df.columns:
            regime_frame = split_df[["date", "Market_Regime"]].drop_duplicates()

        equity_eval = evaluate_equity_curve(
            raw_result["df_account_value"],
            df_actions=raw_result.get("df_action"),
            regime_frame=regime_frame,
        )

        return {
            "raw_result": raw_result,
            "curve": equity_eval["curve"],
            "metrics": equity_eval["metrics"],
            "regime_breakdown": equity_eval["regime_breakdown"],
        }

    def _evaluate_model_path(
        model_path: str,
        fold_env_spec: Mapping[str, Any],
    ) -> dict[str, Any]:
        model = PPO.load(model_path)
        train_eval = _evaluate_model(model, fold_env_spec, "train")
        validation_eval = _evaluate_model(model, fold_env_spec, "validation")
        validation_subdomains = compute_temporal_subdomain_metrics(validation_eval["curve"])

        return {
            "model_path": model_path,
            "train_sharpe": train_eval["metrics"]["sharpe"],
            "train_return_pct": train_eval["metrics"]["return_pct"],
            "train_max_drawdown": train_eval["metrics"]["max_drawdown"],
            "validation_sharpe": validation_eval["metrics"]["sharpe"],
            "validation_return_pct": validation_eval["metrics"]["return_pct"],
            "validation_max_drawdown": validation_eval["metrics"]["max_drawdown"],
            "validation_turnover": validation_eval["metrics"]["turnover"],
            "validation_first_half_sharpe": validation_subdomains["first_half_sharpe"],
            "validation_second_half_sharpe": validation_subdomains["second_half_sharpe"],
            "validation_subdomain_sharpe_floor": validation_subdomains["subdomain_sharpe_floor"],
            "validation_subdomain_sharpe_gap": validation_subdomains["subdomain_sharpe_gap"],
        }

    def _list_checkpoints(checkpoints_dir: str) -> list[str]:
        paths = sorted(Path(checkpoints_dir).glob("*.zip"))
        return [str(path) for path in paths]

    def _train_one_fold_config(
        fold_env_spec: Mapping[str, Any],
        run_dir: str | Path,
        total_timesteps: int,
        seed: int,
        es_mode: str,
        dropout_p: float,
        eval_freq: int,
        checkpoint_freq: int,
        verbose: int,
    ) -> dict[str, Any]:
        set_all_seeds(seed)

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        existing_last_model = run_dir / "last_model.zip"
        existing_best_model = run_dir / "best_model" / "best_model.zip"
        existing_checkpoints_dir = run_dir / "checkpoints"
        if existing_last_model.exists() and existing_checkpoints_dir.exists():
            return {
                "run_dir": str(run_dir),
                "best_model_dir": str(run_dir / "best_model"),
                "best_model_path": str(existing_best_model),
                "checkpoints_dir": str(existing_checkpoints_dir),
                "logs_dir": str(run_dir / "eval_logs"),
                "last_model_path": str(existing_last_model),
                "policy_kwargs": None,
                "reused_existing_run": True,
            }

        env_train, train_env = _make_train_env(fold_env_spec)
        eval_env = _make_eval_env(fold_env_spec, "validation")
        agent = DRLAgent(env=env_train)

        if fold_env_spec["use_custom_policy"]:
            policy_kwargs = build_custom_policy_kwargs(dropout_p=dropout_p)
            model = agent.get_model(
                "ppo",
                model_kwargs=PPO_model_kwargs,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
            )
        else:
            policy_kwargs = None
            model = agent.get_model(
                "ppo",
                model_kwargs=PPO_model_kwargs,
                verbose=verbose,
            )

        callbacks, paths = build_callbacks(
            eval_env=eval_env,
            run_dir=str(run_dir),
            es_mode=es_mode,
            eval_freq=eval_freq,
            checkpoint_freq=checkpoint_freq,
            verbose=max(verbose, 1),
        )

        trained_model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=run_dir.name,
            callback=callbacks,
            progress_bar=True,
        )

        last_model_path = run_dir / "last_model.zip"
        trained_model.save(str(last_model_path))

        paths["last_model_path"] = str(last_model_path)
        paths["policy_kwargs"] = policy_kwargs
        if hasattr(train_env, "get_action_regularization_audit_frame"):
            audit_frame = train_env.get_action_regularization_audit_frame()
            if isinstance(audit_frame, pd.DataFrame) and not audit_frame.empty:
                audit_path = run_dir / "action_regularization_training_audit.csv"
                audit_frame.to_csv(audit_path, index=False)
                paths["action_regularization_training_audit_path"] = str(audit_path)
        if hasattr(train_env, "get_action_regularization_summary"):
            summary = train_env.get_action_regularization_summary()
            if summary.get("enabled"):
                summary_path = run_dir / "action_regularization_training_summary.json"
                summary_path.write_text(
                    json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                paths["action_regularization_training_summary_path"] = str(summary_path)
        meta_path = run_dir / "run_meta.json"
        meta_path.write_text(
            json.dumps(_json_safe(paths), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return paths

    def _build_candidate_table(paths: Mapping[str, Any], fold_env_spec: Mapping[str, Any]) -> pd.DataFrame:
        rows = []
        best_model_path = paths.get("best_model_path")
        last_model_path = paths.get("last_model_path")

        if best_model_path and Path(best_model_path).exists():
            row = _evaluate_model_path(best_model_path, fold_env_spec)
            row["artifact_type"] = "best_model"
            row["timestep"] = np.nan
            rows.append(row)

        if last_model_path and Path(last_model_path).exists():
            row = _evaluate_model_path(last_model_path, fold_env_spec)
            row["artifact_type"] = "last_model"
            row["timestep"] = np.nan
            rows.append(row)

        checkpoints_dir = paths.get("checkpoints_dir")
        if checkpoints_dir:
            for checkpoint_path in _list_checkpoints(checkpoints_dir):
                row = _evaluate_model_path(checkpoint_path, fold_env_spec)
                row["artifact_type"] = "checkpoint"
                row["timestep"] = Path(checkpoint_path).stem
                rows.append(row)

        return pd.DataFrame(rows)

    def _run_single_fold(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: Sequence[str],
        fold: Any,
        seed: int,
        feature_set_name: str,
        *,
        base_config_name: str,
        total_timesteps: int,
        output_dir: str | Path,
        es_mode: str,
        dropout_p: float,
        eval_freq: int,
        checkpoint_freq: int,
        checkpoint_selection_rule: str,
        domain_reward_scaling: Optional[Mapping[str, Any]],
        action_regularization: Optional[Mapping[str, Any]],
        verbose: int,
        benchmark_suite_frame: Optional[pd.DataFrame] = None,
    ) -> Mapping[str, Any]:
        train_scaled, validation_scaled, test_scaled, preprocessing_summary = build_train_only_splits(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            feature_cols=feature_cols,
        )

        fold_env_spec = _build_fold_env_spec(
            train_df=train_scaled,
            validation_df=validation_scaled,
            test_df=test_scaled,
            feature_cols=feature_cols,
            base_config_name=base_config_name,
            print_verbosity=10_000,
            domain_reward_scaling=domain_reward_scaling,
            action_regularization=action_regularization,
        )

        run_dir = (
            Path(output_dir)
            / "fold_runs"
            / base_config_name
            / feature_set_name
            / fold.fold_id
            / f"seed_{seed}"
        )

        paths = _train_one_fold_config(
            fold_env_spec=fold_env_spec,
            run_dir=run_dir,
            total_timesteps=total_timesteps,
            seed=seed,
            es_mode=es_mode,
            dropout_p=dropout_p,
            eval_freq=eval_freq,
            checkpoint_freq=checkpoint_freq,
            verbose=verbose,
        )

        candidate_df = _build_candidate_table(paths, fold_env_spec)
        if candidate_df.empty:
            raise RuntimeError(f"No candidate artifacts were produced in {run_dir}")

        scored_candidate_df, selected_artifact = select_best_artifact(
            candidate_df,
            selection_rule=checkpoint_selection_rule,
            **PROJECT_SELECTION_CONFIG,
        )

        model = PPO.load(selected_artifact["model_path"])
        train_eval = _evaluate_model(model, fold_env_spec, "train")
        validation_eval = _evaluate_model(model, fold_env_spec, "validation")
        test_eval = _evaluate_model(model, fold_env_spec, "test")
        feature_metadata = infer_feature_metadata(feature_set_name)
        primary_benchmark_frame = pd.DataFrame()
        if isinstance(benchmark_suite_frame, pd.DataFrame) and not benchmark_suite_frame.empty:
            primary_benchmark_frame = benchmark_suite_frame[
                benchmark_suite_frame["benchmark_id"] == PRIMARY_BENCHMARK_ID
            ][
                [
                    "date",
                    "benchmark_id",
                    "benchmark_return",
                    "benchmark_turnover",
                    "benchmark_transaction_cost",
                ]
            ].copy()
        daily_test_frame = build_daily_test_export(
            test_eval["curve"],
            raw_test_df=test_df,
            run_key=f"{feature_set_name}__{fold.fold_id}__seed{seed}",
            feature_set=feature_set_name,
            feature_family=feature_metadata["feature_family"],
            is_negative_control=feature_metadata["is_negative_control"],
            fold_id=fold.fold_id,
            seed=seed,
            selected_model_type=selected_artifact.get("artifact_type"),
            selection_rule=checkpoint_selection_rule,
            df_actions=test_eval["raw_result"].get("df_action"),
            benchmark_frame=primary_benchmark_frame,
        )
        test_action_frame = build_test_action_export(
            test_eval["raw_result"].get("df_action"),
            run_key=f"{feature_set_name}__{fold.fold_id}__seed{seed}",
            feature_set=feature_set_name,
            feature_family=feature_metadata["feature_family"],
            is_negative_control=feature_metadata["is_negative_control"],
            fold_id=fold.fold_id,
            seed=seed,
            selected_model_type=selected_artifact.get("artifact_type"),
            selection_rule=checkpoint_selection_rule,
            split_name="test",
        )
        test_observation_frame = build_test_observation_export(
            test_eval["raw_result"].get("df_observation"),
            run_key=f"{feature_set_name}__{fold.fold_id}__seed{seed}",
            feature_set=feature_set_name,
            feature_family=feature_metadata["feature_family"],
            is_negative_control=feature_metadata["is_negative_control"],
            fold_id=fold.fold_id,
            seed=seed,
            selected_model_type=selected_artifact.get("artifact_type"),
            selection_rule=checkpoint_selection_rule,
            split_name="test",
        )

        return {
            "candidate_df": scored_candidate_df,
            "selected_artifact": selected_artifact,
            "selected_artifact_type": selected_artifact.get("artifact_type"),
            "selection_rule": checkpoint_selection_rule,
            "checkpoint_selection_rule": checkpoint_selection_rule,
            "train_metrics": train_eval["metrics"],
            "validation_metrics": validation_eval["metrics"],
            "test_metrics": test_eval["metrics"],
            "regime_breakdown": test_eval["regime_breakdown"],
            "daily_test_frame": daily_test_frame,
            "test_action_frame": test_action_frame,
            "test_observation_frame": test_observation_frame,
            "benchmark_suite_frame": benchmark_suite_frame.copy()
            if isinstance(benchmark_suite_frame, pd.DataFrame)
            else pd.DataFrame(),
            "preprocessing_summary": preprocessing_summary,
            "training_config": {
                "base_config_name": base_config_name,
                "total_timesteps": total_timesteps,
                "es_mode": es_mode,
                "dropout_p": dropout_p,
                "eval_freq": eval_freq,
                "checkpoint_freq": checkpoint_freq,
                "checkpoint_selection_rule": checkpoint_selection_rule,
                "domain_reward_scaling": fold_env_spec.get("domain_reward_scaling_report", {}),
                "action_regularization": fold_env_spec.get("action_regularization_report", {}),
                "seed": seed,
            },
        }

    def run_notebook_research(
        df: pd.DataFrame,
        *,
        base_config_name: str = "custom_custom",
        output_dir: str | Path = "./research_outputs_notebook",
        feature_groups: Optional[Mapping[str, Sequence[str]]] = None,
        candidate_feature_families: Optional[Sequence[str]] = None,
        feature_set_filter: Optional[Sequence[str]] = None,
        seeds: Sequence[int] = (42, 123),
        total_timesteps: int = 50_000,
        max_folds: Optional[int] = 2,
        es_mode: str = "relaxed",
        dropout_p: float = 0.1,
        eval_freq: int = 8192,
        checkpoint_freq: int = 4096,
        checkpoint_selection_rule: str = "checkpoint_robust_score",
        domain_reward_scaling: Optional[Mapping[str, Any]] = None,
        action_regularization: Optional[Mapping[str, Any]] = None,
        verbose: int = 0,
    ) -> dict[str, Any]:
        prepared_df = _prepare_research_df(
            df,
            candidate_feature_families=candidate_feature_families,
        )
        reference_config = build_reference_experiment_config(
            base_config_name,
            seed_list=seeds,
        ).to_dict()
        reference_config["checkpoint_selection_rule"] = str(checkpoint_selection_rule)
        gate = run_research_gate(
            prepared_df,
            output_dir=output_dir,
            reference_config=reference_config,
            feature_ladder=feature_groups or DEFAULT_FEATURE_GROUPS,
            candidate_feature_families=candidate_feature_families,
            feature_set_filter=feature_set_filter,
        )
        folds = gate["folds"]
        if max_folds is not None:
            folds = folds[:max_folds]
        if not folds:
            raise ValueError("No walk-forward folds were generated for the provided dataset.")

        benchmark_suite_cache: dict[str, pd.DataFrame] = {}

        def fold_callback(train_df, validation_df, test_df, feature_cols, fold, seed, feature_set_name):
            if fold.fold_id not in benchmark_suite_cache:
                benchmark_source_df = pd.concat(
                    [train_df.copy(), validation_df.copy(), test_df.copy()],
                    ignore_index=True,
                )
                benchmark_suite_cache[fold.fold_id] = build_fold_benchmark_suite_export(
                    test_df,
                    fold_id=fold.fold_id,
                    benchmark_source_df=benchmark_source_df,
                )
            return _run_single_fold(
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
                feature_cols=feature_cols,
                fold=fold,
                seed=seed,
                feature_set_name=feature_set_name,
                base_config_name=base_config_name,
                total_timesteps=total_timesteps,
                output_dir=output_dir,
                es_mode=es_mode,
                dropout_p=dropout_p,
                eval_freq=eval_freq,
                checkpoint_freq=checkpoint_freq,
                checkpoint_selection_rule=checkpoint_selection_rule,
                domain_reward_scaling=domain_reward_scaling,
                action_regularization=action_regularization,
                verbose=verbose,
                benchmark_suite_frame=benchmark_suite_cache[fold.fold_id],
            )

        results = run_feature_ablation_ladder(
            df=prepared_df,
            folds=folds,
            run_fold_fn=fold_callback,
            feature_ladder=feature_groups or DEFAULT_FEATURE_GROUPS,
            candidate_feature_families=candidate_feature_families,
            feature_set_filter=feature_set_filter,
            seeds=seeds,
            output_dir=output_dir,
            selection_config=PROJECT_SELECTION_CONFIG,
            model_name=base_config_name,
        )

        return {"gate": gate, "results": results}

    return run_notebook_research
