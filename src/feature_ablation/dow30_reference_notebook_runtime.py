from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_NOTEBOOK_PATH = PROJECT_ROOT / "Experiments_Ablation_Ladder_v2.ipynb"

REQUIRED_RUNTIME_NAMES: tuple[str, ...] = (
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
)

REFERENCE_RUNTIME_CELL_PATTERNS: "OrderedDict[str, str]" = OrderedDict(
    (
        ("base_imports", "from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv"),
        ("custom_env", "class CustomStockTradingEnv(StockTradingEnv):"),
        ("zhang_env", "class ZhangStockTradingEnv(StockTradingEnv):"),
        ("runner_imports_and_seeding", "def set_all_seeds(seed: int = 42):"),
        ("custom_extractor", "class CustomDropoutExtractor(BaseFeaturesExtractor):"),
        ("custom_policy", "def build_custom_policy_kwargs("),
        ("ppo_kwargs", '"learning_rate": 1e-4,'),
        ("wrap_eval_env", "def wrap_eval_env(eval_env):"),
        ("callbacks", "def build_callbacks("),
        ("metric_extract", "def extract_metric(stats, metric_name):"),
        ("evaluate_env", "def evaluate_model_on_env(model, environment):"),
    )
)


def _strip_ipython_magics(source: str) -> str:
    cleaned_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip() + "\n"


def _load_notebook_payload(notebook_path: str | Path) -> dict[str, Any]:
    path = Path(notebook_path)
    return json.loads(path.read_text(encoding="utf-8"))


def plan_reference_notebook_runtime_cells(
    notebook_path: str | Path = DEFAULT_REFERENCE_NOTEBOOK_PATH,
) -> list[dict[str, Any]]:
    payload = _load_notebook_payload(notebook_path)
    cells = payload.get("cells", [])
    plan: list[dict[str, Any]] = []

    for key, pattern in REFERENCE_RUNTIME_CELL_PATTERNS.items():
        matched = False
        for cell_index, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            raw_source = "".join(cell.get("source", []))
            if pattern not in raw_source:
                continue
            plan.append(
                {
                    "key": key,
                    "pattern": pattern,
                    "cell_index": cell_index,
                    "source": _strip_ipython_magics(raw_source),
                }
            )
            matched = True
            break
        if not matched:
            raise KeyError(
                f"Could not find notebook runtime cell for `{key}` using pattern `{pattern}`."
            )
    return plan


def load_reference_notebook_runtime(
    *,
    notebook_path: str | Path = DEFAULT_REFERENCE_NOTEBOOK_PATH,
    extra_ns: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "__reference_notebook_runtime__"}
    if extra_ns:
        namespace.update(dict(extra_ns))

    plan = plan_reference_notebook_runtime_cells(notebook_path)
    notebook_file = Path(notebook_path)

    for item in plan:
        code = item["source"]
        exec(
            compile(code, f"{notebook_file.name}:cell_{item['cell_index']}", "exec"),
            namespace,
            namespace,
        )

    missing = [name for name in REQUIRED_RUNTIME_NAMES if name not in namespace]
    if missing:
        raise KeyError(
            "Reference notebook runtime bootstrap did not produce required names: "
            + ", ".join(missing)
        )
    return namespace
