import sys
from pathlib import Path

HORIZON_A_ROOT = Path(r"C:\Users\ivanp\RL for Time-Series Forecasting\GITHUB\Ablation Ladder v2")
if str(HORIZON_A_ROOT) not in sys.path:
    sys.path.insert(0, str(HORIZON_A_ROOT))

from dow30_next_cycle_launch import run_bootstrapped_notebook_launch_from_csv

launch_bundle = run_bootstrapped_notebook_launch_from_csv(
    config_path=r"C:\Users\ivanp\RL for Time-Series Forecasting\GITHUB\configs\next_cycle_candidate_only_rates_credit_vol_risk_state_context.yaml",
    dataset_path=r"C:\Users\ivanp\RL for Time-Series Forecasting\GITHUB\Ablation Ladder v2\research_outputs_next_cycle_rates_credit_vol_risk_state_context\processed_dataset_snapshot.csv",
    output_dir=r"C:\Users\ivanp\RL for Time-Series Forecasting\GITHUB\Ablation Ladder v2\research_outputs_next_cycle_rates_credit_vol_risk_state_context",
    selected_candidate_family="rates_credit_vol_risk_state_context",
    panel_scope="candidate_only",
)

preflight = launch_bundle["preflight"]
research_bundle = launch_bundle["research_bundle"]
