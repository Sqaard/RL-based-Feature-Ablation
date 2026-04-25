from dow30_next_cycle_launch import run_bootstrapped_notebook_launch_from_csv

launch_bundle = run_bootstrapped_notebook_launch_from_csv(
    config_path=r"C:\Users\ivanp\RL for Time-Series Forecasting\GITHUB\configs\next_cycle_candidate_only_xsec_sector_gated_context.yaml",
    dataset_path=r"C:\Users\ivanp\RL for Time-Series Forecasting\GITHUB\processed_final_fixed.csv",
    output_dir=r"Ablation Ladder v2\research_outputs_next_cycle_xsec_sector_gated_context_preflight",
    selected_candidate_family="xsec_sector_gated_context",
    panel_scope="candidate_only",
)

preflight = launch_bundle["preflight"]
research_bundle = launch_bundle["research_bundle"]
