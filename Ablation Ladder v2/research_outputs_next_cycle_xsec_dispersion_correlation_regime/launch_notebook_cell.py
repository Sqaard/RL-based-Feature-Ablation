from dow30_next_cycle_launch import run_bootstrapped_notebook_launch_from_csv

launch_bundle = run_bootstrapped_notebook_launch_from_csv(
    config_path=r"C:\Users\ivanp\RL for Time-Series Forecasting\GITHUB\configs\next_cycle_candidate_only.yaml",
    dataset_path=r"C:\Users\ivanp\RL for Time-Series Forecasting\GITHUB\Ablation Ladder v2\research_outputs_next_cycle_xsec_dispersion_correlation_regime\processed_dataset_snapshot.csv",
    output_dir=r"C:\Users\ivanp\RL for Time-Series Forecasting\GITHUB\Ablation Ladder v2\research_outputs_next_cycle_xsec_dispersion_correlation_regime",
    selected_candidate_family="xsec_dispersion_correlation_regime",
    panel_scope="candidate_only",
)

preflight = launch_bundle["preflight"]
research_bundle = launch_bundle["research_bundle"]
