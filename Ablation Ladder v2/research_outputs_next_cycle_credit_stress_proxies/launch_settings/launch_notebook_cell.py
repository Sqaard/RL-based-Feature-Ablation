from dow30_next_cycle_launch import run_bootstrapped_notebook_launch_from_csv

launch_bundle = run_bootstrapped_notebook_launch_from_csv(
    config_path=r"C:\Users\ivanp\RL for Time-Series Forecasting\GITHUB\configs\next_cycle_candidate_only_credit_stress_proxies.yaml",
    dataset_path=r"C:\Users\ivanp\RL for Time-Series Forecasting\GITHUB\processed_final_fixed_external_lagclean_full.csv",
    output_dir=r"Ablation Ladder v2\research_outputs_next_cycle_credit_stress_proxies_preflight",
    selected_candidate_family="credit_stress_proxies",
    panel_scope="candidate_only",
)

preflight = launch_bundle["preflight"]
research_bundle = launch_bundle["research_bundle"]
