from dow30_next_cycle_launch import run_bootstrapped_notebook_launch_from_csv

launch_bundle = run_bootstrapped_notebook_launch_from_csv(
    config_path=r"/home/ma-user/work/configs/next_cycle_candidate_only_sector_relative_context.yaml",
    dataset_path=r"/home/ma-user/work/Ablation Ladder v2/research_outputs_next_cycle_sector_relative_context/processed_dataset_snapshot.csv",
    output_dir=r"/home/ma-user/work/Ablation Ladder v2/research_outputs_next_cycle_sector_relative_context",
    selected_candidate_family="sector_relative_context",
    panel_scope="candidate_only",
)

preflight = launch_bundle["preflight"]
research_bundle = launch_bundle["research_bundle"]
