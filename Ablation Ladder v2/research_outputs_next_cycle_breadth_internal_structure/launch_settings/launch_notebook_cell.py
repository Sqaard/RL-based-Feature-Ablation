from dow30_next_cycle_launch import run_bootstrapped_notebook_launch_from_csv

launch_bundle = run_bootstrapped_notebook_launch_from_csv(
    config_path=r"/home/ma-user/work/configs/next_cycle_candidate_only_breadth_internal_structure.yaml",
    dataset_path=r"/home/ma-user/work/Ablation Ladder v2/research_outputs_next_cycle_breadth_internal_structure/processed_dataset_snapshot.csv",
    output_dir=r"/home/ma-user/work/Ablation Ladder v2/research_outputs_next_cycle_breadth_internal_structure",
    selected_candidate_family="breadth_internal_structure",
    panel_scope="candidate_only",
)

preflight = launch_bundle["preflight"]
research_bundle = launch_bundle["research_bundle"]
