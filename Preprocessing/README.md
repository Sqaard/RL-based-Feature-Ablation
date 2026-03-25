# Preprocessing

This folder contains the preprocessing pipeline used to build the RL training dataset.

## Contents
- `Data_preprocessing.ipynb` — notebook for data preparation and feature construction
- `dow_30_fundamental_wrds.csv` — WRDS-based fundamental data used in the feature pipeline

## Purpose
The preprocessing stage creates the final state representation for RL experiments by combining:
- price and return information,
- technical indicators,
- WRDS-enriched data,
- macro / market context,
- HMM regime features,
- GRU-based short-horizon forecasts.

## Notes
- HMM regime construction is intended to be causal: trained on the training interval and applied without future leakage.
- This folder is the upstream dependency for both the configuration-comparison experiments and the later walk-forward / ablation studies.
- If preprocessing logic changes, downstream experimental results are not directly comparable unless the change is documented.
