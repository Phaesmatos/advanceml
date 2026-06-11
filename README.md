# FINAL clean preprocessing

This folder contains only the clean preprocessing stage of the final Advanced Machine Learning for Physics project on unsupervised Emerging Jets anomaly detection.

No models, training loops, score production, or Task 3 evaluation are implemented here yet.

## What this stage does

- Reads raw HDF5 files with `jets`, `tracks`, and optional truth groups.
- Uses reconstructed jet and track features only as model inputs.
- Keeps labels, event weights, event numbers, pile-up variables, and truth information as metadata/evaluation-only fields.
- Builds deterministic background-only train/validation/test splits when a single QCD file is provided.
- Computes normalization statistics on training background only.
- Computes track normalization statistics using valid tracks only.
- Re-zeros invalid/padded tracks after normalization.
- Saves padded set inputs and aggregate tabular inputs.

## Configure paths

Edit `configs/preprocessing.yaml`.

Use one of these modes:

1. Single QCD/background file:

```yaml
paths:
  background_path: C:/path/to/pp_output_test_background.h5
  test_signal_path: C:/path/to/pp_output_test_signal.h5
```

The background file is split deterministically with the configured seed.

2. Explicit background splits:

```yaml
paths:
  train_background_path: C:/path/to/train_background.h5
  val_background_path: C:/path/to/val_background.h5
  test_background_path: C:/path/to/test_background.h5
  test_signal_path: C:/path/to/test_signal.h5
```

## Inspect raw files

Open `notebooks/01_data_exploration.ipynb` and set the paths in the first cell. The notebook only inspects HDF5 structure and selected feature availability.

## Build clean datasets

After setting paths:

```bash
cd FINAL
python -m src.datasets --features configs/features_clean.yaml --config configs/preprocessing.yaml
```

Outputs are written to `outputs/preprocessed/` by default:

- `processed_train_background.npz`
- `processed_val_background.npz`
- `processed_test_background.npz`
- `processed_test_signal.npz`, if a signal path is configured
- `preprocessing_summary.json`
- `normalization_stats.json`
- `feature_lists.json`

Each NPZ contains:

- `X_jets`
- `X_tracks`
- `track_mask`
- `X_agg`
- `y`, when available
- feature name arrays
- metadata fields prefixed with `meta_`
- truth-level evaluation summaries prefixed with `eval_`

## Not implemented yet

- AE/VAE
- Normalizing Flow
- Diffusion
- Transformer Set Autoencoder
- Transformer Deep SVDD
- GNN
- supervised baseline
- Task 3 final evaluation
