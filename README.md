# Solar_Prediction

Solar power forecasting — benchmarking statistical baselines, tree models, and Transformer-based models across several public datasets.

## Datasets

| Dataset | Task | EDA | Models |
|---|---|---|---|
| GEFCom2014 | Day-ahead power forecasting | [01](notebook/EDA/01_DEFCom_EDA.ipynb) | [XGBoost](notebook/model/01_GEFCom_baseline_XGBoost.ipynb) / [Transformer](notebook/model/02_GEFCom_baseline_Transformer.ipynb) / [PatchTST](notebook/model/03_GEFCom_patchTST.ipynb) |
| SKIPP'D | 15-min nowcasting | [02](notebook/EDA/02_SKIPPD_EDA.ipynb) | [baseline](notebook/model/04_SKIPPD_baseline.ipynb) |
| NREL | Regional generation | [03](notebook/EDA/03_Renewable-energy-generation.ipynb) | [baseline](notebook/model/05_NationalRenewable_baseline.ipynb) |

## Layout

- [src/baseline_skippd/](src/baseline_skippd/) — SKIPP'D nowcasting pipeline (indexing, windowing, training, evaluation)
- [src/vanilla_transformer/](src/vanilla_transformer/) — dataset-agnostic time-series forecasting package
- [src/helper/](src/helper/) — shared features, metrics, and backtesting utilities
- [configs/](configs/) — experiment configs

## Setup

Python 3.10.

```bash
pip install numpy pandas pyarrow scikit-learn torch xgboost lightgbm matplotlib tqdm pyyaml pillow
```

## Usage

```bash
python src/baseline_skippd/cli.py run \
  --config configs/skippd/vanilla_transformer_pv_only.yaml \
  --model vanilla_transformer
```

Subcommands: `inspect` / `build-index` / `train` / `evaluate` / `run`.

## SKIPP'D test results (n=11086)

| Model | RMSE (kW) | MAE (kW) | acc_rmse |
|---|---|---|---|
| vanilla_transformer | **3.21** | 1.71 | **0.893** |
| pv_only | 3.28 | 1.83 | 0.891 |
| clear_sky_persistence | 3.63 | 1.77 | 0.879 |
| naive_persistence | 3.71 | 2.12 | 0.877 |
| sunset_forecast | 4.06 | 2.91 | 0.865 |

## Notes

`data/` (4.5 GB) and `artifacts/` (model checkpoints) are not tracked — prepare them locally.
