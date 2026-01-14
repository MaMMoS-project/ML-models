# ML model for micromagnetic simulations, h and K aligned in z-direction


## Current version of model
v0.1


## 0. Installation
Use requirements.txt. In addition pytorch, compatible with your system, must be installed



## 1. data preprocessing

run:
PYTHONPATH=$PYTHONPATH:$(pwd) python3.13 scripts/analyze_magnetic_data.py

NEEDS:
- ./data/magnetic_materials.csv

OUTPUT:
- stdout
- ./plots/*.png  # analysis plots
- ./plots/supervised_clustering_model.pkl
- ./plots/supervised_clustering_pipeline.joblib
- ./plots/supervised_metrics.txt

## 2. training of models
run:
PYTHONPATH=$PYTHONPATH:$(pwd) python3.13 scripts/train_model.py   --config config/ml_config_test.yaml

NEEDS:
- ./data/magnetic_materials.csv
- output files ./plots/ of 1

OUTPUT:
- stdout
- ./results/models
- ./results/plots
- ./results/overall_results.json

## 3. plots of metrics
run:
PYTHONPATH=$PYTHONPATH:$(pwd) python3.13 scripts/plot_metrics.py results

NEEDS:
- ./results of 2.

OUTPUT:
- stdout
- ./results/metrics_tables

## Results

### Metrics for the investigated models for the target Hc
| Model | Soft Train              | Soft Test                | Hard Train              | Hard Test               |
| ----- | ----------------------- | ------------------------ | ----------------------- | ----------------------- |
| LR    | MSE: 0.245<br>R²: 0.633 | MSE: 0.282<br>R²: 0.0729 | MSE: 0.122<br>R²: 0.910 | MSE: 0.121<br>R²: 0.923 |
| LASSO | MSE: 0.245<br>R²: 0.633 | MSE: 0.282<br>R²: 0.729  | MSE: 0.122<br>R²: 0.909 | MSE: 0.122<br>R²: 0.922 |
| RF    | MSE: 0.022<br>R²: 0.968 | MSE: 0.157<br>R²: 0.850  | MSE: 0.006<br>R²: 0.996 | MSE: 0.080<br>R²: 0.949 |
| GP    | MSE: 0.000<br>R²: 0.999 | MSE: 0.139<br>R²: 0.862  | MSE: 0.000<br>R²: 0.999 | MSE: 0.045<br>R²: 0.971 |
| FCNN  | MSE: 0.200<br>R²: 0.699 | MSE: 0.368<br>R²: 0.647  | MSE: 0.043<br>R²: 0.968 | MSE: 0.057<br>R²: 0.944 |

### Metrics for the investigated models for the target Mr

| Model | Soft Train              | Soft Test               | Hard Train               | Hard Test               |
| ----- | ----------------------- | ----------------------- | ------------------------ | ----------------------- |
| LR    | MSE: 0.510<br>R²: 0.494 | MSE: 1.965<br>R²: 0.466 | MSE: 0.000<br>R²: 0.999  | MSE: 0.072<br>R²: 0.803 |
| LASSO | MSE: 0.000<br>R²: 0.0   | MSE: 1.192<br>R²: 0.468 | MSE: 0.0002<br>R²: 0.999 | MSE: 0.073<br>R²: 0.802 |
| RF    | MSE: 0.073<br>R²: 0.927 | MSE: 0.883<br>R²: 0.606 | MSE: 0.001<br>R²: 0.999  | MSE: 0.071<br>R²: 0.807 |
| GP    | MSE: 0.000<br>R²: 0.999 | MSE: 0.622<br>R²: 0.723 | MSE: 0.000<br>R²: 0.999  | MSE: 0.072<br>R²: 0.802 |
| FCNN  | MSE: 0.464<br>R²: 0.540 | MSE: 1.360<br>R²: 0.393 | MSE: 0.014<br>R²: 0.950  | MSE: 0.088<br>R²: 0.770 |

### Metrics for the investigated models for the target (BH)max

| Model | Soft Train              | Soft Test                | Hard Train                | Hard Test                 |
| ----- | ----------------------- | ------------------------ | ------------------------- | ------------------------- |
| LR    | MSE: 0.008<br>R²: 0.975 | MSE: 0.0072<br>R²: 0.984 | MSE: 0.002<br>R²: 0.999   | MSE: 0.002<br>R²: 0.999   |
| LASSO | MSE: 0.009<br>R²: 0.975 | MSE: 0.007<br>R²: 0.984  | MSE: 0.002<br>R²: 0.999   | MSE: 0.002<br>R²: 0.998   |
| RF    | MSE: 0.002<br>R²: 0.996 | MSE: 0.006<br>R²: 0.987  | MSE: 0.0004<br>R²: 0.9996 | MSE: 0.0083<br>R²: 0.9933 |
| GP    | MSE: 0.000<br>R²: 0.999 | MSE: 0.004<br>R²: 0.992  | MSE: 0.000<br>R²: 0.992   | MSE: 0.000<br>R²: 0.999   |
| FCNN  | MSE: 0.161<br>R²: 0.518 | MSE: 0.182<br>R²: 0.573  | MSE: 0.012<br>R²: 0.988   | MSE: 0.018<br>R²: 0.986   |

For all three targets, both the FCNN and RF models do not show strong over fitting and the per-
formance is quite comparable.