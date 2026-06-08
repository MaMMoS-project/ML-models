# ML model for micromagnetic simulations, h and K orientation can change independently on sphere.

## Current version of model
v0.1


# 0. Installation
Use requirements.txt. In addition pytorch, compatible with your system, must be installed

# 1. Data pre-processing

Run:

```
python3 -m scripts.analyze_magnetic_data
```

```mermaid
flowchart LR

    subgraph cluster_0["1. Process Micromagnetics Data"]
        direction TB

        A1["./data/mumax3_mindrive_cube_all_params.csv"]
        A2["./data/mumax3_relaxdriver_cube_all_params.csv"]

        A1 --> B0["python3 -m scripts.analyze_magnetic_data"]
        A2 --> B0

        B0 --> O1["./plots/*.png<br/>analysis plots"]
        B0 --> O2["./data/processed/micromagnetics_angle_dependent_symmetries.csv"]
    end
```

NEEDS:
- ./data/mumax3_mindrive_cube_all_params.csv
- ./data/mumax3_relaxdriver_cube_all_params.csv

OUTPUT:
- stdout
- ./plots/*.png  # analysis plots
- ./data/processed/micromagnetics_angle_dependent_symmetries.csv

# 2. Model Training
Run:

```
python3 -m scripts.train_model --config config/ml_config_test.yaml
```

```mermaid
flowchart LR

    subgraph cluster_1["2. Train Model"]
        direction TB

        A1["./data/magnetic_materials.csv"]
        A2["./plots/ (outputs from Step 1)"]

        A1 --> B1["python3 -m scripts.train_model --config config/ml_config_test.yaml"]
        A2 --> B1

        B1 --> O0["stdout"]
        B1 --> O1["./results/models"]
        B1 --> O2["./results/plots"]
        B1 --> O3["./results/overall_results.json"]
    end
```

NEEDS:
- ./data/magnetic_materials.csv
- output files ./plots/ of 1

OUTPUT:
- stdout
- ./results/models
- ./results/plots
- ./results/overall_results.json

# 3. Metric
Run:

```
python3 scripts/plot_metrics.py results
```

NEEDS:
- ./results of 2.

OUTPUT:
- stdout
- ./results/metrics_tables


# 4. Best Model
For all three targets, the RF models does not show strong overfitting and the performance is the best.

![Alt text](https://github.com/MaMMoS-project/ML-models/blob/add-demo-NBs/beyond-stoner-wohlfarth/single-grain-multi-angle-model/results/best_model/random_forest/predictions.png)

![Alt text](https://github.com/MaMMoS-project/ML-models/blob/add-demo-NBs/beyond-stoner-wohlfarth/single-grain-multi-angle-model/results/best_model/random_forest/predictions_jackknife.png)

# 4.1 Feature Importance
![Alt text](https://github.com/MaMMoS-project/ML-models/blob/add-demo-NBs/beyond-stoner-wohlfarth/single-grain-multi-angle-model/results/best_model/random_forest/feature_importance_LogTransformation_all_standard.png)