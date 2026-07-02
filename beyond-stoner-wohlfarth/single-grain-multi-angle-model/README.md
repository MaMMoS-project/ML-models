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
%%{init: {
  "theme": "base",
  "themeVariables": {
    "lineColor": "#94A3B8",
    "clusterBkg": "#FAFAFA",
    "clusterBorder": "#D1D5DB",
    "fontFamily": "Arial, sans-serif"
  }
}}%%

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

    classDef input fill:#EEF4FA,stroke:#A7C4E0,color:#334155,stroke-width:1.2px;
    classDef process fill:#F5F5F4,stroke:#BDBDBD,color:#374151,stroke-width:1.5px;
    classDef output fill:#F0F7F1,stroke:#A8C8A5,color:#334155,stroke-width:1.2px;

    class A1,A2 input;
    class B0 process;
    class O1,O2 output;
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
%%{init: {
  "theme": "base",
  "themeVariables": {
    "lineColor": "#94A3B8",
    "clusterBkg": "#FAFAFA",
    "clusterBorder": "#D1D5DB",
    "fontFamily": "Arial, sans-serif"
  }
}}%%

flowchart LR

    subgraph cluster_1["2. Train Models"]
        direction TB

        A1["./data/magnetic_materials.csv"]
        A2["./plots/ (outputs from Step 1)"]

        A1 --> B1["python3 -m scripts.train_model --config config/ml_config_test.yaml"]
        A2 --> B1

        B1 --> O1["./results/models"]
        B1 --> O2["./results/plots"]
        B1 --> O3["./results/overall_results.json"]
    end

    classDef input fill:#EEF4FA,stroke:#A7C4E0,color:#334155,stroke-width:1.2px;
    classDef process fill:#F5F5F4,stroke:#BDBDBD,color:#374151,stroke-width:1.5px;
    classDef output fill:#F0F7F1,stroke:#A8C8A5,color:#334155,stroke-width:1.2px;

    class A1,A2 input;
    class B1 process;
    class O1,O2,O3 output;
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

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "lineColor": "#94A3B8",
    "clusterBkg": "#FAFAFA",
    "clusterBorder": "#D1D5DB",
    "fontFamily": "Arial, sans-serif"
  }
}}%%

flowchart LR

    subgraph cluster_2["3. Generate Metrics Tables"]
        direction TB

        A1["./results/ (outputs from Step 2)"]

        A1 --> B1["python3 scripts/plot_metrics.py results"]

        B1 --> O1["./results/metrics_tables"]
    end

    classDef input fill:#EEF4FA,stroke:#A7C4E0,color:#334155,stroke-width:1.2px;
    classDef process fill:#F5F5F4,stroke:#BDBDBD,color:#374151,stroke-width:1.5px;
    classDef output fill:#F0F7F1,stroke:#A8C8A5,color:#334155,stroke-width:1.2px;

    class A1 input;
    class B1 process;
    class O1 output;
```

NEEDS:
- ./results of 2.

OUTPUT:
- stdout
- ./results/metrics_tables


# 4. Results Best Model
For all three targets, the RF models does not show strong overfitting and the performance is the best.

### 🏆 Best Model Metrics for target \(H_c\) (A/m)

| Model | All |
| ----- | ----------------------- |
| RF (train) | MSE: **0.0332**<br>R²: **0.9833** |
| RF (test) | MSE: **0.2261**<br>R²: **0.8864** |

### 🏆 Best Model Metrics for target \(M_r\) (A/m)

| Model | All |
| ----- | ----------------------- |
| RF (train) | MSE: **0.0373**<br>R²: **0.9728** |
| RF (test) | MSE: **0.2468**<br>R²: **0.8238** |

### 🏆 Best Model Metrics for target \(BH_{\max}\) (J/m³)

| Model | All |
| ----- | ----------------------- |
| RF (train) | MSE: **0.1185**<br>R²: **0.9781** |
| RF (test) | MSE: **0.8262**<br>R²: **0.8520** |
