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

        A1["./data/processed/micromagnetics_angle_dependent_symmetries.csv"]
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
- ./data/processed/micromagnetics_angle_dependent_symmetries.csv  (output of Step 1)
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

# 5. Inference

Unlike the easy-axis model, this model is **not** split into hard/soft clusters
(`clustering.method: none` in the config): a single random-forest regressor is trained on the
whole dataset. The extra input is the **relative angle between the external field H and the
uniaxial anisotropy axis K1**, given in **radians** (range `[0, pi]`).

To run an inference:

```
python3 ./scripts/load_onnx_models.py
```

Or from Python:

```python
from scripts.load_onnx_models import calculate_extrinsic_properties
res = calculate_extrinsic_properties(Ms=1.0e6, A=1.0e-11, K=4.5e6, angle=0.0)
# -> {"Hc": ..., "Mr": ..., "BHmax": ...}
```

The pipeline applies `log1p` to `Ms`, `A`, `K1` (the angle is left untransformed, matching
`log_exclude_cols`), runs the ONNX random forest (the `StandardScaler` is baked into the ONNX
graph), and inverts the target `log1p` with `expm1`. Inputs outside the training volume are
predicted anyway but trigger an extrapolation warning.

NEEDS:
- ./results/models/LogTransformation_all/random_forest.onnx  (output of Step 2)
- ./data/processed/micromagnetics_angle_dependent_symmetries.csv  (for the training-volume check)

OUTPUT:
- dict with predicted Hc (A/m), Mr (A/m) and (BH)max (J/m³)

# 6. Validation on fresh (held-out) data

The model is validated on a genuinely fresh dataset, using the same strategy as the easy-axis
and inverse models. The model is trained on the V2 angle-dependent dataset; the older V1 file
`data/magnetic_materials.csv` (1,497 single-grain simulations) is used as an external hold-out,
sharing **0 %** of its `(Ms, A, K1)` triples with the training data.

**This validates only the (nearly) aligned slice.** V1 was generated with the field a *constant*
~1° off the easy axis, so it does not exercise the model's full angular range. The relative angle
is **derived exactly from the data** (`arccos(|û_K · û_H|)`, the same definition used in training),
which gives **1.025° (0.0179 rad)** for every V1 point — so the prediction is made at the exact
measured angle, **not** hard-coded to 0°.

Run:

```
python3 scripts/validate_fresh_v1.py
```

```mermaid
flowchart LR

    subgraph cluster_3["6. Validate on fresh data (aligned slice)"]
        direction TB

        A5["./data/magnetic_materials.csv (fresh V1, aligned ~1°, 0 % overlap)"] --> B5["python3 scripts/validate_fresh_v1.py"]
        A6["./results/models/LogTransformation_all/random_forest.onnx"] --> B5

        B5 --> O1["./validation_v1/parity.png"]
        B5 --> O2["./validation_v1/stats.csv"]
    end
```

Of the 1,497 fresh points, 42 fall outside the training volume (excluded as extrapolation),
leaving **1,455** points (1,244 hard, 211 soft by the physical `Mr/Ms > 0.4` criterion).

### Per-target results (fresh V1 data, aligned slice)

| Target | R² | R²(log) | Median rel. error | MAE |
| --- | --- | --- | --- | --- |
| Coercive field `Hc` | 0.886 | 0.866 | 27 % | 4.7×10⁵ A/m |
| Remanence `Mr` | 0.563 | 0.610 | 11 % | 4.1×10⁵ A/m |
| Energy product `(BH)max` | **−0.85** | −7.6 | 68 % | 1.3×10⁶ J/m³ |

**Interpretation.** On the aligned slice the multi-angle model is clearly *weaker than the
dedicated easy-axis model* (which reaches R² ≈ 0.995 for `Hc`/`(BH)max` on the same V1 data).
Two structural reasons, both visible in `validation_v1/parity.png`:

1. **No hard/soft split.** Unlike the easy-axis pipeline, this model is a single unified
   regressor with no classifier. The **hard** magnets (which dominate) track the diagonal
   reasonably, but the **soft** magnets — whose aligned loops have low `Mr`/`(BH)max` — are
   badly scattered, dragging `Mr` down and making `(BH)max` R² negative.
2. **Generalist across all angles.** The model spreads its capacity over the full angular range,
   so at any single angle it is less accurate than a model trained only for that geometry.

**Practical guidance:** for aligned / easy-axis use cases prefer the specialised
`single-grain-easy-axis-model`; the multi-angle model is intended for genuinely off-axis angles
that the specialised model does not cover. A full validation of the off-axis regime still
requires a fresh dataset with a range of relative angles (none is currently available).

OUTPUT (in `./validation_v1/`):
- `parity.png` — predicted vs. true `Hc, Mr, (BH)max` (log–log), coloured by the true hard/soft class
- `stats.csv` — per-target R², R²(log), MAE, RMSE, median relative error

