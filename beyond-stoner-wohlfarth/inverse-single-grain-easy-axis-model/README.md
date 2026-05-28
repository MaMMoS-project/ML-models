# ML surrogate model for micromagnetic simulations, H and K1 aligned in z-direction


## Current version of model
v1.0


## 0. Installation
Use requirements.txt. In addition pytorch, compatible with your system, must be installed.

## Training data generation

- The training data has been created using micromagnetic simulations.
- One hysteresis loop for a cube of 50nm edge length was computed for each combination of material parameters A, Ms, K
- from the hysteresis loops, Hc, Mr and BHmax are computed (that's the input data for the ML model, available at [data/single_grain_cube_50nm_aligned.csv](data/single_grain_cube_50nm_aligned.csv)).
- in total 10388 data points were computed

The generation of the V2 training data and details on the simulation software and method are [described in data-generation](https://github.com/MaMMoS-project/BSW_data_generation).

## 1. Data preprocessing

Run:

```
python3 -m scripts.analyze_magnetic_data
```

NEEDS:
- ./data/magnetic_materials.csv

OUTPUT:
- stdout
- ./plots/*.png  # analysis plots
- ./plots/supervised_clustering_model.pkl
- ./plots/supervised_clustering_pipeline.joblib
- ./plots/supervised_metrics.txt

In this specific case where the anisotropy axis is aligned with the external magnetic field, the dataset can be split into two distinct groups when considering the dimensionless ratio Mr/Ms. Namely, hard and soft magnetic materials. The points for hard magnets corresponds to Mr/Ms≈1 (red points ) while other points lie around Mr/Ms≈0 (blue points). A k-means clustering algorithm is applied to find the cluster centers of Mr/Ms ration. Then a random forest classifier is trained to predict the material class label (hard, soft) from intrinsic properties. 

![Alt text](https://github.com/MaMMoS-project/ML-models/blob/add-demo-NBs/beyond-stoner-wohlfarth/single-grain-easy-axis-model/results/best_model_hard_magnets/random_forest/kmeans_clustering.png)


## 2. Model Training

Linear regression (LR) models, a random forest (RF), the LASSO regression, a Gaussian process and a fully connected neural network (FCNN) have been developed. Note that separate regressors have been trained for the hard and soft magnetic materials. 

Run:

```
python3 -m scripts.train_model --config config/ml_config_test.yaml
```

NEEDS:
- ./data/magnetic_materials.csv
- output files ./plots/ of 1

OUTPUT:
- stdout
- ./results/models
- ./results/plots
- ./results/overall_results.json

## 3. Metric Plots
Run:

```
python3 scripts/plot_metrics.py results
```

NEEDS:
- ./results of 2.

OUTPUT:
- stdout
- ./results/metrics_tables

## Results
For all three targets, both the FCNN and RF models do not show strong over fitting and the performance is quite comparable. Results below shown from Random Forest model.

### Metrics for target Hc
| Model | Soft Train              | Soft Test                | Hard Train              | Hard Test               |
| ----- | ----------------------- | ------------------------ | ----------------------- | ----------------------- |
| LR    | MSE: 0.105<br>R²: 0.530 | MSE: 0.098<br>R²: 0.592  | MSE: 0.124<br>R²: 0.930 | MSE: 0.106<br>R²: 0.939 | x
| LASSO | MSE: 0.105<br>R²: 0.530 | MSE: 0.099<br>R²: 0.589  | MSE: 0.125<br>R²: 0.930 | MSE: 0.106<br>R²: 0.939 | x
| RF    | MSE: 0.006<br>R²: 0.972 | MSE: 0.038<br>R²: 0.842  | MSE: 0.002<br>R²: 0.999 | MSE: 0.012<br>R²: 0.993 | x
| GP    | MSE: 0.000<br>R²: 0.999 | MSE: 0.032<br>R²: 0.867  | MSE: 0.003<br>R²: 0.998 | MSE: 0.045<br>R²: 0.971 | x
| FCNN  | MSE: 0.048<br>R²: 0.784 | MSE: 0.055<br>R²: 0.770  | MSE: 0.005<br>R²: 0.997 | MSE: 0.04<br>R²: 0.998  | x

### Metrics for target Mr
| Model | Soft Train              | Soft Test               | Hard Train               | Hard Test               |
| ----- | ----------------------- | ----------------------- | ------------------------ | ----------------------- |
| LR    | MSE: 0.160<br>R²: 0.427 | MSE: 0.128<br>R²: 0.474 | MSE: 0.013<br>R²: 0.996  | MSE: 0.014<br>R²: 0.987 |x
| LASSO | MSE: 0.161<br>R²: 0.422 | MSE: 0.127<br>R²: 0.478 | MSE: 0.004<br>R²: 0.996  | MSE: 0.014<br>R²: 0.987 |x
| RF    | MSE: 0.011<br>R²: 0.962 | MSE: 0.038<br>R²: 0.846 | MSE: 0.001<br>R²: 0.999  | MSE: 0.012<br>R²: 0.989 |x
| GP    | MSE: 0.000<br>R²: 0.999 | MSE: 0.030<br>R²: 0.770 | MSE: 0.000<br>R²: 0.999  | MSE: 0.011<br>R²: 0.990 |x
| FCNN  | MSE: 0.064<br>R²: 0.769 | MSE: 0.058<br>R²: 0.763 | MSE: 0.004<br>R²: 0.997  | MSE: 0.013<br>R²: 0.989 |x

### Metrics for target (BH)max
| Model | Soft Train              | Soft Test                | Hard Train                | Hard Test                 |
| ----- | ----------------------- | ------------------------ | ------------------------- | ------------------------- |
| LR    | MSE: 0.013<br>R²: 0.985 | MSE: 0.012<br>R²: 0.985  | MSE: 0.003<br>R²: 0.999   | MSE: 0.002<br>R²: 0.999   |x
| LASSO | MSE: 0.013<br>R²: 0.985 | MSE: 0.013<br>R²: 0.988  | MSE: 0.003<br>R²: 0.999   | MSE: 0.002<br>R²: 0.998   |x
| RF    | MSE: 0.001<br>R²: 0.999 | MSE: 0.006<br>R²: 0.995  | MSE: 0.0001<br>R²: 1.000 | MSE: 0.0003<br>R²: 0.99999 | x
| GP    | MSE: 0.000<br>R²: 0.999 | MSE: 0.007<br>R²: 0.993  | MSE: 0.000<br>R²: 0.992   | MSE: 0.000<br>R²: 0.999   |
| FCNN  | MSE: 0.265<br>R²: 0.970 | MSE: 0.041<br>R²: 0.961  | MSE: 0.000<br>R²: 0.999   | MSE: 0.000<br>R²: 0.999   |x

### Plots Hard-Magnet Random Forest Model

![Alt text](https://github.com/MaMMoS-project/ML-models/blob/add-demo-NBs/beyond-stoner-wohlfarth/single-grain-easy-axis-model/results/best_model_hard_magnets/random_forest/predictions.png)


![Alt text](https://github.com/MaMMoS-project/ML-models/blob/add-demo-NBs/beyond-stoner-wohlfarth/single-grain-easy-axis-model/results/best_model_hard_magnets/random_forest/predictions_jackknife.png)

#### Feature Importance via Mean Decrease in Impurity

Feature's contribution in a Random Forest model is measured by the average variance reduction across all trees and splits, ranking features by their predictive power during training. Higher values indicate greater importance.

![Alt text](https://github.com/MaMMoS-project/ML-models/blob/add-demo-NBs/beyond-stoner-wohlfarth/single-grain-easy-axis-model/results/best_model_hard_magnets/random_forest/feature_importance_LogTransformation_cluster1_standard.png)


## 4. Inference

To run an inference please run:
python3 ./scripts/load_onnx_models.py 
