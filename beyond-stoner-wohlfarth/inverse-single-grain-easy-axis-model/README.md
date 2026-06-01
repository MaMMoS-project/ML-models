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
Below results for the best model (Gaussian Process) and second best model (Random Forest) are displayed. Note that the Gaussian Process is likely prone to overfitting.


### 🏆 Best model metrics for target A (J/m)
| Model | Soft Magnet                         |Hard Magnets            | 
| ----- | -----------------------             |------------------------| 
| RF (Train) | MSE: 0.0000  <br> R²:0.8481    | MSE: 0.0000 <br> R²: 0.6179  |
| RF (Test)    | MSE: 0.0000 <br> R²: 0.7248  | MSE: 0.0000 <br> R²: 0.5160  |
| GP (Train)  | MSE: 0.0000 <br> R²: 1.0      | MSE: 0.0000 <br> R²: 1.0  | 
| GP (Test)  | MSE: 0.0000 <br> R²: 0.8210    | MSE: 0.0000 <br> R²: 0.5196  | 

### 🏆 Best model metrics for target K (J/m^3)
| Model | Soft Magnet             | Hard Magnets               | 
| ----- | ----------------------- | ------------------------ | 
| RF (Train) | MSE: 0.1350  <br> R²: 0.9422 | MSE: 0.0052 <br> R²: 0.9983|
| RF (Test)  | MSE: 0.4184  <br> R²: 0.8449 | MSE: 0.0102 <br> R²: 0.9966|
| GP (Train) | MSE: 0.0000  <br> R²: 1.0    | MSE: 0.0000 <br> R²: 1.0| 
| GP (Test)  | MSE: 0.3968  <br> R²: 0.8529 | MSE: 0.0118 <br> R²:  0.9961| 

### 🏆 Best model metrics for target Ms (A/m)
| Model | Soft Magnet                           | Hard Magnets               | 
| ----- | -----------------------               | ------------------------   | 
| RF (Train) | MSE: 0.0081 <br> R²: 0.9563      | MSE: 0.0001 <br> R²:   0.9999           |
| RF (Test)    | MSE: 0.0188 <br> R²: 0.9006    | MSE: 0.0002 <br> R²:   0.9998          |
| GP (Train)  | MSE: 0.0000 <br> R²: 1.0        | MSE: 0.0000 <br> R²:   1.0         | 
| GP (Test)  | MSE: 0.0154 <br> R²: 0.9187      | MSE: 0.0011 <br> R²:   0.9990         | 


## 4. Inference

To run an inference please run:
python3 ./scripts/load_onnx_models.py 
