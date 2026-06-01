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

- Cluster 0 = soft magnetic materials
- Cluster 1 = hard magnetic materials


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

# TODO: UPDATE TABLES BELOW!

### 🏆 Best Model Metrics for target Hc (A/m)
| Model | Soft               |  Hard                 |
| ----- | ----------------------- | ----------------------- |
| RF (train) | MSE: 0.0112<br>R²: 0.9498 |  MSE: 0.0016<br>R²: 0.9991 | 
| RF (test)    | MSE: 0.0347<br>R²: 0.8512 |  MSE: 0.0108<br>R²: 0.9938 | 

 
### 🏆 Best Model Metrics for target Mr (A/m)
| Model | Soft      | Hard |
| ----- | ----------------------- | ----------------------- | 
| RF (train) | MSE: 0.0183 <br>R²: 0.9324| MSE: 0.0016 <br>R²: 0.9991| 
| RF (test) | MSE: 0.0370 <br>R²: 0.8624| MSE: 0.0108 <br>R²: 0.9938| 

### 🏆 Best Model Metrics for target (BH)max 
| Model | Soft      | Hard |
| ----- | ----------------------- | ----------------------- | 
| RF (train) | MSE: 0.0012 <br>R²: 0.9987| MSE: 0.0000 <br>R²: 1.0 | 
| RF (test) | MSE:  0.0047 <br>R²: 0.9952 | MSE: 0.0999 <br>R²: 0.9999| 


## 4. Inference

To run an inference please run:
python3 ./scripts/load_onnx_models.py 
