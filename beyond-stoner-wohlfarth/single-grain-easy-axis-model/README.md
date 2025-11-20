
ML model for micromagnetic simulations, h and K aligned in z-direction

0. Installation
Use requirements.txt. In addition pytorch, compatible with your system, must be installed



1. data preprocessing

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

2. training of models
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

3. plots of metrics
run:
PYTHONPATH=$PYTHONPATH:$(pwd) python3.13 scripts/plot_metrics.py results

NEEDS:
- ./results of 2.

OUTPUT:
- stdout
- ./results/metrics_tables
