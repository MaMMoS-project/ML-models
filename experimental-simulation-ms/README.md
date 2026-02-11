# Spontaneous Magnetization Error Correction

Identify and correct systematic biases in DFT-based simulations of spontaneous magnetization (Ms), using a combination of:
- Statistical modeling of error patterns
- AI-driven correction functions
- Data augmentation for missing experimental values (⚠️ Not yet validated and may introduce bias.)

## Current version of model
v0.1

---

## 📦 Prerequisites

- Python ≥ 3.12.3
- `pip` (Python package installer)
- `venv` (for virtual environments)
- Slurm (for job submission, if running on HPC)
- PyTorch (version matching your hardware, see: https://pytorch.org/get-started/locally/)

---

## 🛠️ Setup

### 1. Create a Virtual Environment

```bash
python -m venv venv/mammos-ms
```

### 2. Activate the Environment

```bash
# On Linux/macOS
source venv/mammos-ms/bin/activate

# On Windows
venv\mammos-ms\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> *Note: Ensure `requirements.txt` is present and up to date. If not, generate it with `pip freeze > requirements.txt` after installing packages.*

---

## 📊 Data Pre-Processing

This step combines experimental and simulation data from different sources, filters out materials with spontaneous magnetization ≤ 50,000 A/m (to reduce noise), and saves processed datasets.

### Run the Data Pre-Processing Pipeline

The datasets are stored in data/data.tar.xz. You can extract them by running `tar -xJf data/data.tar.xz`.

```bash
python -m src.data_pipeline --config configs/data_pipeline_ms.yml
```
#### Output Files

- `pairwise_df_no_Ms_leq_50000.csv`  
  → Contains only paired simulation/experimental data.
- `merged_df_no_Ms_leq_50000.csv`  
  → Contains all materials after thresholding.
- `merged_dfs_augmented_exp.csv`
→ Contains augmented materials data.
---

## 🧪 Model Training

Train machine learning models to correct systematic errors in spontaneous magnetization (Ms) predictions.

### Option 1: Run a Single Experiment

Use the `--configdir` flag to specify different configuration files. Each config defines model type, features, hyperparameters, and training settings.


#### Example Commands

```bash
# Train scikit-learn models (Random Forest, Lasso, Ridge, Linear Regression)
python -m src.run_experiments --configdir configs/scikit_models_config.yml
```

```bash
# Train symbolic regression model (performance baseline)
python -m src.run_experiments --configdir configs/pysr_linear_models_config.yml
```

```bash
# Train neural network (PyTorch)
python -m src.run_experiments --configdir configs/pytorch_mlp_config.yml
```

```bash
# Train scikit models with Mat200 embeddings (Random Forest, Lasso, Ridge, Linear Regression)
python -m src.run_experiments --configdir configs/scikit_models_w_mat200_config.yml
```

```bash
# Train neural network models with Mat200 embeddings
python -m src.run_experiments --configdir configs/pytorch_mlp_w_emb_config.yml
```
---

### Option 2: Submit to Slurm (HPC)

Submit jobs via SLURM scripts:

```bash
sbatch run_experiments_pairwise_ms.sh
```

For experiments on **augmented data**:

```bash
sbatch run_experiments_augmented_ms.sh
```

> ⚠️ **Important**: The current data augmentation strategy is **not yet validated** and may introduce bias. The augmentation logic in `src/data_augmentation.py` needs improvement.

---

## 📊 Post-process results

Post-process results and generate metric tables stored in respective experiment directory in the artifacts folder.

```
python src/post_process_results.py --dir "artifacts/"
```


##  📈 Model Performance Comparison

### Metric Results on Log-Transformed Data

This table summarizes the performance of different models across multiple datasets, comparing results **with and without structural embeddings** (from composition). All metrics are computed on **log-transformed spontaneous magnetization (Ms)** values.

| Dataset           | Best Model         | Embedding   | R²       | RMSE     | MAE      |
|-------------------|--------------------|-------------|----------|----------|----------|
| **All-Pairs**     | **Ridge**          | **Mat200**  | **0.8238** | **0.3535** | **0.1993** |
|                   | Linear Regr.       | —           | 0.7818   | 0.3933   | 0.2387   |
|                   | Symbolic Regr.     | —           | 0.7819   | 0.3933   | 0.2382   |
| **All-Synth**     | **Random Forest**  | **Mat200**  | **0.7690** | **0.3898** | **0.2289** |
|                   | Random Forest      | —           | 0.7660   | 0.3917   | 0.2300   |
|                   | Symbolic Regr.     | —           | 0.7540   | 0.4020   | 0.2370   |
| **RE-Pairs**      | **Ridge**          | **Mat200**  | **0.7462** | **0.4284** | **0.3182** |
|                   | Ridge              | —           | 0.4668   | 0.6209   | 0.4518   |
|                   | Symbolic Regr.     | —           | 0.4113   | 0.6525   | 0.4604   |
| **RE-Synth**      | **Random Forest**  | **Mat200**  | **0.7240** | **0.4304** | **0.2589** |
|                   | Linear Regr.       | —           | 0.7180   | 0.4347   | 0.2657   |
|                   | Symbolic Regr.     | —           | 0.7040   | 0.4460   | 0.2710   |
| **RE-Free Pairs** | **Random Forest**  | **Mat200**  | **0.8900** | **0.2724** | **0.1624** |
|                   | Lasso              | —           | 0.8726   | 0.2934   | 0.1751   |
|                   | Symbolic Regr.     | —           | 0.8725   | 0.2934   | 0.1745   |
| **RE-Free Synth** | **Random Forest**  | **Mat200**  | **0.7818** | **0.3779** | **0.2180** |
|                   | Random Forest      | —           | 0.7790   | 0.3790   | 0.2184   |
|                   | Symbolic Regr.     | —           | 0.7680   | 0.3820   | 0.2240   |

> 🔍 **Legend**:
> - `Mat200`: Uses compositional representations constructed from the Matscholar200 embeddings.
> - `—`: No embedding used.
> - `Symbolic Regr.`: Symbolic regression (PySR) for interpretable correction functions.
> - `R²`: Coefficient of determination (higher = better).
> - `RMSE`, `MAE`: Lower is better.

---

#### Ridge RE (best model pairwise dataset)
![Alt text](https://github.com/MaMMoS-project/ML-models/blob/add-demo-NBs/experimental-simulation-ms/best_models/mat200log_sim2exp-Ridge/re.png)

#### Random Forest RE-Free Materials (best model pairwise dataset)
![Alt text](https://github.com/MaMMoS-project/ML-models/blob/add-demo-NBs/experimental-simulation-ms/best_models/mat200log_sim2exp-RandomForest/re_free.png)

#### Random Forest RE Materials (best model augmented dataset)
![Alt text](https://github.com/MaMMoS-project/ML-models/blob/add-demo-NBs/experimental-simulation-ms/best_models/mat200log_sim2exp-RandomForest/re.png)

#### Random Forest RE-Free Materials (best model augmented dataset)
![Alt text](https://github.com/MaMMoS-project/ML-models/blob/add-demo-NBs/experimental-simulation-ms/best_models/mat200log_sim2exp-RandomForest/re_free.png)

## 📌 Notes & TODOs

- **Data filtering** at 50,000 A/m is currently hardcoded. Consider making this configurable.
- **Config files** possibly improve config files.
- **Input features** possibly add more relevant input features.
- **Data augmentation** is experimental and needs improvement.
- **Models predicting Delta files** not yet ready.
- **Config file paths** should be made more flexible (e.g., use `pathlib` or environment variables).
- Add logging and metrics tracking (e.g., MLflow, TensorBoard) as an option if wanted.
- Consider adding unit tests for data processing and model evaluation.
- **possibly use hydra** for versioning https://hydra.cc/docs/intro/
