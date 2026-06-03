
# Predicting simulated Curie temperatures from compound embeddings

This pipeline trains machine learning models that predict simulated Curie temperatures
(Tc_sim, in Kelvin) directly from stoichiometric compound embeddings — without any
experimental Tc values or data augmentation.

## Pipeline overview

```
preprocessed_data/*.csv
        │
        ▼
1. src/create_embeddings.py        → outputs/*_w_embeddings.pkl
        │
        ▼
2. src/compress_embeddings_pca.py  → outputs/*_w_embeddings_PCA.pkl
        │
        ▼
3a. src/train_exp_tc_re_free.py    → results/exp_tc/RE-Free_results.csv
3b. src/train_exp_tc_re.py         → results/exp_tc/RE_results.csv
3c. src/train_exp_tc_all.py        → results/exp_tc/All_results.csv
```

```mermaid
graph TD
    %% === Cluster 1: Create Embeddings ===
    subgraph cluster_1["1. Create Embeddings"]
        direction TB
        A[preprocessed_data/*.csv] --> B[src/create_embeddings.py]
        B --> C[outputs/*_w_embeddings.pkl]
    end

    %% === Cluster 2: Compress Embeddings ===
    subgraph cluster_2["2. Compress Embeddings (PCA)"]
        direction TB
        C --> D[src/compress_embeddings_pca.py]
        D --> E[outputs/*_w_embeddings_PCA.pkl]
    end

    %% === Cluster 3: Train Models ===
    subgraph cluster_3["3. Train Models"]
        direction TB
        E --> F1[src/train_exp_tc_re_free.py]
        E --> F2[src/train_exp_tc_re.py]
        E --> F3[src/train_exp_tc_all.py]
        F1 --> G1[results/exp_tc/RE-Free_results.csv]
        F2 --> G2[results/exp_tc/RE_results.csv]
        F3 --> G3[results/exp_tc/All_results.csv]
    end

    %% === Styling ===
    classDef input fill:#f0f0f0,stroke:#333,stroke-width:1px,color:#000;
    classDef process fill:#e0e8ff,stroke:#333,stroke-width:1px,color:#000;
    classDef output fill:#d0e8d0,stroke:#333,stroke-width:1px,color:#000;

    class A input
    class B,C,D,F1,F2,F3 process
    class G1,G2,G3 output

    class cluster_1,cluster_2,cluster_3 fill:#ffffff,stroke:#ccc,stroke-width:1px;
```

Three datasets are trained independently (steps 3a–3c can run in any order or in parallel):
- **RE-Free** — rare-earth-free compounds (~6 200 rows)
- **RE** — rare-earth-containing compounds (~9 800 rows)
- **All** — combined dataset (~16 000 rows)

> **Note:** `src/train_exp_tc.py` is still available as a convenience script that runs all
> three datasets in sequence and is the shared library used by the individual scripts.

## 0. Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

PyTorch must be installed separately to match your hardware:

```bash
# CPU-only example — see https://pytorch.org/get-started/locally/ for GPU variants
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 1. Create compound embeddings

Generates element-abundance-weighted compound embeddings from the Matscholar200
element vectors (200-dimensional). For example:

```
Fe2O3 embedding = (2/5) × [Fe vec] + (3/5) × [O vec]
```

Run:

```bash
python src/create_embeddings.py
```

**Needs:**
```
preprocessed_data/Experimental_Tc_RE-Free.csv
preprocessed_data/Experimental_Tc_RE.csv
preprocessed_data/Experimental_Tc_all.csv
data/embeddings/element/matscholar200.json
```

**Outputs:**
```
outputs/Experimental_Tc_RE-Free_w_embeddings.pkl
outputs/Experimental_Tc_RE_w_embeddings.pkl
outputs/Experimental_Tc_all_w_embeddings.pkl
logs/create_embeddings.txt
```

Each pickle contains the original `composition` and `Tc_exp` columns plus a
`compound_embedding` column holding a 200-D numpy array per row. Rows whose
compositions cannot be parsed or contain elements absent from the Matscholar200
vocabulary are dropped.

## 2. Compress embeddings with PCA

Fits PCA on each dataset independently and adds compressed embedding columns for
component sizes 8, 16, 32, and 64.

Run:

```bash
python src/compress_embeddings_pca.py
```

**Needs:**
```
outputs/Experimental_Tc_RE-Free_w_embeddings.pkl
outputs/Experimental_Tc_RE_w_embeddings.pkl
outputs/Experimental_Tc_all_w_embeddings.pkl
```

**Outputs:**
```
outputs/Experimental_Tc_RE-Free_w_embeddings_PCA.pkl
outputs/Experimental_Tc_RE_w_embeddings_PCA.pkl
outputs/Experimental_Tc_all_w_embeddings_PCA.pkl
logs/compress_embeddings_pca.txt
```

Each output pickle extends the input with columns `comp_emb_pca_8`, `comp_emb_pca_16`,
`comp_emb_pca_32`, and `comp_emb_pca_64`.

## 3. Train models

Trains three model families on five embedding variants for each of the three datasets
(15 training runs per dataset, 45 total):

| Model family | Variants |
|---|---|
| Linear (Lasso / Ridge best of two) | all 5 embedding variants |
| Random Forest (randomised CV) | all 5 embedding variants |
| MLP with early stopping (PyTorch) | all 5 embedding variants |

Embedding variants: `raw_200D`, `pca_8`, `pca_16`, `pca_32`, `pca_64`.

Hyperparameters are scaled to the training-set size:
- **RF `n_iter`** scales inversely with n_train (≈40 / 25 / 15 for RE-Free / RE / All).
- **MLP architecture**: `(128, 64, 32)` for n_train < 6 000; `(256, 128, 64)` otherwise.

Each dataset is trained by a dedicated script. Run them individually:

```bash
python src/train_exp_tc_re_free.py   # RE-Free dataset
python src/train_exp_tc_re.py        # RE dataset
python src/train_exp_tc_all.py       # All (combined) dataset
```

Or run all three in one go (backward-compatible):

```bash
python src/train_exp_tc.py
```

**Needs (per script):**
```
outputs/Experimental_Tc_RE-Free_w_embeddings_PCA.pkl   ← train_exp_tc_re_free.py
outputs/Experimental_Tc_RE_w_embeddings_PCA.pkl         ← train_exp_tc_re.py
outputs/Experimental_Tc_all_w_embeddings_PCA.pkl        ← train_exp_tc_all.py
```

**Outputs (per script):**
```
results/exp_tc/<Dataset>_results.csv
results/exp_tc/exp_tc_comparison.csv      (updated from all datasets run so far)
results/exp_tc/exp_tc_best_by_dataset.csv (updated from all datasets run so far)
results/exp_tc/figures/<dataset>_<embedding>_<model>.png
logs/train_exp_tc_re_free.txt  |  train_exp_tc_re.txt  |  train_exp_tc_all.txt
```

---

## Results

All metrics are on a held-out 20 % test split. Metrics are R² (higher is better),
MAE and RMSE in Kelvin (lower is better).

### Best model per dataset


---

### RE-Free — full results table

| Embedding | Model | R² | MAE (K) | RMSE (K) |

---

### RE — full results table

| Embedding | Model | R² | MAE (K) | RMSE (K) |

---

### All — full results table

| Embedding | Model | R² | MAE (K) | RMSE (K) |

