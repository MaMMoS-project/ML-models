
# ML model for systematic errors between simulations and experimental measurements of the Curie temperature

This codebase implements various machine learning models to predict experimental Curie temperatures from simulated values. Additionally, chemical property information is incorporated via an embedding representation. 

## Current version of model
v0.2


## 0. Installation
Use requirements.txt. In addition pytorch, compatible with your system, must be installed
- PyTorch (version matching your hardware, see: https://pytorch.org/get-started/locally/)

# Data Processing


```mermaid
flowchart TB

%% =========================
%% Styles
%% =========================
classDef input fill:#D6EAF8,stroke:#2E86C1,stroke-width:2px,color:#000;
classDef process fill:#D5F5E3,stroke:#27AE60,stroke-width:2px,color:#000;
classDef output fill:#FDEBD0,stroke:#E67E22,stroke-width:2px,color:#000;

%% =========================
%% 0. Build merged dataset (run first if the merged CSV is missing)
%% =========================
subgraph cluster_build["0. Build merged dataset"]
    direction TB

    R0["data/ raw sources:\nm-tcsum_nur_new.csv, sd_tc_data.csv, DS1+DS2.csv,\nliterature_values_prepared.csv, combinded_tables.xlsx, MagneticMaterials_All.csv"]
    Bb["python3 -m src.build_merged_tc"]

    R0 --> Bb
    Bb --> A0
end

%% =========================
%% 1. Data Augmentation
%% =========================
subgraph cluster_0["1. Data Augmentation (Bootstrap Sampling)"]
    direction TB

    A0["./data/merged_curie_temp.csv"]
    B0["python3 -m src.augment_data"]

    A0 --> B0

    B0 --> O1["./outputs/Pairs_*.csv"]
    B0 --> O2["./outputs/Augm_sim_*.csv"]
    B0 --> O3["./outputs/Augm_exp_*.csv"]
    B0 --> O4["./outputs/Augm_combined_*.csv"]
    B0 --> O5["./outputs/distributions_plots/*.png"]
end

%% =========================
%% 2. Create Embeddings
%% =========================
subgraph cluster_1["2. Create Embeddings"]
    direction TB

    A1["./data/embeddings/element/matscholar200.json"]
    A2["./outputs/Pairs_all_emb.csv"]
    A3["./outputs/Augm_combined_all_emb.csv"]

    B1["python3 -m src.create_embeddings"]

    A1 --> B1
    A2 --> B1
    A3 --> B1

    B1 --> O7["./outputs/embeddings_tsne_plots/*.png"]
    B1 --> O8["./outputs/*embeddings.pkl"]
end

%% =========================
%% 3. PCA Compression
%% =========================
subgraph cluster_2["3. PCA Compression of Embeddings"]
    direction TB

    A4["./outputs/*embeddings.pkl"]
    B2["python3 -m src.compress_embedding_PCA"]

    A4 --> B2

    B2 --> O10["./outputs/*embeddings_PCA.pkl"]
end

%% =========================
%% Pipeline Flow
%% =========================
O5 --> A2
O4 --> A3
O8 --> A4

%% =========================
%% Apply Classes
%% =========================
class A0,A1,A2,A3,A4,R0 input;
class B0,B1,B2,Bb process;
class O1,O2,O3,O4,O5,O7,O8,O10 output;

%% =========================
%% Subgraph Styling
%% =========================
style cluster_build fill:#F4F6F7,stroke:#5D6D7E,stroke-width:2px
style cluster_0 fill:#F8F9FA,stroke:#5D6D7E,stroke-width:2px
style cluster_1 fill:#F4F6F7,stroke:#5D6D7E,stroke-width:2px
style cluster_2 fill:#F8F9FA,stroke:#5D6D7E,stroke-width:2px
```

## 0. Build merged dataset

Aggregates the experimental and simulated Curie temperatures from the raw sources into a
single lean training table, `./data/merged_curie_temp.csv`. **Run this first if that file
does not exist** (or when the raw sources change); every later stage depends on it.

For each composition, all simulated (resp. experimental) Tc values from every source are
pooled and reduced with a **single median** (one median, every source included — not a
per-source pre-average and not a median-of-medians).

Run:

```
python3 -m src.build_merged_tc
```

NEEDS (in `./data/`):
- m-tcsum_nur_new.csv, sd_tc_data.csv, DS1+DS2.csv
- literature_values_prepared.csv, combinded_tables.xlsx, MagneticMaterials_All.csv

OUTPUT — `./data/merged_curie_temp.csv`, a plain CSV with columns:
```
composition, Tc_sim, Tc_exp, contains_rare_earth, use_for_emb
```
(`Tc_delta = Tc_exp − Tc_sim` and `pair_exists = both present` are derived downstream.)

## 1. Data augmentation

Executing the code below performs data augmentation on missing experimental values using bootstrap sampling.

Run:

```
python3 -m src.augment_data
```

NEEDS:

- ./data/merged_curie_temp.csv


OUTPUT:
```
- stdout
- ./outputs/Pairs_all.csv
- ./outputs/Pairs_RE.csv
- ./outputs/Pairs_RE_Free.csv
- ./outputs/Pairs_all_emb.csv
- ./outputs/Pairs_RE_emb.csv
- ./outputs/Pairs_RE_Free_emb.csv
- ./outputs/Augm_sim_all.csv          # Phase 1: paired + Tc_sim-only (mock Tc_exp)
- ./outputs/Augm_sim_RE.csv
- ./outputs/Augm_sim_RE_Free.csv
- ./outputs/Augm_sim_all_emb.csv
- ./outputs/Augm_sim_RE_emb.csv
- ./outputs/Augm_sim_RE_Free_emb.csv
- ./outputs/Augm_exp_all.csv          # Phase 2: paired + Tc_exp-only (mock Tc_sim)
- ./outputs/Augm_exp_RE.csv
- ./outputs/Augm_exp_RE_Free.csv
- ./outputs/Augm_exp_all_emb.csv
- ./outputs/Augm_exp_RE_emb.csv
- ./outputs/Augm_exp_RE_Free_emb.csv
- ./outputs/Augm_combined_all.csv     # Phase 3: Phase 1 + Phase 2 (used for training)
- ./outputs/Augm_combined_RE.csv
- ./outputs/Augm_combined_RE_Free.csv
- ./outputs/Augm_combined_all_emb.csv
- ./outputs/Augm_combined_RE_emb.csv
- ./outputs/Augm_combined_RE_Free_emb.csv
- ./outputs/distributions_plots/*.png
```

## 2. Creation of embeddings

Stoichiometric embeddings are created from the Matscholar200 embeddings
using an element-abundance weighted sum approach. For example:
    H2O embedding = 2 × [H embedding] + 1 × [O embedding]

Run:

```
python3 -m src.create_embeddings
```

NEEDS:
- ./data/embeddings/element/matscholar200.json
- ./outputs/Pairs_all_emb.csv
- ./outputs/Pairs_RE_emb.csv
- ./outputs/Pairs_RE_Free_emb.csv
- ./outputs/Augm_combined_all_emb.csv  (required)
- ./outputs/Augm_combined_RE_emb.csv   (required)
- ./outputs/Augm_combined_RE_Free_emb.csv  (required)
- ./outputs/Augm_exp_all_emb.csv       (optional, processed when present)
- ./outputs/Augm_exp_RE_emb.csv        (optional)
- ./outputs/Augm_exp_RE_Free_emb.csv   (optional)
- ./outputs/Augm_sim_all_emb.csv       (optional)
- ./outputs/Augm_sim_RE_emb.csv        (optional)
- ./outputs/Augm_sim_RE_Free_emb.csv   (optional)

OUTPUT:
```
- stdout
- ./outputs/embeddings_tsne_plots/*.png
- ./outputs/*embeddings.pkl
```

## 3. Compress embeddings with PCA
Create PCA-compressed embeddings for the paired Curie temperature dataset.
It computes PCA components of sizes 8, 16, 32, and 64 to ensure they are available
for the training scripts.

Run:

```
python3 -m src.compress_embedding_PCA
```

NEEDS:
- ./outputs/*embeddings.pkl

OUTPUT:
```
- stdout
- ./outputs/*embeddings_PCA.pkl
```
# Modeling

## 4. Model Training

Train baseline models on original (non-augmented, non-embedding) data. Namely, 

· Symbolic regression: stoichiometry was disregarded
· LASSO regression,
· RIDGE regression,
· Random Forest,
· FCNN.

The materials dataset is evaluated separately for RE and RE-free samples to account
for potential differences in data distribution and model behavior. Experiments on the
combined (“All”) dataset are included as a global baseline to assess generalization.

## 4.1 Orginal dataset

Run:

```
python3 -m src.training_original
```

NEEDS:
- ./outputs/Pairs_all.csv
- ./outputs/Pairs_RE.csv
- ./outputs/Pairs_RE_Free.csv

OUTPUT:
```
- stdout
- ./results/figures/All-Pairs_*_no_emb.png
- ./results/figures/RE-Pairs_*_no_emb.png
- ./results/figures/RE-Free-Pairs_*_no_emb.png
- ./results/original_[model]
- ./results/original_comparision/*.csv
```

## 4.2 Orginal dataset with stoichiometric embedding
Train models on original data with stoichiometric embeddings as additional input to the simulate value.

Run:

```
python3 -m src.training_original_emb
```

NEEDS:
- ./outputs/Pairs_RE_Free_emb.csv
- ./outputs/Pairs_RE_emb.csv
- ./outputs/Pairs_all_emb.csv
- ./outputs/Pairs_RE_Free_emb_w_embeddings.pkl
- ./outputs/Pairs_RE_Free_emb_w_embeddings_PCA.pkl
- ./outputs/Pairs_RE_emb_w_embeddings.pkl
- ./outputs/Pairs_RE_emb_w_embeddings_PCA.pkl
- ./outputs/Pairs_all_emb_w_embeddings.pkl
- ./outputs/Pairs_all_emb_w_embeddings_PCA.pkl


OUTPUT:
```
- stdout
- ./results/original_emb_[model]
- ./results/original_emb_comparison/*.csv
- ./results/figures/All-Pairs_[model]_[None|pca_*].png
- ./results/figures/RE_Pairs_[model]_[None|pca_*].png
```

## 4.3 Augmented dataset

Train baseline models on augmented data (no embeddings).

Run:

```
python3 -m src.training_augmented
```

NEEDS:
- ./outputs/Augm_exp_all.csv
- ./outputs/Augm_exp_RE.csv
- ./outputs/Augm_exp_RE_Free.csv
- ./outputs/Augm_sim_all.csv
- ./outputs/Augm_sim_RE.csv
- ./outputs/Augm_sim_RE_Free.csv
- ./outputs/Augm_combined_all.csv
- ./outputs/Augm_combined_RE.csv
- ./outputs/Augm_combined_RE_Free.csv

OUTPUT:
```
- stdout
- ./results/augmented_[model]/{variant}/      (variant: exp_augmented, sim_augmented, combined_augmented)
- ./results/figures/{variant}/[All|RE|RE-Free]-Augm_*_no_emb.png
- ./results/figures/{variant}/[All|RE|RE-Free]-Augm_SR.png
- ./results/augmented_comparison/{variant}/augmented_models_comparison.csv
- ./results/augmented_comparison/{variant}/augmented_best_by_dataset.csv
- ./results/augmented_comparison/{variant}/augmented_comparison_pivot.csv
- ./results/augmented_comparison/augmented_all_variants_comparison.csv
- ./results/augmented_comparison/augmented_all_variants_best.csv
- ./results/augmented_comparison/augmented_cross_variant_pivot.csv
```



## 4.4 Augmented dataset with stoichiometry embedding
Train models on augmented data WITH EMBEDDINGS.

Run:

```
python3 -m src.training_augmented_emb
```

NEEDS:
- ./outputs/Augm_exp_all_emb_w_embeddings[_PCA].pkl
- ./outputs/Augm_exp_RE_emb_w_embeddings[_PCA].pkl
- ./outputs/Augm_exp_RE_Free_emb_w_embeddings[_PCA].pkl
- ./outputs/Augm_sim_all_emb_w_embeddings[_PCA].pkl
- ./outputs/Augm_sim_RE_emb_w_embeddings[_PCA].pkl
- ./outputs/Augm_sim_RE_Free_emb_w_embeddings[_PCA].pkl
- ./outputs/Augm_combined_all_emb_w_embeddings[_PCA].pkl
- ./outputs/Augm_combined_RE_emb_w_embeddings[_PCA].pkl
- ./outputs/Augm_combined_RE_Free_emb_w_embeddings[_PCA].pkl

(For each file the _PCA.pkl variant is preferred; plain .pkl is used as fallback.)

OUTPUT:
```
- stdout
- ./results/augmented_emb_[model]/{variant}/      (variant: exp_augmented, sim_augmented, combined_augmented)
- ./results/figures/{variant}/[All|RE|RE-Free]-Augm_[model]_[None|pca_*].png
- ./results/augmented_emb_comparison/{variant}/augmented_emb_models_comparison.csv
- ./results/augmented_emb_comparison/{variant}/augmented_emb_best_by_dataset.csv
- ./results/augmented_emb_comparison/{variant}/augmented_emb_comparison_pivot.csv
- ./results/augmented_emb_comparison/augmented_emb_all_variants_comparison.csv
- ./results/augmented_emb_comparison/augmented_emb_all_variants_best.csv
- ./results/augmented_emb_comparison/augmented_emb_cross_variant_pivot.csv
```
## 📈 Model Performance Comparison 
(best models and symbolic regression baseline shown)

| Dataset         | Model              | Embedding   | R²    | RMSE    |
|----------------|---------------------|-------------|-------|---------|
| All-Pairs      | **MLP (FCNN)**      | -           | 0.848 | 94.56   |
| All-Pairs      | MLP (FCNN)          | raw_200D    | 0.801 | 107.931 |
| All-Pairs      | Symbolic Regression | -           | 0.841 | 96.758  |
| All-Augm       | MLP (FCNN)          | -           | 0.935 | 69.907  |
| All-Augm       | **Random Forest**   | PCA8        | 0.942 | 64.689  |
| All-Augm       | Symbolic Regression | -           | 0.935 | 70.342  |
| RE-Pairs       | MLP (FCNN)          | -           | 0.913 | 52.197  |
| RE-Pairs       | **MLP (FCNN)**      | PCA8        | 0.946 | 37.26   |
| RE-Pairs       | Symbolic Regression | -           | 0.913 | 52.234  |
| RE-Augm        | Linear (LINEAR)     | -           | 0.980 | 38.240  |
| RE-Augm        | **Random Forest**   | PCA16       | 0.984 | 33.854  |
| RE-Augm        | Symbolic Regression | -           | 0.980 | 38.282  |
| RE-Free-Pairs  | MLP (FCNN)          | -           | 0.792 | 129.820 |
| RE-Free-Pairs  | **Linear (LASSO)**  | raw_200D    | 0.800 | 122.397 |
| RE-Free-Pairs  | Symbolic Regression | -           | 0.789 | 130.646 |
| RE-Free-Augm   | MLP (FCNN)          | -           | 0.829 | 119.460 |
| RE-Free-Augm   | **Random Forest**   | raw_200D    | 0.909 | 83.60   |
| RE-Free-Augm   | Symbolic Regression | -           | 0.827 | 120.166 |


> 🔍 **Note**: The augmented datasets (`All-Augm`, `RE-Augm`, `RE-Free-Augm`) were created by combining **simulated (Tc_sim)** and **experimental (Tc_exp)** data to improve model generalization and performance.

### 📊 Summary of Results

Data augmentation significantly improved model performance across all datasets. The best-performing models were MLP (FCNN) for All-Pairs, RE-Pairs, and RE-Free-Augm, Random Forest for All-Augm and RE-Augm, and LASSO for RE-Free-Pairs. While embeddings were not universally beneficial, low-dimensional PCA embeddings substantially improved performance for RE-Pairs (PCA8) and RE-Augm (PCA16), where they enabled the highest predictive accuracies. In contrast, models trained on the original Mat200 descriptors generally performed better for the remaining datasets. Symbolic regression consistently achieved performance comparable to the best machine learning models while providing interpretable relationships. The lower performance observed for the RE-Free datasets indicates a more challenging prediction task and supports their separate evaluation from the RE datasets.