
# ML model for systematic errors between simulations and experimental measurements of the Curie temperature

This codebase implements various machine learning models to predict experimental Curie temperatures from simulated values. Additionally, chemical property information is incorporated via an embedding representation.

## Current version of model
v0.1


## 0. Installation
Use requirements.txt. In addition pytorch, compatible with your system, must be installed
- PyTorch (version matching your hardware, see: https://pytorch.org/get-started/locally/)

## 1. Data augmentation

Executing the code below performs data augmentation on missing experimental values using bootstrap sampling.

Run:

```
python3 -m src.augment_data
```

NEEDS:

- ./data/EC_curie_temp.csv


OUTPUT:
```
- stdout
- ./outputs/Augm_RE.csv
- ./outputsAugm_RE_Free.csv
- ./outputs/Augm_RE_Free_emb.csv
- ./outputs/Augm_RE_emb.csv
- ./outputs/Augm_all.csv
- ./outputs/Augm_all_emb.csv
- ./outputs/Pairs_RE.csv
- ./outputs/Pairs_RE_Free.csv
- ./outputs/Pairs_RE_Free_emb.csv
- ./outputs/Pairs_RE_emb.csv
- ./outputs/Pairs_all.csv
- ./outputs/Pairs_all_emb.csv
- ./outputs/distribution_plots/*.png
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
- ./outputs/Augm_RE.csv
- ./outputsAugm_RE_Free.csv
- ./outputs/Augm_RE_Free_emb.csv
- ./outputs/Augm_RE_emb.csv
- ./outputs/Augm_all.csv
- ./outputs/Augm_all_emb.csv
- ./outputs/Pairs_RE.csv
- ./outputs/Pairs_RE_Free.csv
- ./outputs/Pairs_RE_Free_emb.csv
- ./outputs/Pairs_RE_emb.csv
- ./outputs/Pairs_all.csv
- ./outputs/Pairs_all_emb.csv

OUTPUT:
```
- stdout
- ./outputs/embeddings_tsne_plots/*.png
- ./outputs/*embeddings.pkl
```

## 3. Compress embeddings with PCA
Create PCA-compressed embeddings for the paired Curie temperature dataset.
It focuses specifically on PCA components of sizes 8, 16, and 32 to ensure they are available
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

## 4. Model Training

Train baseline models on original (non-augmented, non-embedding) data. Namely, 

· Symbolic regression: stoichiometry was disregarded
· LASSO regression,
· RIDGE regression,
· Random Forest,
· FCNN.

## 4.1 Orginal dataset

Run:

```
python3 -m src.training_original
```

NEEDS:
- ./outputs/Pairs_RE.csv
- ./outputs/Pairs_RE_Free.csv
- ./outputs/Pairs_all_emb.csv

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
- ./outputs/Augm_RE_Free.csv
- ./outputs/Augm_RE.csv
- ./outputs/Augm_all.csv

OUTPUT:
```
- stdout
- ./results/augmented_[model]
- ./results/figures/All-Augm_*_no_emb.png
- ./results/figures/All-Augm_SR.png
- ./results/figures/RE-Augm_*_no_emb.png
- ./results/figures/RE-Augm_SR.png
- ./results/figures/RE-Free-Augm_*_no_emb.png
- ./results/figures/RE-Free-Augm_SR.png
- ./results/augmented_comparison/*.csv
```



## 4.4 Augmented dataset with stoichiometry embedding
Train models on augmented data WITH EMBEDDINGS.

Run:

```
python3 -m src.training_augmented_emb
```

NEEDS:
- ./outputs/Augm_RE_Free_emb.csv
- ./outputs/Augm_RE_emb.csv
- ./outputs/Augm_all_emb.csv
- ./outputs/Augm_RE_Free_emb_w_embeddings.pkl
- ./outputs/Augm_RE_Free_emb_w_embeddings_PCA.pkl
- ./outputs/Augm_RE_emb_w_embeddings.pkl
- ./outputs/Augm_RE_emb_w_embeddings_PCA.pkl
- ./outputs/Augm_all_emb_w_embeddings.pkl
- ./outputs/Augm_all_emb_w_embeddings_PCA.pkl



OUTPUT:
```
- stdout
- ./results/augmented_emb_[model]
- ./results/figures/All-Augm_[model]_[None|pca_*].png
- ./results/figures/RE-Augm_[model]_[None|pca_*].png
- ./results/figures/RE-Free-Augm_[model]_[None|pca_*].png
- ./results/augmented_emb_comparison/*.csv
```
## 📈 Model Performance Comparison
| Dataset        | Best Model (Embedding) | Embedding | R2    | RMSE    | Best Model | R2    | RMSE    | Baseline | Baseline R2 | Baseline RMSE |
|----------------|------------------------|-----------|-------|---------|------------|-------|---------|----------|-------------|---------------|
| All-Pairs      | Ridge                  | PCA32     | 0.791 | 110.762 | MLP        | 0.849 | 94.323  | SR       | 0.841       | 96.757        |
| All-Augm       | MLP                    | PCA32     | 0.927 | 74.566  | MLP        | 0.928 | 81.235  | SR       | 0.927       | 81.526        |
| RE-Pairs       | Ridge                  | PCA32     | 0.791 | 110.762 | MLP        | 0.915 | 51.738  | SR       | 0.913       | 52.234        |
| RE-Augm        | MLP                    | PCA32     | 0.929 | 73.819  | MLP        | 0.967 | 33.534  | SR       | 0.967       | 33.538        |
| RE-free Pairs  | Ridge                  | PCA32     | 0.791 | 110.762 | MLP        | 0.791 | 129.950 | SR       | 0.789       | 130.640       |
| RE-free Augm   | MLP                    | PCA8      | 0.927 | 75.583  | MLP        | 0.904 | 111.992 | SR       | 0.901       | 113.760       |

## Current best model
Currently the best models are obtained with the augmented dataset and symbolic regression.
