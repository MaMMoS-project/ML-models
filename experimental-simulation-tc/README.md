
# ML model for systematic errors between simulations and experimental measurements of the Curie temperature


## Current version of model
v0.1


## 0. Installation
Use requirements.txt. In addition pytorch, compatible with your system, must be installed

## 1. data augmentation

run:
```
PYTHONPATH=$PYTHONPATH:$(pwd) python3.13 src/augment_data.py
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

## 2. creation of embeddings
Run:

```
PYTHONPATH=$PYTHONPATH:$(pwd) python3.13 src/create_embeddings.py 
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

## 3. compress embeddings with PCA
run:

```
PYTHONPATH=$PYTHONPATH:$(pwd) python3.13 src/compress_embedding_PCA.py
```

NEEDS:
- ./outputs/*embeddings.pkl

OUTPUT:
```
- stdout
- ./outputs/*embeddings_PCA.pkl
```

## 4. Model Training

## 4.1 orginal dataset

run:
```
PYTHONPATH=$PYTHONPATH:$(pwd) python3.13 src/training_original.py
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

## 4.2 orginal dataset with embedding

run:

```
PYTHONPATH=$PYTHONPATH:$(pwd) python3.13 src/training_original_emb.py
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

run:
```
PYTHONPATH=$PYTHONPATH:$(pwd) python3.13 src/training_augmented.py
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

run:
```
PYTHONPATH=$PYTHONPATH:$(pwd) python3.13 src/training_augmented_emb.py
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

## 5. Currently best model
Currently the best models are obtained with the augmented dataset and
symbolic regression
