# Predicting simulated Curie temperatures from compound embeddings

This pipeline trains machine learning models that predict simulated Curie temperatures
(Tc_sim, in Kelvin) directly from stoichiometric compound embeddings — without any
experimental Tc values or data augmentation.

## Pipeline overview

```mermaid
graph TD

    %% === Cluster 1: Process TC Data ===
    subgraph cluster_1["1. Process TC Data"]
        direction TB
        A0["data/curated/*_simulated.csv"] --> B0[src/process_tc_data.py]
        B0 --> A[preprocessed_data/*.csv]
    end

    %% === Cluster 2: Create Embeddings ===
    subgraph cluster_2["2. Create Embeddings"]
        direction TB
        A --> B[src/create_embeddings.py]
        B --> C[outputs/*_w_embeddings.pkl]
    end

    %% === Cluster 3: Compress Embeddings (PCA) ===
    subgraph cluster_3["3. Compress Embeddings (PCA)"]
        direction TB
        C --> D[src/compress_embeddings_pca.py]
        D --> E[outputs/*_w_embeddings_PCA.pkl]
    end

    %% === Cluster 4: Train Models ===
    subgraph cluster_4["4. Train Models"]
        direction TB
        E --> F1[src/train_sim_tc_re_free.py]
        E --> F2[src/train_sim_tc_re.py]
        E --> F3[src/train_sim_tc_all.py]

        F1 --> G1[results/RE-Free_sim_results.csv]
        F2 --> G2[results/RE_sim_results.csv]
        F3 --> G3[results/All_sim_results.csv]
    end

    %% === Styling ===
    classDef input fill:#f0f0f0,stroke:#333,stroke-width:1px,color:#000;
    classDef process fill:#e0e8ff,stroke:#333,stroke-width:1px,color:#000;
    classDef output fill:#d0e8d0,stroke:#333,stroke-width:1px,color:#000;

    class A0,A input
    class B0,B,C,D,F1,F2,F3 process
    class G1,G2,G3 output

    class cluster_1,cluster_2,cluster_3,cluster_4 fill:#ffffff,stroke:#ccc,stroke-width:1px;
```

Three datasets are trained independently (steps 3a–3c can run in any order or in parallel):
- **RE-Free** — rare-earth-free compounds (392 rows — **curated**, see §1)
- **RE** — rare-earth-containing compounds (1,148 rows)
- **All** — combined dataset (1,540 rows)

> **Note:** `src/train_sim_tc.py` is still available as a convenience script that runs all
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

## 1. Pre-Process Data

> **RE-free data curation (with partner UU).** The rare-earth-free (RE-free) Tc data has been
> curated together with partner UU. Each raw-source entry was reviewed and classified as
> *experimental* or *theoretical* (mean-field / Monte-Carlo / RPA / ...); for RE-free compounds
> this expert classification (`Sim-TC-HH-NH-AV.xlsx`) takes **precedence** over the raw-source
> flag -- correcting many values the public databases had mislabelled as "simulated" -- and
> entries UU flagged as wrong/unusable were dropped. The raw sources were split accordingly
> into experimental/simulated files under `data/curated/`, and this (simulated-Tc) project now
> builds its datasets from the **simulation-only** curated files `data/curated/*_simulated.csv`
> and writes **simulation outputs only**. Net effect: the RE-free simulated set went from 959
> -> **392** reduced compounds (All 2,107 -> **1,540**); the RE set is unchanged (the curation
> is RE-free only). The original raw-source generator is kept at `src/process_tc_data.py.bak`.

1. **Aggregate** the curated simulation-only source files (`data/curated/*_simulated.csv`).  
2. **Clean** Tc values: remove units, symbols, and uncertainties; convert to float.  
3. **Drop** invalid (non-numeric) Tc entries.  
4. **Canonicalise & deduplicate**: reduce each formula to its pymatgen *reduced formula* (so H₂O/H₄O₂, CoFe₂O₄/Fe₂CoO₄ and other spelling / element-ordering / stoichiometric-multiple variants pool together; unparsable strings are dropped), then take the **median Tc** per reduced composition.  
5. **Flag** compositions containing rare-earth elements.  
6. **Split** data into RE-containing and RE-free subsets.  
7. **Save** clean, structured datasets for analysis.


Run:

```bash
python src/process_tc_data.py
```

**Needs:**
```
data/curated/DS1+DS2_simulated.csv
data/curated/MagneticMaterials_All_simulated.csv
data/curated/combinded_tables_simulated.csv
data/curated/literature_values_prepared_simulated.csv
data/curated/m-tcsum_nur_new_simulated.csv
```
**Outputs:**
```
preprocessed_data/Simulation_Tc_all.csv
preprocessed_data/Simulation_Tc_RE.csv
preprocessed_data/Simulation_Tc_RE-Free.csv
preprocessed_data/Simulated_Tc.csv            (copy of Simulation_Tc_all.csv)
```


## 2. Create compound embeddings

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
preprocessed_data/Simulation_Tc_RE-Free.csv
preprocessed_data/Simulation_Tc_RE.csv
preprocessed_data/Simulation_Tc_all.csv
data/embeddings/element/matscholar200.json
```

**Outputs:**
```
outputs/Simulation_Tc_RE-Free_w_embeddings.pkl
outputs/Simulation_Tc_RE_w_embeddings.pkl
outputs/Simulation_Tc_all_w_embeddings.pkl
logs/create_embeddings.txt
```

Each pickle contains the original `composition` and `Tc_sim` columns plus a
`compound_embedding` column holding a 200-D numpy array per row. Rows whose
compositions cannot be parsed or contain elements absent from the Matscholar200
vocabulary are dropped.

## 3. Compress embeddings with PCA

Fits PCA on each dataset independently and adds compressed embedding columns for
component sizes 8, 16, 32, and 64.

Run:

```bash
python src/compress_embeddings_pca.py
```

**Needs:**
```
outputs/Simulation_Tc_RE-Free_w_embeddings.pkl
outputs/Simulation_Tc_RE_w_embeddings.pkl
outputs/Simulation_Tc_all_w_embeddings.pkl
```

**Outputs:**
```
outputs/Simulation_Tc_RE-Free_w_embeddings_PCA.pkl
outputs/Simulation_Tc_RE_w_embeddings_PCA.pkl
outputs/Simulation_Tc_all_w_embeddings_PCA.pkl
logs/compress_embeddings_pca.txt
```

Each output pickle extends the input with columns `comp_emb_pca_8`, `comp_emb_pca_16`,
`comp_emb_pca_32`, and `comp_emb_pca_64`.

## 4. Train models

The framework supports **four model families**, but the shipped `training_config.yaml`
enables only the top-two — **LightGBM and Random Forest** (Linear and MLP are available but
disabled, since they trail badly on Tc). Each (family × embedding) is trained as an
**ensemble** of N members on different random train/test splits (N per family, default 10),
so the shipped default is 2 × 5 × 10 = **100 fits per dataset**.

| Model family | Notes |
|---|---|
| Linear (Lasso / Ridge, best of two) | all 5 embedding variants |
| Random Forest (randomised CV, tuned once per embedding) | all 5 embedding variants |
| MLP with early stopping (PyTorch) | all 5 embedding variants |
| LightGBM (gradient-boosted trees, randomised CV, tuned once per embedding) | all 5 embedding variants |

**Enabled in the shipped config:** LightGBM + Random Forest only (the top-two on Tc). Linear
and MLP remain in the table because they are implemented and can be re-enabled in
`training_config.yaml`.

Embedding variants: `raw_200D`, `pca_8`, `pca_16`, `pca_32`, `pca_64`.

Hyperparameters are scaled to the training-set size:
- **RF / LightGBM `n_iter`** scales inversely with n_train (the search is run **once**
  per (dataset, embedding) and the best params are reused across all ensemble members).
- **MLP architecture**: `(128, 64, 32)` for n_train < 6 000; `(256, 128, 64)` otherwise.

> **ONNX note:** every model — Linear, RF, MLP **and LightGBM** — is exported to ONNX
> (`results/onnx_models/`) for use by `predict_tc.py`. LightGBM export requires the
> `onnxmltools` package (in `requirements.txt`); if it is missing, only LightGBM is
> skipped and training still completes. With `re_features` enabled the ONNX input
> changes from a 200-D embedding to `[embedding | 7 RE feats]` (207-D), and `predict_tc`
> supplies the extra features automatically (see the `re_features` row below).

### Configuration (`training_config.yaml`)

Which families to train, the ensemble size, and the rare-earth feature toggle are all
controlled by `training_config.yaml`:

```yaml
  re_features: true         # shipped default: rare-earth physics features ON (see below)
  models:                   # shipped default enables only the top-two families (LGBM + RF)
    linear:
      enabled: false        # trails badly on Tc -> disabled
      ensemble: 10
    rf:
      enabled: true
      ensemble: 10          # train 10 members on different splits; headline = mean ± std
    mlp:
      enabled: false        # trails badly on Tc -> disabled
      ensemble: 10
    lgbm:                   # LightGBM (gradient-boosted trees)
      enabled: true
      ensemble: 10
```

**Options:**

| Key | Values | Meaning |
|---|---|---|
| `models.<family>.enabled` | `true` / `false` | Train this family or skip it entirely. Families: `linear`, `rf`, `mlp`, `lgbm`. |
| `models.<family>.ensemble` | integer ≥ 1 | Number of ensemble members (different random splits). Reported metrics are the **mean ± std** across members. `ensemble: 1` reproduces a single split (std = 0). |
| `re_features` | `true` / `false` | When `true`, append 7 rare-earth physics features (de Gennes factor, S-state fraction, free-ion moment, …) to the embedding. Zero for RE-free compounds, so safe on every dataset. The exported ONNX then takes a **207-D** input `[embedding \| 7 feats]` (**raw_200D only** — PCA variants are skipped, as they'd need an in-graph ColumnTransformer), written with a **`_refeats`** suffix so it doesn't collide with the embedding-only models; `predict_tc` detects the 207-D input and computes & appends the features from the formula automatically. Code default `false`, but the **shipped config sets it `true`**. |

Shorthands: a family may be given as a bare bool (`rf: true` ⇒ enabled, ensemble 1); an
omitted family defaults to enabled with ensemble 1; if the file is missing, all four
families train with ensemble 1 and `re_features` off. `lgbm` requires the optional
`lightgbm` package (otherwise it is skipped with a note).

Each dataset is trained by a dedicated script. Run them individually:

```bash
python src/train_sim_tc_re_free.py   # RE-Free dataset
python src/train_sim_tc_re.py        # RE dataset
python src/train_sim_tc_all.py       # All (combined) dataset
```

Or run all three in one go (backward-compatible):

```bash
python src/train_sim_tc.py
```

 
**Outputs (per script):**
```
results/<Dataset>_sim_results.csv         (one row per ensemble member)
results/<Dataset>_sim_results_agg.csv     (ensemble mean ± std per model/embedding)
results/sim_tc_comparison.csv             (aggregated, updated from all datasets run so far)
results/sim_tc_best_by_dataset.csv        (best by mean R², updated from all datasets run so far)
results/figures/<dataset>_<embedding>_<model>.png
results/onnx_models/<dataset>_<embedding>_<model>.onnx   (LightGBM needs onnxmltools;
                                                          re_features adds a _refeats, raw_200D-only, 207-D variant)
logs/train_sim_tc_re_free.txt  |  train_sim_tc_re.txt  |  train_sim_tc_all.txt
```

## 5. Predict Tc for new compounds

`src/predict_tc.py` predicts (simulated) Tc for any chemical formula using the exported
ONNX models — you give it a formula and it does all preprocessing (embedding, PCA,
scaling, and the RE features if needed) internally.

```bash
# best model for the compound's type (RE vs RE-free is auto-detected)
python src/predict_tc.py --compound Nd2Fe14B --best

# every applicable model, as a comparison table (ensemble mean ± std)
python src/predict_tc.py --compound Fe --all

# a specific model file
python src/predict_tc.py --compound SmCo5 --model results/onnx_models/RE_raw_200D_lgbm_e0.onnx

# many compounds from a file (one formula per line)
python src/predict_tc.py --compounds-file new_materials.txt --best

# list available models
python src/predict_tc.py --list
```

**Choosing a model:** `--best`/`--all` auto-detect rare-earth content and pick the right
dataset's model(s) — `--best` uses the best **RE** model for a rare-earth compound and the
best **RE-Free** model for a rare-earth-free one. If you pass `--model` yourself, match it
to the chemistry — **RE-Free** or **All** for rare-earth-free compounds (Fe, Co, Ni…),
**RE** or **All** for rare-earth compounds (Nd₂Fe₁₄B, SmCo₅…). The RE and RE-Free models
extrapolate poorly across the RE boundary, so `predict_tc` **refuses** a mismatched
`--model` (a RE model on a RE-free compound, or vice-versa) with an error telling you to
use an `All_*` model; the `All` model is always valid.

**RE-features models:** models trained with `re_features: true` are saved with a
`_refeats` suffix and take a 207-D input. `predict_tc` detects this from the ONNX graph
and computes & appends the 7 features automatically — no extra arguments. `--best`/`--all`
resolve to these `_refeats` files when they are the ones on disk (exact embedding-only
name first, `_refeats` as fallback).

### Validate against a reference set (`src/validate_reference_data.py`)

`src/validate_reference_data.py` scores the models against an external reference list of
compounds with known Curie/Néel temperatures (`data/validation_reference.csv`). For each
compound it predicts Tc with **only the best model for that chemistry** — the best **RE**
model for rare-earth compounds, the best **RE-Free** model otherwise (from
`results/sim_tc_best_by_dataset.csv`) — as the **ensemble mean ± std** over the model's ONNX
members (never a best-of-N pick). It reuses the exact prediction path from `predict_tc.py`,
so it can't drift from the deployed predictor.

```bash
python src/validate_reference_data.py
# or point at a different reference / output file
python src/validate_reference_data.py --ref data/validation_reference.csv --out table.csv
```

It prints a table (`compound | RE? | reference | prediction | std | error | best model`),
writes the same to `results/validation_reference_predictions.csv`, and reports a summary MAE
over the true ferro/ferrimagnetic Curie temperatures — antiferromagnets (Néel T) and
non-magnetic entries are shown for sanity but excluded from the error stat.

> **Note (simulated model):** the reference values are *experimental* Curie/Néel temperatures,
> whereas this model predicts a *simulated* (DFT / spin-dynamics) Tc; read the error column as
> bundling that simulation-vs-experiment offset with model error, not pure accuracy. See
> `validation_idea.txt`.

---

## Results

Metrics are on held-out 20 % test splits, reported as the **ensemble mean ± std** over
the N members (default N = 10) — not the single luckiest split. R² higher is better;
MAE and RMSE in Kelvin lower is better.

> **Current run:** **RE** and **RE-Free** below are from the latest run with **LightGBM**
> and `re_features: true` (rare-earth physics features on). The **All** (combined) dataset
> has not been run yet — run `python src/train_sim_tc_all.py` to produce it. The
> simulated-Tc datasets are small (RE ≈ 1 200 rows, RE-Free ≈ 1 000), so per-split
> variance is sizeable (note the ± std).

### Best model per dataset (ensemble mean ± std)

| Dataset | Embedding | Model | N  | R²    | R² std | MAE     | MAE std | RMSE    | RMSE std |
| ------- | --------- | ----- | -- | ----- | ------ | ------- | ------- | ------- | -------- |
| RE      | raw_200D  | RF    | 10 | 0.760 | 0.068  | 55.856  | 5.427   | 94.302  | 15.091   |
| All     | raw_200D  | RF    | 10 | 0.656 | 0.061  | 93.432  | 6.226   | 156.325 | 14.977   |
| RE-Free | raw_200D  | RF    | 10 | 0.377 | 0.111  | 195.310 | 14.908  | 276.861 | 36.036   |

### All Materials — all models × embeddings (ensemble mean ± std, with RE features)

Latest run: 4 families, re_features: true, N = 10 members.

| Dataset | Embedding | Model | N  | R²    | R² std | MAE     | MAE std | RMSE    | RMSE std |
| ------- | --------- | ----- | -- | ----- | ------ | ------- | ------- | ------- | -------- |
| All     | raw_200D  | RF    | 10 | 0.656 | 0.061  | 93.432  | 6.226   | 156.325 | 14.977   |
| All     | pca_64    | RF    | 10 | 0.640 | 0.050  | 98.729  | 6.297   | 159.996 | 13.300   |
| All     | pca_64    | LGBM  | 10 | 0.634 | 0.059  | 98.975  | 5.987   | 161.211 | 14.890   |
| All     | pca_32    | RF    | 10 | 0.634 | 0.049  | 98.472  | 6.000   | 161.485 | 12.268   |
| All     | raw_200D  | LGBM  | 10 | 0.634 | 0.082  | 93.447  | 7.128   | 160.962 | 17.215   |
| All     | pca_32    | LGBM  | 10 | 0.628 | 0.047  | 100.986 | 4.629   | 162.753 | 11.015   |
| All     | pca_16    | RF    | 10 | 0.624 | 0.057  | 97.929  | 6.440   | 163.429 | 13.066   |
| All     | pca_8     | RF    | 10 | 0.615 | 0.060  | 99.072  | 5.865   | 165.515 | 14.378   |
| All     | pca_16    | LGBM  | 10 | 0.580 | 0.072  | 103.176 | 6.478   | 172.620 | 16.112   |
| All     | pca_8     | LGBM  | 10 | 0.574 | 0.066  | 105.021 | 6.306   | 173.899 | 14.618   |


### RE Materials — all models × embeddings (ensemble mean ± std, with RE features)

Latest run: 4 families, re_features: true, N = 10 members. 

| Dataset | Embedding | Model | N  | R²    | R² std | MAE    | MAE std | RMSE    | RMSE std |
| ------- | --------- | ----- | -- | ----- | ------ | ------ | ------- | ------- | -------- |
| RE      | raw_200D  | RF    | 10 | 0.760 | 0.068  | 55.856 | 5.427   | 94.302  | 15.091   |
| RE      | raw_200D  | LGBM  | 10 | 0.741 | 0.077  | 53.687 | 5.423   | 98.053  | 15.854   |
| RE      | pca_64    | LGBM  | 10 | 0.730 | 0.070  | 58.717 | 5.492   | 100.270 | 12.927   |
| RE      | pca_32    | LGBM  | 10 | 0.727 | 0.073  | 59.457 | 5.701   | 100.578 | 13.140   |
| RE      | pca_32    | RF    | 10 | 0.720 | 0.063  | 62.613 | 5.145   | 102.351 | 12.644   |
| RE      | pca_16    | LGBM  | 10 | 0.717 | 0.091  | 59.925 | 5.939   | 102.139 | 15.447   |
| RE      | pca_8     | RF    | 10 | 0.702 | 0.074  | 62.915 | 5.916   | 105.351 | 14.133   |
| RE      | pca_64    | RF    | 10 | 0.700 | 0.085  | 64.204 | 5.562   | 105.568 | 13.180   |
| RE      | pca_16    | RF    | 10 | 0.697 | 0.087  | 64.278 | 6.099   | 106.005 | 13.868   |
| RE      | pca_8     | LGBM  | 10 | 0.682 | 0.066  | 62.830 | 5.365   | 109.243 | 12.498   |

### RE-Free Materials — all models × embeddings (ensemble mean ± std, with RE features) 

Latest run: 4 families, re_features: true, N = 10 members. (RE features are all-zero for RE-free compounds, so they leave LightGBM/Linear unchanged and only perturb RF/MLP within noise.)

| Dataset | Embedding | Model | N  | R²    | R² std | MAE     | MAE std | RMSE    | RMSE std |
| ------- | --------- | ----- | -- | ----- | ------ | ------- | ------- | ------- | -------- |
| RE-Free | raw_200D  | RF    | 10 | 0.377 | 0.111  | 195.310 | 14.908  | 276.861 | 36.036   |
| RE-Free | pca_32    | RF    | 10 | 0.366 | 0.090  | 204.466 | 14.176  | 279.469 | 32.194   |
| RE-Free | pca_32    | LGBM  | 10 | 0.339 | 0.112  | 207.012 | 15.112  | 285.085 | 32.692   |
| RE-Free | pca_64    | RF    | 10 | 0.336 | 0.087  | 207.609 | 17.324  | 286.538 | 36.045   |
| RE-Free | pca_16    | RF    | 10 | 0.323 | 0.088  | 210.558 | 15.042  | 288.891 | 32.227   |
| RE-Free | pca_8     | RF    | 10 | 0.320 | 0.073  | 214.991 | 12.047  | 289.767 | 29.342   |
| RE-Free | raw_200D  | LGBM  | 10 | 0.320 | 0.057  | 213.459 | 13.095  | 289.960 | 28.133   |
| RE-Free | pca_64    | LGBM  | 10 | 0.302 | 0.104  | 206.535 | 20.142  | 293.596 | 37.587   |
| RE-Free | pca_16    | LGBM  | 10 | 0.295 | 0.068  | 210.959 | 12.913  | 295.001 | 27.611   |
| RE-Free | pca_8     | LGBM  | 10 | 0.272 | 0.073  | 222.973 | 12.046  | 299.298 | 24.338   |
