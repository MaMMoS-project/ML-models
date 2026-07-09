# -*- coding: utf-8 -*-
"""Train models on Phase 3 combined augmented data WITH compound embeddings.

Usage:
    python -m src.training_augmented_emb

Requires:
    outputs/Augm_combined_*_w_embeddings_PCA.pkl
    (run augment_data.py → create_embeddings.py → compress_embedding_PCA.py first)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

from src.log_to_file import log_output

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)


# @log_output('logs/training_augmented_emb.txt')
def training_augmented_emb():

    script_dir = Path(__file__).parent
    training_dir = script_dir / "training"
    sys.path.insert(0, str(training_dir))

    project_root = script_dir.parent
    results_dir = project_root / "results"
    output_dir  = project_root / "outputs"

    from linear_models import LinearModelsTrainer
    from random_forest import RandomForestTrainer
    from fcnn_mlp import FCNNTrainer
    from base_trainer import SIM_COL, EXP_COL, parse_ms_threshold, parse_delta_learning

    ms_threshold = parse_ms_threshold()
    delta_learning = parse_delta_learning()

    print("=" * 80)
    print("AUGMENTED DATA TRAINING WITH EMBEDDINGS")
    print("=" * 80)

    dataset_configs = [
        {"name": "All-Augm",      "type": "all",     "file_prefix": "Augm_combined_all_w_embeddings"},
        {"name": "RE-Augm",       "type": "re",      "file_prefix": "Augm_combined_RE_w_embeddings"},
        {"name": "RE-Free-Augm",  "type": "re-free", "file_prefix": "Augm_combined_RE_Free_w_embeddings"},
    ]

    embedding_types = [None, "pca_16", "pca_32", "pca_8", "pca_64"]
    is_augmented = True

    # ── Locate PKL files ─────────────────────────────────────────────────
    loaded_dataframes = {}
    for cfg in dataset_configs:
        dtype  = cfg["type"]
        prefix = cfg["file_prefix"]
        pca_path  = output_dir / f"{prefix}_PCA.pkl"
        orig_path = output_dir / f"{prefix}.pkl"

        if pca_path.exists():
            loaded_dataframes[dtype] = {"file": pca_path, "loaded": False}
            print(f"Found PCA file for '{dtype}': {pca_path}")
        elif orig_path.exists():
            loaded_dataframes[dtype] = {"file": orig_path, "loaded": False}
            print(f"Found original embedding file for '{dtype}': {orig_path}")
        else:
            print(f"WARNING: No embedding file found for '{dtype}' — "
                  "run augment_data.py, create_embeddings.py, compress_embedding_PCA.py")

    if not loaded_dataframes:
        print("ERROR: No embedding files found.")
        sys.exit(1)

    def load_dataset(dataset_type):
        if dataset_type not in loaded_dataframes:
            return None
        info = loaded_dataframes[dataset_type]
        if not info["loaded"]:
            try:
                df = pd.read_pickle(info["file"])
                info["data"] = df
                info["loaded"] = True
                emb_cols = [c for c in df.columns if 'emb' in c.lower()]
                print(f"  Loaded {len(df)} rows for '{dataset_type}'. Emb cols: {emb_cols}")
            except Exception as e:
                print(f"ERROR loading {info['file']}: {e}")
                return None
        return loaded_dataframes[dataset_type]["data"].copy()

    all_results = []

    for config in dataset_configs:
        
        dataset_name = config["name"]
        dataset_type = config["type"]

        if dataset_type not in loaded_dataframes:
            print(f"\nSkipping {dataset_name} — no embedding file.")
            continue

        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*60}")

        df_data = load_dataset(dataset_type)
        if df_data is None:
            continue

        if ms_threshold is not None:
            before = len(df_data)
            df_data = df_data[
                (df_data[SIM_COL] > ms_threshold) & (df_data[EXP_COL] > ms_threshold)
            ].copy()
            print(f"Ms threshold {ms_threshold:.0f} A/m: {before} → {len(df_data)} rows kept")

        def make_loader_patch(df_ref):
            return lambda dataset_type=None: df_ref.copy()

        # Create trainers once per dataset; each train_and_evaluate call builds
        # a fresh model, so sharing the trainer across embedding types is safe.
        lin_trainer = LinearModelsTrainer(
            output_dir=str(results_dir / "augmented_emb_linear")
        )
        rf_trainer = RandomForestTrainer(
            output_dir=str(results_dir / "augmented_emb_rf")
        )
        mlp_trainer = FCNNTrainer(
            output_dir=str(results_dir / "augmented_emb_fcnn")
        )
        for _t in (lin_trainer, rf_trainer, mlp_trainer):
            _t.loader.load_augmented_data = make_loader_patch(df_data)
            _t.loader.load_pairs_data = make_loader_patch(df_data)
            _t.loader.delta_learning = delta_learning

        for embedding_type in embedding_types:
            emb_name = "raw_200D" if embedding_type is None else embedding_type

            if embedding_type is None:
                if 'compound_embedding' not in df_data.columns:
                    print(f"  Skipping raw_200D — no compound_embedding column.")
                    continue
            else:
                col_name = f'comp_emb_{embedding_type}_components'
                if col_name not in df_data.columns:
                    print(f"  Skipping {emb_name} — column {col_name} not found.")
                    continue

            print(f"\n{'-'*50}")
            print(f"Embedding: {emb_name}")
            print(f"{'-'*50}")

            dataset_emb_results = []

            # 1. Linear models
            try:
                lin_results = lin_trainer.train_and_evaluate(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    is_augmented=is_augmented,
                    use_embedding=True,
                    embedding_type=embedding_type,
                    model_types=["lasso", "ridge"],
                )
                best_model = max(lin_results, key=lambda k: lin_results[k]["R2"])
                best_metrics = lin_results[best_model]
                dataset_emb_results.append({
                    "Model_Family": "Linear", "Model": best_model.upper(),
                    "Dataset": dataset_name, "Embedding": emb_name,
                    "R2": best_metrics["R2"], "RMSE": best_metrics["RMSE"],
                    "MAE": best_metrics["MAE"],
                })
                print(f"  Linear - Best: {best_model.upper()}, R²: {best_metrics['R2']:.4f}")
            except Exception as e:
                print(f"Error running Linear models: {e}")

            # 2. Random Forest
            try:
                rf_metrics = rf_trainer.train_and_evaluate(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    is_augmented=is_augmented,
                    use_embedding=True,
                    embedding_type=embedding_type,
                )
                dataset_emb_results.append({
                    "Model_Family": "RandomForest", "Model": "RF",
                    "Dataset": dataset_name, "Embedding": emb_name,
                    "R2": rf_metrics["R2"], "RMSE": rf_metrics["RMSE"],
                    "MAE": rf_metrics["MAE"],
                })
                print(f"  Random Forest - R²: {rf_metrics['R2']:.4f}")
            except Exception as e:
                print(f"Error running Random Forest: {e}")

            # 3. FCNN/MLP
            try:
                mlp_metrics = mlp_trainer.train_and_evaluate(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    is_augmented=is_augmented,
                    use_embedding=True,
                    embedding_type=embedding_type,
                )
                dataset_emb_results.append({
                    "Model_Family": "MLP", "Model": "FCNN",
                    "Dataset": dataset_name, "Embedding": emb_name,
                    "R2": mlp_metrics["R2"], "RMSE": mlp_metrics["RMSE"],
                    "MAE": mlp_metrics["MAE"],
                })
                print(f"  FCNN/MLP - R²: {mlp_metrics['R2']:.4f}")
            except Exception as e:
                print(f"Error running FCNN/MLP: {e}")

            all_results.extend(dataset_emb_results)

    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values(
            by=["Dataset", "Embedding", "R2"], ascending=[True, True, False]
        ).reset_index(drop=True)

        out_dir = results_dir / "augmented_emb_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(out_dir / "augmented_emb_models_comparison.csv", index=False)

        print("\n" + "=" * 80)
        print("AUGMENTED DATA WITH EMBEDDINGS - MODEL COMPARISON")
        print("=" * 80)
        pd.set_option('display.max_rows', None)
        print(df_results.to_string(index=False))

        best_by_dataset = df_results.loc[
            df_results.groupby('Dataset')['R2'].idxmax()
        ].sort_values('R2', ascending=False)
        best_by_dataset.to_csv(out_dir / "augmented_emb_best_by_dataset.csv", index=False)

        try:
            pivot_df = df_results.pivot_table(
                index=["Dataset", "Embedding"],
                columns="Model_Family",
                values="R2",
                aggfunc='max',
            ).reset_index()
            pivot_df.to_csv(out_dir / "augmented_emb_comparison_pivot.csv", index=False)
            print("\nR² BY EMBEDDING AND MODEL FAMILY:")
            print(pivot_df.to_string(index=False))
        except Exception as e:
            print(f"Could not create pivot table: {e}")
    else:
        print("No results were produced.")


if __name__ == "__main__":
    training_augmented_emb()
