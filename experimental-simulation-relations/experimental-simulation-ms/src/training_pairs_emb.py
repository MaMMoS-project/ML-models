# -*- coding: utf-8 -*-
"""Train models on pairs data WITH compound embeddings.

Usage:
    python -m src.training_pairs_emb

Requires outputs from the embedding pipeline:
    outputs/Pairs_*_w_embeddings_PCA.pkl  (or *_w_embeddings.pkl as fallback)

Trains 3 model families (Linear, Random Forest, FCNN/MLP) on All/RE/RE-free
pairs with raw 200D or PCA-compressed embeddings.

All models operate in log1p space; reported metrics are in original A/m space.
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


@log_output('logs/training_pairs_emb.txt')
def training_pairs_emb():

    script_dir = Path(__file__).parent
    training_dir = script_dir / "training"
    sys.path.insert(0, str(training_dir))

    project_root = script_dir.parent
    results_dir = project_root / "results"
    output_dir = project_root / "outputs"

    from linear_models import LinearModelsTrainer
    from random_forest import RandomForestTrainer
    from fcnn_mlp import FCNNTrainer
    from base_trainer import SIM_COL, EXP_COL, parse_ms_threshold, parse_delta_learning

    ms_threshold = parse_ms_threshold()
    delta_learning = parse_delta_learning()

    print("=" * 80)
    print("PAIRS DATA TRAINING WITH EMBEDDINGS")
    print("=" * 80)

    dataset_configs = [
        {"name": "All-Pairs",     "type": "all",     "file_prefix": "Pairs_all_w_embeddings"},
        {"name": "RE-Pairs",      "type": "re",      "file_prefix": "Pairs_RE_w_embeddings"},
        {"name": "RE-Free-Pairs", "type": "re-free", "file_prefix": "Pairs_RE_Free_w_embeddings"},
    ]

    embedding_types = [
        None,       # raw 200D
        "pca_16",
        "pca_32",
        "pca_8",
        "pca_64",
    ]

    is_augmented = False

    # ── Locate PKL files for each dataset type ──────────────────────────────
    loaded_dataframes = {}
    for cfg in dataset_configs:
        dtype = cfg["type"]
        prefix = cfg["file_prefix"]
        pca_path = output_dir / f"{prefix}_PCA.pkl"
        orig_path = output_dir / f"{prefix}.pkl"

        if pca_path.exists():
            print(f"Found PCA-enriched file for '{dtype}': {pca_path}")
            loaded_dataframes[dtype] = {"file": pca_path, "loaded": False}
        elif orig_path.exists():
            print(f"Found original embedding file for '{dtype}': {orig_path}")
            print(f"  Note: run compress_embedding_PCA.py for better results.")
            loaded_dataframes[dtype] = {"file": orig_path, "loaded": False}
        else:
            print(f"WARNING: No embedding file found for '{dtype}'")
            print(f"  Checked: {pca_path} and {orig_path}")

    if not loaded_dataframes:
        print("ERROR: No embedding files found. Run create_embeddings.py first.")
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
                print(f"  Loaded {len(df)} samples for '{dataset_type}'. "
                      f"Embedding columns: {emb_cols}")
            except Exception as e:
                print(f"ERROR loading {info['file']}: {e}")
                return None
        return loaded_dataframes[dataset_type]["data"].copy()

    all_results = []

    for config in dataset_configs:
        dataset_name = config["name"]
        dataset_type = config["type"]

        if dataset_type not in loaded_dataframes:
            print(f"\nSkipping {dataset_name} — no embedding file available.")
            continue

        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name} (type: {dataset_type})")
        print(f"{'='*60}")

        df_data = load_dataset(dataset_type)
        if df_data is None:
            print(f"Could not load data for '{dataset_type}'. Skipping.")
            continue

        if ms_threshold is not None:
            before = len(df_data)
            df_data = df_data[
                (df_data[SIM_COL] > ms_threshold) & (df_data[EXP_COL] > ms_threshold)
            ].copy()
            print(f"Ms threshold {ms_threshold:.0f} A/m: {before} → {len(df_data)} rows kept")

        for embedding_type in embedding_types:
            emb_name = "raw_200D" if embedding_type is None else embedding_type

            # Check the embedding column actually exists before training
            if embedding_type is None:
                if 'compound_embedding' not in df_data.columns:
                    print(f"  Skipping raw_200D — no compound_embedding column.")
                    continue
            else:
                col_name = f'comp_emb_pca_{embedding_type}_components'
                if col_name not in df_data.columns:
                    print(f"  Skipping {emb_name} — column {col_name} not found.")
                    continue

            print(f"\n{'-'*50}")
            print(f"Embedding: {emb_name}")
            print(f"{'-'*50}")

            dataset_emb_results = []

            # Monkey-patch load_pairs_data to return the pre-loaded PKL DataFrame.
            # prepare_dataset handles the log1p transform and embedding extraction.
            def make_loader_patch(df_ref):
                return lambda dataset_type=None: df_ref.copy()

            # 1. Linear models
            try:
                lin_trainer = LinearModelsTrainer(
                    output_dir=str(results_dir / "pairs_emb_linear")
                )
                lin_trainer.loader.load_pairs_data = make_loader_patch(df_data)
                lin_trainer.loader.delta_learning = delta_learning
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
                rf_trainer = RandomForestTrainer(
                    output_dir=str(results_dir / "pairs_emb_rf")
                )
                rf_trainer.loader.load_pairs_data = make_loader_patch(df_data)
                rf_trainer.loader.delta_learning = delta_learning
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
                mlp_trainer = FCNNTrainer(
                    output_dir=str(results_dir / "pairs_emb_fcnn")
                )
                mlp_trainer.loader.load_pairs_data = make_loader_patch(df_data)
                mlp_trainer.loader.delta_learning = delta_learning
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

    # ── Summary table ────────────────────────────────────────────────────────
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values(
            by=["Dataset", "Embedding", "R2"], ascending=[True, True, False]
        ).reset_index(drop=True)

        out_dir = results_dir / "pairs_emb_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(out_dir / "pairs_emb_models_comparison.csv", index=False)

        print("\n" + "=" * 80)
        print("PAIRS DATA WITH EMBEDDINGS - MODEL COMPARISON")
        print("=" * 80)
        pd.set_option('display.max_rows', None)
        print(df_results.to_string(index=False))

        best_by_dataset = df_results.loc[
            df_results.groupby('Dataset')['R2'].idxmax()
        ].sort_values('R2', ascending=False)
        best_by_dataset.to_csv(out_dir / "pairs_emb_best_by_dataset.csv", index=False)

        try:
            pivot_df = df_results.pivot_table(
                index=["Dataset", "Embedding"],
                columns="Model_Family",
                values="R2",
                aggfunc='max',
            ).reset_index()
            pivot_df.to_csv(out_dir / "pairs_emb_comparison_pivot.csv", index=False)
            print("\nR² BY EMBEDDING AND MODEL FAMILY:")
            print(pivot_df.to_string(index=False))
        except Exception as e:
            print(f"Could not create pivot table: {e}")
    else:
        print("No results were produced.")


if __name__ == "__main__":
    training_pairs_emb()
