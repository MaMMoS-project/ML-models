# -*- coding: utf-8 -*-
"""Train models on augmented data WITH EMBEDDINGS — all augmentation variants.

Run from project root:

    python src/training_augmented_emb.py

Trains Linear, Random Forest, and FCNN/MLP models on three augmentation variants:
  1. Tc_exp augmented  (Augm_exp_*_emb_w_embeddings_PCA.pkl)
  2. Tc_sim augmented  (Augm_sim_*_emb_w_embeddings_PCA.pkl)
  3. Combined          (Augm_combined_*_emb_w_embeddings_PCA.pkl)

For each variant and dataset type (All, RE, RE-Free), models are trained with:
  - Raw 200D embeddings (compound_embedding)
  - PCA-compressed versions: pca_8, pca_16, pca_32, pca_64

Outputs go to variant-specific subdirectories:
  results/figures/{variant}/
  results/augmented_emb_{model}/{variant}/
  results/augmented_emb_comparison/{variant}/
  results/augmented_emb_comparison/   <- combined tables

Prerequisites: augment_data.py → create_embeddings.py → compress_embedding_PCA.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os

from src.log_to_file import log_output

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Augmentation variant definitions
# ---------------------------------------------------------------------------
AUGMENTATION_VARIANTS = [
    {
        "label": "exp_augmented",
        "description": "Tc_exp augmented",
        "file_map": {
            "all":     "Augm_exp_all_emb_w_embeddings",
            "re":      "Augm_exp_RE_emb_w_embeddings",
            "re-free": "Augm_exp_RE_Free_emb_w_embeddings",
        },
    },
    {
        "label": "sim_augmented",
        "description": "Tc_sim augmented",
        "file_map": {
            "all":     "Augm_sim_all_emb_w_embeddings",
            "re":      "Augm_sim_RE_emb_w_embeddings",
            "re-free": "Augm_sim_RE_Free_emb_w_embeddings",
        },
    },
    {
        "label": "combined_augmented",
        "description": "Combined (Tc_exp + Tc_sim) augmented",
        "file_map": {
            "all":     "Augm_combined_all_emb_w_embeddings",
            "re":      "Augm_combined_RE_emb_w_embeddings",
            "re-free": "Augm_combined_RE_Free_emb_w_embeddings",
        },
    },
]

DATASET_CONFIGS = [
    {"name": "All-Augm",     "type": "all"},
    {"name": "RE-Augm",      "type": "re"},
    {"name": "RE-Free-Augm", "type": "re-free"},
]

# Embedding types to try (None = raw 200D)
EMBEDDING_TYPES = [None, "pca_16", "pca_32", "pca_8", "pca_64"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_pkl(output_dir: Path, file_prefix: str) -> Path | None:
    """Return _PCA.pkl if it exists, fall back to .pkl, or None."""
    pca_path = output_dir / f"{file_prefix}_PCA.pkl"
    orig_path = output_dir / f"{file_prefix}.pkl"
    if pca_path.exists():
        return pca_path
    if orig_path.exists():
        print(f"  Note: PCA file not found for {file_prefix}; "
              "using original. Run compress_embedding_PCA.py for better results.")
        return orig_path
    return None


def _make_patched_prepare(original_prepare):
    """Delegate to the loader's prepare_dataset.

    prepare_dataset now resolves PCA embedding column names itself AND applies
    delta-learning / RE-features, so the historical column-resolution monkeypatch
    is no longer needed — delegating keeps every code path (delta, reconstruction)
    consistent.
    """
    def patched(df, dataset_type, use_embedding=False, embedding_type=None):
        return original_prepare(df, dataset_type, use_embedding, embedding_type)

    return patched


@log_output('logs/train_augmented_embedding.txt')
def train_augmented_embedding():

    script_dir = Path(__file__).parent
    training_dir = script_dir / "training"
    sys.path.insert(0, str(training_dir))

    project_root = script_dir.parent
    results_dir = project_root / "results"
    output_dir = project_root / "outputs"

    from linear_models import LinearModelsTrainer
    from random_forest import RandomForestTrainer
    from lightgbm_trainer import LightGBMTrainer, LIGHTGBM_AVAILABLE
    from fcnn_mlp import FCNNTrainer
    from base_trainer import (parse_delta_learning, parse_re_features, parse_cv_folds,
                              model_enabled, SkipModel)

    delta_learning = parse_delta_learning()
    use_re_features = parse_re_features()
    cv_folds = parse_cv_folds()

    print("=" * 80)
    print("AUGMENTED DATA TRAINING WITH EMBEDDINGS — ALL AUGMENTATION VARIANTS")
    print("=" * 80)

    is_augmented = True
    all_results = []

    # ------------------------------------------------------------------
    # Outer loop: augmentation variants
    # ------------------------------------------------------------------
    for variant in AUGMENTATION_VARIANTS:
        label = variant["label"]
        description = variant["description"]
        file_map = variant["file_map"]

        print(f"\n{'#' * 80}")
        print(f"# AUGMENTATION VARIANT: {description}  [{label}]")
        print(f"{'#' * 80}")

        # Resolve pkl files for each dataset type
        resolved_files: dict[str, Path] = {}
        for dtype, prefix in file_map.items():
            path = _resolve_pkl(output_dir, prefix)
            if path:
                resolved_files[dtype] = path
            else:
                print(f"  WARNING: no embedding file found for type '{dtype}' "
                      f"(prefix: {prefix}). Skipping this type.")

        if not resolved_files:
            print(f"  No embedding files found for variant '{label}'. Skipping.")
            continue

        # Cache loaded DataFrames lazily
        loaded: dict[str, pd.DataFrame | None] = {k: None for k in resolved_files}

        def load_df(dtype: str) -> pd.DataFrame | None:
            if dtype not in resolved_files:
                return None
            if loaded[dtype] is None:
                path = resolved_files[dtype]
                print(f"  Loading {path.name} ...")
                try:
                    df = pd.read_pickle(path)
                    emb_cols = [c for c in df.columns if 'emb' in c.lower()]
                    print(f"  Loaded {len(df)} samples. Embedding columns: {emb_cols}")
                    loaded[dtype] = df
                except Exception as e:
                    print(f"  ERROR loading {path}: {e}")
            return loaded[dtype]

        variant_results = []

        # --------------------------------------------------------------
        # Inner loop: dataset types
        # --------------------------------------------------------------
        for config in DATASET_CONFIGS:
            dataset_name = config["name"]
            dataset_type = config["type"]

            if dataset_type not in resolved_files:
                print(f"\n  Skipping {dataset_name} — no file available for type '{dataset_type}'")
                continue

            df_data = load_df(dataset_type)
            if df_data is None:
                continue

            print(f"\n{'=' * 60}")
            print(f"[{description}] Dataset: {dataset_name}")
            print(f"{'=' * 60}")

            # ----------------------------------------------------------
            # Embedding type loop
            # ----------------------------------------------------------
            for embedding_type in EMBEDDING_TYPES:
                emb_name = "raw_200D" if embedding_type is None else embedding_type
                print(f"\n{'-' * 50}")
                print(f"  Embedding: {emb_name}")
                print(f"{'-' * 50}")

                emb_results = []

                def _base_row():
                    return {
                        "Augmentation": description,
                        "Dataset": dataset_name,
                        "Embedding": emb_name,
                    }

                # ---- 1. Linear models --------------------------------
                try:
                    if not model_enabled("linear"):
                        raise SkipModel
                    lin_trainer = LinearModelsTrainer(
                        output_dir=str(results_dir / "augmented_emb_linear" / label)
                    )
                    lin_trainer.evaluator.figures_subdir = label
                    orig_load = lin_trainer.loader.load_augmented_data
                    lin_trainer.loader.delta_learning = delta_learning
                    lin_trainer.loader.use_re_features = use_re_features
                    lin_trainer.loader.cv_folds = cv_folds
                    orig_prep = lin_trainer.loader.prepare_dataset
                    lin_trainer.loader.load_augmented_data = lambda dt=None: df_data.copy()
                    lin_trainer.loader.prepare_dataset = _make_patched_prepare(orig_prep)

                    lin_results = lin_trainer.train_and_evaluate(
                        dataset_name=dataset_name,
                        dataset_type=dataset_type,
                        is_augmented=is_augmented,
                        use_embedding=True,
                        embedding_type=embedding_type,
                        model_types=["lasso", "ridge"],
                    )

                    lin_trainer.loader.load_augmented_data = orig_load
                    lin_trainer.loader.prepare_dataset = orig_prep

                    best_model = max(lin_results, key=lambda m: lin_results[m]["R2"])
                    best = lin_results[best_model]
                    row = _base_row()
                    row.update({"Model_Family": "Linear", "Model": best_model.upper(),
                                "R2": best["R2"], "RMSE": best["RMSE"], "MAE": best["MAE"]})
                    emb_results.append(row)
                    print(f"  Linear ({best_model.upper()}) R²: {best['R2']:.4f}")
                except SkipModel:
                    print("  Linear - disabled in training_config.yaml (skipping)")
                except Exception as e:
                    print(f"  Error running Linear models: {e}")

                # ---- 2. Random Forest --------------------------------
                try:
                    if not model_enabled("rf"):
                        raise SkipModel
                    rf_trainer = RandomForestTrainer(
                        output_dir=str(results_dir / "augmented_emb_rf" / label)
                    )
                    rf_trainer.evaluator.figures_subdir = label
                    orig_load = rf_trainer.loader.load_augmented_data
                    rf_trainer.loader.delta_learning = delta_learning
                    rf_trainer.loader.use_re_features = use_re_features
                    rf_trainer.loader.cv_folds = cv_folds
                    orig_prep = rf_trainer.loader.prepare_dataset
                    rf_trainer.loader.load_augmented_data = lambda dt=None: df_data.copy()
                    rf_trainer.loader.prepare_dataset = _make_patched_prepare(orig_prep)

                    rf_metrics = rf_trainer.train_and_evaluate(
                        dataset_name=dataset_name,
                        dataset_type=dataset_type,
                        is_augmented=is_augmented,
                        use_embedding=True,
                        embedding_type=embedding_type,
                    )

                    rf_trainer.loader.load_augmented_data = orig_load
                    rf_trainer.loader.prepare_dataset = orig_prep

                    row = _base_row()
                    row.update({"Model_Family": "RandomForest", "Model": "RF",
                                "R2": rf_metrics["R2"], "RMSE": rf_metrics["RMSE"],
                                "MAE": rf_metrics["MAE"]})
                    emb_results.append(row)
                    print(f"  Random Forest R²: {rf_metrics['R2']:.4f}")
                except SkipModel:
                    print("  Random Forest - disabled in training_config.yaml (skipping)")
                except Exception as e:
                    print(f"  Error running Random Forest: {e}")

                # ---- 2b. LightGBM (gradient-boosted trees) ----------
                if LIGHTGBM_AVAILABLE:
                    try:
                        if not model_enabled("lgbm"):
                            raise SkipModel
                        gbm_trainer = LightGBMTrainer(
                            output_dir=str(results_dir / "augmented_emb_lightgbm" / label)
                        )
                        gbm_trainer.evaluator.figures_subdir = label
                        orig_load = gbm_trainer.loader.load_augmented_data
                        gbm_trainer.loader.delta_learning = delta_learning
                        gbm_trainer.loader.use_re_features = use_re_features
                        gbm_trainer.loader.cv_folds = cv_folds
                        orig_prep = gbm_trainer.loader.prepare_dataset
                        gbm_trainer.loader.load_augmented_data = lambda dt=None: df_data.copy()
                        gbm_trainer.loader.prepare_dataset = _make_patched_prepare(orig_prep)

                        gbm_metrics = gbm_trainer.train_and_evaluate(
                            dataset_name=dataset_name,
                            dataset_type=dataset_type,
                            is_augmented=is_augmented,
                            use_embedding=True,
                            embedding_type=embedding_type,
                        )

                        gbm_trainer.loader.load_augmented_data = orig_load
                        gbm_trainer.loader.prepare_dataset = orig_prep

                        row = _base_row()
                        row.update({"Model_Family": "LightGBM", "Model": "LGBM",
                                    "R2": gbm_metrics["R2"], "RMSE": gbm_metrics["RMSE"],
                                    "MAE": gbm_metrics["MAE"]})
                        emb_results.append(row)
                        print(f"  LightGBM R²: {gbm_metrics['R2']:.4f}")
                    except SkipModel:
                        print("  LightGBM - disabled in training_config.yaml (skipping)")
                    except Exception as e:
                        print(f"  Error running LightGBM: {e}")
                else:
                    print("  LightGBM not installed — skipping (pip install lightgbm)")

                # ---- 3. FCNN/MLP ------------------------------------
                try:
                    if not model_enabled("mlp"):
                        raise SkipModel
                    mlp_trainer = FCNNTrainer(
                        output_dir=str(results_dir / "augmented_emb_fcnn" / label)
                    )
                    mlp_trainer.evaluator.figures_subdir = label
                    orig_load = mlp_trainer.loader.load_augmented_data
                    mlp_trainer.loader.delta_learning = delta_learning
                    mlp_trainer.loader.use_re_features = use_re_features
                    mlp_trainer.loader.cv_folds = cv_folds
                    orig_prep = mlp_trainer.loader.prepare_dataset
                    mlp_trainer.loader.load_augmented_data = lambda dt=None: df_data.copy()
                    mlp_trainer.loader.prepare_dataset = _make_patched_prepare(orig_prep)

                    mlp_metrics = mlp_trainer.train_and_evaluate(
                        dataset_name=dataset_name,
                        dataset_type=dataset_type,
                        is_augmented=is_augmented,
                        use_embedding=True,
                        embedding_type=embedding_type,
                    )

                    mlp_trainer.loader.load_augmented_data = orig_load
                    mlp_trainer.loader.prepare_dataset = orig_prep

                    row = _base_row()
                    row.update({"Model_Family": "MLP", "Model": "FCNN",
                                "R2": mlp_metrics["R2"], "RMSE": mlp_metrics["RMSE"],
                                "MAE": mlp_metrics["MAE"]})
                    emb_results.append(row)
                    print(f"  FCNN/MLP R²: {mlp_metrics['R2']:.4f}")
                except SkipModel:
                    print("  FCNN/MLP - disabled in training_config.yaml (skipping)")
                except Exception as e:
                    print(f"  Error running FCNN/MLP: {e}")

                variant_results.extend(emb_results)

        # ------------------------------------------------------------------
        # Per-variant summary
        # ------------------------------------------------------------------
        if variant_results:
            df_var = pd.DataFrame(variant_results)
            df_var = df_var.sort_values(
                ["Dataset", "Embedding", "R2"], ascending=[True, True, False]
            ).reset_index(drop=True)

            var_out = results_dir / "augmented_emb_comparison" / label
            var_out.mkdir(parents=True, exist_ok=True)

            var_csv = var_out / "augmented_emb_models_comparison.csv"
            df_var.to_csv(var_csv, index=False)

            print(f"\n{'=' * 80}")
            print(f"RESULTS — {description}")
            print(f"{'=' * 80}")
            pd.set_option('display.max_rows', None)
            print(df_var.to_string(index=False))
            print(f"Saved: {var_csv}")

            best_var = (
                df_var.loc[df_var.groupby(['Dataset', 'Embedding'])['R2'].idxmax()]
                .sort_values(['Dataset', 'R2'], ascending=[True, False])
            )
            best_var.to_csv(var_out / "augmented_emb_best_by_dataset_embedding.csv", index=False)

            best_ds = (
                df_var.loc[df_var.groupby('Dataset')['R2'].idxmax()]
                .sort_values('R2', ascending=False)
            )
            best_ds.to_csv(var_out / "augmented_emb_best_by_dataset.csv", index=False)

            try:
                pivot = df_var.pivot_table(
                    index=["Dataset", "Embedding"],
                    columns="Model_Family",
                    values="R2",
                    aggfunc='max',
                ).reset_index()
                pivot.to_csv(var_out / "augmented_emb_comparison_pivot.csv", index=False)
                print(f"\nR² pivot ({description}):")
                print(pivot.to_string(index=False))
            except Exception as e:
                print(f"  Could not create pivot table: {e}")

        all_results.extend(variant_results)

    # ------------------------------------------------------------------
    # Combined summary across all variants
    # ------------------------------------------------------------------
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all = df_all.sort_values(
            ["Augmentation", "Dataset", "Embedding", "R2"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)

        combined_out = results_dir / "augmented_emb_comparison"
        combined_out.mkdir(parents=True, exist_ok=True)

        combined_csv = combined_out / "augmented_emb_all_variants_comparison.csv"
        df_all.to_csv(combined_csv, index=False)

        print(f"\n{'=' * 80}")
        print("COMBINED RESULTS — ALL AUGMENTATION VARIANTS (WITH EMBEDDINGS)")
        print(f"{'=' * 80}")
        print(df_all.to_string(index=False))
        print(f"Combined results saved: {combined_csv}")

        best_all = (
            df_all.loc[df_all.groupby(['Augmentation', 'Dataset'])['R2'].idxmax()]
            .sort_values(['Augmentation', 'R2'], ascending=[True, False])
        )
        best_all.to_csv(combined_out / "augmented_emb_all_variants_best.csv", index=False)

        try:
            cross_pivot = df_all.pivot_table(
                index=["Dataset", "Embedding", "Model_Family"],
                columns="Augmentation",
                values="R2",
                aggfunc='max',
            ).reset_index()
            cross_csv = combined_out / "augmented_emb_cross_variant_pivot.csv"
            cross_pivot.to_csv(cross_csv, index=False)
            print(f"\nCross-variant R² pivot:")
            print(cross_pivot.to_string(index=False))
            print(f"Saved: {cross_csv}")
        except Exception as e:
            print(f"  Could not create cross-variant pivot: {e}")
    else:
        print("No results were produced; please check error messages above.")


if __name__ == "__main__":
    train_augmented_embedding()
