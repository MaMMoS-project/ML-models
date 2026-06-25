# -*- coding: utf-8 -*-
"""Train baseline models on augmented data (no embeddings).

Run from project root:

    python src/training_augmented.py

Trains all model families on three augmentation variants:
  1. Tc_exp augmented  (Augm_exp_*)
  2. Tc_sim augmented  (Augm_sim_*)
  3. Combined          (Augm_combined_*)

For each variant and each dataset type (All, RE, RE-Free) four model families
are trained: Symbolic Regression, Linear (best of LASSO/Ridge/Linear),
Random Forest, and FCNN/MLP.

Outputs are written to variant-specific subdirectories so results from
different augmentation strategies do not mix:
  results/figures/{variant}/        <- scatter plots
  results/augmented_{model}/{variant}/  <- model artefacts
  results/augmented_comparison/{variant}/  <- per-variant CSVs
  results/augmented_comparison/    <- combined CSV across all variants
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
        "aug_file":        "Augm_exp_all.csv",
        "re_aug_file":     "Augm_exp_RE.csv",
        "re_free_aug_file":"Augm_exp_RE_Free.csv",
    },
    {
        "label": "sim_augmented",
        "description": "Tc_sim augmented",
        "aug_file":        "Augm_sim_all.csv",
        "re_aug_file":     "Augm_sim_RE.csv",
        "re_free_aug_file":"Augm_sim_RE_Free.csv",
    },
    {
        "label": "combined_augmented",
        "description": "Combined (Tc_exp + Tc_sim) augmented",
        "aug_file":        "Augm_combined_all.csv",
        "re_aug_file":     "Augm_combined_RE.csv",
        "re_free_aug_file":"Augm_combined_RE_Free.csv",
    },
]

# Dataset types trained for every augmentation variant
DATASET_CONFIGS = [
    {"name": "All-Augm",      "type": "all"},
    {"name": "RE-Augm",       "type": "re"},
    {"name": "RE-Free-Augm",  "type": "re-free"},
]


@log_output('logs/training_augmented.txt')
def training_augmented():

    script_dir = Path(__file__).parent
    training_dir = script_dir / "training"
    sys.path.insert(0, str(training_dir))

    project_root = script_dir.parent
    results_dir = project_root / "results"

    from symbolic_regression import SymbolicRegressionTrainer
    from linear_models import LinearModelsTrainer
    from random_forest import RandomForestTrainer
    from fcnn_mlp import FCNNTrainer
    from base_trainer import DataLoader

    print("=" * 80)
    print("AUGMENTED DATA TRAINING (NO EMBEDDINGS) — ALL AUGMENTATION VARIANTS")
    print("=" * 80)

    is_augmented = True
    use_embedding = False
    all_results = []  # accumulates rows from every variant

    # ------------------------------------------------------------------
    # Outer loop: augmentation variants
    # ------------------------------------------------------------------
    for variant in AUGMENTATION_VARIANTS:
        label = variant["label"]
        description = variant["description"]

        print(f"\n{'#' * 80}")
        print(f"# AUGMENTATION VARIANT: {description}  [{label}]")
        print(f"{'#' * 80}")

        # Check that files exist for this variant
        required_files = [
            variant["aug_file"],
            variant["re_aug_file"],
            variant["re_free_aug_file"],
        ]
        missing = [f for f in required_files
                   if not (project_root / "outputs" / f).exists()]
        if missing:
            print(f"  WARNING: the following files are missing and this variant "
                  f"will be skipped:\n  " + "\n  ".join(missing))
            continue

        custom_loader = DataLoader(
            augmented_file=variant["aug_file"],
            re_augmented_file=variant["re_aug_file"],
            re_free_augmented_file=variant["re_free_aug_file"],
        )

        variant_results = []

        # --------------------------------------------------------------
        # Inner loop: dataset types
        # --------------------------------------------------------------
        for config in DATASET_CONFIGS:
            dataset_name = config["name"]
            dataset_type = config["type"]

            print(f"\n{'-' * 60}")
            print(f"[{description}] Dataset: {dataset_name} (type: {dataset_type})")
            print(f"{'-' * 60}")

            dataset_results = []

            def _make_base_row():
                return {
                    "Augmentation": description,
                    "Dataset": dataset_name,
                }

            # ---- 1. Symbolic Regression --------------------------------
            try:
                sr_trainer = SymbolicRegressionTrainer(
                    output_dir=str(results_dir / f"augmented_sr" / label)
                )
                sr_trainer.loader = custom_loader
                sr_trainer.evaluator.figures_subdir = label
                sr_metrics = sr_trainer.train_and_evaluate(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    is_augmented=is_augmented,
                )
                row = _make_base_row()
                row.update({
                    "Model_Family": "SymbolicRegression",
                    "Model": "PySR",
                    "R2": sr_metrics["R2"],
                    "RMSE": sr_metrics["RMSE"],
                    "MAE": sr_metrics["MAE"],
                })
                dataset_results.append(row)
                print(f"  Symbolic Regression - R²: {sr_metrics['R2']:.4f}")
            except Exception as e:
                print(f"  Error running Symbolic Regression: {e}")

            # ---- 2. Linear models --------------------------------------
            try:
                lin_trainer = LinearModelsTrainer(
                    output_dir=str(results_dir / f"augmented_linear" / label)
                )
                lin_trainer.loader = custom_loader
                lin_trainer.evaluator.figures_subdir = label
                lin_results = lin_trainer.train_and_evaluate(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    is_augmented=is_augmented,
                    use_embedding=use_embedding,
                    model_types=["lasso", "ridge", "linear"],
                )
                best_model = max(lin_results, key=lambda m: lin_results[m]["R2"])
                best_metrics = lin_results[best_model]
                row = _make_base_row()
                row.update({
                    "Model_Family": "Linear",
                    "Model": best_model.upper(),
                    "R2": best_metrics["R2"],
                    "RMSE": best_metrics["RMSE"],
                    "MAE": best_metrics["MAE"],
                })
                dataset_results.append(row)
                print(f"  Linear - Best: {best_model.upper()}, R²: {best_metrics['R2']:.4f}")
            except Exception as e:
                print(f"  Error running Linear models: {e}")

            # ---- 3. Random Forest --------------------------------------
            try:
                rf_trainer = RandomForestTrainer(
                    output_dir=str(results_dir / f"augmented_rf" / label)
                )
                rf_trainer.loader = custom_loader
                rf_trainer.evaluator.figures_subdir = label
                rf_metrics = rf_trainer.train_and_evaluate(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    is_augmented=is_augmented,
                    use_embedding=use_embedding,
                )
                row = _make_base_row()
                row.update({
                    "Model_Family": "RandomForest",
                    "Model": "RF",
                    "R2": rf_metrics["R2"],
                    "RMSE": rf_metrics["RMSE"],
                    "MAE": rf_metrics["MAE"],
                })
                dataset_results.append(row)
                print(f"  Random Forest - R²: {rf_metrics['R2']:.4f}")
            except Exception as e:
                print(f"  Error running Random Forest: {e}")

            # ---- 4. FCNN/MLP -------------------------------------------
            try:
                mlp_trainer = FCNNTrainer(
                    output_dir=str(results_dir / f"augmented_fcnn" / label)
                )
                mlp_trainer.loader = custom_loader
                mlp_trainer.evaluator.figures_subdir = label
                mlp_metrics = mlp_trainer.train_and_evaluate(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    is_augmented=is_augmented,
                    use_embedding=use_embedding,
                )
                row = _make_base_row()
                row.update({
                    "Model_Family": "MLP",
                    "Model": "FCNN",
                    "R2": mlp_metrics["R2"],
                    "RMSE": mlp_metrics["RMSE"],
                    "MAE": mlp_metrics["MAE"],
                })
                dataset_results.append(row)
                print(f"  FCNN/MLP - R²: {mlp_metrics['R2']:.4f}")
            except Exception as e:
                print(f"  Error running FCNN/MLP: {e}")

            variant_results.extend(dataset_results)

        # ------------------------------------------------------------------
        # Per-variant summary
        # ------------------------------------------------------------------
        if variant_results:
            df_var = pd.DataFrame(variant_results)
            df_var = df_var.sort_values(
                by=["Dataset", "R2"], ascending=[True, False]
            ).reset_index(drop=True)

            var_out_dir = results_dir / "augmented_comparison" / label
            var_out_dir.mkdir(parents=True, exist_ok=True)

            var_csv = var_out_dir / "augmented_models_comparison.csv"
            df_var.to_csv(var_csv, index=False)

            print(f"\n{'=' * 80}")
            print(f"RESULTS — {description}")
            print(f"{'=' * 80}")
            pd.set_option('display.max_rows', None)
            print(df_var.to_string(index=False))
            print(f"{'=' * 80}")
            print(f"Saved: {var_csv}")

            # Best model per dataset for this variant
            best_var = (
                df_var.loc[df_var.groupby('Dataset')['R2'].idxmax()]
                .sort_values('R2', ascending=False)
            )
            best_var_csv = var_out_dir / "augmented_best_by_dataset.csv"
            best_var.to_csv(best_var_csv, index=False)
            print(f"\nBest by dataset saved: {best_var_csv}")

            # Pivot table
            try:
                pivot = df_var.pivot_table(
                    index="Dataset",
                    columns="Model_Family",
                    values="R2",
                    aggfunc='max',
                ).reset_index()
                pivot_csv = var_out_dir / "augmented_comparison_pivot.csv"
                pivot.to_csv(pivot_csv, index=False)
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
            by=["Augmentation", "Dataset", "R2"], ascending=[True, True, False]
        ).reset_index(drop=True)

        combined_out_dir = results_dir / "augmented_comparison"
        combined_out_dir.mkdir(parents=True, exist_ok=True)

        combined_csv = combined_out_dir / "augmented_all_variants_comparison.csv"
        df_all.to_csv(combined_csv, index=False)

        print(f"\n{'=' * 80}")
        print("COMBINED RESULTS — ALL AUGMENTATION VARIANTS")
        print(f"{'=' * 80}")
        print(df_all.to_string(index=False))
        print(f"{'=' * 80}")
        print(f"Combined results saved: {combined_csv}")

        # Best per (Augmentation, Dataset)
        best_all = (
            df_all.loc[df_all.groupby(['Augmentation', 'Dataset'])['R2'].idxmax()]
            .sort_values(['Augmentation', 'R2'], ascending=[True, False])
        )
        best_all_csv = combined_out_dir / "augmented_all_variants_best.csv"
        best_all.to_csv(best_all_csv, index=False)
        print(f"\nBest per (Augmentation, Dataset) saved: {best_all_csv}")

        # Cross-variant pivot: rows = (Dataset, Model_Family), cols = Augmentation
        try:
            cross_pivot = df_all.pivot_table(
                index=["Dataset", "Model_Family"],
                columns="Augmentation",
                values="R2",
                aggfunc='max',
            ).reset_index()
            cross_csv = combined_out_dir / "augmented_cross_variant_pivot.csv"
            cross_pivot.to_csv(cross_csv, index=False)
            print(f"\nCross-variant R² pivot:")
            print(cross_pivot.to_string(index=False))
            print(f"Saved: {cross_csv}")
        except Exception as e:
            print(f"  Could not create cross-variant pivot: {e}")
    else:
        print("No results were produced; please check error messages above.")


if __name__ == "__main__":
    training_augmented()
