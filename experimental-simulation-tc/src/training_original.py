# -*- coding: utf-8 -*-
"""Train baseline models on original (non-augmented, non-embedding) data.

This script is intended to be run from the project root:

    python src/training_original.py

It trains the existing 4 model families on the All-Pairs dataset (from Pairs_all.csv)
without using any embeddings:

1. Symbolic Regression (SR)
2. Linear models (LASSO, Ridge, LinearRegression) -> best submodel
3. Random Forest
4. FCNN/MLP

For each family, it records R², RMSE, and MAE on the test split and
creates a small comparison table for use in the paper/report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def training_original():
    
    # Ensure we can import training modules
    script_dir = Path(__file__).parent
    training_dir = script_dir / "training"
    sys.path.insert(0, str(training_dir))
    
    # Get project root directory for storing results
    project_root = script_dir.parent
    results_dir = project_root / "results"

    # Import trainers lazily after adjusting sys.path
    from symbolic_regression import SymbolicRegressionTrainer
    from linear_models import LinearModelsTrainer
    from random_forest import RandomForestTrainer
    from fcnn_mlp import FCNNTrainer
    from base_trainer import DataLoader
    
    # Create a custom DataLoader with the specific file names for original dataset
    custom_loader = DataLoader(
        pairs_file="Pairs_all.csv",
        re_pairs_file="Pairs_RE.csv",
        re_free_pairs_file="Pairs_RE_Free.csv"
    )

    print("=" * 80)
    print("ORIGINAL DATA TRAINING (NO EMBEDDINGS)")
    print("=" * 80)

    # Define the dataset configurations to train on
    dataset_configs = [
        {"name": "All-Pairs", "type": "all"},
        {"name": "RE-Pairs", "type": "re"},
        {"name": "RE-Free-Pairs", "type": "re-free"}
    ]
    
    is_augmented = False
    use_embedding = False

    results = []

    all_results = []
    
    # Train on each dataset configuration
    for config in dataset_configs:
        dataset_name = config["name"]
        dataset_type = config["type"]
        
        print(f"\n{'-' * 60}")
        print(f"Training on {dataset_name} dataset type: {dataset_type}")
        print(f"{'-' * 60}")
        
        dataset_results = []
        
        # 1. Symbolic Regression
        try:
            sr_trainer = SymbolicRegressionTrainer(
                output_dir=str(results_dir / "original_sr")
            )
            # Use custom loader
            sr_trainer.loader = custom_loader
            sr_metrics = sr_trainer.train_and_evaluate(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                is_augmented=is_augmented
            )
            dataset_results.append({
                "Model_Family": "SymbolicRegression",
                "Model": "PySR",
                "Dataset": dataset_name,
                "R2": sr_metrics["R2"],
                "RMSE": sr_metrics["RMSE"],
                "MAE": sr_metrics["MAE"],
            })
            print(f"  Symbolic Regression - R²: {sr_metrics['R2']:.4f}")
        except Exception as e:
            print(f"Error running Symbolic Regression: {e}")

        # 2. Linear models (no embeddings) ->        # 2. Linear models (LASSO, Ridge, Linear)
        try:
            lin_trainer = LinearModelsTrainer(
                output_dir=str(results_dir / "original_linear")
            )
            # Use custom loader
            lin_trainer.loader = custom_loader
            lin_results = lin_trainer.train_and_evaluate(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                is_augmented=is_augmented,
                use_embedding=use_embedding,
                model_types=["lasso", "ridge", "linear"],
            )
            
            # Find best linear model
            best_model = None
            best_r2 = -np.inf
            for model_name, metrics in lin_results.items():
                if metrics["R2"] > best_r2:
                    best_r2 = metrics["R2"]
                    best_model = model_name
                    
            if best_model:
                best_metrics = lin_results[best_model]
                dataset_results.append({
                    "Model_Family": "Linear",
                    "Model": best_model.upper(),
                    "Dataset": dataset_name,
                    "R2": best_metrics["R2"],
                    "RMSE": best_metrics["RMSE"],
                    "MAE": best_metrics["MAE"],
                })
                print(f"  Linear - Best model: {best_model.upper()}, R²: {best_metrics['R2']:.4f}")
        except Exception as e:
            print(f"Error running Linear models: {e}")

        # 3. Random Forest
        try:
            rf_trainer = RandomForestTrainer(
                output_dir=str(results_dir / "original_rf")
            )
            # Use custom loader
            rf_trainer.loader = custom_loader
            rf_metrics = rf_trainer.train_and_evaluate(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                is_augmented=is_augmented,
                use_embedding=use_embedding
            )
            dataset_results.append({
                "Model_Family": "RandomForest",
                "Model": "RF",
                "Dataset": dataset_name,
                "R2": rf_metrics["R2"],
                "RMSE": rf_metrics["RMSE"],
                "MAE": rf_metrics["MAE"],
            })
            print(f"  Random Forest - R²: {rf_metrics['R2']:.4f}")
        except Exception as e:
            print(f"Error running Random Forest: {e}")

        # 4. FCNN/MLP
        try:
            mlp_trainer = FCNNTrainer(
                output_dir=str(results_dir / "original_fcnn")
            )
            # Use custom loader
            mlp_trainer.loader = custom_loader
            mlp_metrics = mlp_trainer.train_and_evaluate(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                is_augmented=is_augmented,
                use_embedding=use_embedding
            )
            dataset_results.append({
                "Model_Family": "MLP",
                "Model": "FCNN",
                "Dataset": dataset_name,
                "R2": mlp_metrics["R2"],
                "RMSE": mlp_metrics["RMSE"],
                "MAE": mlp_metrics["MAE"],
            })
            print(f"  FCNN/MLP - R²: {mlp_metrics['R2']:.4f}")
        except Exception as e:
            print(f"Error running FCNN/MLP: {e}")
            
        # Add this dataset's results to overall results
        all_results.extend(dataset_results)

    # Build comparison table
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(by=["Dataset", "R2"], ascending=[True, False]).reset_index(drop=True)

        out_dir = results_dir / "original_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "original_models_comparison.csv"
        df.to_csv(out_csv, index=False)

        print("\n" + "=" * 80)
        print("ORIGINAL DATA (NO EMBEDDINGS) - MODEL COMPARISON")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        print(f"Results saved to: {out_csv}")
        
        # Create a table with the best model for each dataset sorted by R2
        best_by_dataset = df.loc[df.groupby('Dataset')['R2'].idxmax()].sort_values('R2', ascending=False)
        out_best_csv = out_dir / "original_best_by_dataset.csv"
        best_by_dataset.to_csv(out_best_csv, index=False)
        
        print("\n" + "=" * 80)
        print("BEST MODEL BY DATASET (SORTED BY R2)")
        print("=" * 80)
        print(best_by_dataset.to_string(index=False))
        print("=" * 80)
        print(f"Best models by dataset saved to: {out_best_csv}")
        
        # Also create a pivot table for clearer comparison
        try:
            pivot_df = df.pivot_table(
                index="Dataset", 
                columns="Model_Family", 
                values="R2",
                aggfunc='max'
            ).reset_index()
            
            pivot_csv = out_dir / "original_comparison_pivot.csv"
            pivot_df.to_csv(pivot_csv, index=False)
            print("\nR² BY DATASET AND MODEL FAMILY:")
            print(pivot_df.to_string(index=False))
            print(f"Pivot table saved to: {pivot_csv}")
        except Exception as e:
            print(f"Could not create pivot table: {e}")
    else:
        print("No results were produced; please check error messages above.")


if __name__ == "__main__":
    training_original()
