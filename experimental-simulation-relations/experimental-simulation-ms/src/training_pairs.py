# -*- coding: utf-8 -*-
"""Train baseline models on pairs data (no embeddings).

Usage:
    python -m src.training_pairs

Trains 4 model families on All/RE/RE-free pairs from merged_df_python.csv
without using compound embeddings:

1. Symbolic Regression (SR)
2. Linear models (LASSO, Ridge, LinearRegression)
3. Random Forest
4. FCNN/MLP

All models operate in log1p space; reported metrics are in original A/m space.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

from src.log_to_file import log_output

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)


@log_output('logs/training_pairs.txt')
def training_pairs():

    script_dir = Path(__file__).parent
    training_dir = script_dir / "training"
    sys.path.insert(0, str(training_dir))

    project_root = script_dir.parent
    results_dir = project_root / "results"

    from symbolic_regression import SymbolicRegressionTrainer
    from linear_models import LinearModelsTrainer
    from random_forest import RandomForestTrainer
    from fcnn_mlp import FCNNTrainer
    from base_trainer import DataLoader, parse_ms_threshold, parse_delta_learning

    ms_threshold = parse_ms_threshold()
    delta_learning = parse_delta_learning()
    custom_loader = DataLoader(ms_threshold=ms_threshold, delta_learning=delta_learning)

    print("=" * 80)
    print("PAIRS DATA TRAINING (NO EMBEDDINGS)")
    print("=" * 80)

    dataset_configs = [
        {"name": "All-Pairs",      "type": "all"},
        {"name": "RE-Pairs",       "type": "re"},
        {"name": "RE-Free-Pairs",  "type": "re-free"},
    ]

    is_augmented = False
    use_embedding = False
    all_results = []

    for config in dataset_configs:
        dataset_name = config["name"]
        dataset_type = config["type"]

        print(f"\n{'-' * 60}")
        print(f"Training on {dataset_name}")
        print(f"{'-' * 60}")

        dataset_results = []

        # 1. Symbolic Regression
        try:
            sr_trainer = SymbolicRegressionTrainer(
                output_dir=str(results_dir / "pairs_sr")
            )
            sr_trainer.loader = custom_loader
            sr_metrics = sr_trainer.train_and_evaluate(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                is_augmented=is_augmented,
            )
            dataset_results.append({
                "Model_Family": "SymbolicRegression", "Model": "PySR",
                "Dataset": dataset_name,
                "R2": sr_metrics["R2"], "RMSE": sr_metrics["RMSE"], "MAE": sr_metrics["MAE"],
            })
            print(f"  Symbolic Regression - R²: {sr_metrics['R2']:.4f}")
        except Exception as e:
            print(f"Error running Symbolic Regression: {e}")

        # 2. Linear models
        try:
            lin_trainer = LinearModelsTrainer(
                output_dir=str(results_dir / "pairs_linear")
            )
            lin_trainer.loader = custom_loader
            lin_results = lin_trainer.train_and_evaluate(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                is_augmented=is_augmented,
                use_embedding=use_embedding,
                model_types=["lasso", "ridge", "linear"],
            )
            best_model = max(lin_results, key=lambda k: lin_results[k]["R2"])
            best_metrics = lin_results[best_model]
            dataset_results.append({
                "Model_Family": "Linear", "Model": best_model.upper(),
                "Dataset": dataset_name,
                "R2": best_metrics["R2"], "RMSE": best_metrics["RMSE"], "MAE": best_metrics["MAE"],
            })
            print(f"  Linear - Best: {best_model.upper()}, R²: {best_metrics['R2']:.4f}")
        except Exception as e:
            print(f"Error running Linear models: {e}")

        # 3. Random Forest
        try:
            rf_trainer = RandomForestTrainer(
                output_dir=str(results_dir / "pairs_rf")
            )
            rf_trainer.loader = custom_loader
            rf_metrics = rf_trainer.train_and_evaluate(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                is_augmented=is_augmented,
                use_embedding=use_embedding,
            )
            dataset_results.append({
                "Model_Family": "RandomForest", "Model": "RF",
                "Dataset": dataset_name,
                "R2": rf_metrics["R2"], "RMSE": rf_metrics["RMSE"], "MAE": rf_metrics["MAE"],
            })
            print(f"  Random Forest - R²: {rf_metrics['R2']:.4f}")
        except Exception as e:
            print(f"Error running Random Forest: {e}")

        # 4. FCNN/MLP
        try:
            mlp_trainer = FCNNTrainer(
                output_dir=str(results_dir / "pairs_fcnn")
            )
            mlp_trainer.loader = custom_loader
            mlp_metrics = mlp_trainer.train_and_evaluate(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                is_augmented=is_augmented,
                use_embedding=use_embedding,
            )
            dataset_results.append({
                "Model_Family": "MLP", "Model": "FCNN",
                "Dataset": dataset_name,
                "R2": mlp_metrics["R2"], "RMSE": mlp_metrics["RMSE"], "MAE": mlp_metrics["MAE"],
            })
            print(f"  FCNN/MLP - R²: {mlp_metrics['R2']:.4f}")
        except Exception as e:
            print(f"Error running FCNN/MLP: {e}")

        all_results.extend(dataset_results)

    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(by=["Dataset", "R2"], ascending=[True, False]).reset_index(drop=True)

        out_dir = results_dir / "pairs_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "pairs_models_comparison.csv", index=False)

        print("\n" + "=" * 80)
        print("PAIRS DATA (NO EMBEDDINGS) - MODEL COMPARISON")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        best_by_dataset = df.loc[df.groupby('Dataset')['R2'].idxmax()].sort_values('R2', ascending=False)
        best_by_dataset.to_csv(out_dir / "pairs_best_by_dataset.csv", index=False)

        try:
            pivot_df = df.pivot_table(
                index="Dataset", columns="Model_Family", values="R2", aggfunc='max'
            ).reset_index()
            pivot_df.to_csv(out_dir / "pairs_comparison_pivot.csv", index=False)
            print("\nR² BY DATASET AND MODEL FAMILY:")
            print(pivot_df.to_string(index=False))
        except Exception as e:
            print(f"Could not create pivot table: {e}")
    else:
        print("No results were produced.")


if __name__ == "__main__":
    training_pairs()
