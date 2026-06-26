"""
Linear Models Training: LASSO, RIDGE, and Linear Regression
"""
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from base_trainer import DataLoader, ModelEvaluator, split_data

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))


class LinearModelsTrainer:
    """Train and evaluate linear models (LASSO, RIDGE, LinearRegression)."""

    def __init__(self, output_dir: str = "results/linear_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DataLoader()
        self.evaluator = ModelEvaluator()

    def train_and_evaluate(
        self,
        dataset_name: str,
        dataset_type: str,
        is_augmented: bool = False,
        use_embedding: bool = False,
        embedding_type: Optional[str] = None,
        model_types: List[str] = ['lasso', 'ridge', 'linear'],
    ) -> Dict[str, Dict]:
        """Train linear models with hyperparameter tuning."""
        print(f"\n{'='*60}")
        print(f"Training Linear Models: {dataset_name}")
        if use_embedding:
            print(f"Using embedding: {embedding_type}")
        else:
            print("Using only Ms_sim (no embedding)")
        print(f"{'='*60}")

        if is_augmented:
            df = self.loader.load_augmented_data(dataset_type)
        else:
            df = self.loader.load_pairs_data(dataset_type)

        X, y = self.loader.prepare_dataset(df, dataset_type, use_embedding, embedding_type)
        X_train, X_test, y_train, y_test = split_data(X, y)

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Feature dimensions: {X_train.shape[1]}")

        results = {}

        for model_type in model_types:
            print(f"\n--- Training {model_type.upper()} ---")

            if model_type == 'lasso':
                model, best_params = self._train_lasso(X_train, y_train)
            elif model_type == 'ridge':
                model, best_params = self._train_ridge(X_train, y_train)
            else:
                model = LinearRegression()
                model.fit(X_train, y_train)
                best_params = {}

            # Predict, then map back to log1p(Ms_exp) space (no-op unless
            # delta_learning is on) so metrics stay comparable across runs.
            y_train_true = self.loader.reconstruct_log_exp(y_train, X_train)
            y_test_true = self.loader.reconstruct_log_exp(y_test, X_test)
            y_train_pred_log = self.loader.reconstruct_log_exp(model.predict(X_train), X_train)
            y_test_pred_log = self.loader.reconstruct_log_exp(model.predict(X_test), X_test)

            train_metrics = self.evaluator.compute_metrics(y_train_true, y_train_pred_log)
            test_metrics = self.evaluator.compute_metrics(y_test_true, y_test_pred_log)

            print(f"Best params: {best_params}")
            print(f"Test R² = {test_metrics['R2']:.4f}")
            print(f"Test RMSE = {test_metrics['RMSE']:.4f}")

            emb_suffix = f"_{embedding_type}" if use_embedding else "_no_emb"
            output_path = self.output_dir / f"{dataset_name}_{model_type}{emb_suffix}.png"
            self.evaluator.plot_predictions(
                y_train_true, y_train_pred_log,
                y_test_true, y_test_pred_log,
                title=f"{model_type.upper()} - {dataset_name}"
                      + (f" ({embedding_type})" if use_embedding else ""),
                output_path=str(output_path),
            )

            results[model_type] = {
                'R2': test_metrics['R2'],
                'RMSE': test_metrics['RMSE'],
                'MAE': test_metrics['MAE'],
                'best_params': best_params,
                'model': model,
            }

        return results

    def _train_lasso(self, X_train, y_train):
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
        lasso = Lasso(max_iter=100000, tol=1e-4, selection='random')
        grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='r2', n_jobs=_N_JOBS)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_

    def _train_ridge(self, X_train, y_train):
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
        ridge = Ridge(max_iter=1000)
        grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2', n_jobs=_N_JOBS)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_


def main():
    """Run linear models for all dataset configurations."""
    trainer = LinearModelsTrainer()

    configs = [
        ('All-Pairs', 'all', False, [(False, None)]),
        ('RE-Pairs', 're', False, [(False, None)]),
        ('RE-Free-Pairs', 're-free', False, [(False, None)]),
    ]

    all_results = []

    for dataset_name, dataset_type, is_augmented, embedding_configs in configs:
        for use_embedding, embedding_type in embedding_configs:
            try:
                results = trainer.train_and_evaluate(
                    dataset_name, dataset_type, is_augmented,
                    use_embedding, embedding_type,
                )
                for model_type, metrics in results.items():
                    all_results.append({
                        'Dataset': dataset_name,
                        'Model': model_type.upper(),
                        'Embedding': embedding_type if use_embedding else 'None',
                        'R2': metrics['R2'],
                        'RMSE': metrics['RMSE'],
                        'MAE': metrics['MAE'],
                    })
            except Exception as e:
                print(f"Error training {dataset_name} with {embedding_type}: {e}")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(trainer.output_dir / "linear_models_summary.csv", index=False)
    print("\n" + "="*80)
    print("Linear Models Summary:")
    print("="*80)
    print(summary_df.to_string())
    return all_results


if __name__ == "__main__":
    main()
