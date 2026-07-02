"""
Random Forest Models Training
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV, RandomizedSearchCV
from base_trainer import DataLoader, ModelEvaluator, split_data, cross_val_report

# Respect SLURM CPU allocation; fall back to all cores when running locally.
_N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))


class RandomForestTrainer:
    """Train and evaluate Random Forest models."""

    def __init__(self, output_dir: str = "results/random_forest"):
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
        use_randomized_search: bool = True,
    ) -> Dict:
        """Train Random Forest model with hyperparameter tuning."""
        print(f"\n{'='*60}")
        print(f"Training Random Forest: {dataset_name}")
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

        print("\nPerforming hyperparameter tuning...")
        if use_randomized_search:
            model, best_params = self._randomized_search(X_train, y_train)
        else:
            model, best_params = self._grid_search(X_train, y_train)
        print(f"Best parameters: {best_params}")

        # Predict, then map back to log1p(Ms_exp) space (no-op unless
        # delta_learning is on) so metrics stay comparable across runs.
        y_train_true = self.loader.reconstruct_log_exp(y_train, X_train)
        y_test_true = self.loader.reconstruct_log_exp(y_test, X_test)
        y_train_pred_log = self.loader.reconstruct_log_exp(model.predict(X_train), X_train)
        y_test_pred_log = self.loader.reconstruct_log_exp(model.predict(X_test), X_test)

        train_metrics = self.evaluator.compute_metrics(y_train_true, y_train_pred_log)
        test_metrics = self.evaluator.compute_metrics(y_test_true, y_test_pred_log)

        print(f"\nSingle-split Test Metrics:")
        print(f"  R² = {test_metrics['R2']:.4f}")
        print(f"  RMSE = {test_metrics['RMSE']:.4f}")
        print(f"  MAE = {test_metrics['MAE']:.4f}")

        # Optional K-fold CV reporting (full dataset, same tuned config).
        cv_folds = getattr(self.loader, 'cv_folds', 0)
        cv = None
        if cv_folds and cv_folds >= 2:
            cv = cross_val_report(model, X, y, self.loader, n_splits=cv_folds)
            print(f"\n{cv_folds}-fold CV Metrics (headline):")
            print(f"  R²   = {cv['R2']:.4f} ± {cv['R2_std']:.4f}")
            print(f"  RMSE = {cv['RMSE']:.4f} ± {cv['RMSE_std']:.4f}")
            print(f"  MAE  = {cv['MAE']:.4f} ± {cv['MAE_std']:.4f}")

        emb_suffix = f"_{embedding_type}" if use_embedding else "_no_emb"
        output_path = self.output_dir / f"{dataset_name}_RF{emb_suffix}.png"
        self.evaluator.plot_predictions(
            y_train_true, y_train_pred_log,
            y_test_true, y_test_pred_log,
            title=f"Random Forest - {dataset_name}"
                  + (f" ({embedding_type})" if use_embedding else ""),
            output_path=str(output_path),
        )

        # CV means become the reported (headline) metrics when CV is on.
        reported = {'R2': cv['R2'], 'RMSE': cv['RMSE'], 'MAE': cv['MAE']} if cv else test_metrics

        info_path = self.output_dir / f"{dataset_name}_RF{emb_suffix}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Embedding: {embedding_type if use_embedding else 'None'}\n")
            f.write(f"Best params: {best_params}\n")
            f.write(f"Single-split R2: {test_metrics['R2']:.4f}\n")
            if cv:
                f.write(f"CV folds: {cv_folds}\n")
                f.write(f"CV R2: {cv['R2']:.4f} +/- {cv['R2_std']:.4f}\n")
                f.write(f"CV RMSE: {cv['RMSE']:.4f} +/- {cv['RMSE_std']:.4f}\n")
                f.write(f"CV MAE: {cv['MAE']:.4f} +/- {cv['MAE_std']:.4f}\n")

        result = {
            'R2': reported['R2'],
            'RMSE': reported['RMSE'],
            'MAE': reported['MAE'],
            'best_params': best_params,
            'model': model,
        }
        if cv:
            result.update({'R2_std': cv['R2_std'], 'RMSE_std': cv['RMSE_std'],
                           'MAE_std': cv['MAE_std'], 'cv_folds': cv_folds})
        return result

    def _randomized_search(self, X_train, y_train):
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
        }
        # CRITICAL: estimator runs SINGLE-THREADED (n_jobs=1) and only the outer
        # search is parallelised (n_jobs=_N_JOBS). Nesting both at _N_JOBS spawns
        # _N_JOBS x _N_JOBS threads (e.g. 72x72 on a full node), oversubscribing
        # the CPUs and crawling — same issue that hung LightGBM.
        rf = RandomForestRegressor(random_state=42, n_jobs=1)
        # HalvingRandomSearchCV: evaluate many candidates on a small data
        # slice, keep the best 1/factor each round, triple the budget.
        # Only the finalists pay the full CV cost → same quality, much less
        # compute than RandomizedSearchCV(n_iter=N) on the full dataset.
        # n_candidates=50 fixes the default 'exhaust' behaviour, which derives
        # the candidate count from how many halving rounds the dataset supports.
        # With small datasets (e.g. RE pairs ~720 samples) 'exhaust' evaluates
        # only 3 random combinations; 50 ensures adequate search at all sizes.
        search = HalvingRandomSearchCV(
            rf, param_distributions,
            n_candidates=50,
            factor=3,
            resource='n_samples',
            min_resources=100,
            cv=5,
            scoring='r2',
            n_jobs=_N_JOBS,
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)
        # Best model: give it all cores for the final refit / predictions / CV
        # report (no nesting at this point).
        best = search.best_estimator_
        best.set_params(n_jobs=_N_JOBS)
        return best, search.best_params_

    def _grid_search(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=1)  # single-threaded inner
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=_N_JOBS, verbose=1)
        grid_search.fit(X_train, y_train)
        best = grid_search.best_estimator_
        best.set_params(n_jobs=_N_JOBS)
        return best, grid_search.best_params_


def main():
    """Run Random Forest for all dataset configurations."""
    trainer = RandomForestTrainer()

    configs = [
        ('All-Pairs', 'all', False, False, None),
        ('RE-Pairs', 're', False, False, None),
        ('RE-Free-Pairs', 're-free', False, False, None),
    ]

    all_results = []

    for dataset_name, dataset_type, is_augmented, use_embedding, embedding_type in configs:
        try:
            result = trainer.train_and_evaluate(
                dataset_name, dataset_type, is_augmented, use_embedding, embedding_type,
            )
            all_results.append({
                'Dataset': dataset_name,
                'Embedding': embedding_type if use_embedding else 'None',
                'R2': result['R2'],
                'RMSE': result['RMSE'],
                'MAE': result['MAE'],
                'Best_Params': str(result['best_params']),
            })
        except Exception as e:
            print(f"Error training {dataset_name} with {embedding_type}: {e}")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(trainer.output_dir / "random_forest_summary.csv", index=False)
    print("\n" + "="*80)
    print("Random Forest Summary:")
    print("="*80)
    print(summary_df[['Dataset', 'Embedding', 'R2', 'RMSE']].to_string())
    return all_results


if __name__ == "__main__":
    main()
