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
        use_randomized_search: bool = True
    ) -> Dict:
        """
        Train Random Forest model with hyperparameter tuning.
        
        Args:
            dataset_name: Name for output files
            dataset_type: 'all', 're', or 're-free'
            is_augmented: Whether to use augmented data
            use_embedding: Whether to use embeddings
            embedding_type: Type of embedding
            use_randomized_search: Use RandomizedSearchCV (faster) vs GridSearchCV
            
        Returns:
            Dictionary with test metrics and best parameters
        """
        print(f"\n{'='*60}")
        print(f"Training Random Forest: {dataset_name}")
        if use_embedding:
            print(f"Using embedding: {embedding_type}")
        else:
            print("Using only Tc_sim (no embedding)")
        print(f"{'='*60}")
        
        # Load data
        if is_augmented:
            df = self.loader.load_augmented_data(dataset_type)
        else:
            df = self.loader.load_pairs_data(dataset_type)
        
        # Prepare dataset
        X, y = self.loader.prepare_dataset(df, dataset_type, use_embedding, embedding_type)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        # Hyperparameter tuning
        print("\nPerforming hyperparameter tuning...")
        if use_randomized_search:
            model, best_params = self._randomized_search(X_train, y_train)
        else:
            model, best_params = self._grid_search(X_train, y_train)
        
        print(f"Best parameters: {best_params}")
        
        # Predictions, then map back to Tc space (no-op unless delta_learning is on).
        y_train_true = self.loader.reconstruct_target(y_train, X_train)
        y_test_true = self.loader.reconstruct_target(y_test, X_test)
        y_train_pred = self.loader.reconstruct_target(model.predict(X_train), X_train)
        y_test_pred = self.loader.reconstruct_target(model.predict(X_test), X_test)

        # Compute metrics
        train_metrics = self.evaluator.compute_metrics(y_train_true, y_train_pred)
        test_metrics = self.evaluator.compute_metrics(y_test_true, y_test_pred)

        print(f"\nSingle-split Test Metrics:")
        print(f"  R² = {test_metrics['R2']:.4f}")
        print(f"  RMSE = {test_metrics['RMSE']:.2f} K")
        print(f"  MAE = {test_metrics['MAE']:.2f} K")

        # Optional K-fold CV reporting (full dataset, same tuned config). RF does
        # not scale its inputs, so the tuned estimator is CV'd directly.
        cv_folds = getattr(self.loader, 'cv_folds', 0)
        cv = None
        if cv_folds and cv_folds >= 2:
            cv = cross_val_report(model, X, y, self.loader, n_splits=cv_folds)
            print(f"\n{cv_folds}-fold CV Metrics (headline):")
            print(f"  R²   = {cv['R2']:.4f} ± {cv['R2_std']:.4f}")
            print(f"  RMSE = {cv['RMSE']:.2f} ± {cv['RMSE_std']:.2f} K")
            print(f"  MAE  = {cv['MAE']:.2f} ± {cv['MAE_std']:.2f} K")

        # Plot results
        emb_suffix = f"_{embedding_type}" if use_embedding else "_no_emb"
        output_path = self.output_dir / f"{dataset_name}_RF{emb_suffix}.png"
        self.evaluator.plot_predictions(
            y_train_true, y_train_pred,
            y_test_true, y_test_pred,
            title=f"Random Forest - {dataset_name}" + 
                  (f" ({embedding_type})" if use_embedding else ""),
            output_path=str(output_path)
        )
        
        # CV means become the reported (headline) metrics when CV is on.
        reported = {'R2': cv['R2'], 'RMSE': cv['RMSE'], 'MAE': cv['MAE']} if cv else test_metrics

        # Save model info
        info_path = self.output_dir / f"{dataset_name}_RF{emb_suffix}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Embedding: {embedding_type if use_embedding else 'None'}\n")
            f.write(f"Best params: {best_params}\n")
            f.write(f"Single-split R2: {test_metrics['R2']:.4f}\n")
            if cv:
                f.write(f"CV folds: {cv_folds}\n")
                f.write(f"CV R2: {cv['R2']:.4f} +/- {cv['R2_std']:.4f}\n")
                f.write(f"CV RMSE: {cv['RMSE']:.2f} +/- {cv['RMSE_std']:.2f}\n")
                f.write(f"CV MAE: {cv['MAE']:.2f} +/- {cv['MAE_std']:.2f}\n")

        result = {
            'R2': reported['R2'],
            'RMSE': reported['RMSE'],
            'MAE': reported['MAE'],
            'best_params': best_params,
            'model': model
        }
        if cv:
            result.update({'R2_std': cv['R2_std'], 'RMSE_std': cv['RMSE_std'],
                           'MAE_std': cv['MAE_std'], 'cv_folds': cv_folds})
        return result
    
    def _randomized_search(self, X_train, y_train):
        """Perform randomized search for hyperparameters."""
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }

        # CRITICAL: estimator runs SINGLE-THREADED (n_jobs=1) and only the outer
        # search is parallelised (n_jobs=_N_JOBS). Nesting both at n_jobs=-1 spawns
        # (n_cores x n_cores) workers (e.g. 72x72 on a full node), oversubscribing
        # the CPUs — this is what made train_augmented_embedding crawl and produced
        # the loky/ResourceTracker "No child processes" spam at teardown.
        rf = RandomForestRegressor(random_state=42, n_jobs=1)

        # HalvingRandomSearchCV: evaluate many candidates on a small data slice,
        # keep the best 1/factor each round, triple the budget. Only the finalists
        # pay the full CV cost → similar quality, far less compute than
        # RandomizedSearchCV(n_iter=100) on the full dataset. n_candidates=50 fixes
        # the default 'exhaust' behaviour, which on small datasets evaluates only a
        # handful of combinations.
        search = HalvingRandomSearchCV(
            rf,
            param_distributions,
            n_candidates=50,
            factor=3,
            resource='n_samples',
            min_resources=100,
            cv=5,
            scoring='r2',
            n_jobs=_N_JOBS,
            random_state=42,
            verbose=1
        )

        search.fit(X_train, y_train)

        # Best model: give it all cores for the final refit / predictions (no
        # nesting at this point).
        best = search.best_estimator_
        best.set_params(n_jobs=_N_JOBS)
        return best, search.best_params_
    
    def _grid_search(self, X_train, y_train):
        """Perform grid search for hyperparameters (more thorough but slower)."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=1)  # single-threaded inner

        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=_N_JOBS,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best = grid_search.best_estimator_
        best.set_params(n_jobs=_N_JOBS)
        return best, grid_search.best_params_


def main():
    """Run Random Forest for all dataset configurations."""
    trainer = RandomForestTrainer()
    
    # Define all configurations
    configs = [
        # Pairs - No embedding
        ('All-Pairs', 'all', False, False, None),
        ('RE-Pairs', 're', False, False, None),
        ('RE-free-Pairs', 're-free', False, False, None),
        
        # Pairs - With embeddings (based on table)
        ('All-Pairs', 'all', False, True, 'pca_32'),
        ('RE-Pairs', 're', False, True, 'kpca_30'),
        ('RE-free-Pairs', 're-free', False, True, 'pca_16'),
        
        # Augmented - No embedding
        ('All-Augm', 'all', True, False, None),
        ('RE-Augm', 're', True, False, None),
        ('RE-free-Augm', 're-free', True, False, None),
        
        # Augmented - With embeddings
        ('All-Augm', 'all', True, True, 'mat200'),
        ('RE-Augm', 're', True, True, 'mat200'),
        ('RE-free-Augm', 're-free', True, True, 'mat200'),
    ]
    
    all_results = []
    
    for dataset_name, dataset_type, is_augmented, use_embedding, embedding_type in configs:
        try:
            result = trainer.train_and_evaluate(
                dataset_name, dataset_type, is_augmented,
                use_embedding, embedding_type
            )
            
            all_results.append({
                'Dataset': dataset_name,
                'Embedding': embedding_type if use_embedding else 'None',
                'R2': result['R2'],
                'RMSE': result['RMSE'],
                'MAE': result['MAE'],
                'Best_Params': str(result['best_params'])
            })
        except Exception as e:
            print(f"Error training {dataset_name} with {embedding_type}: {e}")
    
    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(trainer.output_dir / "random_forest_summary.csv", index=False)
    
    print("\n" + "="*80)
    print("Random Forest Summary:")
    print("="*80)
    print(summary_df[['Dataset', 'Embedding', 'R2', 'RMSE']].to_string())
    
    return all_results


if __name__ == "__main__":
    main()
