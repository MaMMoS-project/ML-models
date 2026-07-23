"""
Linear Models Training: LASSO, RIDGE, and Linear Regression
"""
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from base_trainer import DataLoader, ModelEvaluator, split_data, cross_val_report

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Respect SLURM CPU allocation; fall back to all cores when running locally.
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
        model_types: List[str] = ['lasso', 'ridge', 'linear']
    ) -> Dict[str, Dict]:
        """
        Train linear models with hyperparameter tuning.
        
        Args:
            dataset_name: Name for output files
            dataset_type: 'all', 're', or 're-free'
            is_augmented: Whether to use augmented data
            use_embedding: Whether to use embeddings
            embedding_type: Type of embedding (e.g., 'pca_32', 'kpca_30', 'mat200')
            model_types: List of models to train
            
        Returns:
            Dictionary with results for each model type
        """
        print(f"\n{'='*60}")
        print(f"Training Linear Models: {dataset_name}")
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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        results = {}
        
        # Train each model type
        for model_type in model_types:
            print(f"\n--- Training {model_type.upper()} ---")
            
            if model_type == 'lasso':
                model, best_params = self._train_lasso(X_train_scaled, y_train)
            elif model_type == 'ridge':
                model, best_params = self._train_ridge(X_train_scaled, y_train)
            else:  # linear
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
                best_params = {}
            
            # Predictions, then map back to Tc space (no-op unless delta_learning is
            # on) using the UNSCALED X so the Tc_sim baseline is correct.
            y_train_true = self.loader.reconstruct_target(y_train, X_train)
            y_test_true = self.loader.reconstruct_target(y_test, X_test)
            y_train_pred = self.loader.reconstruct_target(model.predict(X_train_scaled), X_train)
            y_test_pred = self.loader.reconstruct_target(model.predict(X_test_scaled), X_test)

            # Compute metrics
            train_metrics = self.evaluator.compute_metrics(y_train_true, y_train_pred)
            test_metrics = self.evaluator.compute_metrics(y_test_true, y_test_pred)

            print(f"Best params: {best_params}")
            print(f"Single-split Test R² = {test_metrics['R2']:.4f}")
            print(f"Single-split Test RMSE = {test_metrics['RMSE']:.2f} K")

            # Optional K-fold CV reporting (full dataset, same tuned config). The
            # linear models StandardScaler their inputs, so CV must scale per fold:
            # wrap the estimator in a Pipeline(StandardScaler, model) and feed it
            # the UNSCALED X (cross_val_report reads the Tc_sim baseline from
            # X[:, -1], which must be unscaled).
            cv_folds = getattr(self.loader, 'cv_folds', 0)
            cv = None
            if cv_folds and cv_folds >= 2:
                cv_model = make_pipeline(StandardScaler(), model)
                cv = cross_val_report(cv_model, X, y, self.loader, n_splits=cv_folds)
                print(f"{cv_folds}-fold CV R² = {cv['R2']:.4f} ± {cv['R2_std']:.4f} (headline)")

            # Plot results
            emb_suffix = f"_{embedding_type}" if use_embedding else "_no_emb"
            output_path = self.output_dir / f"{dataset_name}_{model_type}{emb_suffix}.png"
            self.evaluator.plot_predictions(
                y_train_true, y_train_pred,
                y_test_true, y_test_pred,
                title=f"{model_type.upper()} - {dataset_name}" + 
                      (f" ({embedding_type})" if use_embedding else ""),
                output_path=str(output_path)
            )
            
            # CV means become the reported (headline) metrics when CV is on.
            reported = {'R2': cv['R2'], 'RMSE': cv['RMSE'], 'MAE': cv['MAE']} if cv else test_metrics
            results[model_type] = {
                'R2': reported['R2'],
                'RMSE': reported['RMSE'],
                'MAE': reported['MAE'],
                'best_params': best_params,
                'model': model,
                'scaler': scaler
            }
            if cv:
                results[model_type].update({'R2_std': cv['R2_std'], 'RMSE_std': cv['RMSE_std'],
                                            'MAE_std': cv['MAE_std'], 'cv_folds': cv_folds})

        # --- ONNX export: best linear submodel, deployable raw_200D variant only ---
        try:
            from onnx_export import maybe_export_onnx
            if results:
                best_type = max(results, key=lambda k: results[k]['R2'])
                maybe_export_onnx(
                    family="linear", model=results[best_type]['model'],
                    scaler=results[best_type]['scaler'], input_dim=X_train.shape[1],
                    dataset_name=dataset_name, use_embedding=use_embedding,
                    embedding_type=embedding_type, loader=self.loader,
                    aug_label=getattr(self.evaluator, "figures_subdir", None),
                    output_dir=self.output_dir,
                )
        except Exception as _onnx_exc:
            print(f"    ONNX export skipped/failed: {_onnx_exc}")

        return results
    
    def _train_lasso(self, X_train, y_train):
        """Train LASSO with grid search."""
        param_grid = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        }

        lasso = Lasso(max_iter=100000, tol=1e-4, selection='random')
        grid_search = GridSearchCV(
            lasso, param_grid,
            cv=5,
            scoring='r2',
            n_jobs=_N_JOBS
        )
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def _train_ridge(self, X_train, y_train):
        """Train RIDGE with grid search."""
        param_grid = {
             'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        }
        
        ridge = Ridge(max_iter=1000)
        grid_search = GridSearchCV(
            ridge, param_grid,
            cv=5,
            scoring='r2',
            n_jobs=_N_JOBS
        )
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_


def main():
    """Run linear models for all dataset configurations."""
    trainer = LinearModelsTrainer()
    
    # Define all configurations
    # Format: (dataset_name, dataset_type, is_augmented, embedding_configs)
    configs = [
        # Pairs - No embedding
        ('All-Pairs', 'all', False, [(False, None)]),
        ('RE-Pairs', 're', False, [(False, None)]),
        ('RE-free-Pairs', 're-free', False, [(False, None)]),
        
        # Pairs - With embeddings
        ('All-Pairs', 'all', False, [(True, 'pca_32'), (True, 'pca_16'), (True, 'kpca_30')]),
        ('RE-Pairs', 're', False, [(True, 'kpca_30'), (True, 'pca_32')]),
        ('RE-free-Pairs', 're-free', False, [(True, 'pca_16'), (True, 'pca_32')]),
        
        # Augmented - No embedding
        ('All-Augm', 'all', True, [(False, None)]),
        ('RE-Augm', 're', True, [(False, None)]),
        ('RE-free-Augm', 're-free', True, [(False, None)]),
        
        # Augmented - With embeddings
        ('All-Augm', 'all', True, [(True, 'mat200'), (True, 'pca_32')]),
        ('RE-Augm', 're', True, [(True, 'mat200'), (True, 'kpca_30')]),
        ('RE-free-Augm', 're-free', True, [(True, 'mat200'), (True, 'pca_16')]),
    ]
    
    all_results = []
    
    for dataset_name, dataset_type, is_augmented, embedding_configs in configs:
        for use_embedding, embedding_type in embedding_configs:
            try:
                results = trainer.train_and_evaluate(
                    dataset_name, dataset_type, is_augmented,
                    use_embedding, embedding_type
                )
                
                # Store best model results
                for model_type, metrics in results.items():
                    all_results.append({
                        'Dataset': dataset_name,
                        'Model': model_type.upper(),
                        'Embedding': embedding_type if use_embedding else 'None',
                        'R2': metrics['R2'],
                        'RMSE': metrics['RMSE'],
                        'MAE': metrics['MAE']
                    })
            except Exception as e:
                print(f"Error training {dataset_name} with {embedding_type}: {e}")
    
    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(trainer.output_dir / "linear_models_summary.csv", index=False)
    
    print("\n" + "="*80)
    print("Linear Models Summary:")
    print("="*80)
    print(summary_df.to_string())
    
    return all_results


if __name__ == "__main__":
    main()
