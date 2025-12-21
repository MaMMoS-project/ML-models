"""
Random Forest Models Training
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from base_trainer import DataLoader, ModelEvaluator, split_data


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
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Compute metrics
        train_metrics = self.evaluator.compute_metrics(y_train, y_train_pred)
        test_metrics = self.evaluator.compute_metrics(y_test, y_test_pred)
        
        print(f"\nTest Metrics:")
        print(f"  R² = {test_metrics['R2']:.4f}")
        print(f"  RMSE = {test_metrics['RMSE']:.2f} K")
        print(f"  MAE = {test_metrics['MAE']:.2f} K")
        
        # Plot results
        emb_suffix = f"_{embedding_type}" if use_embedding else "_no_emb"
        output_path = self.output_dir / f"{dataset_name}_RF{emb_suffix}.png"
        self.evaluator.plot_predictions(
            y_train, y_train_pred,
            y_test, y_test_pred,
            title=f"Random Forest - {dataset_name}" + 
                  (f" ({embedding_type})" if use_embedding else ""),
            output_path=str(output_path)
        )
        
        # Save model info
        info_path = self.output_dir / f"{dataset_name}_RF{emb_suffix}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Embedding: {embedding_type if use_embedding else 'None'}\n")
            f.write(f"Best params: {best_params}\n")
            f.write(f"R2: {test_metrics['R2']:.4f}\n")
            f.write(f"RMSE: {test_metrics['RMSE']:.2f}\n")
            f.write(f"MAE: {test_metrics['MAE']:.2f}\n")
        
        return {
            'R2': test_metrics['R2'],
            'RMSE': test_metrics['RMSE'],
            'MAE': test_metrics['MAE'],
            'best_params': best_params,
            'model': model
        }
    
    def _randomized_search(self, X_train, y_train):
        """Perform randomized search for hyperparameters."""
        param_distributions_old = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        param_distributions = {
            'n_estimators': [50, 100, 200],#, 300, 500],
            'max_depth': [None, 10, 20],#, 30],#, 40, 50],
            'min_samples_split': [2, 5],#, 10, 20],
            'min_samples_leaf': [2, 5],#, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            rf,
            param_distributions,
            n_iter=50,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        '''
        #removing the randomized approach:

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 20, 40],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', None]
        }
    
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        '''


        random_search.fit(X_train, y_train)
        
        return random_search.best_estimator_, random_search.best_params_
    
    def _grid_search(self, X_train, y_train):
        """Perform grid search for hyperparameters (more thorough but slower)."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_


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
