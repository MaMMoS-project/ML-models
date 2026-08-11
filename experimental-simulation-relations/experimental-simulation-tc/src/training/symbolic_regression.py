"""
Symbolic Regression Training - Baseline Model
Uses only Tc_sim as input (stoichiometry disregarded)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import r2_score, mean_squared_error
from base_trainer import DataLoader, ModelEvaluator, split_data, cross_val_report_fn

try:
    from pysr import PySRRegressor
except ImportError:
    print("Warning: PySR not installed. Install with: pip install pysr")
    PySRRegressor = None

# Create log directory 
import os
from src.log_to_file import log_output
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)


class SymbolicRegressionTrainer:
    """Train symbolic regression models as baseline."""
    
    def __init__(self, output_dir: str = "results/symbolic_regression"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DataLoader()
        self.evaluator = ModelEvaluator()
        
    def train_and_evaluate(
        self,
        dataset_name: str,
        dataset_type: str,
        is_augmented: bool = False,
        niterations: int = 100,
        max_train_samples: int = 2000
    ) -> Dict[str, float]:
        """
        Train symbolic regression model.
        
        Args:
            dataset_name: Name for output files (e.g., 'All-Pairs', 'RE-Augm')
            dataset_type: 'all', 're', or 're-free'
            is_augmented: Whether to use augmented data
            niterations: Number of PySR iterations
            max_train_samples: Cap on training set size; random subsample taken when exceeded
            
        Returns:
            Dictionary with test metrics and equation
        """
        print(f"\n{'='*60}")
        print(f"Training Symbolic Regression: {dataset_name}")
        print(f"{'='*60}")
        
        # Load data
        if is_augmented:
            df = self.loader.load_augmented_data(dataset_type)
        else:
            df = self.loader.load_pairs_data(dataset_type)
        
        # Prepare dataset (NO embedding for symbolic regression baseline)
        X, y = self.loader.prepare_dataset(df, dataset_type, use_embedding=False)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        # Subsample training set if too large — PySR does not scale to tens of thousands of rows
        if len(X_train) > max_train_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            print(f"Subsampled training set to {max_train_samples} samples for PySR")

        # Train PySR model
        if PySRRegressor is None:
            raise ImportError("PySR is required for symbolic regression")
        
        '''
        model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "*", "^"],
            unary_operators=["square", "cube"],
            maxsize=10,
            populations=15,
            population_size=33,
            ncycles_per_iteration=550,  # Updated from ncyclesperiteration
            fraction_replaced_hof=0.035
        )

        model_all = PySRRegressor(
            niterations=100,
            binary_operators=["+", "*"],
            maxsize=10
        )
        '''
        # Define PySR model complexity
        model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "*"],  # Keep only + and *
            unary_operators=[],           # Remove square and cube
            maxsize=10,                    # Limit formula complexity (Tc_sim * A + B has 3 nodes)
            populations=15,
            population_size=33,
            ncycles_per_iteration=550,    # Updated parameter name from ncyclesperiteration
            fraction_replaced_hof=0.035,
            constraints={"no_constants": False}  # Allow constants for the A and B values
        )
        
        print("\nFitting PySR model...")
        model.fit(X_train, y_train)
        
        # Get best equation
        best_eq = model.get_best()
        equation_str = best_eq['equation']
        print(f"\nBest equation: {equation_str}")
        
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

        # Optional K-fold CV reporting. Each fold fits a fresh PySR model on the
        # (subsampled) fold-train and predicts the held-out fold. NOTE: this is
        # expensive — it runs the full symbolic search once per fold. SR does not
        # scale its inputs, so the closure works directly on the raw fold arrays.
        cv_folds = getattr(self.loader, 'cv_folds', 0)
        cv = None
        if cv_folds and cv_folds >= 2:
            def _fit_predict(X_tr, y_tr, X_te):
                if len(X_tr) > max_train_samples:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(X_tr), size=max_train_samples, replace=False)
                    X_tr, y_tr = X_tr[idx], y_tr[idx]
                m = PySRRegressor(
                    niterations=niterations,
                    binary_operators=["+", "*"],
                    unary_operators=[],
                    maxsize=10,
                    populations=15,
                    population_size=33,
                    ncycles_per_iteration=550,
                    fraction_replaced_hof=0.035,
                    constraints={"no_constants": False},
                )
                m.fit(X_tr, y_tr)
                return m.predict(X_te)

            print(f"\nRunning {cv_folds}-fold CV for SR (slow — {cv_folds} symbolic searches)...")
            cv = cross_val_report_fn(_fit_predict, X, y, self.loader, n_splits=cv_folds)
            print(f"{cv_folds}-fold CV Metrics (headline):")
            print(f"  R²   = {cv['R2']:.4f} ± {cv['R2_std']:.4f}")
            print(f"  RMSE = {cv['RMSE']:.2f} ± {cv['RMSE_std']:.2f} K")
            print(f"  MAE  = {cv['MAE']:.2f} ± {cv['MAE_std']:.2f} K")

        # Plot results
        output_path = self.output_dir / f"{dataset_name}_SR.png"
        self.evaluator.plot_predictions(
            y_train_true, y_train_pred,
            y_test_true, y_test_pred,
            title=f"Symbolic Regression - {dataset_name}",
            output_path=str(output_path),
            equation=equation_str
        )
        
        # CV means become the reported (headline) metrics when CV is on.
        reported = {'R2': cv['R2'], 'RMSE': cv['RMSE'], 'MAE': cv['MAE']} if cv else test_metrics

        # Save equation to file
        eq_path = self.output_dir / f"{dataset_name}_SR_equation.txt"
        with open(eq_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Equation: {equation_str}\n")
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
            'equation': equation_str
        }
        if cv:
            result.update({'R2_std': cv['R2_std'], 'cv_folds': cv_folds})
        return result


def main():
    """Run symbolic regression for all datasets."""
    trainer = SymbolicRegressionTrainer()
    
    # Define all dataset configurations
    configs = [
        # Pairs datasets
        ('All-Pairs', 'all', False),
        ('RE-Pairs', 're', False),
        ('RE-free-Pairs', 're-free', False),
        
        # Augmented datasets
        ('All-Augm', 'all', True),
        ('RE-Augm', 're', True),
        ('RE-free-Augm', 're-free', True),
    ]
    
    results = {}
    for dataset_name, dataset_type, is_augmented in configs:
        try:
            metrics = trainer.train_and_evaluate(dataset_name, dataset_type, is_augmented)
            results[dataset_name] = metrics
        except Exception as e:
            print(f"Error training {dataset_name}: {e}")
            results[dataset_name] = {'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'equation': 'Error'}
    
    # Save summary
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(trainer.output_dir / "symbolic_regression_summary.csv")
    print("\n" + "="*60)
    print("Symbolic Regression Summary:")
    print("="*60)
    print(summary_df)
    
    return results


if __name__ == "__main__":
    main()
