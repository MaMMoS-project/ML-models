"""
Symbolic Regression Training - Baseline Model
Uses only Tc_sim as input (stoichiometry disregarded)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import r2_score, mean_squared_error
from base_trainer import DataLoader, ModelEvaluator, split_data

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
        niterations: int = 100
    ) -> Dict[str, float]:
        """
        Train symbolic regression model.
        
        Args:
            dataset_name: Name for output files (e.g., 'All-Pairs', 'RE-Augm')
            dataset_type: 'all', 're', or 're-free'
            is_augmented: Whether to use augmented data
            niterations: Number of PySR iterations
            
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
        output_path = self.output_dir / f"{dataset_name}_SR.png"
        self.evaluator.plot_predictions(
            y_train, y_train_pred,
            y_test, y_test_pred,
            title=f"Symbolic Regression - {dataset_name}",
            output_path=str(output_path),
            equation=equation_str
        )
        
        # Save equation to file
        eq_path = self.output_dir / f"{dataset_name}_SR_equation.txt"
        with open(eq_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Equation: {equation_str}\n")
            f.write(f"R2: {test_metrics['R2']:.4f}\n")
            f.write(f"RMSE: {test_metrics['RMSE']:.2f}\n")
            f.write(f"MAE: {test_metrics['MAE']:.2f}\n")
        
        return {
            'R2': test_metrics['R2'],
            'RMSE': test_metrics['RMSE'],
            'MAE': test_metrics['MAE'],
            'equation': equation_str
        }


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
