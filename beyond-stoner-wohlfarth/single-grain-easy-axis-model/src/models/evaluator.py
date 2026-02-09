
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
import torch

from src.models.regression_metrics import multioutput_mape, adjusted_r_squared, gini_coefficient
from src.models.plot_models import plot_predictions_with_metrics_row_confidence, plot_predictions_jackknife
from src.models.plot_diagnostics import plot_diagnostics

class Evaluator:
    """Model evaluator (uses same data as training) """
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.results_dir = Path(config['data']['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, dataset_name, model_name, errors=None):
        """Evaluate a trained model using pre-split and pre-scaled data.
        
        Args:
            model: The trained model to evaluate
            X_train: Training features (already scaled)
            y_train: Training targets
            X_test: Test features (already scaled)
            y_test: Test targets
            dataset_name: Name of dataset for logging
            model_name: Name of model for logging
            errors: Optional jackknife errors for Random Forest models
        """
        # Make predictions directly on the provided data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Convert predictions to DataFrames if needed
        if hasattr(y_train, 'columns'):
            output_names = y_train.columns
            if not isinstance(y_train_pred, pd.DataFrame):
                y_train_pred = pd.DataFrame(y_train_pred, columns=output_names, index=y_train.index)
            if not isinstance(y_test_pred, pd.DataFrame):
                y_test_pred = pd.DataFrame(y_test_pred, columns=output_names, index=y_test.index)
        
        # Compute metrics
        metrics = self._compute_metrics(y_train, y_train_pred, y_test, y_test_pred, X_train, X_test)
        
        # Generate plots if configured
        if self.config['evaluation']['generate_plots']:
            # Convert all data to numpy arrays for plotting
            y_train_np = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train
            y_train_pred_np = y_train_pred.to_numpy() if hasattr(y_train_pred, 'to_numpy') else y_train_pred
            y_test_np = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else y_test
            y_test_pred_np = y_test_pred.to_numpy() if hasattr(y_test_pred, 'to_numpy') else y_test_pred
            
            # Get feature names
            feature_names = y_train.columns if hasattr(y_train, 'columns') else None
            plot_feature_names = feature_names if feature_names is not None else ['Output ' + str(i) for i in range(y_train_np.shape[1])]
            
            # Generate plots
            self._generate_plots(
                y_train_np, y_train_pred_np, y_test_np, y_test_pred_np,
                dataset_name, model_name, plot_feature_names, errors
            )
        
        # Save model if configured
        if self.config['evaluation']['save_models']:
            self._save_model(model, dataset_name, model_name, metrics)
        
        # Print metrics
        self.print_metrics(metrics, dataset_name, model_name)
        
        return metrics
    
    def _compute_metrics(self, y_train, y_train_pred, y_test, y_test_pred, X_train, X_test):
        """Compute evaluation metrics on the exact same data used for training/testing."""
        # Convert to numpy if needed for consistent metric calculation
        y_train_orig = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train
        y_train_pred_orig = y_train_pred.to_numpy() if hasattr(y_train_pred, 'to_numpy') else y_train_pred
        y_test_orig = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else y_test
        y_test_pred_orig = y_test_pred.to_numpy() if hasattr(y_test_pred, 'to_numpy') else y_test_pred
        
        # Get feature names for individual metrics
        feature_names = list(y_train.columns) if hasattr(y_train, 'columns') else [f'output_{i}' for i in range(y_train_orig.shape[1])]
        
        # Calculate overall metrics (averaged across all outputs)
        metrics = {
            'train': {
                'mse': mean_squared_error(y_train_orig, y_train_pred_orig),
                'mae': mean_absolute_error(y_train_orig, y_train_pred_orig),
                'r2': r2_score(y_train_orig, y_train_pred_orig),
                'mape': multioutput_mape(y_train_orig, y_train_pred_orig),
                'gini': gini_coefficient(y_train_orig, y_train_pred_orig),
            },
            'test': {
                'mse': mean_squared_error(y_test_orig, y_test_pred_orig),
                'mae': mean_absolute_error(y_test_orig, y_test_pred_orig),
                'r2': r2_score(y_test_orig, y_test_pred_orig),
                'mape': multioutput_mape(y_test_orig, y_test_pred_orig),
                'gini': gini_coefficient(y_test_orig, y_test_pred_orig),
            }


        }
        
        # Add adjusted R² scores for overall metrics
        n_train, p_train = X_train.shape
        n_test, p_test = X_test.shape
        metrics['train']['adj_r2'] = adjusted_r_squared(metrics['train']['r2'], n_train, p_train)
        metrics['test']['adj_r2'] = adjusted_r_squared(metrics['test']['r2'], n_test, p_test)
        
        # Calculate per-variable metrics
        metrics['per_variable'] = {}
        
        for i, feature_name in enumerate(feature_names):
            # Extract single variable data
            y_train_var = y_train_orig[:, i].reshape(-1, 1) if y_train_orig.ndim > 1 else y_train_orig.reshape(-1, 1)
            y_train_pred_var = y_train_pred_orig[:, i].reshape(-1, 1) if y_train_pred_orig.ndim > 1 else y_train_pred_orig.reshape(-1, 1)
            y_test_var = y_test_orig[:, i].reshape(-1, 1) if y_test_orig.ndim > 1 else y_test_orig.reshape(-1, 1)
            y_test_pred_var = y_test_pred_orig[:, i].reshape(-1, 1) if y_test_pred_orig.ndim > 1 else y_test_pred_orig.reshape(-1, 1)
            
            # Calculate metrics for this variable
            var_metrics = {
                'train': {
                    'mse': mean_squared_error(y_train_var, y_train_pred_var),
                    'mae': mean_absolute_error(y_train_var, y_train_pred_var),
                    'r2': r2_score(y_train_var, y_train_pred_var),
                    'mape': multioutput_mape(y_train_var, y_train_pred_var),
                    'gini': gini_coefficient(y_train_var, y_train_pred_var),
                },
                'test': {
                    'mse': mean_squared_error(y_test_var, y_test_pred_var),
                    'mae': mean_absolute_error(y_test_var, y_test_pred_var),
                    'r2': r2_score(y_test_var, y_test_pred_var),
                    'mape': multioutput_mape(y_test_var, y_test_pred_var),
                    'gini': gini_coefficient(y_test_var, y_test_pred_var),
                }
            }
            
            # Add adjusted R² scores for this variable
            var_metrics['train']['adj_r2'] = adjusted_r_squared(var_metrics['train']['r2'], n_train, p_train)
            var_metrics['test']['adj_r2'] = adjusted_r_squared(var_metrics['test']['r2'], n_test, p_test)
            
            # Store metrics for this variable
            metrics['per_variable'][feature_name] = var_metrics
        
        return metrics
    
    def _generate_plots(self, y_train, y_train_pred, y_test, y_test_pred, 
                       dataset_name, model_name, feature_names, errors=None):
        """Generate evaluation plots."""
        plot_dir = self.results_dir / 'plots' / dataset_name / model_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if this is a Random Forest model with jackknife errors
        is_random_forest = model_name == 'random_forest'
        
        # Generate prediction plots with metrics
        if is_random_forest and errors is not None:
            # For Random Forest models with jackknife errors, use the specialized plot
            fig = plot_predictions_jackknife(
                y_train, y_train_pred,
                y_test, y_test_pred,
                feature_names,
                errors
            )
            plt.savefig(plot_dir / 'predictions_jackknife.png')
            plt.close(fig)
            
            # Also generate the standard prediction plot for comparison
            plot_predictions_with_metrics_row_confidence(
                y_train, y_train_pred,
                y_test, y_test_pred,
                feature_names
            )
            plt.savefig(plot_dir / 'predictions.png')
            plt.close()
        else:
            # For other models, use the standard prediction plot
            plot_predictions_with_metrics_row_confidence(
                y_train, y_train_pred,
                y_test, y_test_pred,
                feature_names
            )
            plt.savefig(plot_dir / 'predictions.png')
            plt.close()
        
        # Generate diagnostic plots
        plot_diagnostics(y_test, y_test_pred, feature_names)
        plt.savefig(plot_dir / 'diagnostics.png')
        plt.close()
        
    def _save_model(self, model, dataset_name, model_name, metrics):
        """Save model and its metrics."""
        model_dir = self.results_dir / 'models' / dataset_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle PyTorch models differently
        if isinstance(model, torch.nn.Module):
            # For PyTorch models, save state_dict
            torch_path = model_dir / f"{model_name}.pt"
            torch.save(model.state_dict(), torch_path)
            print(f"PyTorch model saved to {torch_path}")
            
            # Also save a pickle with metadata for consistency
            model_info = {
                'type': 'pytorch_model',
                'model_path': str(torch_path),
                'input_dim': model.fc1.in_features if hasattr(model, 'fc1') else None,
                'output_dim': model.fc4.out_features if hasattr(model, 'fc4') else None
            }
            model_path = model_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
        else:
            # Save sklearn model
            model_path = model_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metrics
        metrics_path = model_dir / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def print_metrics(self, metrics, dataset_name, model_name):
        """Print evaluation metrics."""
        print(f"\n==== {model_name} Metrics for {dataset_name} ====\n")
        
        # Print overall metrics (averaged across all outputs)
        print("\n=== OVERALL METRICS (Averaged Across All Outputs) ===")
        for split in ['train', 'test']:
            print(f"\n{split.capitalize()} Metrics:")
            print(f"MSE   : {metrics[split]['mse']:.4f}")
            print(f"MAE   : {metrics[split]['mae']:.4f}")
            print(f"R²    : {metrics[split]['r2']:.4f}")
            print(f"Adj R²: {metrics[split]['adj_r2']:.4f}")
            print(f"MAPE  : {metrics[split]['mape']:.2f} %")
            print(f"Gini  : {metrics[split]['gini']:.4f}")
        
        # Print per-variable metrics
        if 'per_variable' in metrics:
            print("\n=== PER-VARIABLE METRICS ===")
            for var_name, var_metrics in metrics['per_variable'].items():
                print(f"\n--- {var_name} ---")
                for split in ['train', 'test']:
                    print(f"\n{split.capitalize()} Metrics:")
                    print(f"MSE   : {var_metrics[split]['mse']:.4f}")
                    print(f"MAE   : {var_metrics[split]['mae']:.4f}")
                    print(f"R²    : {var_metrics[split]['r2']:.4f}")
                    print(f"Adj R²: {var_metrics[split]['adj_r2']:.4f}")
                    print(f"MAPE  : {var_metrics[split]['mape']:.2f} %")
                    print(f"Gini  : {var_metrics[split]['gini']:.4f}")
