from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from typing import Dict, Tuple, Any, Optional
import numpy as np
import forestci as fci
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

def train_random_forest(X, y, **kwargs) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """Train a random forest model with hyperparameter tuning.
    
    Args:
        X: Input features
        y: Target values
        **kwargs: Additional parameters from config
        
    Returns:
        Tuple of (trained model, best parameters)
    """
    param_grid = kwargs.get('param_grid', {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    })

    # Get cross-validation parameters from kwargs
    cv_config = kwargs.get('cv_config', {
        'n_folds': 5,
        'random_state': 42
    })
    
    # Get random state from cv_config
    random_state = cv_config.get('random_state', 42)
    rf = RandomForestRegressor(random_state=random_state)
    
    # Create KFold cross-validator
    from sklearn.model_selection import KFold
    shuffle = cv_config.get('shuffle', True)  # Default to True if not specified
    cv = KFold(n_splits=cv_config['n_folds'], 
               shuffle=shuffle,
               random_state=cv_config['random_state'] if shuffle else None)

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    # Get the best model
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("Best parameters:", best_params)
    print("Best score:", -grid_search.best_score_)

    # Plot feature importance if requested
    show_feature_importance = kwargs.get('show_feature_importance', False)
    if show_feature_importance:
        # Get results directory from kwargs or use default
        results_dir = kwargs.get('results_dir', 'results')
        dataset_name = kwargs.get('dataset_name', 'unknown')
        scaler_type = kwargs.get('scaler_type', 'unknown')
        plot_feature_importance(best_rf, X, results_dir=results_dir, dataset_name=dataset_name, scaler_type=scaler_type)
        
    return best_rf, best_params


def calculate_jackknife_variance(model, X_train, X_test, y_train) -> Optional[np.ndarray]:
    """Calculate jackknife variance for Random Forest models.
    
    Args:
        model: Trained Random Forest model
        X_train: Training features
        X_test: Test features
        y_train: Training target values
        
    Returns:
        Numpy array of jackknife variance estimates or None if calculation fails
    """
    try:
        print("Computing jackknife variance for Random Forest model...")
        # Get number of output dimensions
        n_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # For forestci, we need to use numpy arrays without feature names
        X_train_np = X_train.to_numpy() if hasattr(X_train, 'to_numpy') else X_train
        X_test_np = X_test.to_numpy() if hasattr(X_test, 'to_numpy') else X_test
        
        # Get the first output to initialize the errors array
        output_0 = fci.random_forest_error(model, X_train_np.shape, X_test_np, y_output=0)
        errors = np.empty((output_0.shape[0], n_outputs))
        
        # Calculate errors for each output dimension
        for i in range(n_outputs):
            errors[:,i] = fci.random_forest_error(model, X_train_np.shape, X_test_np, y_output=i)
        
        print(f"Jackknife variance computed successfully. Shape: {errors.shape}")
        return errors
        
    except Exception as e:
        print(f"Warning: Could not compute jackknife variance: {str(e)}")
        return None


def plot_feature_importance(model, X, results_dir='results', dataset_name='unknown', scaler_type='unknown', title=None):
    """
    Plot feature importance for a Random Forest model and save it to the results directory.
    
    Args:
        model: Trained Random Forest model with feature_importances_ attribute
        X: Input features used for training
        results_dir: Directory to save the plot
        dataset_name: Name of the dataset for the plot title and filename
        scaler_type: Type of scaler used for the plot title and filename
        title: Optional custom title for the plot
        
    Returns:
        Dictionary of feature importances
    """
    try:
        if hasattr(model, "feature_importances_"):  # Ensure model supports feature importance
            importances = model.feature_importances_
            feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"Feature {i}" for i in range(X.shape[1])]

            # Sort feature importances
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(importances)), importances[indices], align='center')
            plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
            plt.xlabel("Feature Importance")
            plt.ylabel("Feature Name")
            
            # Use provided title or generate one based on dataset and scaler
            if title is None:
                title = f"Feature Importance ({dataset_name}, {scaler_type})"
            plt.title(title)
            plt.gca().invert_yaxis()  # Most important at the top
            plt.tight_layout()
            
            # Create results directory if it doesn't exist
            results_path = Path(results_dir)
            results_path.mkdir(exist_ok=True)
            
            # Create plots subdirectory
            plots_dir = results_path / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Create dataset-specific subdirectory first
            dataset_dir = plots_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # Create random_forest subdirectory inside dataset directory
            rf_dir = dataset_dir / 'random_forest'
            rf_dir.mkdir(exist_ok=True)
            
            # Save the plot
            filename = f"feature_importance_{dataset_name}_{scaler_type}.png"
            filepath = rf_dir / filename
            plt.savefig(filepath)
            plt.close()
            
            print(f"Feature importance plot saved to {filepath}")
            
            # Return the sorted feature importances as a dictionary
            importance_dict = {str(feature_names[i]): float(importances[i]) for i in indices}
            return importance_dict
        else:
            print("Warning: Model does not have feature_importances_ attribute.")
            return None
    except Exception as e:
        print(f"Warning: Could not plot feature importance: {str(e)}")
        return None
