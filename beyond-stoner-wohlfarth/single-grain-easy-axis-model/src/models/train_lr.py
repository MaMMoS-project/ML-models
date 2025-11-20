from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Any
import numpy as np

class MultiOutputLinearRegression:
    """Custom class for handling multiple linear regression models as one."""
    
    def __init__(self, models, feature_names=None):
        self.models = models
        self.feature_names_ = feature_names
    
    def predict(self, X):
        # Convert to numpy if it's a pandas DataFrame
        X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
        predictions = np.column_stack([model.predict(X_np) for model in self.models])
        return predictions
    
    @property
    def feature_importances_(self):
        # Linear models don't have feature_importances_ but we can use coefficients
        return np.vstack([np.abs(model.coef_) for model in self.models]).mean(axis=0)
    
    @property
    def coef_(self):
        return np.vstack([model.coef_ for model in self.models])
            
    @property
    def intercept_(self):
        return np.array([model.intercept_ for model in self.models])

def train_linear_regression(X, y, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """Train a linear regression model with hyperparameter tuning.
    
    Args:
        X: Input features
        y: Target values
        **kwargs: Additional parameters from config
        
    Returns:
        Tuple of (trained model, best parameters)
    """
    # Get parameter grid from kwargs or use default
    param_grid = kwargs.get('param_grid', {
        'fit_intercept': [True, False]
    })

    # Get cross-validation parameters from kwargs
    cv_config = kwargs.get('cv_config', {
        'n_folds': 5,
        'random_state': 42
    })
    
    # Create base linear regression model
    lr = linear_model.LinearRegression()
    
    # Create KFold cross-validator
    shuffle = cv_config.get('shuffle', True)  # Default to True if not specified
    cv = KFold(n_splits=cv_config['n_folds'], 
               shuffle=shuffle,
               random_state=cv_config['random_state'] if shuffle else None)
    
    # Handle multi-output regression
    if len(y.shape) > 1 and y.shape[1] > 1:
        # For multi-output, train separate models for each output
        models = []
        best_params_list = []
        
        # Convert to numpy if it's a pandas DataFrame
        X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
        y_np = y.to_numpy() if hasattr(y, 'to_numpy') else y
        
        for i in range(y.shape[1]):
            # Create grid search for this output dimension
            grid_search = GridSearchCV(
                linear_model.LinearRegression(),
                param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            # If X is a DataFrame, get feature names for later
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
            else:
                feature_names = None
                
            # Fit the grid search with numpy arrays
            grid_search.fit(X_np, y_np[:, i])
            
            # Save the best model and parameters
            models.append(grid_search.best_estimator_)
            best_params_list.append(grid_search.best_params_)
            
        # Use the MultiOutputLinearRegression class defined outside the function
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = None
            
        model = MultiOutputLinearRegression(models, feature_names)
        # Use the most common parameters as the "best" parameters
        best_params = best_params_list[0]  # Just use the first model's params for simplicity
        
        print("Best parameters for multi-output linear regression:")
        for i, params in enumerate(best_params_list):
            print(f"Output {i}: {params}")
    else:
        # For single output, just fit the grid search
        grid_search = GridSearchCV(
            lr,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X, y)
        
        # Get the best model and parameters
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print("Best parameters:", best_params)
        print("Best score:", -grid_search.best_score_)
    
    return model, best_params


