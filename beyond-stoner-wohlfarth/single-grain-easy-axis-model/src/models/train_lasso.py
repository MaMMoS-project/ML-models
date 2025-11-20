
from sklearn.linear_model import Lasso
from typing import Dict, Tuple, Any
import numpy as np

def train_lasso(X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[Lasso, Dict[str, Any]]:
    """Train a Lasso regression model.
    
    Args:
        X: Input features
        y: Target values
        **kwargs: Additional parameters from config, including:
            - alpha: L1 regularization parameter
            - tol: Tolerance for optimization
        
    Returns:
        Tuple of (trained model, parameters used)
    """
    alpha = float(kwargs.get('alpha', 0.01))
    tol = float(kwargs.get('tol', 1e-4))
    
    model = Lasso(alpha=alpha, tol=tol)
    model.fit(X, y)
    
    params = {
        'alpha': alpha,
        'tol': tol
    }
    
    return model, params
