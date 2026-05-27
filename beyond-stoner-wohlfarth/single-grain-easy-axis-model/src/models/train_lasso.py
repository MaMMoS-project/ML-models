
from sklearn.linear_model import LassoLars, LassoLarsCV
from sklearn.multioutput import MultiOutputRegressor
from typing import Dict, Tuple, Any
import numpy as np

def train_lasso_lars(X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[LassoLars, Dict[str, Any]]:
    """Train a LassoLars regression model.

    Uses the LARS algorithm to compute the exact LASSO solution without
    iterative convergence issues.

    Args:
        X: Input features
        y: Target values
        **kwargs: Additional parameters from config, including:
            - alpha: L1 regularization parameter

    Returns:
        Tuple of (trained model, parameters used)
    """
    alpha = float(kwargs.get('alpha', 0.01))

    model = LassoLars(alpha=alpha)
    model.fit(X, y)

    params = {'alpha': alpha}
    return model, params


def train_lasso_lars_cv(X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[LassoLarsCV, Dict[str, Any]]:
    """Train a LassoLarsCV regression model.

    Uses cross-validation along the LARS regularization path to select alpha
    automatically, avoiding both manual tuning and convergence issues.

    Args:
        X: Input features
        y: Target values
        **kwargs: Additional parameters from config, including:
            - cv: Number of cross-validation folds (default 5)
            - max_iter: Maximum number of iterations / knots in the path (default 500)

    Returns:
        Tuple of (trained model, parameters used)
    """
    cv = int(kwargs.get('cv', 5))
    max_iter = int(kwargs.get('max_iter', 500))

    # LassoLarsCV does not support multi-output, so wrap it.
    model = MultiOutputRegressor(LassoLarsCV(cv=cv, max_iter=max_iter))
    model.fit(X, y)

    alphas = [float(est.alpha_) for est in model.estimators_]
    params = {
        'cv': cv,
        'max_iter': max_iter,
        'alpha_per_output': alphas,
    }
    return model, params
