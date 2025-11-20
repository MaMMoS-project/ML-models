import numpy as np

def multioutput_mape(y_true, y_pred):
    """Mean Absolute Percentage Error (MAPE) for multi-output regression."""
    epsilon = 1e-12  # Prevent division by zero
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100


def adjusted_r_squared(r2, n, p):
    """Compute adjusted R² given sample size n and number of predictors p."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def gini_coefficient(y_true, y_pred):
    """Compute Gini coefficient for regression predictions.
    
    Works with both single-output and multi-output regression.
    For multi-output regression, returns the average Gini coefficient across outputs.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle multi-output case
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        # For multi-output, compute Gini for each output dimension and average
        gini_values = []
        for i in range(y_true.shape[1]):
            gini_values.append(_single_gini(y_true[:, i], y_pred[:, i]))
        return np.mean(gini_values)
    else:
        # For single output
        if len(y_true.shape) > 1:
            y_true = y_true.flatten()
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        return _single_gini(y_true, y_pred)

def _single_gini(y_true, y_pred):
    """Helper function to compute Gini coefficient for a single output."""
    # Sort by predictions
    sorted_indices = np.argsort(y_pred)
    sorted_true = y_true[sorted_indices]
    
    # Avoid division by zero
    sum_true = np.sum(sorted_true)
    if sum_true == 0:
        return 0.0
        
    cumulative_true = np.cumsum(sorted_true) / sum_true
    n = len(y_true)
    gini = (np.sum((np.arange(n) + 1) * cumulative_true) / n) - (n + 1) / 2
    return gini / (n / 2)

def singleoutput_mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE) for a single target.
    """
    epsilon = 1e-12
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

def calculate_mape(y_true, y_pred):
    """Alias for singleoutput_mape."""
    return singleoutput_mape(y_true, y_pred)
