import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

def plot_diagnostics(y_test, y_pred, feature_names):
    """Generate diagnostic plots for model evaluation.
    
    Args:
        y_test: True test values (DataFrame or ndarray)
        y_pred: Predicted test values (DataFrame or ndarray)
        feature_names: List of feature names
    """
    if not isinstance(y_pred, pd.DataFrame):
        y_pred = pd.DataFrame(y_pred, columns=feature_names, index=y_test.index if hasattr(y_test, 'index') else None)
    if not isinstance(y_test, pd.DataFrame):
        y_test = pd.DataFrame(y_test, columns=feature_names, index=y_pred.index)

    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, 3, figsize=(15, 5 * n_features))
    
    for i, feature in enumerate(feature_names):
        residuals = y_test[feature] - y_pred[feature]
        predicted = y_pred[feature]
        
        # Histogram of residuals
        axes[i, 0].hist(residuals, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[i, 0].set_title(f'Histogram of Residuals\n{feature}')
        axes[i, 0].set_xlabel('Residuals')
        axes[i, 0].set_ylabel('Frequency')
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title(f'Q-Q Plot\n{feature}')
        
        # Residuals vs Predicted
        axes[i, 2].scatter(predicted, residuals, alpha=0.5)
        axes[i, 2].axhline(y=0, color='r', linestyle='--')
        
        # Add trend line
        X = predicted.values.reshape(-1, 1)
        y = residuals.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        axes[i, 2].plot(X, reg.predict(X), color='red', alpha=0.8)
        
        axes[i, 2].set_title(f'Residuals vs Predicted\n{feature}')
        axes[i, 2].set_xlabel('Predicted Values')
        axes[i, 2].set_ylabel('Residuals')
    
    plt.tight_layout()
    return fig
