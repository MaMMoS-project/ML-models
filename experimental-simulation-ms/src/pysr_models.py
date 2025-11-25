import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import os 

import pdb

sns.set_style("whitegrid") 

def optimize_sr_model(X, y,  
                      model,
                      xlabel, ylabel,
                      param_grid=None,
                      test_size: float = 0.2,
                      random_state=42,
                      plot_name=None):

    if X.ndim == 1:
        X = X.reshape(-1, 1)
 
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )     
 
    # -----------------------
    # Fit PySR model
    # -----------------------   
    model.fit(X_train, y_train)

    # -----------------------
    # Predictions
    # -----------------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # -----------------------
    # Compute residuals and outliers (based on prediction error)
    # -----------------------
    res_train = y_train - y_train_pred
    res_test  = y_test - y_test_pred
    threshold_train = 2 * np.std(res_train)
    threshold_test  = 2 * np.std(res_test)

    outliers_train_idx = np.where(np.abs(res_train) > threshold_train)[0]
    outliers_test_idx  = np.where(np.abs(res_test) > threshold_test)[0]
    
    # -----------------------
    # Metrics
    # -----------------------
    metrics_train = {
        "R2": r2_score(y_train, y_train_pred),
        "MAE": mean_absolute_error(y_train, y_train_pred),
        "RMSE": root_mean_squared_error(y_train, y_train_pred)
    }
    metrics_test = {
        "R2": r2_score(y_test, y_test_pred),
        "MAE": mean_absolute_error(y_test, y_test_pred),
        "RMSE": root_mean_squared_error(y_test, y_test_pred)
    }

    # -----------------------
    # Best PySR equation
    # -----------------------
    best_eq = model.get_best()['equation']  
    best_eq_single_line = " ".join(best_eq.split()) 
    
    # -----------------------
    # Create 2x2 grid plot
    # -----------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: X vs y (Train)
    axes[0,0].scatter(X_train, y_train, alpha=0.3, label="Train data")
    axes[0,0].scatter(X_train[outliers_train_idx], y_train[outliers_train_idx],
                      facecolors='none', edgecolors='red', s=100, label='Outliers')
    X_sorted = np.sort(X, axis=0)
    y_sorted_pred = model.predict(X_sorted)
    axes[0,0].plot(X_sorted, y_sorted_pred, color="red", linewidth=2, label="PySR fit")
    axes[0,0].set_xlabel("Tc_sim")
    axes[0,0].set_ylabel("Tc_exp")
    axes[0,0].legend()
    axes[0,0].grid(True)
    metrics_text_train = (
        f"Train Metrics:\n"
        f"R2={metrics_train['R2']:.3f}, MAE={metrics_train['MAE']:.3f}, RMSE={metrics_train['RMSE']:.3f}\n"
        f"Eq: {best_eq}"
    )
    axes[0,0].text(0.05, 0.95, metrics_text_train, transform=axes[0,0].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Top-right: X vs y (Test)
    axes[0,1].scatter(X_test, y_test, alpha=0.7, color="orange", label="Test data")
    axes[0,1].scatter(X_test[outliers_test_idx], y_test[outliers_test_idx],
                     facecolors='none', edgecolors='red', s=100, label='Outliers')
    axes[0,1].plot(X_sorted, y_sorted_pred, color="red", linewidth=2, label="PySR fit")
    axes[0,1].set_xlabel("Tc_sim")
    axes[0,1].set_ylabel("Tc_exp")
    axes[0,1].legend()
    axes[0,1].grid(True)
    metrics_text_test = (
        f"Test Metrics:\n"
        f"R2={metrics_test['R2']:.3f}, MAE={metrics_test['MAE']:.3f}, RMSE={metrics_test['RMSE']:.3f}"
    )
    axes[0,1].text(0.05, 0.95, metrics_text_test, transform=axes[0,1].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Bottom-left: y_true vs y_pred (Train)
    axes[1,0].scatter(y_train, y_train_pred, alpha=0.5, label="Train predictions")
    axes[1,0].scatter(y_train[outliers_train_idx], y_train_pred[outliers_train_idx],
                     facecolors='none', edgecolors='red', s=100, label='Outliers')
    axes[1,0].plot([y.min(), y.max()], [y.min(), y.max()],
                   color="black", linestyle="--", label="y=x")
    axes[1,0].set_xlabel(ylabel)
    axes[1,0].set_ylabel(f"{ylabel}_hat")
    axes[1,0].legend()
    axes[1,0].grid(True)
    axes[1,0].set_title("Train: y_true vs y_pred")

    # Bottom-right: y_true vs y_pred (Test)
    axes[1,1].scatter(y_test, y_test_pred, alpha=0.7, color="orange", label="Test predictions")
    axes[1,1].scatter(y_test[outliers_test_idx], y_test_pred[outliers_test_idx],
                      facecolors='none', edgecolors='red', s=100, label='Outliers')
    axes[1,1].plot([y.min(), y.max()], [y.min(), y.max()],
                   color="black", linestyle="--", label="y=x")
    axes[1,1].set_xlabel(ylabel)
    axes[1,1].set_ylabel(f"{ylabel}_hat")
    axes[1,1].legend()
    axes[1,1].grid(True)
    axes[1,1].set_title("Test: y_true vs y_pred")
    
    fig.suptitle("Symbolic Regression", fontsize=16) 

    plt.tight_layout()
    plt.savefig(f'{plot_name}.png')
    # plt.show()
    plt.close()
    
    return best_eq, metrics_train, metrics_test, None
    
# Compute metrics helper
def compute_metrics(y_true, y_pred):
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}  