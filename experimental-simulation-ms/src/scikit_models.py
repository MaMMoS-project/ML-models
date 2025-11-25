# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

sns.set_style("whitegrid") 

# Scikit-learn models
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Scikit-learn utilities
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.base import RegressorMixin

# Typing
from typing import Union, List, Dict, Tuple, Type, Optional, Any

# Custom modules
import mammos_entity as me
import mammos_units as u

def optimize_scikit_model(
    X: Union[str, List[str]],
    y: str,
    xlabel: str,
    ylabel: str,
    model: Type[RegressorMixin], 
    param_grid: Dict,
    test_size: float = 0.2,
    random_state: int = 42,
    scoring: str = 'neg_root_mean_squared_error',
    cv: int = 3,
    plot_name: str = 'default'
) -> Tuple[RegressorMixin, Dict[str, float]]:
    """
    Train and optimize a scikit-learn regression model using GridSearchCV.
    """

    grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if y.ndim == 2:
        y = y.reshape(-1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    grid_search.fit(X_train, y_train) 

    # Evaluate
    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters for {model.__class__.__name__}: {grid_search.best_params_}")

    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Compute metrics helper
    def compute_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    def draw_metrics_block(ax, metrics, x=0.05, y=0.95):
        textstr = (
            f"RMSE: {metrics['RMSE']:.3f}\n"
            f"MAE: {metrics['MAE']:.3f}\n"
            f"R2: {metrics['R2']:.3f}"
        )
        props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7)
        ax.text(x, y, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    # Outliers in train
    train_residuals = y_train - y_train_pred
    train_std_resid = np.std(train_residuals)
    train_outliers = np.abs(train_residuals) > 2 * train_std_resid

    # Outliers in test
    test_residuals = y_test - y_test_pred
    test_std_resid = np.std(test_residuals)
    test_outliers = np.abs(test_residuals) > 2 * test_std_resid

    sns.set_style("whitegrid") 

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=100, sharex=False, sharey=True)
    
    ((ax1, ax2), (ax3, ax4)) = axes

    # === Fit Line on Train === 
    if hasattr(best_model, 'coef_') and X_train.shape[1] <=1:

        x_vals = np.linspace(min(X_train.min(), X_test.min()), 
                             max(X_train.max(), X_test.max()), 100)

        y_line = best_model.coef_[0] * x_vals + best_model.intercept_
               
        ax1.scatter(X_train, y_train, alpha=0.6, label="Train Data")
        ax1.scatter(X_train[train_outliers], y_train[train_outliers],
                    facecolors='none', edgecolors='red', s=100, label='Outliers')
        ax1.plot(x_vals, y_line, color='red', 
                 label=f"Fit: Y = {best_model.coef_[0]:.2f}X + {best_model.intercept_:.2f}")
        ax1.set_title("Fit Line: Training Set")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.legend()
        ax1.grid(True)

        # === Fit Line on Test ===
        ax2.scatter(X_test, y_test, alpha=0.6, label="Test Data", color='orange')
        ax2.scatter(X_test[test_outliers], y_test[test_outliers],
                    facecolors='none', edgecolors='red', s=100, label='Outliers')
        ax2.plot(x_vals, y_line, color='red', label=f"Fit: Y = {best_model.coef_[0]:.2f}X + {best_model.intercept_:.2f}")
        ax2.set_title("Fit Line: Test Set")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.legend()
        ax2.grid(True)
    
    elif hasattr(best_model, 'estimators_') and X_train.shape[1] <=1:
        x_vals = np.linspace(min(X_train.min(), X_test.min()), 
                             max(X_train.max(), X_test.max()), 100)

        # Plot x-axis: Tc_sim, y-axis: f(Tc_sim_train, comp_emb_train) = y_pred_train
        ax1.scatter(X_train, y_train_pred, alpha=0.6, label="Train Data")
        
        # Fit linear model to add to plot
        # huber_regression_train = HuberRegressor()
        # huber_regression_train.fit(X_train, y_train_pred)
        # y_line_train = huber_regression_train.predict(x_vals.reshape(-1,1))
        # ax1.plot(x_vals, y_line_train, color='red', label=f"Fit: Tc_exp_train = f_hat(Tc_sim_train), Y = {huber_regression_train.coef_[0]:.2f}X + {huber_regression_train.intercept_:.2f}")
        # ax1.set_title("Fit Line: Training Set")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Ms_exp_predicted_train (A/m)")
        ax1.legend()
        ax1.grid(True) 

        # === Fit Line on Test ===
        ax2.scatter(X_test, y_test_pred, alpha=0.6, label="Train Data")
        
        # Fit linear model to add to plot
        # huber_regression_test = HuberRegressor()
        # huber_regression_test.fit(X_test, y_test_pred)
        # y_line_test = huber_regression_test.predict(x_vals.reshape(-1,1))
        # ax2.plot(x_vals, y_line_test, color='red', label=f"Fit: Tc_exp_test = f_hat(Tc_sim_test), Y = {huber_regression_test.coef_[0]:.2f}X + {huber_regression_test.intercept_:.2f}")
        ax2.set_title("Fit Line: Test Set")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Tc_exp_predicted_test (A/m)")
        ax2.legend()
        ax2.grid(True)       
        
        # === Add metrics box ===
        draw_metrics_block(ax1, train_metrics)
        draw_metrics_block(ax2, test_metrics)

    if X_train.shape[1] > 1:       
        # ax3, ax4 = ax1, ax2 
        # fig.delaxes(axes[1,0])
        # fig.delaxes(axes[1,1])
        plt.close()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 12), dpi=150, 
                                 sharex=False, sharey=True)
        ((ax3, ax4)) = axes
        
        # === Add metrics box ===
        draw_metrics_block(ax3, train_metrics)
        draw_metrics_block(ax4, test_metrics)

    # === TRAIN SET: y vs y_hat ===
    lims = [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())]

    ax3.scatter(y_train, y_train_pred, alpha=0.6)
    ax3.plot(lims, lims, 'k--', linewidth=1.5)

    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_aspect('equal', adjustable='box')   

    ax3.set_title('Training Set: Predicted vs Actual')
    ax3.set_xlabel(ylabel)
    ax3.set_ylabel(f'{ylabel}_predicted')
    ax3.scatter(y_train[train_outliers], y_train_pred[train_outliers],
                facecolors='none', edgecolors='red', s=100, label='Train Outliers')
    ax3.legend()
    
    # === TEST SET: y vs y_hat ===
    ax4.scatter(y_test, y_test_pred, alpha=0.6, color='orange')
    ax4.plot(lims, lims, 'k--', linewidth=1.5)
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    ax4.set_aspect('equal', adjustable='box') 
    ax4.set_title('Test Set: Predicted vs Actual')
    ax4.set_xlabel(ylabel)
    ax4.set_ylabel(f'{ylabel}_predicted')
    ax4.scatter(y_test[test_outliers], y_test_pred[test_outliers],
                facecolors='none', edgecolors='red', s=100, label='Test Outliers')
    ax4.legend()
    ax4.grid(True)
            
    plt.tight_layout()
    plt.savefig(f'{plot_name}.png', dpi=150)
    # plt.show()

    # Residual histogram for test set
    plt.figure(figsize=(6, 4), dpi=100)
    plt.hist(test_residuals, bins=30, color='salmon', edgecolor='black')
    plt.title('Residual Distribution (Test Set)')
    plt.xlabel(f'Error ({ylabel} - Predicted)')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.close()
    
    grid = {}
    grid['best_params'] = best_model
    
    return best_model, train_metrics, test_metrics, grid

def evaluate():
    pass