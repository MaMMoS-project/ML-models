import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.regression_metrics import adjusted_r_squared, calculate_mape, singleoutput_mape, gini_coefficient

def plot_predictions_jackknife(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    feature_names: list,
    errors: np.ndarray,
    outlier_std: float = 3.0
) -> None:
    """Plot predictions with jackknife error bars for Random Forest models.
    
    Args:
        y_true_train: True training values
        y_pred_train: Predicted training values
        y_true_test: True test values
        y_pred_test: Predicted test values
        feature_names: Names of target features
        errors: Jackknife variance for error bars
        outlier_std: Standard deviation threshold for outliers
    """
    # Convert arrays to numpy if needed
    y_true_train = np.asarray(y_true_train)
    y_pred_train = np.asarray(y_pred_train)
    y_true_test = np.asarray(y_true_test)
    y_pred_test = np.asarray(y_pred_test)
    errors = np.asarray(errors)
    
    num_features = y_true_train.shape[1] if len(y_true_train.shape) > 1 else 1
    
    # Handle single feature case
    if num_features == 1 and len(y_true_train.shape) == 1:
        y_true_train = y_true_train.reshape(-1, 1)
        y_pred_train = y_pred_train.reshape(-1, 1)
        y_true_test = y_true_test.reshape(-1, 1)
        y_pred_test = y_pred_test.reshape(-1, 1)
        if errors.ndim == 1:
            errors = errors.reshape(-1, 1)
    
    # Create figure with one row per feature
    fig, axes = plt.subplots(num_features, 2, figsize=(14, 5 * num_features))
    
    # Handle single feature case for axes
    if num_features == 1:
        axes = np.array([axes])
    
    for i in range(num_features):
        # --- Training metrics ---
        y_train_actual = y_true_train[:, i]
        y_train_pred = y_pred_train[:, i]
        
        mse_train = mean_squared_error(y_train_actual, y_train_pred)
        mae_train = mean_absolute_error(y_train_actual, y_train_pred)
        r2_train = r2_score(y_train_actual, y_train_pred)
        adj_r2_train = adjusted_r_squared(r2_train, len(y_train_actual), 1)
        mape_train = singleoutput_mape(y_train_actual, y_train_pred)
        
        # --- Testing metrics ---
        y_test_actual = y_true_test[:, i]
        y_test_pred = y_pred_test[:, i]
        
        mse_test = mean_squared_error(y_test_actual, y_test_pred)
        mae_test = mean_absolute_error(y_test_actual, y_test_pred)
        r2_test = r2_score(y_test_actual, y_test_pred)
        adj_r2_test = adjusted_r_squared(r2_test, len(y_test_actual), 1)
        mape_test = singleoutput_mape(y_test_actual, y_test_pred)
        
        # ~~~~~~~~~~~~~~ TRAINING PLOT ~~~~~~~~~~~~~~~~~~
        ax_train = axes[i, 0]
        
        # Scatter plot
        ax_train.scatter(y_train_actual, y_train_pred, alpha=0.5, color='blue')
        
        # Perfect prediction line
        min_val = min(np.min(y_train_actual), np.min(y_train_pred))
        max_val = max(np.max(y_train_actual), np.max(y_train_pred))
        ax_train.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
        # Identify outliers
        residuals_train = y_train_pred - y_train_actual
        std_resid_train = np.std(residuals_train)
        threshold_train = outlier_std * std_resid_train
        mask_outliers_train = np.abs(residuals_train) > threshold_train
        
        # Highlight outliers
        if np.any(mask_outliers_train):
            ax_train.scatter(
                y_train_actual[mask_outliers_train],
                y_train_pred[mask_outliers_train],
                edgecolors='red',
                facecolors='none',
                s=100,
                label='Outliers'
            )
        
        # Add metrics text box
        ax_train.text(
            0.05,
            0.95,
            f'MSE: {mse_test:.2f}\nR²: {r2_test:.2f}',
            verticalalignment='top',
            transform=ax_train.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
        )
        
        # Set labels and title
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i}'
        ax_train.set_xlabel('Actual Values')
        ax_train.set_ylabel('Predicted Values')
        ax_train.set_title(f'Training: {feature_name}')
        ax_train.legend()
        
        # ~~~~~~~~~~~~~~ TESTING PLOT WITH JACKKNIFE ERROR BARS ~~~~~~~~~~~~~~~~~~
        ax_test = axes[i, 1]
        
        # Scatter plot
        ax_test.scatter(y_test_actual, y_test_pred, alpha=0.5, color='green')
        
        # Perfect prediction line
        min_val = min(np.min(y_test_actual), np.min(y_test_pred))
        max_val = max(np.max(y_test_actual), np.max(y_test_pred))
        ax_test.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
        # Identify outliers
        residuals_test = y_test_pred - y_test_actual
        std_resid_test = np.std(residuals_test)
        threshold_test = outlier_std * std_resid_test
        mask_outliers_test = np.abs(residuals_test) > threshold_test
        
        # Highlight outliers
        if np.any(mask_outliers_test):
            ax_test.scatter(
                y_test_actual[mask_outliers_test],
                y_test_pred[mask_outliers_test],
                edgecolors='red',
                facecolors='none',
                s=100,
                label='Outliers'
            )
        
        # Add jackknife error bars
        error_values = np.sqrt(errors[:, i]) if errors.ndim > 1 else np.sqrt(errors)
        ax_test.errorbar(y_test_actual, y_test_pred, yerr=error_values, fmt='o', color='g', alpha=0.5, label='Jackknife Error')
        
        # Add metrics text box
        ax_test.text(
            0.05,
            0.95,
            f'MSE: {mse_test:.2f}\nR²: {r2_test:.2f}',
            verticalalignment='top',
            transform=ax_test.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
        )
        
        # Set labels and title
        ax_test.set_xlabel('Actual Values')
        ax_test.set_ylabel('Predicted Values')
        ax_test.set_title(f'Test with Jackknife Error: {feature_name}')
        ax_test.legend()
    
    plt.tight_layout()
    return fig

def plot_predictions_with_metrics_row_confidence(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    feature_names: list,
    y_std_train: np.ndarray = None,
    y_std_test: np.ndarray = None,
    outlier_std: float = 3.0,
    errors: np.ndarray = None
) -> None:

    """Plot predictions with confidence intervals.
    
    Args:
        y_true_train: True training values
        y_pred_train: Predicted training values
        y_true_test: True test values
        y_pred_test: Predicted test values
        feature_names: Names of target features
        y_std_train: Standard deviation of training predictions
        y_std_test: Standard deviation of test predictions
        outlier_std: Standard deviation multiplier for outlier detection
    """

    # Convert arrays to numpy if needed
    y_true_train = np.asarray(y_true_train)
    y_pred_train = np.asarray(y_pred_train)
    y_true_test = np.asarray(y_true_test)
    y_pred_test = np.asarray(y_pred_test)
    if y_std_train is not None:
        y_std_train = np.asarray(y_std_train)
    if y_std_test is not None:
        y_std_test = np.asarray(y_std_test)
    if errors is not None:
        errors = np.asarray(errors)
    
    num_features = y_true_train.shape[1]
    fig, axes = plt.subplots(num_features, 2, figsize=(15, 5 * num_features))
    
    for i in range(num_features):
        # Training plot
        ax_train = axes[i, 0]
        ax_train.scatter(y_true_train[:, i], y_pred_train[:, i], alpha=0.5, color='blue', label='Predictions')
        
        if y_std_train is not None:
            ax_train.fill_between(
                y_true_train[:, i],
                y_pred_train[:, i] - 2 * y_std_train[:, i],
                y_pred_train[:, i] + 2 * y_std_train[:, i],
                color='blue', alpha=0.2, label='95% CI'
            )
        
        # Perfect prediction line
        min_val = min(y_true_train[:, i].min(), y_pred_train[:, i].min())
        max_val = max(y_true_train[:, i].max(), y_pred_train[:, i].max())
        ax_train.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
        # Metrics
        mse = mean_squared_error(y_true_train[:, i], y_pred_train[:, i])
        mae = mean_absolute_error(y_true_train[:, i], y_pred_train[:, i])
        r2 = r2_score(y_true_train[:, i], y_pred_train[:, i])
        adj_r2 = adjusted_r_squared(r2, y_true_train.shape[0], y_true_train.shape[1])
        
        ax_train.set_title(f'Training: {feature_names[i]}')
        ax_train.set_xlabel('True Values')
        ax_train.set_ylabel('Predicted Values')
        ax_train.text(0.05, 0.95,
                     #f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}\nAdj R²: {adj_r2:.4f}',
                     f'MSE: {mse:.4f}\nR²: {r2:.4f}',
                     transform=ax_train.transAxes, verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.8))
        ax_train.legend()
        
        # Test plot with regression line and confidence interval
        ax_test = axes[i, 1]
        sns.regplot(
            x=y_true_test[:, i],
            y=y_pred_test[:, i],
            ax=ax_test,
            #ci=95,
            fit_reg=False,
            scatter_kws={'alpha': 0.5, 'color': 'green'},
            truncate=False,
            label='Predictions'
        )
        sns.regplot(
            x=y_true_test[:, i],
            y=y_pred_test[:, i],
            ax=ax_test,
            ci=95,
            scatter=False,
            scatter_kws={'alpha': 0.5, 'color': 'green'},
            fit_reg=True,
            line_kws={'color': 'red'},
            truncate=False,
            label='Regression Line'
        )
        
        # Add error bars if provided
        if errors is not None:
            ax_test.errorbar(
                y_true_test[:, i],
                y_pred_test[:, i],
                yerr=np.sqrt(errors[:, i]) if errors.ndim > 1 else np.sqrt(errors),
                fmt='none',
                color='g',
                alpha=0.3,
                label='Prediction Error'
            )
        

        # Add confidence intervals if provided
        if y_std_test is not None:
            ax_test.fill_between(
                y_true_test[:, i],
                y_pred_test[:, i] - 2 * y_std_test[:, i],
                y_pred_test[:, i] + 2 * y_std_test[:, i],
                color='green',
                alpha=0.2,
                label='95% CI'
            )
        
        # Perfect prediction line
        min_val = min(y_true_test[:, i].min(), y_pred_test[:, i].min())
        max_val = max(y_true_test[:, i].max(), y_pred_test[:, i].max())
        ax_test.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
        # Outlier detection
        residuals_test = y_pred_test[:, i] - y_true_test[:, i]
        std_resid_test = np.std(residuals_test)
        threshold_test = outlier_std * std_resid_test
        mask_outliers_test = np.abs(residuals_test) > threshold_test
        
        if np.any(mask_outliers_test):
            ax_test.scatter(
                y_true_test[mask_outliers_test, i],
                y_pred_test[mask_outliers_test, i],
                edgecolors='red',
                facecolors='none',
                s=100,
                label='Outliers'
            )
        
        # Metrics
        mse = mean_squared_error(y_true_test[:, i], y_pred_test[:, i])
        mae = mean_absolute_error(y_true_test[:, i], y_pred_test[:, i])
        mape = calculate_mape(y_true_test[:, i], y_pred_test[:, i])
        r2 = r2_score(y_true_test[:, i], y_pred_test[:, i])
        adj_r2 = adjusted_r_squared(r2, y_true_test.shape[0], y_true_test.shape[1])
        
        ax_test.set_title(f'Test: {feature_names[i]}')
        ax_test.set_xlabel('True Values')
        ax_test.set_ylabel('Predicted Values')
        ax_test.text(
            0.05,
            0.95,
            f'MSE: {mse:.4f}\nR²: {r2:.4f}',
            transform=ax_test.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.7)
        )
        ax_test.legend(loc='lower right')
    
    plt.tight_layout()
    
    backend = plt.get_backend()
    if "inline" not in backend.lower():
        plt.show()
    else:
        plt.ioff()
    plt.close(fig)
    plt.close()