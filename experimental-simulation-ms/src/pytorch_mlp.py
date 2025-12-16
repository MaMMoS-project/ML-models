 # Standard libraries
import pandas as pd
import json
import numpy as np
import argparse
import copy

import pdb

# Plotting / visualizing
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# PyTorch MLP Architecture
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

class SimpleMLP(nn.Module):
    def __init__(self, input_size=None, hidden_size=50):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # first hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # second hidden layer
        self.fc3 = nn.Linear(hidden_size, 1) # output layer
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimplerMLP(nn.Module):
    def __init__(self, input_size=None, hidden_size=50):
        super(SimplerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # first hidden layer
        self.fc2 = nn.Linear(hidden_size, 1) # second hidden layer
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Early stopping helper
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience       # how many epochs to wait without improvement
        self.min_delta = min_delta     # minimum change to qualify as an improvement
        self.best_loss = float('inf')  # best validation loss observed so far
        self.counter = 0               # counts epochs without improvement
        self.should_stop = False       # flag to indicate stopping
    
    def step(self, val_loss):
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
def get_param_grid(dataset_size):
    """
    Returns a parameter grid suitable for the dataset size.
    
    Args:
        dataset_size (int): number of samples in the dataset
    
    Returns:
        List[Dict]: list of hyperparameter dictionaries
    """
    grid = []

    if dataset_size <= 300:  # small dataset: RE
        hidden_sizes = [16, 32]
        lrs = [0.001]
        epochs = 2000
        patience = 100
    elif dataset_size <= 500:  # medium dataset: RE-Free
        hidden_sizes = [32, 64]
        lrs = [0.001]
        epochs = 2000
        patience = 100
    else:  # large dataset: All
        hidden_sizes = [64, 128]
        lrs = [0.001]
        epochs = 2000
        patience = 100

    # Build the grid
    for hs in hidden_sizes:
        for lr in lrs:
            grid.append({
                "hidden_size": hs,
                "lr": lr,
                "epochs": epochs,
                "patience": patience
            })

    return grid


architectures = {
    "SimpleMLP": SimpleMLP,
    "SimplerMLP": SimplerMLP
}

# Compute metrics helper
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

def optimize_mlp(X, y, xlabel, ylabel, 
                 model, param_grid, plot_name):
    
    # Training loop with parameter grid
    param_grid = get_param_grid(X.shape[0])
    best_val_loss = float('inf')
    best_model_state = None
    best_arch_name = None
    best_params = None
    best_metrics = None

    for arch_name, ArchClass in architectures.items(): 
        
        print(f"\nBest architecture: {best_arch_name}")
        print(f"Best params: {best_params}")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Best metrics: {best_metrics}")
        
        for i, params in enumerate(param_grid):
    
            # -------------------------
            # Split Data
            # -------------------------
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32).view(-1,1)

            # -------------------------
            # Model, optimizer, loss
            # -------------------------
            if len(X_train.shape) == 1:
                X_train = X_train.view(-1, 1)
                X_val = X_val.view(-1, 1)
                
            model = ArchClass(input_size=X_train.shape[1], 
                              hidden_size=params["hidden_size"])
            
            optimizer = optim.Adam(model.parameters(), lr=params["lr"])
            criterion = nn.MSELoss()
            early_stopping = EarlyStopping(patience=params["patience"])

            # -------------------------
            # Training loop
            # -------------------------
            train_losses = []
            val_losses = []
            early_stop = False
            for epoch in range(params["epochs"]): # note: perhaps use mini-batching
                
                model.train()
                optimizer.zero_grad()
                outputs_train = model(X_train)

                loss_train = criterion(outputs_train.view(-1), y_train.view(-1))
                loss_train.backward()
                optimizer.step()

                # Validation loss
                model.eval()
                with torch.no_grad():
                    outputs_val = model(X_val)
                    loss_val = criterion(outputs_val.view(-1), y_val.view(-1)).item()

                # Store losses for plotting later
                train_losses.append(loss_train.item())
                val_losses.append(loss_val)
                
                # Track the best performing model on validation
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_arch_name = arch_name
                    best_params = params.copy()
                    test_metrics = compute_metrics(outputs_val.view(-1), y_val.view(-1))
                    best_metrics = test_metrics  
                    
                    # update in early stopper
                    early_stopping.best_loss = best_val_loss

                # Early stopping check
                early_stopping.step(loss_val)
                if early_stopping.should_stop:
                    print(f"Early stopping at epoch {epoch} for {arch_name}")
                    break
                    
                # Print progress
                # if epoch % 10 == 0 or early_stop:  # Print every 10 epochs
                print(f"Epoch [{epoch}/{params['epochs']}], Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val:.4f}")

            
    # -------------------------------------------------------
    # After all architectures & params are done:
    # Load and plot best model
    # -------------------------------------------------------
    print(f"\nBest architecture: {best_arch_name}")
    print(f"Best params: {best_params}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best metrics: {best_metrics}")
            
    # Rebuild best model
    best_model = architectures[best_arch_name](input_size=X_train.shape[1],
                                               hidden_size=best_params["hidden_size"])
    best_model.load_state_dict(best_model_state)
    best_model.eval()
            
    # Predictions on validation set
    val_preds = best_model(X_val)
    
    # Calculate metrics
    train_metrics = compute_metrics(y_train.detach().cpu().numpy(), outputs_train.detach().cpu().numpy())
    test_metrics = compute_metrics(y_val.detach().cpu().numpy(), val_preds.detach().cpu().numpy())
                                            
    # --------------------------------------------------
    # Compute residuals (prediction errors)
    # --------------------------------------------------
                
    residuals_train = (y_train - outputs_train).detach().cpu().numpy()
    threshold_train = 2 * np.std(residuals_train)  # Define threshold for outliers in val set
    
    residuals_val = (y_val - val_preds).detach().cpu().numpy()
    threshold_val = 2 * np.std(residuals_val)  # Define threshold for outliers in val set

    # Find outliers
    outliers_val_idx = np.where(np.abs(residuals_val) > threshold_val)[0]
    outliers_train_idx = np.where(np.abs(residuals_train) > threshold_train)[0]

    # ---------------------------------------------------------------------------
    # Create plots for training, validation sets, and Tc_exp vs Tc_sim
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Left plot: Train set
    axes[0].scatter(y_train.detach().cpu().numpy(), 
                    outputs_train.detach().cpu().numpy(), 
                    alpha=0.7, label="Train predictions")
    axes[0].scatter(y_train[outliers_train_idx].detach().cpu().numpy(), 
                    outputs_train[outliers_train_idx].detach().cpu().numpy(),
                    facecolors='none', edgecolors='red', s=100, label='Outliers')
    axes[0].plot([y_train.min(), y_train.max()], 
                        [y_train.min(), y_train.max()], 
                        color="black", linestyle="--", label="y=x")
    axes[0].set_xlabel("Ms_e (A/m)")
    axes[0].set_ylabel("Ms_e_hat (A/m)")
    axes[0].set_title("Train")
    axes[0].legend()
    axes[0].grid(True)

    # Create text box with validation metrics
    metrics_text_train = (
                            f"R2={train_metrics["R2"]:.3f}\n"
                            f"MAE={train_metrics["MAE"]:.3f}\n"
                            f"RMSE={train_metrics["RMSE"]:.3f}"
                        )

    # Add text box to validation plot
    axes[0].text(0.05, 0.75, metrics_text_train, transform=axes[0].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='grey', alpha=0.8))
                                    
    # Right plot: Validation set
    axes[1].scatter(y_val.detach().cpu().numpy(),
                    val_preds.detach().cpu().numpy(), alpha=0.7, 
                     color='orange',
                    label="Validation predictions")
    axes[1].scatter(y_val[outliers_val_idx].detach().cpu().numpy(), 
                    val_preds[outliers_val_idx].detach().cpu().numpy(),  color='orange',
                    facecolors='none', edgecolors='red', s=100, label='Outliers')
    
    axes[1].plot([y_val.min(), y_val.max()], 
                                     [y_val.min(), y_val.max()], 
                                     color="black", linestyle="--", label="y=x")
    axes[1].set_xlabel("Ms_e (A/m)")
    axes[1].set_ylabel("Ms_e_hat (A/m)")
    axes[1].set_title("Validation")
    axes[1].legend()
    axes[1].grid(True)

    # Create text box with validation metrics
    metrics_text_val = (f"R2={test_metrics["R2"]:.3f}\n"
                        f"MAE={test_metrics["MAE"]:.3f}\n"
                        f"RMSE={test_metrics["RMSE"]:.3f}")

    # Add text box to validation plot
    axes[1].text(0.05, 0.75, metrics_text_val, transform=axes[1].transAxes,
                 fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='grey', alpha=0.8))

    # Add a title for the whole figure
    fig.suptitle(f"Model Performance - {arch_name}")
    plt.savefig(plot_name + '.png')
    # plt.savefig(plot_name + '.pdf')
    # plt.show()
    plt.close()
                                          
    grid = {}
    grid['best_params'] = dict(model.named_modules()) 
            
    # Rebuild the best model architecture
    best_model = architectures[best_arch_name](input_size=X_train.shape[1], hidden_size=best_params["hidden_size"])
    best_model.load_state_dict(best_model_state)

    grid = {
                "best_architecture": best_arch_name,
                "best_params": best_params,
                "best_val_loss": best_val_loss,
                "best_metrics": best_metrics,       
                "trained_for_epochs": epoch
            }

    return best_model, {}, best_metrics, grid            