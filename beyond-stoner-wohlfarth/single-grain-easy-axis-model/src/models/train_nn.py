"""
Neural network model training module.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

def train_neural_network(X, y, **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a neural network model.
    
    Args:
        X: Input features
        y: Target values
        **kwargs: Additional parameters including:
            - hidden_dim: Number of units in hidden layer
            - num_epochs: Number of training epochs
            - learning_rate: Learning rate for optimizer
            - batch_size: Batch size for training
            - verbose: Whether to print progress
    
    Returns:
        Tuple containing:
            - Trained neural network model
            - Dictionary of best parameters
    """
    # Extract parameters from kwargs
    hidden_dim = kwargs.get('hidden_dim', 64)
    num_epochs = kwargs.get('num_epochs', 50)
    learning_rate = kwargs.get('learning_rate', 1e-3)
    batch_size = kwargs.get('batch_size', 32)
    verbose = kwargs.get('verbose', True)
    
    # Ensure parameters are of the correct type
    hidden_dim = int(hidden_dim) if hidden_dim is not None else 64
    num_epochs = int(num_epochs) if num_epochs is not None else 50
    learning_rate = float(learning_rate) if learning_rate is not None else 1e-3
    batch_size = int(batch_size) if batch_size is not None else 32
    
    # Get param grid if provided for consistency with other models
    param_grid = kwargs.get('param_grid', {})
    
    # Use param_grid values if provided (overrides defaults)
    if 'hidden_dim' in param_grid:
        hidden_dim = int(param_grid['hidden_dim']) if param_grid['hidden_dim'] is not None else 64
    if 'num_epochs' in param_grid:
        num_epochs = int(param_grid['num_epochs']) if param_grid['num_epochs'] is not None else 50
    if 'learning_rate' in param_grid:
        learning_rate = float(param_grid['learning_rate']) if param_grid['learning_rate'] is not None else 1e-3
    if 'batch_size' in param_grid:
        batch_size = param_grid['batch_size']
    
    # Create a simple train/validation split for monitoring
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model, y_train_pred, y_val_pred = train_pytorch_nn(
        X_train, y_train, X_val, y_val,
        hidden_dim=hidden_dim,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Return the best parameters for consistency with other models
    best_params = {
        'hidden_dim': hidden_dim,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }
    
    return model, best_params


def train_pytorch_nn(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    input_dim=None, 
    hidden_dim=64, 
    num_epochs=50, 
    learning_rate=1e-3, 
    batch_size=32,
    verbose=True
):
    # Ensure parameters are of the correct type
    hidden_dim = int(hidden_dim) if hidden_dim is not None else 64
    num_epochs = int(num_epochs) if num_epochs is not None else 50
    learning_rate = float(learning_rate) if learning_rate is not None else 1e-3
    batch_size = int(batch_size) if batch_size is not None else 32
    """
    Trains a simple feedforward neural network (PyTorch) for regression.
    Returns the trained model and predictions for train/test sets.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n_samples, n_features)
    y_train : np.ndarray
        Training targets, shape (n_samples, n_outputs)
    X_test  : np.ndarray
        Testing features, shape (m_samples, n_features)
    y_test  : np.ndarray
        Testing targets, shape (m_samples, n_outputs)
    input_dim : int, optional
        Dimensionality of input features; if None, inferred from X_train.shape[1].
    hidden_dim : int
        Number of units in the hidden layer.
    num_epochs : int
        Number of training epochs.
    learning_rate : float
        Optimizer learning rate.
    batch_size : int
        Batch size for mini-batch gradient descent.
    verbose : bool
        Whether to print epoch progress.
    
    Returns
    -------
    model : nn.Module
        Trained PyTorch model.
    y_train_pred : np.ndarray
        Predictions on X_train (shape = (n_samples, n_outputs)).
    y_test_pred : np.ndarray
        Predictions on X_test (shape = (m_samples, n_outputs)).
    """
    # Convert data to PyTorch tensors
    # First convert DataFrames to numpy arrays if needed
    X_train_np = X_train.to_numpy() if hasattr(X_train, 'to_numpy') else X_train
    y_train_np = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train
    X_test_np = X_test.to_numpy() if hasattr(X_test, 'to_numpy') else X_test
    y_test_np = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else y_test
    
    # Create PyTorch tensors from numpy arrays
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_np, dtype=torch.float32)
    
    # Infer dimensions if not provided
    if input_dim is None:
        input_dim = X_train_np.shape[1]
    
    # Handle output dimension properly
    if len(y_train_np.shape) > 1:
        output_dim = y_train_np.shape[1]
    else:
        output_dim = 1
        
    # Ensure dimensions are integers, not tuples or other types
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    # Simple Feedforward Neural Network
    class Net(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(Net, self).__init__()
            # Ensure all dimensions are integers
            input_dim = int(input_dim)
            hidden_dim = int(hidden_dim)
            output_dim = int(output_dim)
            
            # Create the network layers
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
            #self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
            #self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            return x
        
        def predict(self, X):
            """Predict method to make the model compatible with scikit-learn API
            
            Args:
                X: Input features (numpy array or pandas DataFrame)
                
            Returns:
                Numpy array of predictions
            """
            # Convert to numpy if it's a DataFrame
            if hasattr(X, 'to_numpy'):
                X = X.to_numpy()
                
            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
            # Set model to evaluation mode
            self.eval()
            
            # Make predictions
            with torch.no_grad():
                predictions = self.forward(X_tensor)
                
            # Convert to numpy array
            return predictions.numpy()

    
    # Instantiate the model
    model = Net(input_dim, hidden_dim, output_dim)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create DataLoader for mini-batch training
    # Reshape y_train_t if it's a 1D tensor
    if len(y_train_t.shape) == 1:
        y_train_t = y_train_t.view(-1, 1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if verbose and (epoch+1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    # Evaluation: Predictions
    model.eval()
    with torch.no_grad():
        y_train_pred_t = model(X_train_t)
        y_test_pred_t  = model(X_test_t)
        
        # Reshape predictions if needed
        if output_dim == 1 and len(y_train_pred_t.shape) > 1:
            y_train_pred_t = y_train_pred_t.squeeze()
        if output_dim == 1 and len(y_test_pred_t.shape) > 1:
            y_test_pred_t = y_test_pred_t.squeeze()
    
    # Convert back to numpy arrays
    y_train_pred = y_train_pred_t.cpu().numpy()
    y_test_pred  = y_test_pred_t.cpu().numpy()
    
    # Ensure the output shape matches the input shape for consistency
    if len(y_train_np.shape) > 1 and len(y_train_pred.shape) == 1:
        y_train_pred = y_train_pred.reshape(-1, 1)
    if len(y_test_np.shape) > 1 and len(y_test_pred.shape) == 1:
        y_test_pred = y_test_pred.reshape(-1, 1)

    return model, y_train_pred, y_test_pred


def calculate_feature_importance(model, X):
    """
    Calculate feature importance for neural network using sensitivity analysis.
    
    Args:
        model: Trained neural network model
        X: Input features
    
    Returns:
        Array of feature importance scores
    """
    # Convert to PyTorch tensor if not already
    if not isinstance(X, torch.Tensor):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X
    
    # Put model in eval mode
    model.eval()
    
    # Get baseline predictions
    with torch.no_grad():
        baseline_preds = model(X_tensor)
    
    # Calculate importance for each feature
    importances = []
    for i in range(X.shape[1]):
        # Create perturbed input
        X_perturbed = X_tensor.clone()
        X_perturbed[:, i] = X_perturbed[:, i] * 1.1  # Perturb by 10%
        
        # Get predictions for perturbed input
        with torch.no_grad():
            perturbed_preds = model(X_perturbed)
        
        # Calculate change in output
        importance = torch.mean(torch.abs(perturbed_preds - baseline_preds)).item()
        importances.append(importance)
    
    # Normalize importances
    importances = np.array(importances)
    if np.sum(importances) > 0:
        importances = importances / np.sum(importances)
    
    return importances


def plot_feature_importance(importances, X, results_dir='results', dataset_name='unknown', scaler_type='unknown', title=None):
    """
    Plot feature importance for neural network model and save it to the results directory.
    
    Args:
        importances: Feature importance scores
        X: Input features used for training
        results_dir: Directory to save the plot
        dataset_name: Name of the dataset for the plot title and filename
        scaler_type: Type of scaler used for the plot title and filename
        title: Optional custom title for the plot
    
    Returns:
        Dictionary of feature importances
    """
    try:
        # Get feature names
        feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importances)), importances[indices], align='center')
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature Name")
        
        # Use provided title or generate one based on dataset and scaler
        if title is None:
            title = f"Neural Network Feature Importance ({dataset_name}, {scaler_type})"
        plt.title(title)
        plt.gca().invert_yaxis()  # Most important at the top
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)
        
        # Create plots subdirectory
        plots_dir = results_path / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Save the plot
        filename = f"nn_feature_importance_{dataset_name}_{scaler_type}.png"
        filepath = plots_dir / filename
        plt.savefig(filepath)
        plt.close()
        
        print(f"Neural network feature importance plot saved to {filepath}")
        
        # Return the sorted feature importances as a dictionary
        importance_dict = {str(feature_names[i]): float(importances[i]) for i in indices}
        return importance_dict
    
    except Exception as e:
        print(f"Warning: Could not plot neural network feature importance: {str(e)}")
        return None
