"""
Fully Connected Neural Network (FCNN/MLP) Training with PyTorch
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from base_trainer import DataLoader, ModelEvaluator, split_data


class MLP(nn.Module):
    """Multi-layer perceptron for regression."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32]):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


class FCNNTrainer:
    """Train and evaluate FCNN/MLP models."""
    
    def __init__(self, output_dir: str = "results/fcnn"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DataLoader()
        self.evaluator = ModelEvaluator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def train_and_evaluate(
        self,
        dataset_name: str,
        dataset_type: str,
        is_augmented: bool = False,
        use_embedding: bool = True,
        embedding_type: Optional[str] = 'mat200',
        hidden_dims: list = [128, 64, 32],
        batch_size: int = 32,
        num_epochs: int = 200,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Train FCNN/MLP model.
        
        Args:
            dataset_name: Name for output files
            dataset_type: 'all', 're', or 're-free'
            is_augmented: Whether to use augmented data
            use_embedding: Whether to use embeddings (FCNN works best with embeddings)
            embedding_type: Type of embedding
            hidden_dims: List of hidden layer dimensions
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Dictionary with test metrics
        """
        print(f"\n{'='*60}")
        print(f"Training FCNN/MLP: {dataset_name}")
        if use_embedding:
            print(f"Using embedding: {embedding_type}")
        else:
            print("Using only Tc_sim (no embedding)")
        print(f"{'='*60}")
        
        # Load data
        if is_augmented:
            df = self.loader.load_augmented_data(dataset_type)
        else:
            df = self.loader.load_pairs_data(dataset_type)
        
        # Prepare dataset
        X, y = self.loader.prepare_dataset(df, dataset_type, use_embedding, embedding_type)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        print(f"Hidden dimensions: {hidden_dims}")
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test)
        )
        
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        print("\nTraining...")
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(test_loader)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 
                          self.output_dir / f"{dataset_name}_MLP_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(
            torch.load(self.output_dir / f"{dataset_name}_MLP_best.pt")
        )
        
        # Final predictions
        model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
            
            y_train_pred = model(X_train_tensor).cpu().numpy()
            y_test_pred = model(X_test_tensor).cpu().numpy()
        
        # Compute metrics
        train_metrics = self.evaluator.compute_metrics(y_train, y_train_pred)
        test_metrics = self.evaluator.compute_metrics(y_test, y_test_pred)
        
        print(f"\nFinal Test Metrics:")
        print(f"  R² = {test_metrics['R2']:.4f}")
        print(f"  RMSE = {test_metrics['RMSE']:.2f} K")
        print(f"  MAE = {test_metrics['MAE']:.2f} K")
        
        # Plot results
        emb_suffix = f"_{embedding_type}" if use_embedding else "_no_emb"
        output_path = self.output_dir / f"{dataset_name}_MLP{emb_suffix}.png"
        self.evaluator.plot_predictions(
            y_train, y_train_pred,
            y_test, y_test_pred,
            title=f"FCNN/MLP - {dataset_name}" + 
                  (f" ({embedding_type})" if use_embedding else ""),
            output_path=str(output_path)
        )
        
        # Save model info
        info_path = self.output_dir / f"{dataset_name}_MLP{emb_suffix}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Embedding: {embedding_type if use_embedding else 'None'}\n")
            f.write(f"Hidden dims: {hidden_dims}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Learning rate: {learning_rate}\n")
            f.write(f"R2: {test_metrics['R2']:.4f}\n")
            f.write(f"RMSE: {test_metrics['RMSE']:.2f}\n")
            f.write(f"MAE: {test_metrics['MAE']:.2f}\n")
        
        return {
            'R2': test_metrics['R2'],
            'RMSE': test_metrics['RMSE'],
            'MAE': test_metrics['MAE'],
            'model': model,
            'scaler': scaler
        }


def main():
    """Run FCNN/MLP for all dataset configurations."""
    trainer = FCNNTrainer()
    
    # Define configurations (MLP typically works best with embeddings)
    configs = [
        # Pairs with embeddings
        ('All-Pairs', 'all', False, True, 'pca_32'),
        ('RE-Pairs', 're', False, True, 'kpca_30'),
        ('RE-free-Pairs', 're-free', False, True, 'pca_16'),
        
        # Augmented with embeddings (mat200 according to table)
        ('All-Augm', 'all', True, True, 'mat200'),
        ('RE-Augm', 're', True, True, 'mat200'),
        ('RE-free-Augm', 're-free', True, True, 'mat200'),
    ]
    
    all_results = []
    
    for dataset_name, dataset_type, is_augmented, use_embedding, embedding_type in configs:
        try:
            result = trainer.train_and_evaluate(
                dataset_name, dataset_type, is_augmented,
                use_embedding, embedding_type
            )
            
            all_results.append({
                'Dataset': dataset_name,
                'Embedding': embedding_type,
                'R2': result['R2'],
                'RMSE': result['RMSE'],
                'MAE': result['MAE']
            })
        except Exception as e:
            print(f"Error training {dataset_name} with {embedding_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(trainer.output_dir / "fcnn_summary.csv", index=False)
    
    print("\n" + "="*80)
    print("FCNN/MLP Summary:")
    print("="*80)
    print(summary_df.to_string())
    
    return all_results


if __name__ == "__main__":
    main()
