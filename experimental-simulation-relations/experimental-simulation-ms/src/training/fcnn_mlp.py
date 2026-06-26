"""
Fully Connected Neural Network (FCNN/MLP) Training with PyTorch
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from base_trainer import DataLoader, ModelEvaluator, split_data

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Stub classes used when torch is absent ───────────────────────────────────
class MLP:
    """Placeholder — replaced below when torch is available."""
    pass


class FCNNTrainer:
    """Placeholder — replaced below when torch is available."""
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "PyTorch is not installed. Install torch to use FCNNTrainer."
        )


# ── Real implementations — only defined when torch is available ──────────────
if TORCH_AVAILABLE:

    class MLP(nn.Module):
        """Multi-layer perceptron for regression."""

        def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.0):
            super(MLP, self).__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
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
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True

        def train_and_evaluate(
            self,
            dataset_name: str,
            dataset_type: str,
            is_augmented: bool = False,
            use_embedding: bool = False,
            embedding_type: Optional[str] = None,
            hidden_dims: Optional[list] = None,
            batch_size: Optional[int] = None,
            num_epochs: int = 2000,
            learning_rate: float = 0.001,
        ) -> Dict:
            """Train FCNN/MLP model. Training and metrics are in log1p space.

            hidden_dims and batch_size are auto-sized from the training set when
            not provided. hidden_size scales as min(128, max(16, n_train // 50)),
            giving ~16 neurons for ~800 samples up to 128 for ~6400+ samples.
            """
            print(f"\n{'='*60}")
            print(f"Training FCNN/MLP: {dataset_name}")
            if use_embedding:
                print(f"Using embedding: {embedding_type}")
            else:
                print("Using only Ms_sim (no embedding)")
            print(f"{'='*60}")

            if is_augmented:
                df = self.loader.load_augmented_data(dataset_type)
            else:
                df = self.loader.load_pairs_data(dataset_type)

            X, y = self.loader.prepare_dataset(df, dataset_type, use_embedding, embedding_type)
            X_train, X_test, y_train, y_test = split_data(X, y)

            n_train = len(X_train)

            # Auto-size architecture to the training set
            if hidden_dims is None:
                hidden_size = min(128, max(16, n_train // 50))
                hidden_dims = [hidden_size, hidden_size]

            # Auto-size batch: ~10 batches per epoch for small sets, cap at 256
            if batch_size is None:
                batch_size = min(256, max(16, n_train // 10))

            print(f"Training samples: {n_train}")
            print(f"Test samples: {len(X_test)}")
            print(f"Feature dimensions: {X_train.shape[1]}")
            print(f"Hidden dimensions: {hidden_dims} (auto)")
            print(f"Batch size: {batch_size} (auto)")

            train_dataset = TensorDataset(
                torch.FloatTensor(X_train), torch.FloatTensor(y_train)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test), torch.FloatTensor(y_test)
            )
            pin = self.device.type == 'cuda'
            train_loader = TorchDataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin,
            )
            test_loader = TorchDataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin,
            )

            model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=30
            )

            print("\nTraining in log1p space...")
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 100
            best_model_path = self.output_dir / f"{dataset_name}_MLP_best.pt"

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

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()
                val_loss /= len(test_loader)
                scheduler.step(val_loss)

                if (epoch + 1) % 20 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            model.load_state_dict(torch.load(best_model_path, weights_only=True))

            model.eval()
            with torch.no_grad():
                y_train_pred_log = model(
                    torch.FloatTensor(X_train).to(self.device)
                ).cpu().numpy()
                y_test_pred_log = model(
                    torch.FloatTensor(X_test).to(self.device)
                ).cpu().numpy()

            # Map predictions and targets back to log1p(Ms_exp) space (no-op
            # unless delta_learning is on) so metrics stay comparable across runs.
            y_train_true = self.loader.reconstruct_log_exp(y_train, X_train)
            y_test_true = self.loader.reconstruct_log_exp(y_test, X_test)
            y_train_pred_log = self.loader.reconstruct_log_exp(y_train_pred_log, X_train)
            y_test_pred_log = self.loader.reconstruct_log_exp(y_test_pred_log, X_test)

            train_metrics = self.evaluator.compute_metrics(y_train_true, y_train_pred_log)
            test_metrics = self.evaluator.compute_metrics(y_test_true, y_test_pred_log)

            print(f"\nFinal Test Metrics (log1p space):")
            print(f"  R² = {test_metrics['R2']:.4f}")
            print(f"  RMSE = {test_metrics['RMSE']:.4f}")
            print(f"  MAE = {test_metrics['MAE']:.4f}")

            emb_suffix = f"_{embedding_type}" if use_embedding else "_no_emb"
            output_path = self.output_dir / f"{dataset_name}_MLP{emb_suffix}.png"
            self.evaluator.plot_predictions(
                y_train_true, y_train_pred_log,
                y_test_true, y_test_pred_log,
                title=f"FCNN/MLP - {dataset_name}"
                      + (f" ({embedding_type})" if use_embedding else ""),
                output_path=str(output_path),
            )

            info_path = self.output_dir / f"{dataset_name}_MLP{emb_suffix}_info.txt"
            with open(info_path, 'w') as f:
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Embedding: {embedding_type if use_embedding else 'None'}\n")
                f.write(f"Hidden dims: {hidden_dims}\n")
                f.write(f"Batch size: {batch_size}\n")
                f.write(f"Learning rate: {learning_rate}\n")
                f.write(f"R2: {test_metrics['R2']:.4f}\n")
                f.write(f"RMSE: {test_metrics['RMSE']:.4f}\n")
                f.write(f"MAE: {test_metrics['MAE']:.4f}\n")

            return {
                'R2': test_metrics['R2'],
                'RMSE': test_metrics['RMSE'],
                'MAE': test_metrics['MAE'],
                'model': model,
            }


def main():
    """Run FCNN/MLP for all dataset configurations."""
    trainer = FCNNTrainer()

    configs = [
        ('All-Pairs', 'all', False, False, None),
        ('RE-Pairs', 're', False, False, None),
        ('RE-Free-Pairs', 're-free', False, False, None),
    ]

    all_results = []

    for dataset_name, dataset_type, is_augmented, use_embedding, embedding_type in configs:
        try:
            result = trainer.train_and_evaluate(
                dataset_name, dataset_type, is_augmented, use_embedding, embedding_type,
            )
            all_results.append({
                'Dataset': dataset_name,
                'Embedding': embedding_type if use_embedding else 'None',
                'R2': result['R2'],
                'RMSE': result['RMSE'],
                'MAE': result['MAE'],
            })
        except Exception as e:
            print(f"Error training {dataset_name} with {embedding_type}: {e}")
            import traceback
            traceback.print_exc()

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(trainer.output_dir / "fcnn_summary.csv", index=False)
    print("\n" + "="*80)
    print("FCNN/MLP Summary:")
    print("="*80)
    print(summary_df.to_string())
    return all_results


if __name__ == "__main__":
    main()
