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
from base_trainer import DataLoader, ModelEvaluator, split_data, cross_val_report_fn


class MLP(nn.Module):
    """Multi-layer perceptron for regression."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64]):
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
        
    def _fit_torch(self, X_train_scaled, y_train, X_val_scaled, y_val,
                   input_dim, hidden_dims, batch_size, num_epochs, learning_rate,
                   ckpt_path, verbose=True):
        """Train an MLP on scaled arrays, early-stopping on the (scaled) val set.

        Saves the best-by-val-loss weights to ckpt_path, reloads them, and returns
        the fitted model. Both the single split and each CV fold go through here,
        so they share identical training logic.
        """
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)
        )
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = MLP(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15
        )

        if verbose:
            print("\nTraining...")
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 30

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
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load(ckpt_path))
        return model

    def _predict(self, model, X_scaled):
        """Predict (target space) from a fitted MLP on already-scaled inputs."""
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            return model(X_tensor).cpu().numpy()

    def train_and_evaluate(
        self,
        dataset_name: str,
        dataset_type: str,
        is_augmented: bool = False,
        use_embedding: bool = True,
        embedding_type: Optional[str] = 'mat200',
        hidden_dims: list = [256, 128, 64],
        batch_size: Optional[int] = None,
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
            batch_size: Training batch size. If None, auto-selected: 32 for
                training sets < 5000 samples, 256 otherwise.
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
        
        if batch_size is None:
            batch_size = 32 if len(X_train) < 5000 else 256

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        print(f"Hidden dimensions: {hidden_dims}")
        print(f"Batch size: {batch_size}")
        
        # Train (single split uses the held-out test set as the early-stopping
        # validation set, preserving the original behaviour).
        ckpt_path = self.output_dir / f"{dataset_name}_MLP_best.pt"
        model = self._fit_torch(
            X_train_scaled, y_train, X_test_scaled, y_test,
            input_dim=X_train.shape[1], hidden_dims=hidden_dims,
            batch_size=batch_size, num_epochs=num_epochs,
            learning_rate=learning_rate, ckpt_path=ckpt_path, verbose=True,
        )

        # Final predictions (scaled inputs)
        y_train_pred = self._predict(model, X_train_scaled)
        y_test_pred = self._predict(model, X_test_scaled)

        # Map predictions and targets back to Tc space (no-op unless delta_learning
        # is on) using the UNSCALED X so the Tc_sim baseline is correct.
        y_train_true = self.loader.reconstruct_target(y_train, X_train)
        y_test_true = self.loader.reconstruct_target(y_test, X_test)
        y_train_pred = self.loader.reconstruct_target(y_train_pred, X_train)
        y_test_pred = self.loader.reconstruct_target(y_test_pred, X_test)

        # Compute metrics
        train_metrics = self.evaluator.compute_metrics(y_train_true, y_train_pred)
        test_metrics = self.evaluator.compute_metrics(y_test_true, y_test_pred)

        print(f"\nSingle-split Test Metrics:")
        print(f"  R² = {test_metrics['R2']:.4f}")
        print(f"  RMSE = {test_metrics['RMSE']:.2f} K")
        print(f"  MAE = {test_metrics['MAE']:.2f} K")

        # --- ONNX export (only the deployable raw_200D embedding variant) ---
        try:
            from onnx_export import maybe_export_onnx
            maybe_export_onnx(
                family="mlp", model=model, scaler=scaler, input_dim=X_train.shape[1],
                dataset_name=dataset_name, use_embedding=use_embedding,
                embedding_type=embedding_type, loader=self.loader,
                aug_label=getattr(self.evaluator, "figures_subdir", None),
                output_dir=self.output_dir,
            )
        except Exception as _onnx_exc:
            print(f"    ONNX export skipped/failed: {_onnx_exc}")

        # Optional K-fold CV reporting. Each fold scales on its own train data,
        # carves a validation split from the fold's TRAIN for early stopping (so
        # the held-out fold stays unseen), and trains a fresh MLP via _fit_torch.
        cv_folds = getattr(self.loader, 'cv_folds', 0)
        cv = None
        if cv_folds and cv_folds >= 2:
            cv_ckpt = self.output_dir / f"{dataset_name}_MLP_cvfold.pt"

            def _fit_predict(X_tr, y_tr, X_te):
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)
                Xt, Xv, yt, yv = split_data(X_tr_s, y_tr)
                bs = batch_size if batch_size is not None else (32 if len(Xt) < 5000 else 256)
                m = self._fit_torch(
                    Xt, yt, Xv, yv, input_dim=X_tr.shape[1], hidden_dims=hidden_dims,
                    batch_size=bs, num_epochs=num_epochs, learning_rate=learning_rate,
                    ckpt_path=cv_ckpt, verbose=False,
                )
                return self._predict(m, X_te_s)

            cv = cross_val_report_fn(_fit_predict, X, y, self.loader, n_splits=cv_folds)
            print(f"\n{cv_folds}-fold CV Metrics (headline):")
            print(f"  R²   = {cv['R2']:.4f} ± {cv['R2_std']:.4f}")
            print(f"  RMSE = {cv['RMSE']:.2f} ± {cv['RMSE_std']:.2f} K")
            print(f"  MAE  = {cv['MAE']:.2f} ± {cv['MAE_std']:.2f} K")

        # Plot results
        emb_suffix = f"_{embedding_type}" if use_embedding else "_no_emb"
        output_path = self.output_dir / f"{dataset_name}_MLP{emb_suffix}.png"
        self.evaluator.plot_predictions(
            y_train_true, y_train_pred,
            y_test_true, y_test_pred,
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
            f.write(f"Single-split R2: {test_metrics['R2']:.4f}\n")
            if cv:
                f.write(f"CV folds: {cv_folds}\n")
                f.write(f"CV R2: {cv['R2']:.4f} +/- {cv['R2_std']:.4f}\n")
                f.write(f"CV RMSE: {cv['RMSE']:.2f} +/- {cv['RMSE_std']:.2f}\n")
                f.write(f"CV MAE: {cv['MAE']:.2f} +/- {cv['MAE_std']:.2f}\n")

        # CV means become the reported (headline) metrics when CV is on.
        reported = {'R2': cv['R2'], 'RMSE': cv['RMSE'], 'MAE': cv['MAE']} if cv else test_metrics
        result = {
            'R2': reported['R2'],
            'RMSE': reported['RMSE'],
            'MAE': reported['MAE'],
            'model': model,
            'scaler': scaler
        }
        if cv:
            result.update({'R2_std': cv['R2_std'], 'RMSE_std': cv['RMSE_std'],
                           'MAE_std': cv['MAE_std'], 'cv_folds': cv_folds})
        return result


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
