"""
Base training utilities for Curie Temperature Error Correction models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Tuple, Dict, Optional

sns.set_style("whitegrid")


class DataLoader:
    """Load and prepare datasets for training."""
    
    def __init__(self, data_dir: str = "../../data", pairs_file: str = "Pairs_all.csv", 
                 augmented_file: str = "Augm_all.csv", re_pairs_file: str = "Pairs_RE.csv", 
                 re_free_pairs_file: str = "Pairs_RE_Free.csv", re_augmented_file: str = "Augm_RE.csv", 
                 re_free_augmented_file: str = "Augm_RE_Free.csv"):
        # Interpret data_dir relative to this file's directory, not the CWD
        base_dir = Path(__file__).parent
        self.data_dir = (base_dir / data_dir).resolve()
        
        # Store dataset file names
        self.pairs_file = pairs_file
        self.augmented_file = augmented_file
        self.re_pairs_file = re_pairs_file
        self.re_free_pairs_file = re_free_pairs_file
        self.re_augmented_file = re_augmented_file
        self.re_free_augmented_file = re_free_augmented_file
        
    def load_pairs_data(self, dataset_type: str = "all") -> pd.DataFrame:
        """Load the original pairs dataset.

        This must be the MammoS-style CSV created by augment_data.py.
        
        Args:
            dataset_type: 'all', 're', or 're-free' to specify which dataset to load
        """
        # Select the appropriate file based on dataset_type
        if dataset_type == 'all':
            filename = self.pairs_file
        elif dataset_type == 're':
            filename = self.re_pairs_file
        elif dataset_type == 're-free':
            filename = self.re_free_pairs_file
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'all', 're', or 're-free'.")
            
        # Use project root outputs directory
        project_root = Path(__file__).parent.parent.parent
        df_path = project_root / "outputs" / filename

        if not df_path.exists():
            raise FileNotFoundError(
                f"Required input not found: {df_path}. "
                "Please run augment_data.py to generate it."
            )

        print(f"Loading original (pairs) data for {dataset_type} dataset from: {df_path}")

        # MammoS-style CSV has a 4-line header; skip those rows
        df = pd.read_csv(df_path, skiprows=4, engine="python")

        # Filter for pairs only
        #if 'pair_exists' in df.columns:
        #    df = df[df['pair_exists'] == True].copy()

        return df
    
    def load_augmented_data(self, dataset_type: str = "all") -> pd.DataFrame:
        """Load the augmented dataset.

        This file is MammoS-style with a 4-line header and an extra info row.
        
        Args:
            dataset_type: 'all', 're', or 're-free' to specify which dataset to load
        """
        # Select the appropriate file based on dataset_type
        if dataset_type == 'all':
            filename = self.augmented_file
        elif dataset_type == 're':
            filename = self.re_augmented_file
        elif dataset_type == 're-free':
            filename = self.re_free_augmented_file
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'all', 're', or 're-free'.")
            
        # Use project root outputs directory
        project_root = Path(__file__).parent.parent.parent
        df_path = project_root / "outputs" / filename

        if not df_path.exists():
            raise FileNotFoundError(
                f"Required input not found: {df_path}. "
                "Please run augment_data.py to generate it."
            )

        print(f"Loading augmented data for {dataset_type} dataset from: {df_path}")

        # MammoS-style CSV: skip 4-line header, then drop the info row
        df = pd.read_csv(df_path, skiprows=4, engine="python")

        if 'info' in df.columns:
            # Drop the info row which contains the comment
            # The info starts with "# Info:" and appears in the first row
            df = df[~df['info'].astype(str).str.startswith('# Info:')].copy()
            df = df.drop(columns=['info'])
            print(f"  Removed info/comment row from augmented data")

        return df
    
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        use_embedding: bool = False,
        embedding_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare X and y for training.
        
        Args:
            df: DataFrame with the data
            dataset_type: 'all', 're', or 're-free'
            use_embedding: Whether to use compound embeddings
            embedding_type: Type of embedding (e.g., 'pca_32', 'mat200', 'kpca_30')
            
        Returns:
            X, y arrays
        """
        '''
        # Filter by dataset type and select target column - always use Tc_exp as target
        if dataset_type == 're':
            df = df[df['contains_rare_earth'] == True].copy()
            #target_col = 'Tc_exp' if 'Tc_exp' in df.columns else 'Tc_exp_re_w_mock'
            target_col = 'Tc_exp'
        elif dataset_type == 're-free':
            df = df[df['contains_rare_earth'] == False].copy()
            #target_col = 'Tc_exp' if 'Tc_exp' in df.columns else 'Tc_exp_re_free_w_mock'
            target_col = 'Tc_exp'
        else:  # 'all'
            #target_col = 'Tc_exp' # if 'Tc_exp' in df.columns else 'Tc_exp_all_w_mock'
            target_col = 'Tc_exp'
        '''
        # always use Tc_exp as target
        target_col = 'Tc_exp'

        print(f"\nDataset diagnostics:")
        print(f"  - Dataset type: {dataset_type}")
        print(f"  - Target column: {target_col}")
        print(f"  - Total rows: {len(df)}")
        
        # Check for NaNs in individual columns
        if 'Tc_sim' not in df.columns:
            raise ValueError("Expected column 'Tc_sim' not found in dataframe.")
        
        sim_nans = df['Tc_sim'].isna().sum()
        target_nans = df[target_col].isna().sum()
        print(f"  - Rows with NaN in Tc_sim: {sim_nans}")
        print(f"  - Rows with NaN in {target_col}: {target_nans}")
        
        # Get all columns with 'exp' in the name
        #exp_cols = [col for col in df.columns if 'exp' in col.lower()]
        #print(f"  - Available exp columns: {exp_cols}")
        #for col in exp_cols:
        #    nan_count = df[col].isna().sum()
        #    print(f"    {col}: {nan_count} NaNs ({nan_count/len(df):.1%})")

        # Only drop rows with NaNs in Tc_sim or target column, keep all other rows even if they have NaNs
        # in other columns as requested
        before_len = len(df)
        df = df[df['Tc_sim'].notna() & df[target_col].notna()].copy()
        if len(df) != before_len:
            print(
                f"Dropped {before_len - len(df)} rows with NaN in Tc_sim or {target_col} "
                f"(remaining: {len(df)})"
            )
            print(f"Note: Rows with NaNs in other columns are preserved as requested.")

        y = df[target_col].values
        
        # Prepare X
        if use_embedding:
            if embedding_type is None:
                # Use raw compound_embedding
                if 'compound_embedding' in df.columns:
                    X = np.vstack(df['compound_embedding'].values)
                else:
                    raise ValueError("No compound_embedding column found. Run create_embeddings.py first.")
            else:
                # Use specific embedding type
                col_name = f'comp_emb_{embedding_type}_components'
                if col_name in df.columns:
                    X = np.vstack(df[col_name].values)
                else:
                    # Try without prefix
                    if embedding_type in df.columns:
                        X = np.vstack(df[embedding_type].values)
                    elif 'comp_emp' in df.columns:
                        # Fall back to raw embedding with warning
                        print(f"Warning: {col_name} not found, using raw compound_embedding")
                        X = np.vstack(df['comp_emb'].values)
                    else:
                        raise ValueError(
                            f"Embedding column {col_name} not found. "
                            "Available columns with 'embedding': " +
                            str([c for c in df.columns if 'emb' in c.lower()])
                        )
            
            # Stack with Tc_sim
            Tc_sim = df['Tc_sim'].values.reshape(-1, 1)
            X = np.hstack([X, Tc_sim])
        else:
            # Just use Tc_sim
            X = df['Tc_sim'].values.reshape(-1, 1)
        
        return X, y


class ModelEvaluator:
    """Evaluate and plot model results."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute R2, MAE, and RMSE."""
        return {
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    @staticmethod
    def plot_predictions(
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        title: str,
        output_path: Optional[str] = None,
        equation: Optional[str] = None
    ):
        """
        Create scatter plots of true vs predicted values.
        
        Args:
            y_train: True training values
            y_train_pred: Predicted training values
            y_test: True test values
            y_test_pred: Predicted test values
            title: Plot title
            output_path: Path to save the figure
            equation: Optional equation to display
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Compute metrics
        train_metrics = ModelEvaluator.compute_metrics(y_train, y_train_pred)
        test_metrics = ModelEvaluator.compute_metrics(y_test, y_test_pred)
        
        # Remove parentheses from equation if present
        clean_equation = None
        if equation:
            clean_equation = equation.replace('(', '').replace(')', '')
        
        # Identify outliers (points more than 2 standard deviations away from perfect prediction)
        def identify_outliers(y_true, y_pred):
            residuals = y_true - y_pred
            std_residuals = np.std(residuals)
            is_outlier = np.abs(residuals) > 2 * std_residuals
            return is_outlier
        
        train_outliers = identify_outliers(y_train, y_train_pred)
        test_outliers = identify_outliers(y_test, y_test_pred)
        
        # Generate points for the equation line (red line)
        if 'SR' in title or 'Symbolic' in title:  # Only for symbolic regression plots
            x_range = np.linspace(min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max()), 100)
            
            # Simple linear fit as fallback if we can't evaluate the equation
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(y_train.reshape(-1, 1), y_train_pred)
            y_line = model.predict(x_range.reshape(-1, 1))
            
            # Training plot - Add the equation line in red
            axes[0].plot(x_range, y_line, 'r-', lw=2, label='Best fit equation')
        
        # Training plot - regular points
        #axes[0].scatter(y_train[~train_outliers], y_train_pred[~train_outliers], 
        #              alpha=0.5, label='Train prediction

        axes[0].scatter(y_train, y_train_pred,
                      alpha=0.5, label='Train prediction')
        
        # Training plot - outlier points with circle markers
        if np.any(train_outliers):
            axes[0].scatter(y_train[train_outliers], y_train_pred[train_outliers], 
                         alpha=0.7, edgecolors='red', facecolors='none', s=80, 
                         linewidths=1.5, label='Outliers')
        
        # Identity line
        axes[0].plot([y_train.min(), y_train.max()], 
                     [y_train.min(), y_train.max()],
                     'k--', lw=2, label='y=x')
        
        axes[0].set_xlabel('Tc_exp (K)', fontsize=12)
        axes[0].set_ylabel('Tc_exp_pred (K)', fontsize=12)
        axes[0].set_title('Training Set', fontsize=14)
        axes[0].legend()
        axes[0].grid(True)
        
        metrics_text_train = (
            f"R² = {train_metrics['R2']:.4f}\n"
            f"MAE = {train_metrics['MAE']:.2f} K\n"
            f"RMSE = {train_metrics['RMSE']:.2f} K"
        )
        if clean_equation:
            metrics_text_train = f"Eq: {clean_equation}\n" + metrics_text_train
        
        axes[0].text(0.05, 0.95, metrics_text_train, 
                    transform=axes[0].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Test plot - regular points
        axes[1].scatter(y_test, y_test_pred,
                      alpha=0.7, color='orange', label='Test predictions')
        
        # Test plot - outlier points with circle markers
        if np.any(test_outliers):
            axes[1].scatter(y_test[test_outliers], y_test_pred[test_outliers], 
                         alpha=0.7, edgecolors='red', facecolors='none', s=80, 
                         linewidths=1.5, label='Outliers')
            
        # Test plot - Add the equation line in red for Symbolic Regression plots
        if 'SR' in title or 'Symbolic' in title:  # Only for symbolic regression plots
            axes[1].plot(x_range, y_line, 'r-', lw=2, label='Best fit equation')
        
        # Identity line
        axes[1].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()],
                     'k--', lw=2, label='y=x')
                     
        axes[1].set_xlabel('Tc_exp (K)', fontsize=12)
        axes[1].set_ylabel('Tc_exp_pred (K)', fontsize=12)
        axes[1].set_title('Test Set', fontsize=14)
        axes[1].legend()
        axes[1].grid(True)
        
        metrics_text_test = (
            f"R² = {test_metrics['R2']:.4f}\n"
            f"MAE = {test_metrics['MAE']:.2f} K\n"
            f"RMSE = {test_metrics['RMSE']:.2f} K"
        )
        axes[1].text(0.05, 0.95, metrics_text_test, 
                    transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Create output directory in project root /results/figures/ if it doesn't exist
        project_root = Path(__file__).parent.parent.parent
        figures_dir = project_root / "results" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # If output_path is provided, use it but update the directory
        if output_path:
            # If output_path is a Path object
            if isinstance(output_path, Path):
                filename = output_path.name
            else:  # If it's a string
                filename = Path(output_path).name
            
            # Save to the figures directory
            save_path = figures_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to: {save_path}")
        else:
            # Generate a filename if none provided
            sanitized_title = title.replace(' ', '_').replace('-', '_').lower()
            save_path = figures_dir / f"{sanitized_title}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to: {save_path}")
        
        # Close the figure to prevent display and free memory
        plt.close(fig)
        
        return train_metrics, test_metrics


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Split data into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
