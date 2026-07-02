"""
Base training utilities for Curie Temperature Error Correction models.
"""
import atexit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Tuple, Dict, Optional

sns.set_style("whitegrid")


def parse_delta_learning() -> bool:
    """Parse --delta-learning from command line arguments.

    When set, models train on the correction Tc_exp - Tc_sim instead of Tc_exp
    directly. Metrics are still reported in Tc space (the Tc_sim baseline is added
    back via DataLoader.reconstruct_target), so they stay comparable.
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--delta-learning', action='store_true',
                        help='Train on the correction Tc_exp - Tc_sim instead of Tc_exp.')
    args, _ = parser.parse_known_args()
    return args.delta_learning


def parse_re_features() -> bool:
    """Parse --re-features from command line arguments.

    When set, prepare_dataset appends 7 rare-earth physics features (free-ion
    Hund's-rules quantities incl. the de Gennes factor) derived from the
    'composition' column. The features are zero for RE-free rows, so they are
    safe on any dataset. Tc_sim is kept LAST in X, so delta reconstruction
    (reconstruct_target -> X[:, -1]) stays valid.
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--re-features', action='store_true',
                        help='Append rare-earth physics features (from composition).')
    args, _ = parser.parse_known_args()
    return args.re_features


def parse_cv_folds(default: int = 0) -> int:
    """Parse --cv N from command line arguments.

    N >= 2 reports K-fold cross-validated metrics (mean +/- std over folds) as the
    headline numbers instead of a single 80/20 split. N = 0 (default) keeps the
    single-split behaviour. Useful for the small RE split, whose single-split R2
    swings noticeably from fold to fold.
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cv', type=int, default=default, dest='cv',
                        help='K-fold CV for reporting (N>=2). 0 = single split (default).')
    args, _ = parser.parse_known_args()
    return args.cv if args.cv and args.cv >= 2 else 0

# Explicitly shut down loky's worker pool at exit so that Python 3.13's
# resource tracker process is still alive when __del__ handlers run,
# avoiding the "Exception ignored in ResourceTracker.__del__" race condition.
try:
    from joblib.externals.loky import get_reusable_executor
    atexit.register(lambda: get_reusable_executor().shutdown(wait=True))
except Exception:
    pass


class DataLoader:
    """Load and prepare datasets for training."""
    
    def __init__(self, data_dir: str = "../../data", pairs_file: str = "Pairs_all.csv",
                 augmented_file: str = "Augm_combined_all.csv", re_pairs_file: str = "Pairs_RE.csv",
                 re_free_pairs_file: str = "Pairs_RE_Free.csv", re_augmented_file: str = "Augm_combined_RE.csv",
                 re_free_augmented_file: str = "Augm_combined_RE_Free.csv",
                 delta_learning: bool = False, use_re_features: bool = False,
                 cv_folds: int = 0):
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

        # Training-config flags (ported from my_ms).
        # delta_learning: predict the correction (Tc_exp - Tc_sim) instead of Tc_exp.
        # use_re_features / cv_folds are wired in later steps.
        self.delta_learning = delta_learning
        self.use_re_features = use_re_features
        self.cv_folds = cv_folds

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

        # Target: Tc_exp, or the correction Tc_exp - Tc_sim when delta_learning is on.
        # Tc_sim is the baseline; reconstruct_target() adds it back before metrics so
        # reported numbers stay in Tc space.
        if self.delta_learning:
            y = df[target_col].values - df['Tc_sim'].values
            print(f"  - Target: delta = Tc_exp - Tc_sim")
        else:
            y = df[target_col].values
            print(f"  - Target: Tc_exp")

        # Prepare X. Feature column order is always [embedding?, RE features?, Tc_sim],
        # i.e. Tc_sim is ALWAYS the last column so reconstruct_target (X[:, -1])
        # stays valid even when RE features are appended.
        if use_embedding:
            if embedding_type is None:
                # Use raw compound_embedding
                if 'compound_embedding' in df.columns:
                    X = np.vstack(df['compound_embedding'].values)
                else:
                    raise ValueError("No compound_embedding column found. Run create_embeddings.py first.")
            else:
                # Resolve the embedding column name (PCA-compressed or raw). Matching
                # these candidates here lets the *_emb scripts delegate to this method
                # instead of monkeypatching prepare_dataset.
                candidates = [
                    f'comp_emb_pca_{embedding_type}_components',
                    f'comp_emb_{embedding_type}_components',
                    embedding_type,
                ]
                col_name = next((c for c in candidates if c in df.columns), None)
                if col_name is None:
                    raise ValueError(
                        f"Embedding column for type {embedding_type} not found. "
                        f"Available embedding columns: "
                        f"{[c for c in df.columns if 'emb' in c.lower()]}"
                    )
                X = np.vstack(df[col_name].values)
            feature_blocks = [X]
        else:
            feature_blocks = []

        # Rare-earth physics features are appended BEFORE Tc_sim (which must stay
        # last). Features are zero for RE-free rows, so this is safe on any dataset.
        if self.use_re_features:
            import sys as _sys
            _src_dir = str(Path(__file__).parent.parent)  # src/
            if _src_dir not in _sys.path:
                _sys.path.insert(0, _src_dir)
            from re_features import compute_re_features, RE_FEATURE_NAMES
            comp = (df['composition'] if 'composition' in df.columns
                    else df.index.to_series())
            re_mat = np.array([
                [compute_re_features(c)[k] for k in RE_FEATURE_NAMES]
                for c in comp.values
            ])
            feature_blocks.append(re_mat)
            n_active = int((re_mat[:, 0] > 0).sum())  # rows with nonzero RE fraction
            print(f"  - RE physics features: ON ({len(RE_FEATURE_NAMES)} cols, "
                  f"{n_active}/{len(df)} rows with RE content)")

        # Tc_sim goes last so X[:, -1] is always the simulation baseline.
        Tc_sim = df['Tc_sim'].values.reshape(-1, 1)
        feature_blocks.append(Tc_sim)
        X = np.hstack(feature_blocks) if len(feature_blocks) > 1 else feature_blocks[0]

        return X, y

    def reconstruct_target(self, y_values: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Map values from the training-target space back to Tc space.

        When delta_learning is on, adds the Tc_sim baseline back. Tc_sim is always
        the LAST column of X, so this is simply y + X[:, -1]. When off, returns
        y_values unchanged. IMPORTANT: pass the UNSCALED X (the raw X_train/X_test),
        not a StandardScaler-transformed array, or the baseline will be wrong.
        Apply to both y_true and y_pred before computing metrics.
        """
        if not self.delta_learning:
            return y_values
        return y_values + X[:, -1]


class ModelEvaluator:
    """Evaluate and plot model results."""

    def __init__(self):
        self.figures_subdir: Optional[str] = None

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute R2, MAE, and RMSE."""
        return {
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def plot_predictions(
        self,
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
        
        # Create output directory in project root /results/figures/ (or subdir) if it doesn't exist
        project_root = Path(__file__).parent.parent.parent
        figures_dir = project_root / "results" / "figures"
        if self.figures_subdir:
            figures_dir = figures_dir / self.figures_subdir
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


def cross_val_report(model, X: np.ndarray, y: np.ndarray, loader,
                     n_splits: int = 5, n_repeats: int = 1, random_state: int = 42) -> Dict:
    """K-fold CV metrics in Tc space, delta-aware.

    A fresh clone of `model` (same hyperparameters, unfitted) is trained on each
    training fold and scored on the held-out fold. Predictions and targets are
    mapped back through loader.reconstruct_target(), so metrics match the
    single-split ones and stay comparable whether or not delta-learning is on.

    Used by the sklearn-compatible estimators (Linear, RandomForest, LightGBM).
    For models that StandardScaler their inputs (Linear, MLP), pass an estimator
    that includes the scaler (e.g. a Pipeline) so each fold scales on its own
    train data; reconstruct_target reads Tc_sim from the UNSCALED X[:, -1], which
    is exactly the raw X handed to this function. MLP and Symbolic Regression are
    not sklearn-cloneable and use cross_val_report_fn() instead.

    Returns mean and std of R2/RMSE/MAE across folds.
    """
    if n_repeats > 1:
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=random_state)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2s, rmses, maes = [], [], []
    for tr, te in splitter.split(X):
        est = clone(model)
        est.fit(X[tr], y[tr])
        y_true = loader.reconstruct_target(y[te], X[te])
        y_pred = loader.reconstruct_target(est.predict(X[te]), X[te])
        m = ModelEvaluator.compute_metrics(y_true, y_pred)
        r2s.append(m['R2']); rmses.append(m['RMSE']); maes.append(m['MAE'])

    r2s, rmses, maes = np.array(r2s), np.array(rmses), np.array(maes)
    return {
        'R2': float(r2s.mean()),   'R2_std': float(r2s.std()),
        'RMSE': float(rmses.mean()), 'RMSE_std': float(rmses.std()),
        'MAE': float(maes.mean()),  'MAE_std': float(maes.std()),
        'n_splits': n_splits, 'n_repeats': n_repeats,
    }


def cross_val_report_fn(fit_predict_fn, X: np.ndarray, y: np.ndarray, loader,
                        n_splits: int = 5, n_repeats: int = 1, random_state: int = 42) -> Dict:
    """K-fold CV for non-sklearn models (MLP, Symbolic Regression).

    Like cross_val_report, but instead of cloning an estimator it calls
    fit_predict_fn(X_train, y_train, X_test) -> y_pred_test, where y_pred_test is
    in the TARGET space (delta if delta-learning is on). Predictions and targets
    are mapped back through loader.reconstruct_target() so metrics match the
    single-split / sklearn-CV ones. This lets torch and PySR models be CV'd
    without an sklearn-compatible clone(). Any scaling must happen INSIDE
    fit_predict_fn (fit on the fold's train data only).
    """
    if n_repeats > 1:
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=random_state)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2s, rmses, maes = [], [], []
    for tr, te in splitter.split(X):
        y_pred_target = fit_predict_fn(X[tr], y[tr], X[te])
        y_true = loader.reconstruct_target(y[te], X[te])
        y_pred = loader.reconstruct_target(np.asarray(y_pred_target), X[te])
        m = ModelEvaluator.compute_metrics(y_true, y_pred)
        r2s.append(m['R2']); rmses.append(m['RMSE']); maes.append(m['MAE'])

    r2s, rmses, maes = np.array(r2s), np.array(rmses), np.array(maes)
    return {
        'R2': float(r2s.mean()),   'R2_std': float(r2s.std()),
        'RMSE': float(rmses.mean()), 'RMSE_std': float(rmses.std()),
        'MAE': float(maes.mean()),  'MAE_std': float(maes.std()),
        'n_splits': n_splits, 'n_repeats': n_repeats,
    }
