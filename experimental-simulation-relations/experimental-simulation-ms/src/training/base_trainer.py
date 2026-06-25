"""
Base training utilities for Ms Error Correction models.

Values are large (A/m), so training operates in log1p-space.
DataLoader.prepare_dataset returns log1p-transformed X and y.
Metrics and plots are computed in log1p-space.
"""
import atexit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Tuple, Dict, Optional

sns.set_style("whitegrid")

def _cleanup_multiprocessing():
    # Shut down loky workers; kill_workers=True avoids a hang when workers
    # are already dead (e.g. after SLURM step ends).
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception:
        pass
    # Python 3.13: ResourceTracker.__del__ calls os.waitpid on its sentinel
    # subprocess. If loky already reaped that process, _stop() throws
    # ChildProcessError before it can set _pid=None, so __del__ retries and
    # the traceback leaks out. Setting _pid=None directly is safe: __del__
    # checks _pid first and skips _stop() when it is None.
    try:
        from multiprocessing import resource_tracker
        rt = resource_tracker._resource_tracker
        if rt is not None:
            rt._pid = None
    except Exception:
        pass

atexit.register(_cleanup_multiprocessing)

SIM_COL = 'Ms (ampere/meter)_s'
EXP_COL = 'Ms (ampere/meter)_e'

# Binary feature threshold (Option B): rows with Ms_sim below this value are flagged
# as "poor-DFT regime" so models can learn separate correction factors per regime.
REGIME_THRESHOLD = 10_000  # A/m — must match augment_data.py


def to_original_space(y_log: np.ndarray) -> np.ndarray:
    """Convert log1p-transformed values back to original A/m space."""
    return np.expm1(y_log)


def parse_ms_threshold(default: float = 50_000) -> Optional[float]:
    """Parse --ms-threshold from command line arguments.

    Returns the threshold value (A/m). Rows with Ms_sim or Ms_exp <= threshold
    are dropped before training. Pass --ms-threshold 0 to disable filtering.
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ms-threshold', type=float, default=default,
                        help='Drop rows with Ms_sim or Ms_exp <= this value (A/m). '
                             'Default: %(default)s. Set to 0 to disable.')
    args, _ = parser.parse_known_args()
    return args.ms_threshold if args.ms_threshold > 0 else None


def parse_delta_learning() -> bool:
    """Parse --delta-learning from command line arguments.

    When set, models train on the residual log1p(Ms_exp) - log1p(Ms_sim)
    instead of log1p(Ms_exp) directly. Metrics are still reported in
    log1p(Ms_exp) space (the baseline is added back), so they stay comparable.
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--delta-learning', action='store_true',
                        help='Train on the correction log1p(Ms_exp)-log1p(Ms_sim) '
                             'instead of log1p(Ms_exp) directly.')
    args, _ = parser.parse_known_args()
    return args.delta_learning


class DataLoader:
    """Load and prepare Ms datasets for training."""

    def __init__(self, ms_threshold: Optional[float] = 50_000,
                 delta_learning: bool = False):
        project_root = Path(__file__).parent.parent.parent
        self.csv_path = project_root / "data" / "merged_df_python.csv"
        self.ms_threshold = ms_threshold
        self.delta_learning = delta_learning
        # Column index of log1p(Ms_sim) within X, recorded by prepare_dataset so
        # reconstruct_log_exp() can add the baseline back when delta_learning is on.
        self.sim_log_index_ = 0

    def _apply_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where Ms_sim or Ms_exp is at or below the threshold (in A/m).

        Applied in original space before any log transform.
        """
        if self.ms_threshold is None:
            return df
        before = len(df)
        df = df[
            (df[SIM_COL] > self.ms_threshold) & (df[EXP_COL] > self.ms_threshold)
        ].copy()
        dropped = before - len(df)
        if dropped:
            print(f"Ms threshold {self.ms_threshold:.0f} A/m: "
                  f"dropped {dropped} rows ({before} → {len(df)} kept)")
        return df

    def load_pairs_data(self, dataset_type: str = "all") -> pd.DataFrame:
        """Load rows that have both Ms_sim and Ms_exp from merged_df_python.csv."""
        df = pd.read_csv(self.csv_path, index_col=0)
        df = df[df[SIM_COL].notna() & df[EXP_COL].notna()].copy()
        df = self._apply_threshold(df)

        if dataset_type == 're':
            df = df[df['has_rare_earth'] == True].copy()
        elif dataset_type == 're-free':
            df = df[df['has_rare_earth'] == False].copy()
        elif dataset_type != 'all':
            raise ValueError(
                f"Invalid dataset_type: {dataset_type}. Must be 'all', 're', or 're-free'."
            )

        print(f"Loaded {len(df)} pairs for dataset_type='{dataset_type}'")
        return df

    def load_augmented_data(self, dataset_type: str = "all") -> pd.DataFrame:
        """Load Phase 3 combined augmented dataset from outputs/.

        Falls back to pairs data if the augmented file does not yet exist
        (i.e. augment_data.py has not been run).
        """
        file_map = {
            'all':     'Augm_combined_all.csv',
            're':      'Augm_combined_RE.csv',
            're-free': 'Augm_combined_RE_Free.csv',
        }
        if dataset_type not in file_map:
            raise ValueError(
                f"Invalid dataset_type: {dataset_type}. Must be 'all', 're', or 're-free'."
            )
        project_root = Path(__file__).parent.parent.parent
        path = project_root / 'outputs' / file_map[dataset_type]

        if not path.exists():
            print(
                f"Warning: Augmented file not found at {path}. "
                "Falling back to pairs data. Run src/augment_data.py to generate it."
            )
            return self.load_pairs_data(dataset_type)

        df = pd.read_csv(path)
        df = df[df[SIM_COL].notna() & df[EXP_COL].notna()].copy()
        df = self._apply_threshold(df)
        print(f"Loaded {len(df)} augmented rows for dataset_type='{dataset_type}'")
        return df

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        use_embedding: bool = False,
        embedding_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare X and y for training.

        Features (X):
          - log1p(Ms_sim)      — primary DFT feature
          - is_zero_sim        — binary flag: 1 if Ms_sim < REGIME_THRESHOLD (Option B)
          - embedding (optional) — prepended if use_embedding=True

        Target (y): log1p(Ms_exp)
        Call to_original_space() on predictions before reporting metrics.

        Args:
            df: DataFrame with SIM_COL, EXP_COL, and optionally embedding columns
            dataset_type: 'all', 're', or 're-free' (informational only here)
            use_embedding: Whether to prepend compound embeddings to X
            embedding_type: 'pca_8', 'pca_16', 'pca_32', or None for raw 200D

        Returns:
            X, y arrays in log1p space
        """
        before_len = len(df)
        df = df[df[SIM_COL].notna() & df[EXP_COL].notna()].copy()
        if len(df) != before_len:
            print(f"Dropped {before_len - len(df)} rows with NaN in sim or exp column")

        print(f"\nDataset diagnostics:")
        print(f"  - Dataset type: {dataset_type}")
        print(f"  - Total rows: {len(df)}")
        n_poor = (df[SIM_COL] < REGIME_THRESHOLD).sum()
        print(f"  - Poor-DFT rows (Ms_sim < {REGIME_THRESHOLD:.0e}): {n_poor} ({100*n_poor/len(df):.1f}%)")

        Ms_sim_log = np.log1p(df[SIM_COL].values).reshape(-1, 1)

        # Target: log1p(Ms_exp), or the residual correction to Ms_sim when
        # delta_learning is on. In delta mode the simulation is the baseline and
        # the model only predicts the deviation from it; reconstruct_log_exp()
        # adds the baseline back so reported metrics stay in log1p(Ms_exp) space.
        if self.delta_learning:
            y = np.log1p(df[EXP_COL].values) - Ms_sim_log.ravel()
            print(f"  - Target: delta = log1p(Ms_exp) - log1p(Ms_sim)")
        else:
            y = np.log1p(df[EXP_COL].values)
            print(f"  - Target: log1p(Ms_exp)")

        # Include the poor-DFT regime flag only when the Ms threshold allows
        # rows with Ms_sim < REGIME_THRESHOLD to reach training. With the
        # default threshold of 50,000 A/m the flag is always 0 and adds nothing.
        include_regime_flag = (
            self.ms_threshold is None or self.ms_threshold < REGIME_THRESHOLD
        )
        if include_regime_flag:
            is_zero_sim = (df[SIM_COL].values < REGIME_THRESHOLD).astype(float).reshape(-1, 1)
            print(f"  - Regime flag: ON (ms_threshold={self.ms_threshold} < {REGIME_THRESHOLD})")
        else:
            print(f"  - Regime flag: OFF (ms_threshold={self.ms_threshold:.0f} >= {REGIME_THRESHOLD})")

        if use_embedding:
            if embedding_type is None:
                if 'compound_embedding' not in df.columns:
                    raise ValueError(
                        "No compound_embedding column found. Run create_embeddings.py first."
                    )
                X_emb = np.vstack(df['compound_embedding'].values)
            else:
                col_name = f'comp_emb_pca_{embedding_type}_components'
                if col_name not in df.columns:
                    available = [c for c in df.columns if 'emb' in c.lower()]
                    raise ValueError(
                        f"Column {col_name} not found. Run compress_embedding_PCA.py first.\n"
                        f"Available embedding columns: {available}"
                    )
                X_emb = np.vstack(df[col_name].values)

            X = np.hstack([X_emb, Ms_sim_log, is_zero_sim] if include_regime_flag
                          else [X_emb, Ms_sim_log])
            # log1p(Ms_sim) sits right after the embedding block
            self.sim_log_index_ = X_emb.shape[1]
        else:
            X = np.hstack([Ms_sim_log, is_zero_sim]) if include_regime_flag else Ms_sim_log
            # log1p(Ms_sim) is the first column
            self.sim_log_index_ = 0

        return X, y

    def reconstruct_log_exp(self, y_values: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Map values from the training target space back to log1p(Ms_exp) space.

        When delta_learning is on, adds the log1p(Ms_sim) baseline (taken from
        column sim_log_index_ of X) back to y_values. When off, returns y_values
        unchanged. Apply to both y_true and y_pred before computing metrics so
        results are always reported in log1p(Ms_exp) space.
        """
        if not self.delta_learning:
            return y_values
        return y_values + X[:, self.sim_log_index_]


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
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        }

    def plot_predictions(
        self,
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        title: str,
        output_path: Optional[str] = None,
        equation: Optional[str] = None,
    ):
        """
        Create scatter plots of true vs predicted values.

        Expects values in log1p space.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        train_metrics = ModelEvaluator.compute_metrics(y_train, y_train_pred)
        test_metrics = ModelEvaluator.compute_metrics(y_test, y_test_pred)

        clean_equation = None
        if equation:
            clean_equation = equation.replace('(', '').replace(')', '')

        def identify_outliers(y_true, y_pred):
            residuals = y_true - y_pred
            return np.abs(residuals) > 2 * np.std(residuals)

        train_outliers = identify_outliers(y_train, y_train_pred)
        test_outliers = identify_outliers(y_test, y_test_pred)

        if 'SR' in title or 'Symbolic' in title:
            x_range = np.linspace(
                min(y_train.min(), y_test.min()),
                max(y_train.max(), y_test.max()),
                100,
            )
            from sklearn.linear_model import LinearRegression
            _m = LinearRegression().fit(y_train.reshape(-1, 1), y_train_pred)
            y_line = _m.predict(x_range.reshape(-1, 1))
            axes[0].plot(x_range, y_line, 'r-', lw=2, label='Best fit equation')

        axes[0].scatter(y_train, y_train_pred, alpha=0.5, label='Train prediction')
        if np.any(train_outliers):
            axes[0].scatter(
                y_train[train_outliers], y_train_pred[train_outliers],
                alpha=0.7, edgecolors='red', facecolors='none', s=80,
                linewidths=1.5, label='Outliers',
            )
        axes[0].plot(
            [y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
            'k--', lw=2, label='y=x',
        )
        axes[0].set_xlabel('log1p(Ms_exp)', fontsize=12)
        axes[0].set_ylabel('log1p(Ms_exp_pred)', fontsize=12)
        axes[0].set_title('Training Set', fontsize=14)
        axes[0].legend()
        axes[0].grid(True)

        metrics_text_train = (
            f"R² = {train_metrics['R2']:.4f}\n"
            f"MAE = {train_metrics['MAE']:.4f}\n"
            f"RMSE = {train_metrics['RMSE']:.4f}"
        )
        if clean_equation:
            metrics_text_train = f"Eq: {clean_equation}\n" + metrics_text_train
        axes[0].text(
            0.05, 0.95, metrics_text_train,
            transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        axes[1].scatter(y_test, y_test_pred, alpha=0.7, color='orange', label='Test predictions')
        if np.any(test_outliers):
            axes[1].scatter(
                y_test[test_outliers], y_test_pred[test_outliers],
                alpha=0.7, edgecolors='red', facecolors='none', s=80,
                linewidths=1.5, label='Outliers',
            )
        if 'SR' in title or 'Symbolic' in title:
            axes[1].plot(x_range, y_line, 'r-', lw=2, label='Best fit equation')
        axes[1].plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'k--', lw=2, label='y=x',
        )
        axes[1].set_xlabel('log1p(Ms_exp)', fontsize=12)
        axes[1].set_ylabel('log1p(Ms_exp_pred)', fontsize=12)
        axes[1].set_title('Test Set', fontsize=14)
        axes[1].legend()
        axes[1].grid(True)

        metrics_text_test = (
            f"R² = {test_metrics['R2']:.4f}\n"
            f"MAE = {test_metrics['MAE']:.4f}\n"
            f"RMSE = {test_metrics['RMSE']:.4f}"
        )
        axes[1].text(
            0.05, 0.95, metrics_text_test,
            transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        fig.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()

        project_root = Path(__file__).parent.parent.parent
        figures_dir = project_root / "results" / "figures"
        if self.figures_subdir:
            figures_dir = figures_dir / self.figures_subdir
        figures_dir.mkdir(parents=True, exist_ok=True)

        if output_path:
            filename = Path(output_path).name
        else:
            sanitized = title.replace(' ', '_').replace('-', '_').lower()
            filename = f"{sanitized}.png"

        save_path = figures_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
        plt.close(fig)

        return train_metrics, test_metrics


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Split data into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
