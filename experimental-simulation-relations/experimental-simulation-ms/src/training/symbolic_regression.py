"""
Symbolic Regression Training - Baseline Model
Uses only Ms_sim as input (stoichiometry disregarded)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.metrics import r2_score, mean_squared_error
from base_trainer import DataLoader, ModelEvaluator, split_data, cross_val_report_fn

try:
    from pysr import PySRRegressor
except ImportError:
    print("Warning: PySR not installed. Install with: pip install pysr")
    PySRRegressor = None

import os
import sys
from src.log_to_file import log_output
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)


class SymbolicRegressionTrainer:
    """Train symbolic regression models as baseline."""

    def __init__(self, output_dir: str = "results/symbolic_regression"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DataLoader()
        self.evaluator = ModelEvaluator()

    def _build_model(self, niterations):
        """Construct a fresh PySR regressor (shared by single-split and CV)."""
        return PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "*"],
            unary_operators=[],
            maxsize=10,
            populations=15,
            population_size=33,
            ncycles_per_iteration=550,
            fraction_replaced_hof=0.035,
        )

    @staticmethod
    def _subsample(X, y, max_n):
        """Subsample to at most max_n rows (PySR is expensive on large sets)."""
        if len(X) <= max_n:
            return X, y
        idx = np.random.default_rng(42).choice(len(X), size=max_n, replace=False)
        return X[idx], y[idx]

    def train_and_evaluate(
        self,
        dataset_name: str,
        dataset_type: str,
        is_augmented: bool = False,
        niterations: int = 100,
        max_train_samples: int = 2000,
    ) -> Dict[str, float]:
        """
        Train symbolic regression model.

        The model operates in log1p space (matching the feature/target transform).
        Reported metrics are in log1p space.
        """
        print(f"\n{'='*60}")
        print(f"Training Symbolic Regression: {dataset_name}")
        print(f"{'='*60}")

        if is_augmented:
            df = self.loader.load_augmented_data(dataset_type)
        else:
            df = self.loader.load_pairs_data(dataset_type)

        X, y = self.loader.prepare_dataset(df, dataset_type, use_embedding=False)
        X_train, X_test, y_train, y_test = split_data(X, y)

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        if len(X_train) > max_train_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            print(f"Subsampled training set to {max_train_samples} samples for PySR")

        if PySRRegressor is None:
            raise ImportError("PySR is required for symbolic regression")

        model = self._build_model(niterations)

        print("\nFitting PySR model (in log1p space)...")
        model.fit(X_train, y_train)

        best_eq = model.get_best()
        equation_str = best_eq['equation']
        print(f"\nBest equation (log1p space): {equation_str}")

        # Predict, then map back to log1p(Ms_exp) space (no-op unless
        # delta_learning is on) so metrics stay comparable across runs.
        y_train_true = self.loader.reconstruct_log_exp(y_train, X_train)
        y_test_true = self.loader.reconstruct_log_exp(y_test, X_test)
        y_train_pred_log = self.loader.reconstruct_log_exp(model.predict(X_train), X_train)
        y_test_pred_log = self.loader.reconstruct_log_exp(model.predict(X_test), X_test)

        train_metrics = self.evaluator.compute_metrics(y_train_true, y_train_pred_log)
        test_metrics = self.evaluator.compute_metrics(y_test_true, y_test_pred_log)

        print(f"\nSingle-split Test Metrics (log1p space):")
        print(f"  R² = {test_metrics['R2']:.4f}")
        print(f"  RMSE = {test_metrics['RMSE']:.4f}")
        print(f"  MAE = {test_metrics['MAE']:.4f}")

        # Optional K-fold CV reporting. Each fold fits a fresh PySR model on the
        # (subsampled) fold-train and predicts the held-out fold. NOTE: this is
        # expensive — it runs the full symbolic search once per fold.
        cv_folds = getattr(self.loader, 'cv_folds', 0)
        cv = None
        if cv_folds and cv_folds >= 2:
            def _fit_predict(X_tr, y_tr, X_te):
                Xs, ys = self._subsample(X_tr, y_tr, max_train_samples)
                m = self._build_model(niterations)
                m.fit(Xs, ys)
                return m.predict(X_te)

            print(f"\nRunning {cv_folds}-fold CV for SR (slow — {cv_folds} symbolic searches)...")
            cv = cross_val_report_fn(_fit_predict, X, y, self.loader, n_splits=cv_folds)
            print(f"{cv_folds}-fold CV Metrics (headline):")
            print(f"  R²   = {cv['R2']:.4f} ± {cv['R2_std']:.4f}")
            print(f"  RMSE = {cv['RMSE']:.4f} ± {cv['RMSE_std']:.4f}")
            print(f"  MAE  = {cv['MAE']:.4f} ± {cv['MAE_std']:.4f}")

        reported = {'R2': cv['R2'], 'RMSE': cv['RMSE'], 'MAE': cv['MAE']} if cv else test_metrics

        output_path = self.output_dir / f"{dataset_name}_SR.png"
        self.evaluator.plot_predictions(
            y_train_true, y_train_pred_log,
            y_test_true, y_test_pred_log,
            title=f"Symbolic Regression - {dataset_name}",
            output_path=str(output_path),
            equation=equation_str,
        )

        eq_path = self.output_dir / f"{dataset_name}_SR_equation.txt"
        with open(eq_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Equation (log1p space): {equation_str}\n")
            f.write(f"Single-split R2: {test_metrics['R2']:.4f}\n")
            if cv:
                f.write(f"CV folds: {cv_folds}\n")
                f.write(f"CV R2: {cv['R2']:.4f} +/- {cv['R2_std']:.4f}\n")
                f.write(f"CV RMSE: {cv['RMSE']:.4f} +/- {cv['RMSE_std']:.4f}\n")
                f.write(f"CV MAE: {cv['MAE']:.4f} +/- {cv['MAE_std']:.4f}\n")

        result = {
            'R2': reported['R2'],
            'RMSE': reported['RMSE'],
            'MAE': reported['MAE'],
            'equation': equation_str,
        }
        if cv:
            result.update({'R2_std': cv['R2_std'], 'cv_folds': cv_folds})
        return result


def main():
    """Run symbolic regression for all datasets."""
    trainer = SymbolicRegressionTrainer()

    configs = [
        ('All-Pairs', 'all', False),
        ('RE-Pairs', 're', False),
        ('RE-Free-Pairs', 're-free', False),
    ]

    results = {}
    for dataset_name, dataset_type, is_augmented in configs:
        try:
            metrics = trainer.train_and_evaluate(dataset_name, dataset_type, is_augmented)
            results[dataset_name] = metrics
        except Exception as e:
            print(f"Error training {dataset_name}: {e}")
            results[dataset_name] = {'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'equation': 'Error'}

    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(trainer.output_dir / "symbolic_regression_summary.csv")
    print("\n" + "="*60)
    print("Symbolic Regression Summary:")
    print("="*60)
    print(summary_df)
    return results


if __name__ == "__main__":
    main()
