"""
LightGBM (gradient-boosted trees) Training — Curie temperature (Tc).

Gradient boosting is additive residual modelling, so it composes naturally with
delta-learning (the target is already a residual). It typically beats a plain
Random Forest and is well regularised on small data. It operates in linear Tc
(kelvin) space; with delta-learning the target is the correction Tc_exp - Tc_sim
and is reconstructed (reconstruct_target) before metrics. Rare-earth physics
features are added automatically when the loader has use_re_features=True.
"""
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from sklearn.model_selection import RandomizedSearchCV
from base_trainer import DataLoader, ModelEvaluator, split_data, cross_val_report

# Respect SLURM CPU allocation; fall back to all cores when running locally.
_N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))

# LightGBM's sklearn wrapper records internal feature names on fit, so predicting
# on the plain numpy arrays we pass triggers a benign "X does not have valid
# feature names" UserWarning. Silence it to keep the continuous logs clean.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LightGBMTrainer:
    """Train and evaluate LightGBM gradient-boosted tree models."""

    def __init__(self, output_dir: str = "results/lightgbm"):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DataLoader()
        self.evaluator = ModelEvaluator()

    def train_and_evaluate(
        self,
        dataset_name: str,
        dataset_type: str,
        is_augmented: bool = False,
        use_embedding: bool = False,
        embedding_type: Optional[str] = None,
    ) -> Dict:
        """Train LightGBM with light hyperparameter tuning."""
        print(f"\n{'='*60}")
        print(f"Training LightGBM: {dataset_name}")
        if use_embedding:
            print(f"Using embedding: {embedding_type}")
        else:
            print("Using only Tc_sim (no embedding)")
        print(f"{'='*60}")

        if is_augmented:
            df = self.loader.load_augmented_data(dataset_type)
        else:
            df = self.loader.load_pairs_data(dataset_type)

        X, y = self.loader.prepare_dataset(df, dataset_type, use_embedding, embedding_type)
        X_train, X_test, y_train, y_test = split_data(X, y)

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Feature dimensions: {X_train.shape[1]}")

        print("\nPerforming hyperparameter tuning...")
        model, best_params = self._randomized_search(X_train, y_train)
        print(f"Best parameters: {best_params}")

        # Predictions, then map back to Tc space (no-op unless delta_learning is on)
        # so metrics stay comparable across runs.
        y_train_true = self.loader.reconstruct_target(y_train, X_train)
        y_test_true = self.loader.reconstruct_target(y_test, X_test)
        y_train_pred = self.loader.reconstruct_target(model.predict(X_train), X_train)
        y_test_pred = self.loader.reconstruct_target(model.predict(X_test), X_test)

        train_metrics = self.evaluator.compute_metrics(y_train_true, y_train_pred)
        test_metrics = self.evaluator.compute_metrics(y_test_true, y_test_pred)

        print(f"\nTest Metrics:")
        print(f"  R² = {test_metrics['R2']:.4f}")
        print(f"  RMSE = {test_metrics['RMSE']:.2f} K")
        print(f"  MAE = {test_metrics['MAE']:.2f} K")

        # Optional K-fold CV reporting (on the full dataset, same tuned config).
        # LightGBM does not scale its inputs, so the tuned estimator is CV'd directly.
        cv_folds = getattr(self.loader, 'cv_folds', 0)
        cv = None
        if cv_folds and cv_folds >= 2:
            cv = cross_val_report(model, X, y, self.loader, n_splits=cv_folds)
            print(f"\n{cv_folds}-fold CV Metrics (headline):")
            print(f"  R²   = {cv['R2']:.4f} ± {cv['R2_std']:.4f}")
            print(f"  RMSE = {cv['RMSE']:.2f} ± {cv['RMSE_std']:.2f} K")
            print(f"  MAE  = {cv['MAE']:.2f} ± {cv['MAE_std']:.2f} K")

        emb_suffix = f"_{embedding_type}" if use_embedding else "_no_emb"
        output_path = self.output_dir / f"{dataset_name}_LGBM{emb_suffix}.png"
        self.evaluator.plot_predictions(
            y_train_true, y_train_pred,
            y_test_true, y_test_pred,
            title=f"LightGBM - {dataset_name}"
                  + (f" ({embedding_type})" if use_embedding else ""),
            output_path=str(output_path),
        )

        # CV means become the reported (headline) metrics when CV is on.
        reported = {'R2': cv['R2'], 'RMSE': cv['RMSE'], 'MAE': cv['MAE']} if cv else test_metrics

        info_path = self.output_dir / f"{dataset_name}_LGBM{emb_suffix}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Embedding: {embedding_type if use_embedding else 'None'}\n")
            f.write(f"Best params: {best_params}\n")
            f.write(f"Single-split R2: {test_metrics['R2']:.4f}\n")
            if cv:
                f.write(f"CV folds: {cv_folds}\n")
                f.write(f"CV R2: {cv['R2']:.4f} +/- {cv['R2_std']:.4f}\n")
                f.write(f"CV RMSE: {cv['RMSE']:.2f} +/- {cv['RMSE_std']:.2f}\n")
                f.write(f"CV MAE: {cv['MAE']:.2f} +/- {cv['MAE_std']:.2f}\n")

        # --- ONNX export (only the deployable raw_200D embedding variant) ---
        try:
            from onnx_export import maybe_export_onnx
            maybe_export_onnx(
                family="lgbm", model=model, scaler=None, input_dim=X_train.shape[1],
                dataset_name=dataset_name, use_embedding=use_embedding,
                embedding_type=embedding_type, loader=self.loader,
                aug_label=getattr(self.evaluator, "figures_subdir", None),
                output_dir=self.output_dir,
            )
        except Exception as _onnx_exc:
            print(f"    ONNX export skipped/failed: {_onnx_exc}")

        result = {
            'R2': reported['R2'],
            'RMSE': reported['RMSE'],
            'MAE': reported['MAE'],
            'best_params': best_params,
            'model': model,
        }
        if cv:
            result.update({'R2_std': cv['R2_std'], 'RMSE_std': cv['RMSE_std'],
                           'MAE_std': cv['MAE_std'], 'cv_folds': cv_folds})
        return result

    def _randomized_search(self, X_train, y_train):
        param_distributions = {
            'n_estimators': [200, 400, 600],
            'learning_rate': [0.01, 0.02, 0.05, 0.1],
            'num_leaves': [15, 31, 63],
            'max_depth': [-1, 4, 6, 8],
            'min_child_samples': [5, 10, 20, 40],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'reg_lambda': [0.0, 0.1, 1.0, 5.0],
        }
        # CRITICAL: the estimator runs SINGLE-THREADED (n_jobs=1) and only the
        # outer search is parallelised (n_jobs=_N_JOBS). Giving both n_jobs=_N_JOBS
        # spawns _N_JOBS x _N_JOBS threads (e.g. 72x72 on a full node), which
        # oversubscribes the CPUs and makes LightGBM hang / crawl.
        base = LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1,
                             force_col_wise=True)
        search = RandomizedSearchCV(
            base, param_distributions,
            n_iter=30, cv=5, scoring='r2',
            n_jobs=_N_JOBS, random_state=42,
        )
        search.fit(X_train, y_train)
        # Best model keeps n_jobs=1 from `base`; allow it all cores for the final
        # refit / predictions (no nesting at this point).
        best = search.best_estimator_
        best.set_params(n_jobs=_N_JOBS)
        return best, search.best_params_


def main():
    """Run LightGBM for all dataset configurations."""
    trainer = LightGBMTrainer()

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
                'R2': result['R2'], 'RMSE': result['RMSE'], 'MAE': result['MAE'],
            })
        except Exception as e:
            print(f"Error training {dataset_name} with {embedding_type}: {e}")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(trainer.output_dir / "lightgbm_summary.csv", index=False)
    print("\n" + "="*80)
    print("LightGBM Summary:")
    print("="*80)
    print(summary_df.to_string())
    return all_results


if __name__ == "__main__":
    main()
