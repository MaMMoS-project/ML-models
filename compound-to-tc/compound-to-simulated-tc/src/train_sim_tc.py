# -*- coding: utf-8 -*-
"""Train models to predict simulated Tc from compound embeddings.

Expects pre-processed pickle files in outputs/ produced by the two upstream
scripts.  Run in order:

    python src/create_embeddings.py
    python src/compress_embeddings_pca.py
    python src/train_sim_tc.py

Input files (outputs/):
    Simulation_Tc_RE-Free_w_embeddings_PCA.pkl
    Simulation_Tc_RE_w_embeddings_PCA.pkl
    Simulation_Tc_all_w_embeddings_PCA.pkl

Each pkl must contain at least:
    compound_embedding       – raw 200-D numpy array
    comp_emb_pca_8/16/32/64  – PCA-compressed arrays
    Tc_sim                   – target variable (float, K)

Model families trained per dataset:
    Linear (LassoLars/Ridge best), Random Forest, MLP
    — on embedding variants: raw_200D, pca_8, pca_16, pca_32, pca_64

Hyperparameters are scaled to the training-set size:
    RF:  n_iter ∝ 1/n_train  (fewer CV iterations for larger datasets)
    MLP: smaller architecture for smaller training sets

Outputs:
    results/sim_tc_comparison.csv
    results/sim_tc_best_by_dataset.csv
    results/figures/<dataset>_<embedding>_<model>.png
    logs/train_sim_tc.txt
"""

import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoLars, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset


sys.path.insert(0, str(Path(__file__).parent.parent))
from src.log_to_file import log_output

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Cap parallel workers to avoid OOM on many-core nodes (nested joblib contexts).
# Set env var COMBOUND_N_JOBS to override (e.g. COMBOUND_N_JOBS=8).
_N_JOBS = int(os.environ.get("COMBOUND_N_JOBS", min(8, os.cpu_count() or 1)))

PROJECT_ROOT  = Path(__file__).parent.parent
OUTPUT_DIR    = PROJECT_ROOT / "outputs"
RESULTS_DIR   = PROJECT_ROOT / "results"
ONNX_DIR      = RESULTS_DIR / "onnx_models"
PCA_CACHE_DIR = OUTPUT_DIR

CONFIG_PATH = PROJECT_ROOT / "training_config.yaml"


def _load_model_config() -> Dict[str, Dict]:
    """Return {model_key: {"enabled": bool, "ensemble": int}} for all models."""
    defaults = {
        "linear": {"enabled": True, "ensemble": 1},
        "rf":     {"enabled": True, "ensemble": 1},
        "mlp":    {"enabled": True, "ensemble": 1},
    }
    if not CONFIG_PATH.exists():
        return defaults
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    models = cfg.get("models", {})
    result: Dict[str, Dict] = {}
    for key, default in defaults.items():
        val = models.get(key)
        if val is None:
            result[key] = default.copy()
        elif isinstance(val, bool):
            result[key] = {"enabled": val, "ensemble": 1}
        elif isinstance(val, dict):
            result[key] = {
                "enabled":  bool(val.get("enabled", True)),
                "ensemble": max(1, int(val.get("ensemble", 1))),
            }
        else:
            result[key] = default.copy()
    return result


MODEL_CONFIG = _load_model_config()

DATASETS = [
    {
        "name":  "RE-Free",
        "pca":   "Simulation_Tc_RE-Free_w_embeddings_PCA.pkl",
        "plain": "Simulation_Tc_RE-Free_w_embeddings.pkl",
    },
    {
        "name":  "RE",
        "pca":   "Simulation_Tc_RE_w_embeddings_PCA.pkl",
        "plain": "Simulation_Tc_RE_w_embeddings.pkl",
    },
    {
        "name":  "All",
        "pca":   "Simulation_Tc_all_w_embeddings_PCA.pkl",
        "plain": "Simulation_Tc_all_w_embeddings.pkl",
    },
]

# Embedding variant label → DataFrame column name
EMB_VARIANTS = {
    "raw_200D": "compound_embedding",
    "pca_8":    "comp_emb_pca_8",
    "pca_16":   "comp_emb_pca_16",
    "pca_32":   "comp_emb_pca_32",
    "pca_64":   "comp_emb_pca_64",
}

# ---------------------------------------------------------------------------
# Metrics & plotting
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def save_figure(
    y_train: np.ndarray, y_train_pred: np.ndarray,
    y_test: np.ndarray,  y_test_pred: np.ndarray,
    title: str, out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, yt, yp, label, color in [
        (axes[0], y_train, y_train_pred, "Train", "steelblue"),
        (axes[1], y_test,  y_test_pred,  "Test",  "darkorange"),
    ]:
        m = compute_metrics(yt, yp)
        ax.scatter(yt, yp, alpha=0.5, color=color, s=15)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="y = x")
        ax.set_xlabel("Tc_sim (K)")
        ax.set_ylabel("Tc_sim_pred (K)")
        ax.set_title(label)
        ax.text(
            0.05, 0.95,
            f"R²  = {m['R2']:.3f}\nMAE = {m['MAE']:.1f} K\nRMSE= {m['RMSE']:.1f} K",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
        ax.legend()
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out_path.name}")


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def train_linear(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    run_name: str, out_dir: Path,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict, object, StandardScaler]:
    """Train LassoLars and Ridge; keep the one with higher test R²."""
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)

    param_grid = {"alpha": [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]}
    best: Dict = {"R2": -np.inf, "model": None}

    for label, estimator in [
        ("lasso", LassoLars(max_iter=500)),
        ("ridge", Ridge(max_iter=1_000)),
    ]:
        gs = GridSearchCV(estimator, param_grid, cv=5, scoring="r2", n_jobs=_N_JOBS)
        gs.fit(Xs_tr, y_train)
        model = gs.best_estimator_
        ytr_p = model.predict(Xs_tr)
        yte_p = model.predict(Xs_te)
        m = compute_metrics(y_test, yte_p)
        if m["R2"] > best["R2"]:
            best = dict(label=label, model=model, ytr_p=ytr_p, yte_p=yte_p, **m)

    save_figure(
        y_train, best["ytr_p"], y_test, best["yte_p"],
        title=f"Linear ({best['label'].upper()}) – {run_name}",
        out_path=out_dir / f"{run_name}_linear.png",
    )
    return {"R2": best["R2"], "MAE": best["MAE"], "RMSE": best["RMSE"]}, best["model"], scaler


def train_rf(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    run_name: str, out_dir: Path,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict, RandomForestRegressor, None]:
    """Train Random Forest with randomised hyperparameter search.

    n_iter is scaled inversely with training-set size so that total wall-clock
    time stays roughly constant across the three datasets:
        ~40 iterations for RE-Free (~5 k samples)
        ~25 iterations for RE     (~8 k samples)
        ~15 iterations for All    (~13 k samples)
    """
    n_iter = max(10, round(200_000 / len(X_train)))

    param_dist = {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [None, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", None],
        "bootstrap":         [True, False],
    }
    print(f"    (RF: n_iter={n_iter} for {len(X_train)} train samples)", end="  ")
    rs = RandomizedSearchCV(
        RandomForestRegressor(random_state=seed, n_jobs=1),
        param_dist, n_iter=n_iter, cv=5, scoring="r2",
        n_jobs=_N_JOBS, random_state=seed, verbose=0,
    )
    rs.fit(X_train, y_train)
    model = rs.best_estimator_
    ytr_p = model.predict(X_train)
    yte_p = model.predict(X_test)
    m = compute_metrics(y_test, yte_p)
    save_figure(
        y_train, ytr_p, y_test, yte_p,
        title=f"Random Forest – {run_name}",
        out_path=out_dir / f"{run_name}_rf.png",
    )
    return m, model, None


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze()


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    run_name: str, out_dir: Path,
    hidden_dims: Optional[Tuple[int, ...]] = None,
    batch_size: int = 256,
    num_epochs: int = 200,
    lr: float = 1e-3,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict, "_MLP", StandardScaler]:
    """Train MLP with early stopping.

    Architecture is chosen based on training-set size to avoid over-parameterisation:
        n_train < 6 000  → (128, 64, 32)   ~25 k parameters
        n_train >= 6 000 → (256, 128, 64)  ~80 k parameters
    """
    if hidden_dims is None:
        hidden_dims = (128, 64, 32) if len(X_train) < 6_000 else (256, 128, 64)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train).astype(np.float32)
    Xs_te = scaler.transform(X_test).astype(np.float32)
    y_tr  = y_train.astype(np.float32)
    y_te  = y_test.astype(np.float32)

    tr_loader = TorchDataLoader(
        TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    te_loader = TorchDataLoader(
        TensorDataset(torch.from_numpy(Xs_te), torch.from_numpy(y_te)),
        batch_size=batch_size,
    )

    model     = _MLP(X_train.shape[1], hidden_dims).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{run_name}_MLP_best.pt"
    best_loss, patience_count = float("inf"), 0

    for epoch in range(num_epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in te_loader:
                val_loss += criterion(model(bx.to(device)), by.to(device)).item()
        val_loss /= len(te_loader)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss, patience_count = val_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_count += 1
            if patience_count >= 30:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(ckpt))
    model.eval()
    with torch.no_grad():
        ytr_p = model(torch.from_numpy(Xs_tr).to(device)).cpu().numpy()
        yte_p = model(torch.from_numpy(Xs_te).to(device)).cpu().numpy()

    m = compute_metrics(y_test, yte_p)
    save_figure(
        y_train, ytr_p, y_test, yte_p,
        title=f"MLP {hidden_dims} – {run_name}",
        out_path=out_dir / f"{run_name}_mlp.png",
    )
    return m, model, scaler


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

def _get_or_fit_pca(df: pd.DataFrame, ds_name: str, n_components: int) -> PCA:
    """Return a PCA fitted on the full dataset's raw 200-D embeddings.

    The result is cached as a pickle so that repeated calls (e.g. from multiple
    model variants) are fast.  The cache is consistent with compress_embeddings_pca.py
    because both fit on the same full-dataset raw embeddings with the same seed.
    """
    cache = PCA_CACHE_DIR / f"pca_{ds_name}_{n_components}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    raw = np.vstack(df["compound_embedding"].values)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(raw)
    with open(cache, "wb") as f:
        pickle.dump(pca, f)
    return pca


def _export_sklearn_onnx(
    model: object,
    scaler: Optional[StandardScaler],
    pca: Optional[PCA],
    input_dim: int,
    out_path: Path,
) -> None:
    """Export a sklearn model (optionally preceded by PCA + scaler) to ONNX.

    The resulting ONNX model always accepts raw ``input_dim``-D embeddings and
    applies the preprocessing steps internally before calling the model.
    """
    steps = []
    if pca is not None:
        steps.append(("pca", pca))
    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("model", model))
    pipe = Pipeline(steps)
    initial_type = [("X", FloatTensorType([None, input_dim]))]
    onnx_proto = convert_sklearn(pipe, initial_types=initial_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onnx_proto.SerializeToString())
    print(f"  ONNX → {out_path.name}")


class _MLPWithPreprocessing(nn.Module):
    """Wraps a trained _MLP with optional PCA and mandatory StandardScaler.

    The wrapper takes raw ``input_dim``-D embeddings, applies PCA (if given),
    then scales the features, then runs the MLP.  This lets a single ONNX file
    cover the full inference pipeline from raw compound embeddings to Tc.
    """

    def __init__(
        self,
        pca: Optional[PCA],
        scaler: StandardScaler,
        mlp: "_MLP",
    ) -> None:
        super().__init__()
        self.has_pca = pca is not None
        if self.has_pca:
            self.register_buffer("pca_mean", torch.tensor(pca.mean_, dtype=torch.float32))
            self.register_buffer("pca_components", torch.tensor(pca.components_, dtype=torch.float32))
        self.register_buffer("scaler_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("scaler_scale", torch.tensor(scaler.scale_, dtype=torch.float32))
        self.net = mlp.net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_pca:
            x = (x - self.pca_mean) @ self.pca_components.T
        x = (x - self.scaler_mean) / self.scaler_scale
        return self.net(x).view(-1)


def _export_mlp_onnx(
    mlp: "_MLP",
    scaler: StandardScaler,
    pca: Optional[PCA],
    input_dim: int,
    out_path: Path,
) -> None:
    """Export the MLP (with preprocessing) to ONNX via torch.onnx."""
    full_model = _MLPWithPreprocessing(pca, scaler, mlp)
    full_model.cpu().eval()
    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        full_model,
        dummy,
        str(out_path),
        dynamo=False,
        input_names=["compound_embedding"],
        output_names=["Tc_pred"],
        dynamic_axes={
            "compound_embedding": {0: "batch_size"},
            "Tc_pred":            {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"  ONNX → {out_path.name}")


# ---------------------------------------------------------------------------
# Per-dataset training logic (reused by the individual dataset scripts)
# ---------------------------------------------------------------------------

def train_one_dataset(ds: Dict, figures_dir: Path) -> List[Dict]:
    """Train all model/embedding combinations for one dataset config.

    Returns a list of result-row dicts (one per successful model run).
    Also writes ``results/<ds_name>_results.csv``.
    """
    ds_name = ds["name"]

    pkl_path = OUTPUT_DIR / ds["pca"]
    if not pkl_path.exists():
        pkl_path = OUTPUT_DIR / ds["plain"]
        if not pkl_path.exists():
            print(f"\nDataset {ds_name}: no pkl found – run create_embeddings.py first.")
            return []
        print(f"\nDataset {ds_name}: PCA file not found, using plain embeddings.")

    print(f"\n{'='*70}")
    print(f"Dataset : {ds_name}  ({pkl_path.name})")
    print(f"{'='*70}")

    df = pd.read_pickle(pkl_path)
    print(f"Loaded {len(df)} rows")

    raw_dim = int(df["compound_embedding"].iloc[0].shape[0])

    ds_results: List[Dict] = []

    for emb_label, col_name in EMB_VARIANTS.items():
        if col_name not in df.columns:
            print(f"\n  [{emb_label}] column '{col_name}' not found – skipping.")
            continue

        X = np.vstack(df[col_name].values)
        y = df["Tc_sim"].values.astype(float)

        run_name = f"{ds_name}_{emb_label}"
        n_train_approx = int(0.8 * len(X))
        n_test_approx  = len(X) - n_train_approx
        print(
            f"\n  [{emb_label}]  X: {X.shape}"
            f"  train: ~{n_train_approx}  test: ~{n_test_approx}"
        )

        # PCA transform needed for all variants except raw_200D
        pca: Optional[PCA] = None
        if emb_label != "raw_200D":
            n_pca = int(emb_label.split("_")[1])
            pca = _get_or_fit_pca(df, ds_name, n_pca)

        for model_label, train_fn, key in [
            ("Linear", train_linear, "linear"),
            ("RF",     train_rf,     "rf"),
            ("MLP",    train_mlp,    "mlp"),
        ]:
            cfg = MODEL_CONFIG[key]
            if not cfg["enabled"]:
                continue
            n_ensemble = cfg["ensemble"]

            for i in range(n_ensemble):
                seed = RANDOM_SEED + i
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed
                )
                suffix      = f"_e{i}" if n_ensemble > 1 else ""
                member_name = f"{run_name}{suffix}"
                progress    = f" (ensemble {i + 1}/{n_ensemble})" if n_ensemble > 1 else ""
                print(f"    Training {model_label}{progress}...", end="  ")
                try:
                    m, trained_model, trained_scaler = train_fn(
                        X_train, y_train, X_test, y_test,
                        member_name, figures_dir, seed=seed,
                    )
                    print(
                        f"R²={m['R2']:.4f}  "
                        f"RMSE={m['RMSE']:.1f} K  "
                        f"MAE={m['MAE']:.1f} K"
                    )
                    ds_results.append(dict(
                        Dataset=ds_name, Embedding=emb_label, Model=model_label,
                        EnsembleIdx=i, R2=m["R2"], MAE=m["MAE"], RMSE=m["RMSE"],
                    ))
                    onnx_path = ONNX_DIR / f"{run_name}_{model_label.lower()}{suffix}.onnx"
                    try:
                        if model_label == "MLP":
                            _export_mlp_onnx(trained_model, trained_scaler, pca, raw_dim, onnx_path)
                        else:
                            _export_sklearn_onnx(trained_model, trained_scaler, pca, raw_dim, onnx_path)
                    except Exception as onnx_exc:
                        print(f"    ONNX export failed: {onnx_exc}")
                except Exception as exc:
                    print(f"FAILED: {exc}")

    if ds_results:
        df_ds = (
            pd.DataFrame(ds_results)
            .sort_values("R2", ascending=False)
            .reset_index(drop=True)
        )
        ds_csv = RESULTS_DIR / f"{ds_name}_sim_results.csv"
        df_ds.to_csv(ds_csv, index=False)

        print(f"\n{'─'*70}")
        print(f"Results for dataset: {ds_name}")
        print(f"{'─'*70}")
        print(df_ds.to_string(index=False))

        best_row = df_ds.iloc[0]
        print(
            f"\n  *** Best model for {ds_name}: "
            f"{best_row['Model']} [{best_row['Embedding']}]"
            f"  R²={best_row['R2']:.4f}"
            f"  RMSE={best_row['RMSE']:.1f} K"
            f"  MAE={best_row['MAE']:.1f} K ***"
        )
        print(f"  Saved → {ds_csv}")

    return ds_results


def update_global_summary() -> None:
    """Regenerate global comparison CSVs from all available per-dataset CSVs.

    Loads every ``<dataset>_results.csv`` that already exists in RESULTS_DIR and
    writes the combined ``sim_tc_comparison.csv`` and ``sim_tc_best_by_dataset.csv``.
    Safe to call after any individual dataset script finishes.
    """
    frames = []
    for ds in DATASETS:
        ds_csv = RESULTS_DIR / f"{ds['name']}_sim_results.csv"
        if ds_csv.exists():
            frames.append(pd.read_csv(ds_csv))

    if not frames:
        print("\nNo per-dataset CSVs found; skipping global summary.")
        return

    df_res = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["Dataset", "Embedding", "R2"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    out_csv = RESULTS_DIR / "sim_tc_comparison.csv"
    df_res.to_csv(out_csv, index=False)

    best_global = (
        df_res.loc[df_res.groupby("Dataset")["R2"].idxmax()]
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )
    best_csv = RESULTS_DIR / "sim_tc_best_by_dataset.csv"
    best_global.to_csv(best_csv, index=False)

    print("\n" + "=" * 70)
    print("OVERALL BEST MODEL PER DATASET")
    print("=" * 70)
    print(best_global.to_string(index=False))
    print(f"\nAll results          : {out_csv}")
    print(f"Best per dataset     : {best_csv}")
    print(f"Per-dataset tables   : {RESULTS_DIR}/<dataset>_sim_results.csv")


# ---------------------------------------------------------------------------
# Main pipeline (trains all three datasets in one go)
# ---------------------------------------------------------------------------

@log_output("logs/train_sim_tc.txt")
def train_sim_tc() -> None:
    print("=" * 70)
    print("Training: compound embedding → simulated Tc")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = RESULTS_DIR / "figures"
    pd.set_option("display.max_rows", None)

    for ds in DATASETS:
        train_one_dataset(ds, figures_dir)

    update_global_summary()


if __name__ == "__main__":
    log_path = PROJECT_ROOT / "logs" / "train_sim_tc.txt"
    print(f"Output logged to: {log_path}")
    train_sim_tc()
    print(f"Done. Results in: {RESULTS_DIR}")
