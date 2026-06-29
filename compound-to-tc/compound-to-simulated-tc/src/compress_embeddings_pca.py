# -*- coding: utf-8 -*-
"""Compress compound embeddings with PCA for the simulated Tc datasets.

Reads from outputs/:
    Simulation_Tc_RE-Free_w_embeddings.pkl
    Simulation_Tc_RE_w_embeddings.pkl
    Simulation_Tc_all_w_embeddings.pkl

Saves to outputs/:
    Simulation_Tc_RE-Free_w_embeddings_PCA.pkl
    Simulation_Tc_RE_w_embeddings_PCA.pkl
    Simulation_Tc_all_w_embeddings_PCA.pkl

Each output pickle adds columns comp_emb_pca_8, comp_emb_pca_16,
comp_emb_pca_32, and comp_emb_pca_64 to the existing DataFrame.

Usage:
    python src/compress_embeddings_pca.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.log_to_file import log_output

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

DATASETS = [
    {
        "name": "RE-Free",
        "in":  "Simulation_Tc_RE-Free_w_embeddings.pkl",
        "out": "Simulation_Tc_RE-Free_w_embeddings_PCA.pkl",
    },
    {
        "name": "RE",
        "in":  "Simulation_Tc_RE_w_embeddings.pkl",
        "out": "Simulation_Tc_RE_w_embeddings_PCA.pkl",
    },
    {
        "name": "All",
        "in":  "Simulation_Tc_all_w_embeddings.pkl",
        "out": "Simulation_Tc_all_w_embeddings_PCA.pkl",
    },
]

PCA_SIZES = [8, 16, 32, 64]

# Create log directory 
from src.log_to_file import log_output
import os
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

@log_output("logs/compress_embeddings_pca.txt")
def compress_embeddings_pca() -> None:
    print("=" * 70)
    print("PCA compression of compound embeddings")
    print("=" * 70)
    print(f"Component sizes: {PCA_SIZES}")

    for ds in DATASETS:
        in_path  = OUTPUT_DIR / ds["in"]
        out_path = OUTPUT_DIR / ds["out"]

        print(f"\n{'-'*60}")
        print(f"Dataset : {ds['name']}")

        if not in_path.exists():
            print(f"  Input not found: {in_path}")
            print("  Run create_embeddings.py first.")
            continue

        df = pd.read_pickle(in_path)
        print(f"  Loaded {len(df)} rows")

        if "compound_embedding" not in df.columns:
            print("  'compound_embedding' column missing – skipping.")
            continue

        raw = np.vstack(df["compound_embedding"].values)
        print(f"  Raw embeddings shape: {raw.shape}")

        for n in PCA_SIZES:
            pca = PCA(n_components=n, random_state=RANDOM_SEED)
            compressed = pca.fit_transform(raw)
            col = f"comp_emb_pca_{n}"
            df[col] = list(compressed)
            var = pca.explained_variance_ratio_.sum()
            print(f"  PCA({n:2d} components): explained variance = {var:.3f}  → '{col}'")

        df.to_pickle(out_path)
        print(f"  Saved → {out_path}")

    print("\nDone. Next step: python src/train_sim_tc.py  (or the per-dataset variants)")


if __name__ == "__main__":
    log_path = PROJECT_ROOT / "logs" / "compress_embeddings_pca.txt"
    print(f"Output logged to: {log_path}")
    compress_embeddings_pca()
    print(f"Done. Results in: {OUTPUT_DIR}")
