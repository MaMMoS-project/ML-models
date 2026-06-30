# -*- coding: utf-8 -*-
"""Create compound embeddings for the simulated Tc datasets.

Reads from preprocessed_data/:
    Simulation_Tc_RE-Free.csv
    Simulation_Tc_RE.csv
    Simulation_Tc_all.csv

Saves to outputs/:
    Simulation_Tc_RE-Free_w_embeddings.pkl
    Simulation_Tc_RE_w_embeddings.pkl
    Simulation_Tc_all_w_embeddings.pkl

Each output pickle contains the original columns (composition, Tc_sim) plus a
compound_embedding column holding a 200-D numpy array per row.  Rows whose
compositions cannot be parsed or whose elements are absent from the matscholar200
vocabulary are dropped.

Usage:
    python src/create_embeddings.py
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.log_to_file import log_output

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "preprocessed_data"
EMB_FILE     = PROJECT_ROOT / "data" / "embeddings" / "element" / "matscholar200.json"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

DATASETS = [
    {
        "name": "RE-Free",
        "csv":  "Simulation_Tc_RE-Free.csv",
        "out":  "Simulation_Tc_RE-Free_w_embeddings.pkl",
    },
    {
        "name": "RE",
        "csv":  "Simulation_Tc_RE.csv",
        "out":  "Simulation_Tc_RE_w_embeddings.pkl",
    },
    {
        "name": "All",
        "csv":  "Simulation_Tc_all.csv",
        "out":  "Simulation_Tc_all_w_embeddings.pkl",
    },
]


def _load_elem_features(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _compound_embedding(comp_str: str, elem_features: Dict) -> Optional[np.ndarray]:
    """Element-abundance weighted mean of matscholar200 vectors."""
    try:
        comp   = Composition(comp_str)
        el_amt = comp.get_el_amt_dict()
        amounts = np.array(list(el_amt.values()), dtype=float)
        weights = amounts / amounts.sum()
        dim = len(next(iter(elem_features.values())))
        vec = np.zeros(dim)
        for el, w in zip(el_amt.keys(), weights):
            if el not in elem_features:
                return None
            vec += w * np.array(elem_features[el])
        return vec
    except Exception:
        return None
    
# Create log directory 
from src.log_to_file import log_output
import os
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)


@log_output("logs/create_embeddings.txt")
def create_embeddings() -> None:
    print("=" * 70)
    print("Creating compound embeddings for simulated Tc datasets")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EMB_FILE.exists():
        print(f"ERROR: element embedding file not found: {EMB_FILE}")
        sys.exit(1)

    elem_features = _load_elem_features(EMB_FILE)
    dim = len(next(iter(elem_features.values())))
    print(f"\nElement embeddings: {EMB_FILE}")
    print(f"Vocabulary: {len(elem_features)} elements  |  dimension: {dim}")

    for ds in DATASETS:
        csv_path = DATA_DIR / ds["csv"]
        out_path = OUTPUT_DIR / ds["out"]

        print(f"\n{'-'*60}")
        print(f"Dataset : {ds['name']}  ({ds['csv']})")

        if not csv_path.exists():
            print(f"  File not found – skipping.")
            continue

        df = pd.read_csv(csv_path)
        df = df[df["Tc_sim"].notna()].copy()
        df["Tc_sim"] = df["Tc_sim"].astype(float)
        print(f"  Rows with valid Tc_sim : {len(df)}")

        df["compound_embedding"] = df["composition"].apply(
            lambda c: _compound_embedding(c, elem_features)
        )
        before = len(df)
        df = df[df["compound_embedding"].notna()].copy()
        print(f"  Dropped (un-embeddable): {before - len(df)}")
        print(f"  Embeddable rows        : {len(df)}")

        df.to_pickle(out_path)
        print(f"  Saved → {out_path}")

    print("\nDone. Next step: python src/compress_embeddings_pca.py  (simulated Tc)")


if __name__ == "__main__":
    log_path = PROJECT_ROOT / "logs" / "create_embeddings.txt"
    print(f"Output logged to: {log_path}")
    create_embeddings()
    print(f"Done. Results in: {OUTPUT_DIR}")
