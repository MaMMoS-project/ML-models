# -*- coding: utf-8 -*-
"""Train models to predict simulated Tc — RE-Free dataset.

Run in order:

    python src/create_embeddings.py
    python src/compress_embeddings_pca.py
    python src/train_sim_tc_re_free.py

Input file (outputs/):
    Simulation_Tc_RE-Free_w_embeddings_PCA.pkl

Outputs:
    results/RE-Free_sim_results.csv
    results/sim_tc_comparison.csv      (updated from all available datasets)
    results/sim_tc_best_by_dataset.csv (updated from all available datasets)
    results/figures/RE-Free_*.png
    logs/train_sim_tc_re_free.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.log_to_file import log_output
from src.train_sim_tc import (
    DATASETS,
    PROJECT_ROOT,
    RESULTS_DIR,
    train_one_dataset,
    update_global_summary,
)

_DS = next(d for d in DATASETS if d["name"] == "RE-Free")


@log_output("logs/train_sim_tc_re_free.txt")
def main() -> None:
    print("=" * 70)
    print("Training (RE-Free): compound embedding → simulated Tc")
    print("=" * 70)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_one_dataset(_DS, RESULTS_DIR / "figures")
    update_global_summary()


if __name__ == "__main__":
    log_path = PROJECT_ROOT / "logs" / "train_sim_tc_re_free.txt"
    print(f"Output logged to: {log_path}")
    main()
    print(f"Done. Results in: {RESULTS_DIR}")
