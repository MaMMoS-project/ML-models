# -*- coding: utf-8 -*-
"""Train models to predict experimental Tc — All (combined) dataset.

Run in order:

    python src/create_embeddings.py
    python src/compress_embeddings_pca.py
    python src/train_exp_tc_all.py

Input file (outputs/):
    Experimental_Tc_all_w_embeddings_PCA.pkl

Outputs:
    results/exp_tc/All_results.csv
    results/exp_tc/exp_tc_comparison.csv      (updated from all available datasets)
    results/exp_tc/exp_tc_best_by_dataset.csv (updated from all available datasets)
    results/exp_tc/figures/All_*.png
    logs/train_exp_tc_all.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.log_to_file import log_output
from src.train_exp_tc import (
    DATASETS,
    PROJECT_ROOT,
    RESULTS_DIR,
    train_one_dataset,
    update_global_summary,
)

_DS = next(d for d in DATASETS if d["name"] == "All")


@log_output("logs/train_exp_tc_all.txt")
def main() -> None:
    print("=" * 70)
    print("Training (All): compound embedding → experimental Tc")
    print("=" * 70)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_one_dataset(_DS, RESULTS_DIR / "figures")
    update_global_summary()


if __name__ == "__main__":
    log_path = PROJECT_ROOT / "logs" / "train_exp_tc_all.txt"
    print(f"Output logged to: {log_path}")
    main()
    print(f"Done. Results in: {RESULTS_DIR}")
