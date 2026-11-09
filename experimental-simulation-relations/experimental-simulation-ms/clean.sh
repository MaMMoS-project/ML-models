#!/bin/bash
# clean.sh — remove ALL files produced by a previous pipeline run, leaving a clean
# slate so a fresh run cannot pick up stale .pkl / PCA / ONNX leftovers.
#
# Deletes the generated CONTENTS of (directories are kept, empty):
#   preprocessed_data/   Experimental_Ms_*.csv            (preprocess_ms_data.py)
#   outputs/             *_w_embeddings[_PCA].pkl, pca_*   (create_embeddings / compress_embeddings_pca)
#   results/             *.csv, onnx_models/, figures/     (train_ms_*.py)
#   logs/                per-stage *.txt logs
#   figures/             deliverable parity PNGs
# plus the SLURM job logs (*.out.*, *.err.*) at the project root and src/__pycache__/.
#
# NEVER touches inputs or source:
#   data/ (incl. data/raw, data/embeddings), src/*.py, training_config.yaml,
#   run_1node-*.sh, clean.sh, *.md, *.txt, demonstrator_notebook/
#
# After cleaning, re-run from scratch:
#   sbatch run_1node-prep.sh      # then, once it finishes:
#   sbatch run_1node-ALL.sh ; sbatch run_1node-RE.sh ; sbatch run_1node-RE-free.sh

set -u
cd "$(dirname "$0")" || exit 1
echo "Cleaning pipeline outputs in: $(pwd)"

removed=0
for d in preprocessed_data outputs results logs figures; do
    if [ -d "$d" ]; then
        n=$(find "$d" -mindepth 1 -type f 2>/dev/null | wc -l)
        find "$d" -mindepth 1 -delete 2>/dev/null || true
        [ "$n" -gt 0 ] && echo "  $d/ : removed $n files"
        removed=$((removed + n))
    fi
done

# SLURM job logs at the project root (prep / ALL / RE / RE-free / predict .out & .err)
s=$(ls ./*.out.* ./*.err.* 2>/dev/null | wc -l)
if [ "$s" -gt 0 ]; then
    rm -f ./*.out.* ./*.err.* 2>/dev/null || true
    echo "  SLURM logs : removed $s files"
    removed=$((removed + s))
fi

# Python bytecode cache
rm -rf src/__pycache__ 2>/dev/null || true

echo "Done — removed $removed generated files. Inputs and source left untouched."
