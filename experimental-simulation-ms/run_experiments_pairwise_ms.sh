#!/bin/bash
#SBATCH --job-name=Magnetization
#SBATCH --output=logs-slurm/magntization-error-correction%j.out
#SBATCH --error=logs-slurm/magntization-error-correction%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_partition  # Replace with the appropriate partition name
#SBATCH --gres=gpu:1  # Request 1 GPU

# Load necessary modules 
module load python-waterboa/2025.06

# Set the path to the virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv/mammos-ms"

echo "SCRIPT_DIR = $SCRIPT_DIR"
echo "VENV_PATH  = $VENV_PATH"
echo "Set Paths"

# Activate the virtual environment
source $VENV_PATH/bin/activate

echo "Activated Venv"

# Execute python commands

echo "Data pipeline ..."
srun --cpu-bind=none python src/data_pipeline.py --configdir "configs/augment_data_ms.yml"
echo "Data pipeline Done!"

echo "Run scikit experiments ..."
srun --cpu-bind=none python src/run_experiments.py --configdir "configs/scikit_models_config.yml"
echo "Finished with scikit."

echo "Run scikit experiments with embeddings ..."
srun --cpu-bind=none python src/run_experiments.py --configdir "configs/scikit_models_w_mat200_config.yml"
echo "Finished scikit with embds."

echo "Run PySR linear fit experiments ..."
srun --cpu-bind=none python src/run_experiments.py --configdir "configs/pysr_linear_models_config.yml"
echo "Finished with PySR."

echo "Run PyTorch experiments ..."
srun --cpu-bind=none python src/run_experiments.py --configdir "configs/pytorch_mlp.yml"
echo "Finished PyTorch experiments."

echo "Run PyTorch experiments with embeddings ..."
srun --cpu-bind=none python src/run_experiments.py --configdir "configs/pytorch_mlp_w_emb.yml"
echo "Finished PyTorch experiments with embeddings."

echo "Post process results ..."
srun --cpu-bind=none python src/post_process_results.py --dir "artifacts/"
echo "Finished post-processing."