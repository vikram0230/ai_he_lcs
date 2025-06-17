#!/bin/bash
#SBATCH --job-name=rad_dino_inf                          # Job name
#SBATCH --output=logs/slurm_logs/rad_dino_inf_%j.out     # Name of output file (%j expands to jobId)
#SBATCH --error=logs/slurm_logs/rad_dino_inf_%j.out      # Name of error file (%j expands to jobId)
#SBATCH --time=24:00:00                                # Run time (hh:mm:ss) - 24 hours should be enough for inference
#SBATCH --nodes=1                                      # Number of nodes
#SBATCH --ntasks=1                                     # Number of tasks
#SBATCH --cpus-per-task=4                              # Number of CPU cores per task
#SBATCH --mem=32G                                      # Memory per node
#SBATCH --gres=gpu:3g.40gb:1                           # Request 1 GPU
#SBATCH --partition=batch_gpu                          # Specify partition/queue name

# Load necessary modules
module load CUDA/11.7.0

# Activate python environment
source dinov2_env/bin/activate

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Print some information about the job
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"
nvidia-smi

pip install -r requirements.txt

# Run the inference script
python src/evaluation/inference.py --config src/config/rad_dino_config.yaml --run_id e4e75c936382449696c5b6dc2efa4cba

# Print completion message
echo "Job finished at $(date)" 