#!/bin/bash
#SBATCH --job-name=dinov2_train    # Job name
#SBATCH --output=logs/dinov2_%j.out     # Name of output file (%j expands to jobId)
#SBATCH --error=logs/dinov2_%j.out      # Name of error file (%j expands to jobId)
#SBATCH --time=24:00:00            # Run time (hh:mm:ss) - 24 hours
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=32G                  # Memory per node
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --partition=batch_gpu      # Specify partition/queue name

# Load necessary modules (adjust these based on your cluster's setup)
module load CUDA/11.7.0

# Activate python environment
source dinov2_env/bin/activate

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# Print some information about the job
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "GPU assigned: $CUDA_VISIBLE_DEVICES"

# Install requirements
# pip install -r requirements.txt

# Run the training script
python training/dinov2_train.py

# Print completion message
echo "Job finished at $(date)" 