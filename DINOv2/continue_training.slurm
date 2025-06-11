#!/bin/bash
#SBATCH --job-name=dinov2_cont    # Job name
#SBATCH --output=logs/slurm_logs/dinov2_cont_%j.out     # Name of output file (%j expands to jobId)
#SBATCH --error=logs/slurm_logs/dinov2_cont_%j.out      # Name of error file (%j expands to jobId)
#SBATCH --time=48:00:00            # Run time (hh:mm:ss) - 48 hours
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=32G                  # Memory per node
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --partition=batch_gpu      # Specify partition/queue name

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

# Check if run_id is provided
if [ -z "$1" ]; then
    echo "Error: Please provide the MLflow run ID as an argument"
    echo "Usage: sbatch continue_training.slurm <run_id>"
    exit 1
fi

# Store the run_id
RUN_ID=$1
echo "Continuing training from MLflow run ID: $RUN_ID"

# Run the continuation training script with the run_id
python src/training/continue_training.py --run_id $RUN_ID

# Print completion message
echo "Job finished at $(date)" 