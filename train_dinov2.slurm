#!/bin/bash
#SBATCH --job-name=dinov2_train    # Job name
#SBATCH --output=logs/slurm_logs/dinov2_train_%j.out     # Name of output file (%j expands to jobId)
#SBATCH --error=logs/slurm_logs/dinov2_train_%j.out      # Name of error file (%j expands to jobId)
#SBATCH --time=72:00:00            # Run time (hh:mm:ss) - 72 hours
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (single task for single MIG slice)
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=32G                  # Memory per node
#SBATCH --gres=gpu:3g.40gb:1       # Request 40GB GPU
#SBATCH --partition=batch_gpu      # Specify partition/queue name
#SBATCH --gpu-bind=closest         # Bind to closest GPU

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
echo "Initial GPU Status:"
nvidia-smi

# Start GPU monitoring in background
(
    while true; do
        echo "=== GPU Status at $(date) ==="
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
        sleep 300  # Check every 5 minutes
    done
) > gpu_monitor.log 2>&1 &

# Install requirements
# pip install -r requirements.txt

# Run training with single MIG slice
python training/train_model.py

# Run training with distributed training across MIG slices
# srun python -m torch.distributed.launch --nproc_per_node=3 training/train_model.py

# Print completion message
echo "Job finished at $(date)"