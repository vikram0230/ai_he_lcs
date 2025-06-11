# DINOv2 Medical Imaging Project

*Last updated 06/10/2025 by Vikram Harikrishnan*

This repository contains the implementation of a DINOv2-based medical imaging classification system. The project is organized following best practices for machine learning projects, with clear separation of concerns and modular design.

## Project Structure

```
src/
├── models/          # Model architecture definitions
├── data/           # Data loading and processing
├── training/       # Training scripts and utilities
├── evaluation/     # Evaluation and monitoring tools
├── utils/          # Helper functions and utilities
└── analysis/       # Testing and experimental code
```

### Directory Descriptions

#### `models/`
Contains the model architecture implementations:
- DINOv2 classifier implementations
- Custom model architectures
- Model-related utilities

#### `data/`
Handles all data-related operations:
- Dataset loaders
- Data preprocessing
- Data augmentation
- Data validation

#### `training/`
Contains training-related code:
- Training scripts
- Training utilities
- Model checkpointing
- Training configuration

#### `evaluation/`
Evaluation and monitoring tools:
- Model evaluation scripts
- Performance metrics
- Bias monitoring
- Model validation

#### `utils/`
Common utilities and helper functions:
- Logging utilities
- Common functions
- Configuration management
- File handling

#### `analysis/`
A sandbox directory for testing and experimental code:
- Quick experiments and prototypes
- Temporary analysis scripts
- Test implementations
- Code snippets for debugging
- Not intended for production use

## Implementation Details

### Model Architecture
The project uses DINOv2 (Self-Distillation with No Labels) as the base model for medical image classification. The implementation includes:
- Custom classifier heads for specific medical imaging tasks
- Transfer learning capabilities
- Model checkpointing and saving

### Data Processing
- DICOM image loading and preprocessing
- Data augmentation for medical images
- Dataset splitting and validation
- Data normalization and standardization

### Training Pipeline
- Configurable training parameters
- Multi-GPU training support
- Early stopping and model checkpointing
- Learning rate scheduling
- Gradient accumulation

### Evaluation
- Comprehensive model evaluation metrics
- Bias monitoring and analysis
- Performance visualization
- Model validation tools

## Usage

### Setup
1. Create a virtual environment:
```bash
python -m venv dinov2_env
source dinov2_env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training
To train a new model using SLURM:
```bash
sbatch train_dinov2.slurm
```

The training job uses the following SLURM configuration:
- 72 hours runtime
- 1 node with 1 task
- 4 CPU cores per task
- 32GB memory
- 1 GPU (40GB)
- GPU monitoring every 5 minutes

For local training with a custom config:
```bash
python src/training/train_model.py --config path/to/custom_config.yaml
```

### Inference
To run inference using SLURM:
```bash
sbatch inference_dinov2.slurm
```

The inference job uses the following SLURM configuration:
- 24 hours runtime
- 1 node with 1 task
- 4 CPU cores per task
- 32GB memory
- 1 GPU
- Batch GPU partition

For local inference with custom parameters:
```bash
python src/evaluation/inference.py \
    --config path/to/custom_config.yaml \
    --run_id <mlflow_run_id> \
    --visualize_attention
```

Arguments:
- `--config`: Path to the configuration file (default: src/config/config.yaml)
- `--run_id`: MLflow run ID to load the model from (required)
- `--visualize_attention`: Flag to enable attention map visualization (optional)

## Configuration

The project uses YAML configuration files for managing parameters. Key configuration categories include:
- Model architecture parameters
- Training hyperparameters
- Data processing settings
- Evaluation metrics
- Logging configuration