# MLOps: Data Validation & Experiment Tracking

A comprehensive MLOps project demonstrating best practices for data validation, experiment tracking, and reproducible machine learning pipelines using industry-standard tools.

##  Overview

This repository implements a complete MLOps workflow focused on:
- **Data Validation**: Automated testing of data quality and integrity using pytest
- **Experiment Tracking**: Systematic tracking of experiments, metrics, and artifacts with Weights & Biases
- **Pipeline Orchestration**: Modular, reusable ML pipeline components with MLflow
- **Configuration Management**: Flexible hyperparameter management using Hydra
- **Reproducibility**: Version-controlled code, data, and environments

##  Features

### Data Validation
- **Deterministic Tests**: Schema validation, column presence, data type checks, range validation
- **Non-Deterministic Tests**: Statistical tests (Kolmogorov-Smirnov) with Bonferroni correction for multiple testing
- **Automated Quality Checks**: Integrated into the pipeline to catch data drift and quality issues early

### Experiment Tracking
- **Comprehensive Logging**: All experiments tracked with W&B including configurations, metrics, and artifacts
- **Artifact Versioning**: Automatic versioning of datasets, models, and preprocessing artifacts
- **Hyperparameter Sweeps**: Grid search and parallel execution support via Hydra
- **Model Registry**: Production-ready model management with proper tagging

### Pipeline Components
- **Modular Design**: Each component is independently executable and reusable
- **MLflow Projects**: Standardized component structure with explicit dependencies
- **Artifact Lineage**: Full tracking of data and model provenance
- **Environment Management**: Reproducible conda environments with pinned dependencies

##  Prerequisites

- Python 3.8+
- Conda or Miniconda
- Weights & Biases account (free tier available)
- MLflow

##  Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create and activate the conda environment:
```bash
# Install miniconda (if not already installed)
# Create an account on Weights & Biases
conda create --name mlops_env python=3.8 mlflow jupyter pandas matplotlib requests -c conda-forge
conda activate mlops_env
pip install wandb pytest hydra-core
```

3. Configure W&B:
```bash
wandb login
```

##  Usage

### Running the Complete Pipeline

Execute the full pipeline with default configuration:
```bash
mlflow run .
```

### Custom Configuration

Override parameters using Hydra:
```bash
# Change experiment name and test size
mlflow run . -P hydra_options="main.experiment_name=prod data.test_size=0.2"

# Run hyperparameter sweep
mlflow run . -P hydra_options="-m model.max_depth=5,10,15 model.n_estimators=50,100,150"
```

##  Key Concepts

- **MLflow**: Orchestrates the entire ML pipeline and tracks experiments
- **Weights & Biases**: Manages artifacts, datasets, and model versions
- **Hydra**: Handles configuration and enables hyperparameter sweeps
- **Pytest**: Validates data quality with deterministic and statistical tests
