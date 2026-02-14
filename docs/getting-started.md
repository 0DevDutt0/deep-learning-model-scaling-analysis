# Getting Started

This guide will help you get started with Deep Learning Model Scaling Analysis.

## Quick Start

### 1. Run Training Experiments

```bash
dml-scale train run --output data/experiments.csv
```

This will:
- Train 3 model architectures (Small, Medium, Large)
- Test 3 dataset sizes (2000, 5000, 8000 samples)
- Try 2 epoch counts (3, 5)
- Test 2 learning rates (0.001, 0.0005)
- Save results to CSV

**Total experiments**: 36 (3 × 3 × 2 × 2)

### 2. Analyze Results

```bash
dml-scale analyze run --input data/experiments.csv
```

This will:
- Load experiment results
- Fit Double Machine Learning model
- Estimate causal effect of model size on accuracy
- Display results table

## Python API

### Training Experiments

```python
from dml_model_scaling import ExperimentRunner
from dml_model_scaling.config import ExperimentConfig

# Configure experiments
config = ExperimentConfig(
    model_names=["small", "medium"],
    dataset_sizes=[1000, 2000],
    epochs_list=[3],
    learning_rates=[0.001],
    batch_size=64,
    device="auto",  # Automatically select best device
)

# Run experiments
runner = ExperimentRunner(config)
results_path = runner.run()
print(f"Results saved to: {results_path}")
```

### Causal Analysis

```python
from dml_model_scaling.analysis import DMLAnalyzer
from dml_model_scaling.config import AnalysisConfig

# Configure analysis
config = AnalysisConfig(
    input_path="data/experiments.csv",
    n_estimators=200,
    max_depth=5,
    cv_folds=3,
)

# Run analysis
analyzer = DMLAnalyzer(config)
results = analyzer.analyze()

# Display results
print(results)
print(f"\nInterpretation: Adding 1M parameters improves accuracy by {results.effect_per_million:.4f}")
```

### Using Individual Models

```python
import torch
from dml_model_scaling.models import SmallCNN, get_model_by_name
from dml_model_scaling.models.utils import count_parameters

# Method 1: Direct instantiation
model = SmallCNN()

# Method 2: By name
model = get_model_by_name("small")

# Check parameters
num_params = count_parameters(model)
print(f"Parameters: {num_params:,}")

# Forward pass
x = torch.randn(1, 1, 28, 28)
output = model(x)
print(f"Output shape: {output.shape}")  # (1, 10)
```

## Configuration

### Environment Variables

Set environment variables with the `DML_` prefix:

```bash
export DML_DATA_DIR=/path/to/data
export DML_OUTPUT_DIR=/path/to/outputs
export DML_LOG_LEVEL=INFO
export DML_DEVICE=cuda
export DML_RANDOM_SEED=42
```

### Configuration Files

Create a `config.yaml` file:

```yaml
model_names:
  - small
  - medium
  - large

dataset_sizes:
  - 2000
  - 5000
  - 8000

epochs_list:
  - 3
  - 5

learning_rates:
  - 0.001
  - 0.0005

batch_size: 64
device: auto
random_seed: 42
```

Load it in Python:

```python
import yaml
from dml_model_scaling.config import ExperimentConfig

with open("config.yaml") as f:
    config_dict = yaml.safe_load(f)

config = ExperimentConfig(**config_dict)
```

## Docker Usage

### Run with Docker

```bash
# Pull image
docker pull deep-learning-model-scaling-analysis:latest

# Run training
docker run -v $(pwd)/data:/app/data deep-learning-model-scaling-analysis:latest train run

# Run analysis
docker run -v $(pwd)/data:/app/data deep-learning-model-scaling-analysis:latest analyze run
```

### Docker Compose

```bash
# Start service
docker-compose up dml-scale

# Run custom command
docker-compose run dml-scale train run --models small,medium
```

## Advanced Usage

For more advanced topics, see:

- [Configuration Guide](configuration.md)
- [Basic Tutorial](tutorials/basic.md)
- [Advanced Tutorial](tutorials/advanced.md)
- [API Reference](api/models.md)
