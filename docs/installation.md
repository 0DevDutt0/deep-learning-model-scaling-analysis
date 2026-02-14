# Installation

## Requirements

- Python 3.9 or higher
- pip or conda

## Install from PyPI

```bash
pip install deep-learning-model-scaling-analysis
```

## Install from Source

### Clone the repository

```bash
git clone https://github.com/0DevDutt0/deep-learning-model-scaling-analysis.git
cd deep-learning-model-scaling-analysis
```

### Install in development mode

```bash
pip install -e ".[dev]"
```

### Install pre-commit hooks

```bash
pre-commit install
```

## Optional Dependencies

### Documentation

To build documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

### Jupyter Notebooks

For interactive exploration:

```bash
pip install -e ".[notebook]"
jupyter notebook
```

### All Dependencies

Install everything:

```bash
pip install -e ".[all]"
```

## Docker Installation

### Pull from Docker Hub

```bash
docker pull deep-learning-model-scaling-analysis:latest
```

### Build from source

```bash
docker build -t deep-learning-model-scaling-analysis:latest .
```

### Run with Docker Compose

```bash
docker-compose up dml-scale
```

## Verification

Verify the installation:

```bash
# Check CLI is available
dml-scale --version

# Run a quick test
python -c "from dml_model_scaling import SmallCNN; print('OK')"
```

## Troubleshooting

### PyTorch Installation

If you have issues with PyTorch, install it separately:

```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### NumPy Compatibility

If you encounter NumPy version conflicts:

```bash
pip install "numpy>=1.24.0,<3.0.0"
```

### Permission Issues

On some systems, you may need to use `--user`:

```bash
pip install --user deep-learning-model-scaling-analysis
```

## Next Steps

Continue to the [Getting Started Guide](getting-started.md) to learn how to use the package.
