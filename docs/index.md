# Deep Learning Model Scaling Analysis

**Causal inference analysis of neural network scaling laws using Double Machine Learning**

[![CI](https://github.com/0DevDutt0/deep-learning-model-scaling-analysis/workflows/CI/badge.svg)](https://github.com/0DevDutt0/deep-learning-model-scaling-analysis/actions)
[![codecov](https://codecov.io/gh/0DevDutt0/deep-learning-model-scaling-analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/0DevDutt0/deep-learning-model-scaling-analysis)
[![PyPI version](https://badge.fury.io/py/deep-learning-model-scaling-analysis.svg)](https://badge.fury.io/py/deep-learning-model-scaling-analysis)
[![Python versions](https://img.shields.io/pypi/pyversions/deep-learning-model-scaling-analysis.svg)](https://pypi.org/project/deep-learning-model-scaling-analysis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Deep Learning Model Scaling Analysis is a professional Python package that combines deep learning experimentation with causal inference to analyze neural network scaling laws. It provides:

- 🧠 **CNN Architectures**: Three carefully designed models (Small ~10K, Medium ~40K, Large ~160K parameters)
- 🔬 **Controlled Experiments**: Systematic training across multiple configurations
- 📊 **Causal Analysis**: Double Machine Learning to estimate true causal effects
- 🎯 **Type-Safe**: Full type hints and comprehensive testing
- 🚀 **Production-Ready**: CLI, Docker support, and CI/CD pipelines

## Key Features

### 🎓 Research-Oriented

Apply rigorous causal inference methods to understand how model size **causally** affects performance, controlling for confounders like training time and dataset size.

### 🛠️ Developer-Friendly

- **Type hints** throughout for better IDE support
- **Comprehensive tests** with >80% coverage
- **CLI interface** with rich output
- **Pydantic validation** for configuration
- **Professional logging** and error handling

### 📦 Production-Ready

- **Docker support** for reproducible deployments
- **CI/CD pipelines** with GitHub Actions
- **Pre-commit hooks** for code quality
- **Comprehensive documentation** with examples
- **Semantic versioning** and changelog

## Quick Example

```python
from dml_model_scaling import ExperimentRunner, DMLAnalyzer
from dml_model_scaling.config import ExperimentConfig, AnalysisConfig

# Run experiments
config = ExperimentConfig(
    model_names=["small", "medium", "large"],
    dataset_sizes=[2000, 5000, 8000],
    epochs_list=[3, 5],
    learning_rates=[0.001, 0.0005],
)
runner = ExperimentRunner(config)
results_path = runner.run()

# Analyze causal effects
analysis_config = AnalysisConfig(input_path=results_path)
analyzer = DMLAnalyzer(analysis_config)
results = analyzer.analyze()

print(f"Causal effect: {results.effect_per_million:.4f} accuracy per 1M parameters")
```

## CLI Usage

```bash
# Run training experiments
dml-scale train run --models small,medium --dataset-sizes 1000,2000

# Analyze results
dml-scale analyze run --input data/experiments.csv

# Get help
dml-scale --help
```

## Architecture

```
deep-learning-model-scaling-analysis/
├── src/dml_model_scaling/     # Main package
│   ├── models/                # CNN architectures
│   ├── experiments/           # Training pipeline
│   ├── analysis/              # DML causal inference
│   ├── config/                # Configuration models
│   ├── utils/                 # Utilities
│   └── cli/                   # Command-line interface
├── tests/                     # Comprehensive test suite
├── docs/                      # MkDocs documentation
└── .github/workflows/         # CI/CD pipelines
```

## Installation

See [Installation Guide](installation.md) for detailed instructions.

```bash
pip install deep-learning-model-scaling-analysis
```

## Next Steps

- 📖 [Getting Started Guide](getting-started.md)
- 📚 [API Reference](api/models.md)
- 🎓 [Tutorials](tutorials/basic.md)
- 🤝 [Contributing Guide](contributing.md)

## Citation

If you use this package in your research, please cite:

```bibtex
@software{deep_learning_model_scaling_analysis,
  title = {Deep Learning Model Scaling Analysis: Causal Inference for Neural Network Scaling Laws},
  author = {Deep Learning Model Scaling Analysis Contributors},
  year = {2024},
  url = {https://github.com/0DevDutt0/deep-learning-model-scaling-analysis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/0DevDutt0/deep-learning-model-scaling-analysis/blob/main/LICENSE) file for details.
