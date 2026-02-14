# Configuration API Reference

## Experiment Configuration

::: dml_model_scaling.config.experiment_config.ExperimentConfig

::: dml_model_scaling.config.experiment_config.AnalysisConfig

## Settings

::: dml_model_scaling.config.settings.Settings

::: dml_model_scaling.config.settings.get_settings

## Example

```python
from dml_model_scaling.config import ExperimentConfig, AnalysisConfig, get_settings

# Experiment configuration
exp_config = ExperimentConfig(
    model_names=["small", "medium"],
    dataset_sizes=[1000, 2000],
    epochs_list=[3],
    learning_rates=[0.001],
)

# Analysis configuration
analysis_config = AnalysisConfig(
    n_estimators=200,
    cv_folds=3,
)

# Global settings
settings = get_settings()
print(f"Data directory: {settings.data_dir}")
print(f"Device: {settings.device}")
```
