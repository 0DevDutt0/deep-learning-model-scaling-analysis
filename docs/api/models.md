# Models API Reference

## Model Architectures

::: dml_model_scaling.models.small_cnn.SmallCNN

::: dml_model_scaling.models.medium_cnn.MediumCNN

::: dml_model_scaling.models.large_cnn.LargeCNN

## Utilities

::: dml_model_scaling.models.utils.count_parameters

::: dml_model_scaling.models.utils.get_model_by_name

## Model Registry

```python
from dml_model_scaling.models import MODEL_REGISTRY

# Available models
for name, model_class in MODEL_REGISTRY.items():
    print(f"{name}: {model_class}")
```
