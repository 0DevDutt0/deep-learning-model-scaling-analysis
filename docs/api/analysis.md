# Analysis API Reference

## DML Analyzer

::: dml_model_scaling.analysis.dml.DMLAnalyzer

## Results

::: dml_model_scaling.analysis.dml.DMLResults

## Example Usage

```python
from pathlib import Path
from dml_model_scaling.analysis import DMLAnalyzer
from dml_model_scaling.config import AnalysisConfig

# Configure analysis
config = AnalysisConfig(
    input_path=Path("data/experiments.csv"),
    n_estimators=200,
    max_depth=5,
    cv_folds=3,
    random_seed=42,
)

# Run analysis
analyzer = DMLAnalyzer(config)
results = analyzer.analyze()

# Display results
print(results)
print(f"Causal effect per 1M params: {results.effect_per_million:.4f}")
```
