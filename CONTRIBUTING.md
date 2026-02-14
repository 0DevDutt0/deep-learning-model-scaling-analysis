# Contributing to Deep Learning Model Scaling Analysis

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/0DevDutt0/deep-learning-model-scaling-analysis.git
   cd deep-learning-model-scaling-analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/unit/test_models.py

# Run specific test
pytest tests/unit/test_models.py::test_small_cnn_parameter_count
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Type check
mypy src tests
```

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit. To run manually:

```bash
pre-commit run --all-files
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Write Google-style docstrings
- Keep line length to 100 characters
- Use meaningful variable names

### Example:

```python
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int,
    learning_rate: float = 0.001,
) -> tuple[float, float]:
    """Train a PyTorch model on the provided data.

    Args:
        model: The neural network model to train.
        data_loader: DataLoader containing training data.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        A tuple containing (final_accuracy, training_time).

    Raises:
        ValueError: If epochs is less than 1.
    """
    # Implementation
    pass
```

## Testing Guidelines

- Write tests for all new features
- Maintain test coverage above 80%
- Use descriptive test names
- Use fixtures for common setup
- Mock external dependencies

### Test Structure:

```python
def test_feature_name_expected_behavior():
    """Test that feature_name produces expected behavior."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_value
```

## Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Link to related issues
   - Test results
   - Documentation updates

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant code snippets or error messages

## Questions?

Feel free to open an issue for questions or clarifications!
