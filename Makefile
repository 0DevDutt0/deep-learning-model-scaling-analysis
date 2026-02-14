.PHONY: help install install-dev test test-cov lint format type-check clean docs serve-docs build docker-build docker-run pre-commit all

help:
    @echo "Available commands:"
    @echo "  make install        Install package in production mode"
    @echo "  make install-dev    Install package with development dependencies"
    @echo "  make test           Run tests"
    @echo "  make test-cov       Run tests with coverage report"
    @echo "  make lint           Run linting checks"
    @echo "  make format         Format code with ruff"
    @echo "  make type-check     Run mypy type checking"
    @echo "  make clean          Remove build artifacts and caches"
    @echo "  make docs           Build documentation"
    @echo "  make serve-docs     Serve documentation locally"
    @echo "  make build          Build distribution packages"
    @echo "  make docker-build   Build Docker image"
    @echo "  make docker-run     Run Docker container"
    @echo "  make pre-commit     Run pre-commit hooks"
    @echo "  make all            Run format, lint, type-check, and test"

install:
    pip install -e .

install-dev:
    pip install -e ".[dev,docs]"
    pre-commit install

test:
    pytest tests/ -v

test-cov:
    pytest tests/ --cov=dml_model_scaling --cov-report=html --cov-report=term

lint:
    ruff check src tests

format:
    ruff format src tests
    ruff check --fix src tests

type-check:
    mypy src tests

clean:
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info
    rm -rf .pytest_cache/
    rm -rf .mypy_cache/
    rm -rf .ruff_cache/
    rm -rf htmlcov/
    rm -rf .coverage
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete

docs:
    mkdocs build

serve-docs:
    mkdocs serve

build: clean
    python -m build

docker-build:
    docker build -t deep-learning-model-scaling-analysis:latest .

docker-run:
    docker run -it --rm deep-learning-model-scaling-analysis:latest

pre-commit:
    pre-commit run --all-files

all: format lint type-check test

.DEFAULT_GOAL := help
