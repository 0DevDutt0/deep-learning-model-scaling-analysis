# Changelog

All notable changes to Deep Learning Model Scaling Analysis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-02-14

### Added
- Initial release with professional package structure
- Type-hinted CNN models (SmallCNN ~10K params, MediumCNN ~40K params, LargeCNN ~160K params)
- Experiment runner with configurable training pipeline
- Double Machine Learning causal analysis using econML
- CLI interface with Typer (`dml-scale` command)
- Comprehensive test suite with pytest
- GitHub Actions CI/CD workflows
- Docker support with multi-stage builds
- MkDocs documentation
- Pre-commit hooks for code quality
- Pydantic-based configuration management
- Rich console output with progress tracking

### Changed
- Restructured codebase to src-layout
- Added full type hints throughout
- Implemented Google-style docstrings
- Enhanced error handling with custom exceptions

### Removed
- Legacy scripts and directories (causal_analysis/, experiments/, models/)
- Old configuration files (requirements.txt, test_env.py, test_installation.py)
- Temporary documentation (README_NEW.md, TRANSFORMATION_SUMMARY.md, QUICK_START.md, WINDOWS_SETUP.md)
- Old experimental data (data/experiments.csv)

[0.1.0]: https://github.com/0DevDutt0/deep-learning-model-scaling-analysis/releases/tag/v0.1.0
