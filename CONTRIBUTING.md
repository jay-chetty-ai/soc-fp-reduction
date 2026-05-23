# Contributing

Thanks for your interest in contributing to the SOC False Positive Reduction project.

## Development Setup

1. Fork and clone the repo
2. Run the setup script or follow [docs/setup.md](docs/setup.md)
3. Create a branch for your work: `git checkout -b feature/your-feature`

## Development Rules

- **Type hints** on all function signatures
- **Docstrings** on all public functions
- **No hardcoded paths** — use `config.yaml`
- **Logging** via Python `logging` module, not print statements
- **Tests required** — every change must include tests

## Testing

```bash
# Run the full suite
pytest tests/ -v --tb=short

# Run a specific epic
pytest tests/test_epic1_data.py -v
```

All tests must pass before submitting a PR.

## Pull Request Process

1. Update tests for any changed functionality
2. Run the full test suite and confirm all tests pass
3. Update documentation if you changed interfaces or behavior
4. Fill out the PR template completely
5. Request review

## Commit Messages

Format: `epic[N] story[N.M]: brief description`

Examples:
- `epic1 story1.1: add CICIDS2017 dataset loader`
- `epic2 story2.3: implement adversarial validation agent`

## Code Style

- Python 3.11+
- Line length: 100 characters
- Use `ruff` for linting if available
- Follow existing patterns in the codebase

## Reporting Issues

Use the issue templates provided. Include:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Environment details (OS, Python version, GPU)
