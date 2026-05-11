# Contributing to scikit-rec

Thank you for your interest in contributing to `scikit-rec`! We welcome bug reports, feature requests, and code contributions.

## How to contribute

1. Fork the repository and create a new branch.
2. Make your changes in a focused branch.
3. Add tests for bug fixes and new functionality.
4. Run the test suite locally:

```bash
git clone https://github.com/intuit/scikit-rec.git
cd scikit-rec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

5. Commit with a clear message and open a pull request against `main`.

## Guidelines

- Keep pull requests small and focused.
- Follow the existing code style and conventions.
- Add or update documentation when behavior changes.
- Use descriptive commit messages.

## Reporting issues

If you find a bug or have a feature idea, please open an issue at:

https://github.com/intuit/scikit-rec/issues

Include a clear description of the problem, a minimal reproduction example, and the expected behavior.

## Community

By contributing, you agree to abide by the project’s Code of Conduct:

- `CODE_OF_CONDUCT.md`

We aim to make this project welcoming, inclusive, and respectful.
