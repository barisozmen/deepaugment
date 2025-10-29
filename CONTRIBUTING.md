# Contributing to DeepAugment

We welcome contributions! Follow these steps to get started.

## Quick Setup

```bash
git clone https://github.com/barisozmen/deepaugment.git
cd deepaugment
make setup  # Installs dependencies + git hooks
```

## Development Workflow

```bash
vim src/deepaugment/optimizer.py

make test

git commit -m "Improve optimizer convergence"

git push origin your-branch
```

## Code Style

- Match existing style (read the code first)
- Docstrings: one-line summary, then details
- Functions do one thing well (Unix philosophy)

## Testing

Lightweight end-to-end behavioral tests only. We don't prefer unit tests in most cases as they slow down development.

```bash
make test           # Run all tests
uv run pytest -v    # Verbose output
uv run tests/test_e2e.py::test_name  # Single test
```

## Pull Requests

1. **Fork & branch** from `master`
2. **One feature per PR** - small, focused changes
3. **Test locally** - `make test` must pass
4. **Describe why** - what problem does this solve?
5. **Link issues** - reference related issues

## Version Management

- **Single source of truth** - version lives only in `pyproject.toml`
- **Version updates** - update `pyproject.toml`'s `version` field, the `.git/hooks/post-commit` will generatea tags automatically. Later, push them by `git push --tags`


## Questions?

Open an issue or check existing discussions.

## Philosophy

DeepAugment follows these principles:
- **Minimize complexity** - Bayesian optimization over RL (100× fewer iterations)
- **Be modular** - users can plug in their own models
- **Optimize for users** - simple API, powerful configuration

Code is designed to be very readable. If you find a part hard to understand, please open an issue, or open a PR with your commits that makes the code simpler and more understandable.

Inspired from:
- [Unix philosophy](https://en.wikipedia.org/wiki/Unix_philosophy)
- [Rails Doctrine](https://rubyonrails.org/doctrine)