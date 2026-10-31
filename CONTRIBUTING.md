# Contributing to toxic_comment_classifier

Thanks for helping improve `toxic_comment_classifier`!

## Development setup

```bash
git clone https://github.com/irfanalidv/toxic_comment_classifier.git
cd toxic_comment_classifier
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .          # lint
ruff format --check . # formatting
pytest                # tests
```

- Add or update tests for any behaviour change.
- Add a line to `CHANGELOG.md` under **Unreleased**.
- Keep pull requests focused: one fix or feature per PR.

## Reporting bugs and requesting features

Use the [issue templates](https://github.com/irfanalidv/toxic_comment_classifier/issues/new/choose).
