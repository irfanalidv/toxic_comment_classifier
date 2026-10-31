# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-10-31

### Changed

- **Tokenizer no longer depends on Keras internals or pickle.** 0.3.0 unpickled a Keras
  `Tokenizer` whose class lives at the private path `keras.src.legacy.preprocessing.text`,
  which breaks whenever Keras moves it (and needs `tensorflow.keras.preprocessing`, which
  Keras 3 deprecates). The vocabulary now ships as JSON with a small built-in tokenizer and
  padding function. Outputs are identical to 0.3.0 (verified: same token ids, max score
  difference 1.2e-7), and the package is about 9 MB smaller because only the 19,999 words
  the model can use are kept.
- Model is built with the public Keras 3 API (`import keras`).
- `predict_batch()` runs one batched model call instead of one call per text.
- `predict()`/`predict_batch()` return plain Python `float` instead of `numpy.float64`.
- Requires Python 3.10+ (tested on 3.10-3.13) and `tensorflow>=2.16` / `keras>=3`.
  Removed the unused `pandas` and `scikit-learn` dependencies.
- Packaging modernized (`pyproject.toml` only, PEP 639 `license = "MIT"`, `LICENSE.txt` is
  included in distributions again).

### Added

- `classify_batch(texts)` and `flagged_labels(text, threshold=0.5)`.
- `ToxicCommentClassifier(clean=True)` to apply `clean_text()` automatically.
- `toxic-classifier` command-line tool (JSON Lines output, reads stdin).
- `toxic_classifier.__version__`, `LIST_CLASSES` and `clean_text` exported at package level,
  type hints and `py.typed`.
- Input validation with clear `TypeError`s.
- pytest suite (tokenizer parity, reference scores, batch/single consistency, CLI) and GitHub
  Actions CI on Python 3.10-3.13 with Ruff; community files.

### Fixed

- `clean_text` URL pattern `www.\S+` matched any character after `www`; now `www\.\S+`.

## [0.3.0] - 2025

- Previous release on PyPI.
