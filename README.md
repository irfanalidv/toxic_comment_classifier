<div align="center">

# Toxic Comment Classifier

**A pre-trained deep-learning model that scores comments for six kinds of toxicity, offline, in one line of Python.**

[![PyPI version](https://img.shields.io/pypi/v/toxic-comment-classifier.svg)](https://pypi.org/project/toxic-comment-classifier/)
[![Python versions](https://img.shields.io/pypi/pyversions/toxic-comment-classifier.svg)](https://pypi.org/project/toxic-comment-classifier/)
[![Tests](https://github.com/irfanalidv/toxic_comment_classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/irfanalidv/toxic_comment_classifier/actions/workflows/tests.yml)
[![PyPI Downloads](https://static.pepy.tech/badge/toxic-comment-classifier)](https://pepy.tech/projects/toxic-comment-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/irfanalidv/toxic_comment_classifier/blob/main/LICENSE.txt)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

A Python library for classifying toxic comments using deep learning.
It supports detecting multiple types of toxicity including obscene language, threats, and identity hate.

## ✨ Features

- **Six labels, multi-label**: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`, each scored independently from 0 to 1.
- **Pre-trained and bundled**: the model weights and vocabulary ship inside the package, so there is nothing to download and it works offline.
- **Simple API**: `classify()` for per-label scores, `predict()` for one overall score, batch versions of both, and `flagged_labels()` for threshold-based moderation.
- **Fast batch scoring**: batches go through the network in a single call.
- **Command line**: `toxic-classifier "some comment"` prints JSON.
- **Small model**: a BiLSTM (about 4.6 MB of weights) that runs comfortably on CPU.

**The model:** a bidirectional LSTM (embedding 20,000 x 50, BiLSTM 2 x 50, global max pooling, dense layers) with sigmoid outputs for the six labels of the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset. Its vocabulary was built from that dataset's 159,571 training comments.

## 📦 Installation

```bash
pip install toxic-comment-classifier
```

Requires Python 3.10-3.13. TensorFlow (>= 2.16, with Keras 3) is installed automatically.

## 🚀 Usage

### 🔹 Import and Initialize the Model

```python
from toxic_classifier import ToxicCommentClassifier

# Load the classifier
model = ToxicCommentClassifier()
```

(`from toxic_classifier.model import ToxicCommentClassifier` still works too.)

---

### 🔹 Classify a Single Comment

```python
text = "You are so dumb and stupid!"
scores = model.classify(text)

scores
```

**Example Output:**

```python
{'toxic': 0.9889402985572815,
 'severe_toxic': 0.07256769388914108,
 'obscene': 0.6204292178153992,
 'threat': 0.019348442554473877,
 'insult': 0.8664074540138245,
 'identity_hate': 0.040729500353336334}
```

---

### 🔹 Get Overall Toxicity Score

The overall score is the mean of the six label scores.

```python
toxicity = model.predict(text)
print(f"Overall Toxicity Score: {toxicity:.4f}")
```

**Example Output:**

```python
Overall Toxicity Score: 0.4347
```

---

### 🔹 Classify Multiple Comments

```python
texts = [
    "I hate you so much!",
    "This is wonderful news.",
    "You're disgusting!",
    "Absolutely love your energy!",
    "You're the worst person ever!",
    "Have a nice day :)"
]

scores = model.predict_batch(texts)

for txt, score in zip(texts, scores):
    print(f"Text: {txt} --> Toxicity Score: {score:.4f}")
```

**Example Output:**

```python
Text: I hate you so much! --> Toxicity Score: 0.1395
Text: This is wonderful news. --> Toxicity Score: 0.0013
Text: You're disgusting! --> Toxicity Score: 0.3110
Text: Absolutely love your energy! --> Toxicity Score: 0.0088
Text: You're the worst person ever! --> Toxicity Score: 0.0937
Text: Have a nice day :) --> Toxicity Score: 0.0115
```

Use `model.classify_batch(texts)` to get the per-label dictionaries for a list of texts.

---

### 🔹 Flag Comments Above a Threshold

```python
model.flagged_labels("You are so dumb and stupid!", threshold=0.5)
# ['toxic', 'obscene', 'insult']
```

---

### 🔹 Command Line

```bash
toxic-classifier "go kill yourself"
# {"text": "go kill yourself", "toxicity": 0.374754, "flagged": ["toxic", "insult"], "scores": {"toxic": 0.888507, ...}}

# One comment per line from stdin, JSON Lines out
cat comments.txt | toxic-classifier --threshold 0.7
```

## 💡 Use Cases

- **Comment moderation**: hold back or review comments where `flagged_labels()` is non-empty.
- **Community health dashboards**: track the share of toxic messages over time with `predict_batch()`.
- **Dataset filtering**: drop toxic samples before training or fine-tuning another model.
- **Triage**: route `threat` or `identity_hate` hits to a human reviewer first.

Scores are model estimates, not ground truth. Like any model trained on Jigsaw data, it can be wrong on sarcasm, reclaimed slurs, non-English text, and identity terms in neutral contexts, so keep a human in the loop for consequential decisions.

## 🧩 API Overview

| Method | Returns |
| --- | --- |
| `ToxicCommentClassifier(clean=False, batch_size=256)` | Loads the bundled model. `clean=True` applies `clean_text()` before scoring. |
| `classify(text)` | `dict[str, float]`: score per label |
| `classify_batch(texts)` | `list[dict[str, float]]` |
| `predict(text)` | `float`: mean of the six label scores |
| `predict_batch(texts)` | `list[float]` |
| `flagged_labels(text, threshold=0.5)` | `list[str]`: labels with score >= threshold |
| `toxic_classifier.clean_text(text)` | Lowercases and removes URLs and special characters |
| `toxic_classifier.LIST_CLASSES` | The six label names, in output order |

## ⚙️ Configuration

- `TF_CPP_MIN_LOG_LEVEL`: the library sets it to `2` (if unset) to hide TensorFlow's C++ info logs.
- `KERAS_BACKEND`: the model is built with Keras 3. TensorFlow is the default and tested backend.

## 🤝 Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/irfanalidv/toxic_comment_classifier/blob/main/CONTRIBUTING.md).

```bash
git clone https://github.com/irfanalidv/toxic_comment_classifier.git
cd toxic_comment_classifier
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

See the [CHANGELOG](https://github.com/irfanalidv/toxic_comment_classifier/blob/main/CHANGELOG.md) for release notes.

## 📄 License

MIT. See [LICENSE.txt](https://github.com/irfanalidv/toxic_comment_classifier/blob/main/LICENSE.txt).
