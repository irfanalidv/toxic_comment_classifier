# Toxic Comment Classifier

[![GitHub license](https://img.shields.io/github/license/irfanalidv/toxic_comment_classifier)](https://github.com/irfanalidv/toxic_comment_classifier/blob/main/LICENSE)
[![Python version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://pypi.org/project/toxic-comment-classifier/)
[![PyPI version](https://badge.fury.io/py/toxic-comment-classifier.svg)](https://pypi.org/project/toxic-comment-classifier/)
[![Downloads](https://static.pepy.tech/badge/toxic-comment-classifier)](https://pepy.tech/project/toxic-comment-classifier)


````markdown
# Toxic Comment Classifier

A Python library for classifying toxic comments using deep learning.
It supports detecting multiple types of toxicity including obscene language, threats, and identity hate.


````
## 📦 Installation

```python
pip install toxic-comment-classifier
```
---

## 🚀 Usage

### 🔹 Import and Initialize the Model

```python
from toxic_classifier.model import ToxicCommentClassifier

# Load the classifier
model = ToxicCommentClassifier()
```

---

### 🔹 Classify a Single Comment

```python
text = "You are so dumb and stupid!"
scores = model.classify(text)

print("Toxicity Scores:", scores)
```

**Example Output:**

```python
{
    'toxic': 0.5004,
    'severe_toxic': 0.4987,
    'obscene': 0.4989,
    'threat': 0.5021,
    'insult': 0.4979,
    'identity_hate': 0.5006
}
```

---

### 🔹 Get Overall Toxicity Score

```python
toxicity = model.predict(text)
print(f"Overall Toxicity Score: {toxicity:.4f}")
```

**Example Output:**

```python
Overall Toxicity Score: 0.4998
```

---

### 🔹 Classify Multiple Comments

```python
texts = [
    "I hate this!",
    "You're amazing!",
    "This is the worst thing ever!"
]

scores = model.predict_batch(texts)

for txt, score in zip(texts, scores):
    print(f"Text: {txt} --> Toxicity Score: {score:.4f}")
```

**Example Output:**

```python
Text: I hate this! --> Toxicity Score: 0.5002
Text: You're amazing! --> Toxicity Score: 0.5000
Text: This is the worst thing ever! --> Toxicity Score: 0.5008
```

---

