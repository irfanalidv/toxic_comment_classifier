# Toxic Comment Classifier

[![GitHub license](https://img.shields.io/github/license/irfanalidv/toxic_comment_classifier)](https://github.com/irfanalidv/toxic_comment_classifier/blob/main/LICENSE)
[![Python version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://pypi.org/project/toxic-comment-classifier/)
[![PyPI version](https://badge.fury.io/py/toxic-comment-classifier.svg)](https://pypi.org/project/toxic-comment-classifier/)
[![Downloads](https://static.pepy.tech/badge/toxic-comment-classifier)](https://pepy.tech/project/toxic-comment-classifier)

```markdown
# Toxic Comment Classifier

A Python library for classifying toxic comments using deep learning.
It supports detecting multiple types of toxicity including obscene language, threats, and identity hate.
```

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

scores
```

**Example Output:**

```python
{'toxic': 0.9889402985572815,
 'severe_toxic': 0.07256772369146347,
 'obscene': 0.620429277420044,
 'threat': 0.01934845559298992,
 'insult': 0.8664075136184692,
 'identity_hate': 0.04072948172688484}
```

---

### 🔹 Get Overall Toxicity Score

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

---
