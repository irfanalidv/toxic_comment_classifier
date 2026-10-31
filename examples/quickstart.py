"""Quick start: score one comment and a small batch.

Shows: ``classify()``, ``predict()``, ``predict_batch()`` and ``flagged_labels()``.
Needs: only the package (the model ships with it, no network or API key).

Run: python examples/quickstart.py
"""

from toxic_classifier import ToxicCommentClassifier

classifier = ToxicCommentClassifier()

text = "You are so dumb and stupid!"
print(f"Comment: {text!r}")
for label, score in classifier.classify(text).items():
    print(f"  {label:<14} {score:.4f}")
print(f"Overall toxicity (mean of labels): {classifier.predict(text):.4f}")
print(f"Flagged at 0.5: {classifier.flagged_labels(text, threshold=0.5)}")

print("\nBatch:")
texts = [
    "I hate you so much!",
    "This is wonderful news.",
    "You're disgusting!",
    "Absolutely love your energy!",
    "You're the worst person ever!",
    "Have a nice day :)",
]
for txt, score in zip(texts, classifier.predict_batch(texts), strict=True):
    print(f"  {score:.4f}  {txt}")
