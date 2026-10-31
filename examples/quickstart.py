"""Quick start for toxic_comment_classifier (run: python examples/quickstart.py)."""

from toxic_classifier import ToxicCommentClassifier, clean_text

classifier = ToxicCommentClassifier()

# === Single comment ===
raw_text = "YOU are an amazing person!"
cleaned = clean_text(raw_text)
print(f"\nCleaned text: {cleaned}")

scores = classifier.classify(cleaned)
print("Toxicity scores (single):")
for label, score in scores.items():
    print(f"  {label}: {score:.4f}")

avg = classifier.predict(cleaned)
print(f"Average toxicity score: {avg:.4f}")

# === Batch ===
print("\nBatch predictions:")
texts = [
    "I hate you so much!",
    "This is wonderful news.",
    "You're disgusting!",
    "Absolutely love your energy!",
    "You're the worst person ever!",
    "Have a nice day :)",
]

for txt, score in zip(texts, classifier.predict_batch(texts), strict=True):
    print(f"Text: {txt}\n-> Toxicity Score: {score:.4f}\n")

# Labels at or above a threshold
print(classifier.flagged_labels("You are so dumb and stupid!", threshold=0.5))
