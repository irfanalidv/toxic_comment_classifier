from toxic_classifier.model import ToxicCommentClassifier
from toxic_classifier.utils import clean_text

# === Initialize Classifier ===
classifier = ToxicCommentClassifier()

# === Test: Single Comment ===
raw_text = "YOU are an amazing person!"
cleaned = clean_text(raw_text)
print(f"\n🔹 Cleaned text: {cleaned}")

scores = classifier.classify(cleaned)
print("🔹 Toxicity scores (single):")
for label, score in scores.items():
    print(f"  {label}: {score:.4f}")

avg = classifier.predict(cleaned)
print(f"🔹 Average toxicity score: {avg:.4f}")

# === Test: Bulk / Batch Comments ===
print("\n🔸 Bulk Predictions:")
texts = [
    "I hate you so much!",
    "This is wonderful news.",
    "You're disgusting!",
    "Absolutely love your energy!",
    "You're the worst person ever!",
    "Have a nice day :)"
]

# Optionally clean text in batch
cleaned_texts = [clean_text(t) for t in texts]
batch_scores = classifier.predict_batch(cleaned_texts)

for txt, score in zip(texts, batch_scores):
    print(f"Text: {txt}\n→ Toxicity Score: {score:.4f}\n")
