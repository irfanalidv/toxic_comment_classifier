from toxic_classifier.model import ToxicCommentClassifier
from toxic_classifier.utils import clean_text

# Clean some text
raw_text = "YOU are such an IDIOT!!"
cleaned = clean_text(raw_text)
print(f"Cleaned text: {cleaned}")

# Load and use model
classifier = ToxicCommentClassifier()
result = classifier.classify(cleaned)
print("Toxicity scores:", result)

# Get average toxicity
avg_score = classifier.predict(cleaned)
print("Average toxicity score:", avg_score)
