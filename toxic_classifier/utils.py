import re

def clean_text(text: str) -> str:
    """
    Cleans text by lowercasing, removing special characters and extra spaces.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9!?.,]", " ", text)  # Keep alphanumerics and some punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text
