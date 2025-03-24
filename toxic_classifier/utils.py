import re

def clean_text(text: str) -> str:
    """
    Cleans text by removing special characters and multiple spaces.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9!?.,]", " ", text)  # Remove special chars
    return text.strip()
