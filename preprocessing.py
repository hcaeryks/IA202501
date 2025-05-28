import re

def preprocess_text(texts):
    """Simple text preprocessing function"""
    processed = []
    for text in texts:
        # Lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        processed.append(text)
    return processed 