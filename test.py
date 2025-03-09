import unicodedata
import re
import emoji

# Dictionary to convert English numbers to Nepali numbers
eng_to_nepali_numbers = str.maketrans("0123456789", "०१२३४५६७८९")

def preprocess(text):
    if not isinstance(text, str):
        return ""

    # Normalize Unicode (NFC)
    text = unicodedata.normalize("NFC", text)

    # Remove ".pdf" (case-insensitive)
    text = re.sub(r"\.pdf", "", text, flags=re.IGNORECASE)

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Convert English numbers to Nepali numbers
    text = text.translate(eng_to_nepali_numbers)

    # Remove special symbols except basic Nepali script and punctuation
    text = re.sub(r"[^०१२३४५६७८९अ-ह‍ि-्ॐक़-ॡ০-৯a-zA-Z\s\.\,\?\!]", " ", text)

    # Remove multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Test Cases
test_sentences = [
    "यो एउटा परीक्षण .pdf ✅ 12345",
    "कसैलाई 🙏🤔🤩😂😂 123 हेल्लो ३४५६ !",
    "PDF FILE.pdf should be removed!",
    "सिर्जना ✨🌟 #$%@%^&**()*<>?/|😡😂"
]

for sentence in test_sentences:
    print("Original:", sentence)
    print("Processed:", preprocess(sentence))
    print("--------")
