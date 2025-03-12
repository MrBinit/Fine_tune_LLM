from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("bpe_tokenizer.model")
text = "नेपाल एक सुन्दर देश हो।"
encoded = tokenizer.encode(text)

print("Encoded tokens:", encoded.tokens)
decoded = tokenizer.decode(encoded.ids)
print("Decoded text:", decoded)
