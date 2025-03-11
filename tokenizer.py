from tokenizers import ByteLevelBPETokenizer
import os
from tokenizers.pre_tokenizers import Whitespace
save_dir = "nepali_tokenizer"

os.makedirs(save_dir, exist_ok = True)


tokenizer = ByteLevelBPETokenizer()
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(files=["/home/binit/fine_tune_LLama/nepali_text_utf8.txt"], vocab_size=32000, min_frequency=2, special_tokens=[
    "<|begin_of_text|>", "<|end_of_text|>", "<|pad|>", "<|unk|>", "<|mask|>"
])

# Save model
tokenizer.save_model(save_dir)

tokenizer = ByteLevelBPETokenizer.from_file(
    os.path.join(save_dir, "vocab.json"),
    os.path.join(save_dir, "merges.txt")
)
output = tokenizer.encode('पृथ्वीनारायण शाह नेपालका पहले राजा थिए')
print("Tokens:", output.tokens)