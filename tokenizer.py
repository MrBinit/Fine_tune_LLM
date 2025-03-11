from tokenizers import ByteLevelBPETokenizer
import os
from tokenizers.pre_tokenizers import Whitespace
save_dir = "nepali_tokenizer"

os.makedirs(save_dir, exist_ok = True)


tokenizer = ByteLevelBPETokenizer()
tokenizer.pre_tokenizer = Whitespace()

special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>", 
]
tokenizer.train(files=["/home/binit/fine_tune_LLama/nepali_text_utf8.txt"], vocab_size=32000, min_frequency=2, special_tokens=special_tokens)

# Save model
tokenizer.save_model(save_dir)

tokenizer = ByteLevelBPETokenizer.from_file(
    os.path.join(save_dir, "vocab.json"),
    os.path.join(save_dir, "merges.txt")
)
output = tokenizer.encode('पृथ्वीनारायण शाह नेपालका पहले राजा थिए')
print("Tokens:", output.tokens)