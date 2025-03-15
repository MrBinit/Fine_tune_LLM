import sentencepiece as spm
from transformers import LlamaTokenizer
from typing import List

class sentence_piece_tokenizer:
    def __init__(self, input_file, model_prefix, vocab_size):
        """
        Initializes the SentencePieceTokenizer with the input file, model prefix, and vocab size.
        Args:
            input_file (str): Path to the input text file for training.
            model_prefix (str): Prefix for the generated model and vocab files.
            vocab_size (int): The desired size of the vocabulary.
        """
        self.input_file = input_file
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size

    def train(self):
        """
        Train the SentencePiece model using the provided input file, model prefix, and vocab size.
        """
        spm.SentencePieceTrainer.train(
            f"--input={self.input_file} --model_prefix={self.model_prefix} --vocab_size={self.vocab_size} --character_coverage=1.0 --model_type=bpe"
        )
        print(f"Model training completed! Model and vocab saved as {self.model_prefix}.model")

    def create_llama_tokenizer(self, vocab_file):
        """
        Create a LLaMA-compatible tokenizer using the trained SentencePiece model.
        Args:
            vocab_file (str): Path to the trained SentencePiece model file.
        """
        tokenizer = LlamaTokenizer(vocab_file=vocab_file)
        tokenizer.save_pretrained("custom_nepali_tokenizer")
        print("LLaMA tokenizer saved as 'custom_nepali_tokenizer'.")


class ChatFormat:
    def __init__(self, tokenizer: sentence_piece_tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: dict) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: dict) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(self, dialog: List[dict]) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens


input_file = '/home/binit/fine_tune_LLama/nepali_text.txt'
model_prefix = 'spm_model'
vocab_size = 655
vocab_file = "/home/binit/fine_tune_LLama/tokenizer_script/spm_model.model"

sp_tokenizer = sentence_piece_tokenizer(input_file, model_prefix, vocab_size)
sp_tokenizer.train()
sp_tokenizer.create_llama_tokenizer(vocab_file)