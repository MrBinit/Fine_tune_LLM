import sentencepiece as spm
from transformers import LlamaTokenizer
from typing import List, Sequence
import logging 

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
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, 256)
        ]

        self.special_tokens = {
            token : 100 + i for i, token in enumerate(special_tokens)   # after the text is increase I need increase the size
        }

        self.num_reserved_special_tokens = 256
        self.model = None


        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{self.model_prefix}.model")  # Load the trained model



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

    def encode(self, s: str, bos=True, eos=True) -> List[int]:
        """
        Encode a string into token IDs, including special tokens.
        Uses SentencePieceProcessor for tokenization.
        """
        tokens = []
        if bos:
            tokens.append(self.special_tokens["<|begin_of_text|>"])  # Add BOS token if required

        # Use SentencePieceProcessor's encode function
        tokens.extend(self.sp_model.encode(s))  # Tokenize the string

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])  # Add EOS token

        return tokens

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Decode the token list using SentencePieceProcessor
        decoded_string = self.sp_model.decode(t)
        return decoded_string




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

if __name__ == '__main__':
    input_file = '/home/binit/fine_tune_LLama/nepali_text.txt'
    model_prefix = 'spm_model'
    vocab_size = 655
    vocab_file = "/home/binit/fine_tune_LLama/tokenizer_script/spm_model.model"

    sp_tokenizer = sentence_piece_tokenizer(input_file, model_prefix, vocab_size)
    sp_tokenizer.train()
    sp_tokenizer.create_llama_tokenizer(vocab_file)

    chat_format = ChatFormat(sp_tokenizer)

    chat = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]

    message_1 = {"role": "user", "content": "नमस्ते, तपाईलाई कस्तो छ?"}
    print("chat_format_encode:", chat_format.encode_message(chat[0]))
    tokens = sp_tokenizer.encode("Hello, how are you?", bos=True, eos=True)
    print("chat_encode:", tokens)
    print("decode:", sp_tokenizer.decode(tokens))


    nepali_word = "तपाईलाई कस्तो छ?"
    encoded_word = sp_tokenizer.encode(nepali_word, bos=True, eos=True)
    print("nepali_word_encode:", nepali_word)
    print("nepali_word_decode:", sp_tokenizer.decode(encoded_word))

    nepali_word_format = {"role": "user", "content": "तपाईलाई कस्तो छ?"}
    encoded_word = chat_format.encode_message(nepali_word_format)
    print("nepali_word_format_encode:", nepali_word_format)
    print("nepali_word_format_decode:", sp_tokenizer.decode(encoded_word))


    nepali_dialog = [
    {
        "role": "system",
        "content": "तपाई एक फुटबल विशेषज्ञ हुनुहुन्छ।",
    },
    {
        "role": "user",
        "content": "तपाईलाई के लाग्छ, नेपाली फुटबल कस्तो छ?",
    }
    ]

    encoded_word = chat_format.encode_dialog_prompt(nepali_dialog)
    print("nepali_dialog_format_encode:", nepali_dialog)
    print("nepali_dialog_format_decode:", sp_tokenizer.decode(encoded_word))

    # Tokenizing Nepali word
    encoded_word = sp_tokenizer.encode(nepali_word, bos=True, eos=True)
    print("nepali_word:", nepali_word)
    print("nepali_encoded_word:", encoded_word)
    print("nepali_word_decode:", sp_tokenizer.decode(encoded_word))
    print("nepali_word_decode[1:-1]:", sp_tokenizer.decode(encoded_word[1:-1]))