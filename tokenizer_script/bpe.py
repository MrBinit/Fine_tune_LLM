from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from typing import List, Dict, Union, Literal
from pathlib import Path

class BPETokenizer:
    """
    Tokenizing and encoding/decoding text using a pre-trained or custom BPE tokenizer.
    """
    def __init__(self, bpe_model_path: str = None):
        """
        Initializes the Tokenizer with a BPE model, either pre-trained or custom.
        
        Args:
            bpe_model_path (str): Path to a pre-trained BPE tokenizer model file (optional).
        """
        if bpe_model_path:
            # Load pre-trained model if available
            self.tokenizer = Tokenizer.from_file(bpe_model_path)
            print(f"Loaded pre-trained BPE model from {bpe_model_path}")
        else:
            # Initialize a new BPE tokenizer
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def train(self, corpus: List[str], vocab_size: int = 10000):
        """
        Train a new BPE tokenizer on a given corpus.
        
        Args:
            corpus (List[str]): List of sentences for training the tokenizer.
            vocab_size (int): The size of the vocabulary to be learned.
        """
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, show_progress=True)
        self.tokenizer.train_from_iterator(corpus, trainer)
        print(f"Trained BPE tokenizer with vocab size: {vocab_size}")

    def save(self, model_path: str, vocab_path: str):
        """
        Save the tokenizer model and vocab to files.
        
        Args:
            model_path (str): Path to save the BPE model file.
            vocab_path (str): Path to save the vocab file.
        """
        # Save the tokenizer model and vocab
        self.tokenizer.save(model_path)
        print(f"BPE model saved to {model_path}")
        
        with open(vocab_path, "w", encoding="utf-8") as vocab_file:
            vocab = self.tokenizer.get_vocab()
            for token, id in vocab.items():
                vocab_file.write(f"{token} {id}\n")
        print(f"Vocabulary saved to {vocab_path}")

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of token IDs.
        
        Args:
            text (str): The input string to encode.
        
        Returns:
            List[int]: The list of token IDs.
        """
        encoded = self.tokenizer.encode(text)
        return encoded.ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back to a string.
        
        Args:
            token_ids (List[int]): A list of token IDs.
        
        Returns:
            str: The decoded string.
        """
        decoded = self.tokenizer.decode(token_ids)
        return decoded

if __name__ == '__main__':
    corpus = [
        "नेपाल एक सुन्दर देश हो।",
        "नेपालमा धेरै सुंदर स्थलहरू छन्।",
        "तपाईंलाई कस्तो लाग्छ?",
        "तपाईंको नाम के हो?"
    ]
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.train(corpus, vocab_size=5000)
    bpe_tokenizer.save("bpe_tokenizer.model", "bpe_vocab.vocab")

    # example
    text = "नेपाल एक सुन्दर देश हो।"
    encoded_text = bpe_tokenizer.encode(text)
    decoded_text = bpe_tokenizer.decode(encoded_text)

    print("Original Text:", text)
    print("Encoded Text:", encoded_text)
    print("Decoded Text:", decoded_text)
