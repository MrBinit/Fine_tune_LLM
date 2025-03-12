import logging
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from typing import List, Dict, Union, Literal


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)



class BPETokenizer:
    """
    BPE tokenizer, encode and decode using pre-trained model
    """

    def __init__(self, bpe_model_path: str= None, special_tokens: Dict[str, str] = None):
        """
        Initializes the Tokenizer with a BPE model, either pre-trained or custom.
        
        Args:
            bpe_model_path (str): Path to a pre-trained BPE tokenizer model file (optional).
            special_tokens (dict): A dictionary of special tokens to add to the tokenizer.
        """
        if bpe_model_path:
            # load pre-trained model

            self.tokenizer = Tokenizer.from_file(bpe_model_path)
            logging.info(f"Loaded pre-trained BPE model from {bpe_model_path}")

        else:
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            logger.info("Initialized new BPE tokenizer")

        if special_tokens:
            self.add_special_tokens(special_tokens)

    def add_special_tokens(self, special_tokens: Dict[str, str]):
        """
        Special tokens to the tokenizer.
        
        Args:
            special_tokens (dict): A dictionary of special tokens and their IDs.
        """

        for token , id in special_tokens.items():
            self.tokenizer.add_special_tokens([token])
        
        logger.info(f"Special tokens added: {list(special_tokens.keys())}")

    def train(self,corpus: List[str],  vocab_size: int): 
        """
        Train a new BPE tokenizer on a corpus.
        
        Args:
            corpus (List[str]): List of sentences for training the tokenizer.
            vocab_size (int): The size of the vocabulary to be learned.
        """

        trainer = trainers.BpeTrainer(vocab_size = vocab_size, min_frequency = 2, show_progress= True)
        self.tokenizer.train_from_iterator(corpus, trainer)
        logger.info(f"Trained BPE tokenizer with vocab size: {vocab_size}")

    def save(self, model_path: str, vocab_path: str):
        """
        Save the tokenizer model and vocab to files.
        
        Args:
            model_path (str): Path to save the BPE model file.
            vocab_path (str): Path to save the vocab file.
        """
        self.tokenizer.save(model_path)

        with open(vocab_path, 'w', encoding = 'utf-8') as vocab_file:
            vocab = self.tokenizer.get_vocab()

            for token , id in vocab.items():
                vocab_file.write(f"{token} {id}\n")
            logger.info(f"Vocabulary saved to {vocab_path}")

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of token IDs.
        
        Args:
            text (str): The input string to encode.
        
        Returns:
            List[int]: The list of token IDs.
        """
        encoded = self.tokenizer.encode(text)
        logger.debug(f"Encoded text '{text}' into token IDs: {encoded.ids}")
        return encoded.ids

    def decode(self, token_ids: List[int]) -> str:
        decoded = self.tokenizer.decode(token_ids)
        logger.debug(f"Decoded token IDs {token_ids} back to text: '{decoded}'")
        return decoded


if __name__ == '__main__':
    special_tokens = {
        "<|begin_of_text|>": 0,
        "<|end_of_text|>": 1,
        "<|pad|>": 2,
        "<|unk|>": 3
    }

    corpus = [
        "नेपाल एक सुन्दर देश हो।",
        "नेपालमा धेरै सुंदर स्थलहरू छन्।",
        "तपाईंलाई कस्तो लाग्छ?",
        "तपाईंको नाम के हो?"
    ]

    bpe_tokenizer = BPETokenizer(special_tokens=special_tokens)
    bpe_tokenizer.train(corpus, vocab_size=5000)
    bpe_tokenizer.save("bpe_tokenizer.model", "bpe_vocab.vocab")


    text = "नेपाल एक सुन्दर देश हो।"
    encoded_text = bpe_tokenizer.encode(text)
    decoded_text = bpe_tokenizer.decode(encoded_text)


    logger.info(f"Original Text: {text}")
    logger.info(f"Encoded Text: {encoded_text}")
    logger.info(f"Decoded Text: {decoded_text}")

