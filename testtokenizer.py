import os
import json
import logging
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

# Install these libraries beforehand:
# pip install sentencepiece tiktoken

import sentencepiece as spm
from tiktoken.load import load_tiktoken_bpe

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- STEP 1: TRAIN YOUR NEPALI TOKENIZER USING SENTENCEPIECE -----
def train_nepali_tokenizer(input_file: str, model_prefix: str, vocab_size: int = 32000):
    """
    Trains a SentencePiece BPE tokenizer on your Nepali corpus.
    """
    spm.SentencePieceTrainer.train(
        f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0"
    )
    logger.info(f"Training complete. Models saved with prefix: {model_prefix}")

# ----- STEP 2: CONVERT SENTENCEPIECE MODEL TO A TIKTOKEN-COMPATIBLE FORMAT -----
def convert_sentencepiece_to_tiktoken(model_prefix: str, output_file: str):
    """
    Converts a SentencePiece model to a mergeable ranks file expected by tiktoken.
    NOTE: This conversion is illustrative. In a real BPE, merge rules are ranked by frequency.
    Here we simply assign the SentencePiece order as the rank.
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
    
    mergeable_ranks = {}
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        # Use the piece and its index as the rank (this is a simplistic approach)
        mergeable_ranks[piece] = i
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mergeable_ranks, f, ensure_ascii=False, indent=2)
    logger.info(f"Converted SentencePiece model to tiktoken format: {output_file}")

# ----- TYPES & CHAT MESSAGE DEFINITIONS -----
Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

# ----- STEP 3: DEFINE A NEPALI TOKENIZER CLASS (ADAPTED FROM LLaMA) -----
class NepaliTokenizer:
    """
    A tokenizer for Nepali that loads a Tiktoken BPE model.
    """
    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256

    # Modify regex for Nepali (Devanagari) text.
    # Here we use a simple pattern that captures sequences of Nepali characters or non-space tokens.
    # (For production, you may need a more sophisticated regex or use a Unicode library like "regex".)
    pat_str = r"[\u0900-\u097F]+|\S"

    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise ValueError(f"Model file {model_path} not found")
        
        # Load the mergeable ranks (BPE merge rules) in tiktoken format.
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
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
            "<|eot_id|>",  # end-of-turn marker
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        import tiktoken
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Loaded tiktoken model from {model_path}")
        self.n_words = self.model.n_vocab
        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        self.pad_id = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        """
        assert type(s) is str

        TIKTOKEN_MAX_ENCODE_CHARS = 400_000
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()
            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


# ----- STEP 4: DEFINE THE CHAT FORMAT CLASS -----
class ChatFormat:
    def __init__(self, tokenizer: NepaliTokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Append a header for the assistant's next message
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens

# ----- MAIN EXECUTION -----
if __name__ == "__main__":
    # File paths
    corpus_file = "nepali_corpus.txt"              # Your Nepali text file
    model_prefix = "nepali_bpe"                      # Prefix for SentencePiece output
    mergeable_ranks_file = "nepali_mergeable_ranks.json"  # File to hold tiktoken-compatible merge rules

    # Step 1: Train the SentencePiece tokenizer (only if the model file does not exist)
    if not os.path.exists(f"{model_prefix}.model"):
        logger.info("Training SentencePiece tokenizer...")
        train_nepali_tokenizer(corpus_file, model_prefix, vocab_size=32000)
    else:
        logger.info("SentencePiece model already exists.")

    # Step 2: Convert the SentencePiece model to a tiktoken-compatible mergeable ranks file
    if not os.path.exists(mergeable_ranks_file):
        logger.info("Converting SentencePiece model to tiktoken format...")
        convert_sentencepiece_to_tiktoken(model_prefix, mergeable_ranks_file)
    else:
        logger.info("Mergeable ranks file already exists.")

    # Step 3: Load the NepaliTokenizer using the converted mergeable ranks file
    tokenizer = NepaliTokenizer(mergeable_ranks_file)
    chat_format = ChatFormat(tokenizer)

    # Example dialog
    dialog: Dialog = [
        {"role": "user", "content": "नमस्ते, तपाईंलाई कस्तो छ?"},
        {"role": "assistant", "content": "म ठीक छु, धन्यवाद! तपाईंलाई कस्तो छ?"},
    ]

    # Encode the dialog prompt
    encoded = chat_format.encode_dialog_prompt(dialog)
    logger.info("Encoded tokens: %s", encoded)

    # Decode back to text for verification
    decoded = tokenizer.decode(encoded)
    logger.info("Decoded text: %s", decoded)
