import sentencepiece as spm
import os
import logging
from tiktoken.load import load_tiktoken_bpe



#setup logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

# Train nepali tokenizer using SentencePiece
def train_nepali_tokenizer(input_file: str, model_prefix: str, vocab_size: int = 110):
    """
    Train SentencePiece BPE tokenizer on nepali text 
    """
    spm.SentencePieceTrainer.train(
        f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=0.9995'
        )
    logger.info(f"Training completed. Models saved with prefix: {model_prefix}")

# convert sentencepiece  model to tiktoken compatible format

corpus_file = '/home/binit/fine_tune_LLama/nepali_text.txt'
model_prefix = 'nepali_tokenizer'
if not os.path.exists(f"{model_prefix}.model"):
    train_nepali_tokenizer(corpus_file, model_prefix)
else:
    print("already exists")
