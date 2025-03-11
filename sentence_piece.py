import sentencepiece as spm
import os
import logging
from tiktoken.load import load_tiktoken_bpe

import json

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

def convert_sentencepiece_to_tiktoken(model_prefix: str, output_file:str):
    """
    Converts a SentencePiece model to a mergeable ranks file expected by tiktoken.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    mergeable_ranks = {}
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        mergeable_ranks[piece] = i

    with open(output_file, "w", encoding = "utf-8") as f:
        json.dump(mergeable_ranks, f, ensure_ascii=False, indent=2)
    logger.info(f"Converted SentencePiece model to tiktoken format: {output_file}")

corpus_file = '/home/binit/fine_tune_LLama/nepali_text.txt'
model_prefix = 'nepali_tokenizer'
mergeable_ranks_file = "nepali_mergeable_ranks.json"
if not os.path.exists(f"{model_prefix}.model"):
    train_nepali_tokenizer(corpus_file, model_prefix)
else:
    print("already exists")

if not os.path.exists(mergeable_ranks_file):
    logger.info("Converting sentencePiece model to tiktoken format.....")
    convert_sentencepiece_to_tiktoken(model_prefix, mergeable_ranks_file)
else:
    logger.info("Mergeable ranks file already exists....")
