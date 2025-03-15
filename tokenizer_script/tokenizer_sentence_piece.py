import sentencepiece as spm

from transformers import LlamaTokenizer

def sentence_piece_tokenizer(input_file, model_prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=1.0 --model_type=bpe"
    )
    print("Model training completed! Model and vocab saved as spm_model.model")


def llama_tokenizer_sentence_piece(vocab_file):
    tokenizer = LlamaTokenizer(vocab_file=vocab_file)
    tokenizer.save_pretrained("custom_nepali_tokenizer")



input_file = '/home/binit/fine_tune_LLama/nepali_text.txt'
model_prefix = 'spm_model' 
vocab_size = 655
vocab_file = "/home/binit/fine_tune_LLama/tokenizer_script/spm_model.model"


sentence_piece_tokenizer(input_file, model_prefix, vocab_size)
llama_tokenizer_sentence_piece(vocab_file)


