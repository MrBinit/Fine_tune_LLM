import sentencepiece as spm

corpus_file = '/home/binit/fine_tune_LLama/nepali_text.txt'
model_prefix = 'nepali_tokenizer'
vocab_size = 110

spm.SentencePieceTrainer.train(f'--input={corpus_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=0.9995')

sp = spm.SentencePieceProcessor(model_file='nepali_tokenizer.model')

nepali_sentence = 'पृथ्वीनारायण शाह नेपालका पहले राजा थिए'
tokens = sp.encode(nepali_sentence, out_type=str)

print("Tokens:", tokens)
