from transformers import AutoTokenizer

model_name = "/home/binit/fine_tune_LLama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "नेपालको राजधानी काठमाडौँ हो।"
tokens = tokenizer.tokenize(text)

print(tokens)
