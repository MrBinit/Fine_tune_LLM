from datasets import Dataset
from huggingface_hub import login

text_data_path = "/home/binit/fine_tune_LLama/nepali.txt"
with open(text_data_path, "r") as file:
    text_data = file.readlines()
text_data = [line.strip() for line in text_data]
dataset = Dataset.from_dict({"text": text_data})
dataset.push_to_hub("MrBinit/Nepali-Language-Text")
