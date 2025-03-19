from datasets import Dataset
from huggingface_hub import login

with open("/home/binit/fine_tune_LLama/extracted_text.txt", "r") as file:
    text_data = file.readlines()
dataset = Dataset.from_dict({"text": text_data})
dataset.push_to_hub("MrBinit/Nepali-Text")
