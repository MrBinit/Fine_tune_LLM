# from datasets import load_dataset, Dataset, DatasetDict

# # Load streaming dataset
# streamed = load_dataset("text", data_files="/home/binit/fine_tune_LLama/extracted_text.txt", streaming=True)
# limited_data = streamed['train'].take(20000000) 
# data_list = list(limited_data)
# dataset = Dataset.from_list(data_list)
# DatasetDict({"train": dataset}).push_to_hub("MrBinit/Nepali_text_dattest")


from datasets import load_dataset, Dataset, DatasetDict
from itertools import islice
import itertools

file_path = "/home/binit/fine_tune_LLama/extracted_text.txt"
repo_base = "MrBinit/Nepali_text_dataset"
chunk_size = 20_000_000

streamed = load_dataset("text", data_files=file_path, split="train", streaming=True)
chunk_num = 0
iterator = iter(streamed)

while True:
    chunk = list(islice(iterator, chunk_size))
    if not chunk:
        print("All chunks uploaded!")
        break
    dataset = Dataset.from_list(chunk)
    chunk_repo_id = f"{repo_base}_chunk{chunk_num}"

    print(f"Uploading chunk {chunk_num} to {chunk_repo_id} ...")

    DatasetDict({"train": dataset}).push_to_hub(chunk_repo_id)
    print(f"Chunk {chunk_num} uploaded!")

    chunk_num += 1
