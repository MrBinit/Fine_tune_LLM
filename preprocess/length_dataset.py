from datasets import load_dataset

# Load text dataset
dataset = load_dataset("text", data_files="/home/binit/fine_tune_LLama/extracted_text.txt")
dataset_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
print("Train length:", len(dataset_split['train']))
print("Test length:", len(dataset_split['test']))


ds = concatenate_datasets([Dataset.from_file(arrow_file) for arrow_file in arrow_files])
