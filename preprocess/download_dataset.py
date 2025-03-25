# from datasets import load_dataset

# # Load the dataset from Hugging Face

# ds = load_dataset("MrBinit/nepali_dataset_test")

# # Convert to Pandas DataFrame and save as CSV
# df = ds['train'].to_pandas()
# df.to_csv("/home/binit/fine_tune_LLama/dataset/train_corrected_nepali_text.csv", index=False)

# print("Dataset saved as nepalitext_dataset.csv")


from datasets import load_dataset

# Point to the zip file containing the dataset
ds = load_dataset("csv", cache_dir="/home/binit/fine_tune_LLama/data.zip")

