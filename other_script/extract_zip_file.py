import zipfile
import os

zip_file_path = '/home/binit/fine_tune_LLama/nepali_dataset_test/compressed_file.zip'
extract_dir = '/home/binit/fine_tune_LLama/nepali_dataset_test/'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Files extracted to {extract_dir}")
