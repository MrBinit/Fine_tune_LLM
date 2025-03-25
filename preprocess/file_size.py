import os

# Define the file path
file_path = '/home/binit/fine_tune_LLama/nepali.txt'

# Get the file size in bytes
file_size = os.path.getsize(file_path)

# Convert bytes to a more readable format (KB, MB, GB)
file_size_kb = file_size / 1024  # KB
file_size_mb = file_size_kb / 1024  # MB
file_size_gb = file_size_mb / 1024  # GB

print(f"File size in bytes: {file_size}")
print(f"File size in KB: {file_size_kb:.2f} KB")
print(f"File size in MB: {file_size_mb:.2f} MB")
print(f"File size in GB: {file_size_gb:.2f} GB")
