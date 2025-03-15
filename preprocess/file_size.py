import os

def get_folder_size_in_gb(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    
    return total_size / (1024 ** 3)  # Convert bytes to GB

# Example usage
folder_path = "/home/binit/fine_tune_LLama/dataset"  # Change this to the folder you want to check
folder_size_gb = get_folder_size_in_gb(folder_path)
print(f"Total size of '{folder_path}': {folder_size_gb:.2f} GB")
