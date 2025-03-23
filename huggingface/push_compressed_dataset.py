#push zip file to huggingface
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, Repository

repo_name = "MrBinit/nepali_dataset_text"
dataset_path = "/home/binit/fine_tune_LLama/compressed.zip" 
repo_url = f"https://huggingface.co/datasets/{repo_name}"

api = HfApi()
api.create_repo(repo_name, repo_type="dataset")
repo = Repository(local_dir=repo_name, clone_from=repo_url)
import shutil

# Move the .zip file to the repo directory
shutil.copy(dataset_path, repo.local_dir)

# Commit the file and push to Hugging Face hub
repo.git_add()
repo.git_commit("Initial commit with compressed .zip file")
repo.git_push()

print(f"Dataset pushed to {repo_url} successfully!")
