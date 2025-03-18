from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/binit/fine_tune_LLama/fine_tune/Llama-3.2_3B_Nepali_language"  
repo_name = "MrBinit/Llama3.2B-Nepali-Language-Model"  


create_repo(repo_name, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"Model pushed to Hugging Face at: {repo_name}")
