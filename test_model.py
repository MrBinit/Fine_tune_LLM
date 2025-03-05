from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_path = "/home/binit/fine_tune_LLama/Llama-3.2-3B"
adapter_path = "/home/binit/fine_tune_LLama/Llama-3.2-3B_fined_tuned"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
