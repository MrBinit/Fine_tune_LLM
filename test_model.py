from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Define paths
base_model = "/home/binit/fine_tune_LLama/Llama-3.2-3B"
fine_tuned_model = "/home/binit/fine_tune_LLama/Llama-3.2-3B_fined_tuned/checkpoint-40"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)

# BitsAndBytes configuration (ensure proper quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model with quantization settings
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load fine-tuned weights using PEFT
model = PeftModel.from_pretrained(model, fine_tuned_model)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model Loaded Successfully!")
