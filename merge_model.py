from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from trl import setup_chat_format

base_model_path = "/home/binit/fine_tune_LLama/Llama-3.2-3B"
fine_tuned_model_path = "/home/binit/fine_tune_LLama/Llama-3.2-3B_fined_tuned"
final_model_path = "llama-3.2-3b-it-Ecommerce-ChatBot"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print("Loading base model...")
base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    tokenizer.chat_template = None


base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
model = PeftModel.from_pretrained(base_model_reload, fine_tuned_model_path)
model = model.merge_and_unload()
instruction = "You are a top-rated customer service agent. Be polite and helpful."


messages = [{"role": "system", "content": instruction},
    {"role": "user", "content": "I have to see what payment modalities are accepted"}]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text.split("assistant")[1])

final_model_path = "llama-3.2-3b-it-Ecommerce-ChatBot"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
