# Fine-Tuned Nepali Language Model using Llama 3.2 3B

I have fine-tuned the Llama 3.2 3B model specifically to support the Nepali language effectively. This model has been enhanced to better understand and generate responses in Nepali, making it more accurate for Nepali speakers. I have been working diligently to improve the model's accuracy and ensure that it can handle diverse Nepali language tasks, ranging from conversational queries to more complex language processing needs.

If you are interested in contributing to this project or improving the model further, I highly encourage you to check out this GitHub repository and explore how you can contribute to the code and model improvements.

## Huggingface Repository:
[Fine-tune Llama 3.2 3B for Nepali Language](https://huggingface.co/MrBinit/Llama-3.2_3B_Nepali_language)

## Code Example:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/binit/fine_tune_LLama/fine_tune/Llama-3.2_3B_Nepali_language"

# Load the tokenizer and set the padding token to the eos_token.
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).to("cuda")

def generate_response(user_input):
    instruction = """You are a top-rated customer service agent named John. 
    Be polite to customers and answer all their questions."""
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text.split("assistant")[1].strip()

user_query = "राणा शासनले नेपाल कसरी कब्जा गर्यो भनेर व्याख्या गर्न सक्नुहुन्छ?"
response = generate_response(user_query)
print("Chatbot:", response)
