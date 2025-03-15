from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/binit/fine_tune_LLama/llama-3.2-3b-it-Ecommerce-ChatBot"
tokenizer = AutoTokenizer.from_pretrained(model_path)
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

