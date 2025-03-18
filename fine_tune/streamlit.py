import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
model_path = "/home/binit/fine_tune_LLama/fine_tune/Llama-3.2_3B_Nepali_language"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = model.to_empty("cuda")

def generate_response(user_input):
    instruction = """You are a Nepali chatbot and you are fluent in Nepalese language."""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=500, num_return_sequences=1)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response_text.split("assistant")[1].strip()
    
st.title("Nepali Language Chatbot")
user_input = st.text_input("Enter your query in Nepali:")
if st.button("Submit"):
    if user_input:
        response = generate_response(user_input)
        st.write("Chatbot response: ", response)
    else:
        st.write("Please enter a query.")
