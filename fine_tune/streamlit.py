import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/binit/fine_tune_LLama/fine_tune/Llama-3.2_3B_Nepali_language"


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).to("cuda")

def generate_response(user_input):
    instruction = """You are chatbot proficient in Nepalese Language."""
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=500, num_return_sequences=1)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text.split("assistant")[1].strip()

iface = gr.Interface(
    fn=generate_response, 
    inputs=gr.Textbox(label="Enter your question in Nepali"),
    outputs=gr.Textbox(label="Chatbot Response"),
    title="Nepali Language Chatbot",
    description="Chat with the model in Nepali. The model is proficient in understanding and responding in Nepali."
)

iface.launch()
