from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
import bitsandbytes as bnb
import torch
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

base_model = "/home/binit/fine_tune_LLama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(base_model)
new_model = "/home/binit/fine_tune_LLama/Llama-3.2-3B_fined_tuned"
dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 0 else torch.float16
attn_implementation = "flash_attention_2" if torch_dtype == torch.bfloat16 else "eager"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch_dtype,
    attn_implementation=attn_implementation,
)

dataset = load_dataset(dataset_name, split="train").shuffle(seed=65).select(range(100))

instruction = """You are a top-rated customer service agent named John. 
    Be polite to customers and answer all their questions.
"""

def format_chat_template(row):
    row_json = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": row["response"]},
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(format_chat_template, num_proc=4)

train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, len(dataset)))

modules = ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj', 'lm_head']

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.5,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)

if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    tokenizer.chat_template = None

model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    logging_dir="./logs",
    lr_scheduler_type="cosine",
    max_steps=200,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

messages = [{"role": "system", "content": instruction},
    {"role": "user", "content": "I bought the same item twice, cancel order {{Order Number}}"}]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text.split("assistant")[1])

trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    tokenizer.chat_template = None


base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
model = PeftModel.from_pretrained(base_model_reload, new_model)
model = model.merge_and_unload()

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
