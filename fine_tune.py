from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
import bitsandbytes as bnb
import torch
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format


base_model = "/home/binit/fine_tune_LLama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(base_model)
new_model = "/home/binit/fine_tune_LLama/Llama-3.2-3B_fined_tuned"
dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

#returns the major compute capability of the current GPU 
if torch.cuda.get_device_capability()[0] >= 0:
    torch_dtype = torch.bfloat16 #if the condition is passed then dtype will be converted to bfloat16 which is more efficient on modern GPU 
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager" #Otherwise, it falls back to "eager", which is the standard PyTorch implementation.


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
    torch_dtype = torch.bfloat16,
    attn_implementation=attn_implementation
)
print(model)

dataset = load_dataset(dataset_name, split="train")
dataset = dataset.shuffle(seed=65).select(range(100)) 
instruction = """You are a top-rated customer service agent named Sahara Chutiya. 
    Be polite to customers and answer all their questions.
    """
def format_chat_template(row):
    
    row_json = [{"role": "system", "content": instruction },
               {"role": "user", "content": row["instruction"]},
               {"role": "assistant", "content": row["response"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc= 4,
)

train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, len(dataset)))

modules = ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']

peft_config = LoraConfig(
    r = 16, 
    lora_alpha= 32,
    lora_dropout= 0.5,
    bias = "none",
    task_type="CAUSAL_LM",
    target_modules=modules
)
if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    tokenizer.chat_template = None
model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config)

#Hyperparamter
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=0.2,e
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    group_by_length=True,
    logging_dir="./logs",
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    max_steps= 200,

)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,  
    args=training_arguments,
)
trainer.train()

trainer.model.save_pretrained(new_model)