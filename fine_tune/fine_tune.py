from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
import bitsandbytes as bnb
import torch
from datasets import load_dataset,  load_from_disk
from trl import SFTTrainer, setup_chat_format
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Llama_trainer:
    def __init__(self, base_model, txt_file, new_model):
        """
        Initialize the trainer with model paths and configuration.

        Args:
            base_model (str): Path to the pre-trained base model.
            txt_file (str): Path to the plain text file with training data.
            new_model (str): Directory to save the fine-tuned model.
            num_examples (int): Maximum number of examples to use from the dataset.
            seed (int): Seed for shuffling the dataset.
        """

        self.base_model = base_model
        self.txt_file = txt_file
        self.new_model = new_model
        self.seed = 65 
        self.instruction = "You are a chatbot who is trained for Nepalese language.\n"
        self.tokenizer= AutoTokenizer.from_pretrained(self.base_model)
        logger.info("Loaded tokenizer from base model.")
        self.model = None
        self.train_dataset = None
        self.test_dataset = None

    def setup_model(self):
        """
        Set up the model using BitsAndBytes for 4-bit quantization.
        """

        torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 0 else torch.float16
        attn_implementation = "flash_attention_2" if torch_dtype == torch.bfloat16 else "eager"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        logger.info("Loaded base model with 4-bit quantization.")

    def prepare_dataset(self):
        """
        Load, shuffle, and format the dataset from a plain text file.
        """
        train_arrow = "train_split.arrow"
        test_arrow = "test_split.arrow"
        if os.path.exists(train_arrow) and os.path.exists(test_arrow):
            logger.info("Arrow files found. Loading dataset splits from disk. ")
            self.train = load_from_disk(train_arrow)
            self.test = load_from_disk(test_arrow)
        else:
            logger.info("Arrow files not found. Loading dataset and creating train/test split.")
            dataset = load_dataset("text", data_files = self.txt_file, split="train").shuffle(seed = self.seed)
            dataset = dataset.map(self.format_chat_template, num_proc=1, load_from_cache_file=False)
            logger.info("Applied chat formatting to the dataset.")
            dataset_split = dataset.train_test_split(test_size=0.2, seed= self.seed)
            self.train_dataset = dataset_split["train"]
            self.test_dataset = dataset_split["test"]
            logger.info(f"Train dataset size: {len(self.train_dataset)}, Test dataset size: {len(self.test_dataset)}")

            self.train_dataset.save_to_disk(train_arrow)
            self.test_dataset.save_to_disk(test_arrow)
            logger.info(f"Train dataset size: {len(self.train_dataset)}, Test dataset size: {len(self.test_dataset)}")



    def format_chat_template(self, row):
        """
        Format each row by applying a chat template.

        This function creates a prompt by combining a system instruction and the original text
        as a user message, then uses the tokenizer's chat template functionality.

        Args:
            row (dict): A dictionary with a 'text' field.

        Returns:
            dict: The updated row with formatted 'text'.
        """
        row_json = [
            {"role": "system", "content": self.instruction},
            {"role": "assistant", "content": row["text"]},
        ]
        row["text"] = self.tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    def setup_peft(self):
        modules = ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj', 'lm_head']
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.5,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules,
        )
        logger.info("PEFT configuration set up with LoRA.")

        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            self.tokenizer.chat_template = None

        # Set up the chat format for the model and tokenizer
        self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
        logger.info("Chat format set up for model and tokenizer.")

        # Wrap the model using PEFT with the LoRA configuration
        self.model = get_peft_model(self.model, self.peft_config)
        logger.info("PEFT model created using LoRA.")
    
    def setup_training_arguments(self):
        """
        Define training arguments for the fine-tuning process.
        """
        self.training_arguments = TrainingArguments(
            output_dir=self.new_model,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            optim="paged_adamw_32bit",
            num_train_epochs=4,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_steps=1000,
            logging_steps=1000,
            warmup_steps=100,
            logging_strategy="steps",
            learning_rate=1e-6,
            fp16=False,
            bf16=False,
            group_by_length=True,
            logging_dir="/home/binit/fine_tune_LLama/logs",
            lr_scheduler_type="cosine",
            # max_steps=200,
            save_total_limit=3  
        )
        logger.info("Training arguments configured.")

    def train(self):
        """
        Train the model using the SFTTrainer.
        """
        # Initialize the trainer with the model, datasets, and training arguments
        trainer = SFTTrainer(
            model=self.model,
            train_dataset= self.train_dataset,
            eval_dataset= self.test_dataset,
            peft_config=self.peft_config,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
        )
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete.")

        # Save the trained model and tokenizer
        trainer.model.save_pretrained(self.new_model)
        self.tokenizer.save_pretrained(self.new_model)

        logger.info(f"Model and tokenizer saved to {self.new_model}.")


    def merge_model(self):
        if hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
            logger.info("LoRA weights merged into the base model.")
        else:
            logger.warning("Model does not support merge_and_unload; skipping merge.")



    def generate_response(self, user_message):
        """
        Generate a response from the model given a user message.

        Args:
            user_message (str): The message from the user.

        Returns:
            str: The generated response from the model.
        """
        # Create a chat prompt with system instruction and user message
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": user_message}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)
        decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract and return the assistant's response (split based on your prompt format)
        response = decoded_text.split("assistant")[-1].strip()
        return response





if __name__ == '__main__':
    base_model_path = "/home/binit/fine_tune_LLama/Llama-3.2-3B"
    text_file_path = "/home/binit/fine_tune_LLama/nepali_text.txt"
    new_model_path = "/home/binit/fine_tune_LLama/Llama-3.2-3B_fined_tuned"

    final_model_path = "Llama-3.2_3B_Nepali_language"
    # Create an instance of ChatbotTrainer
    trainer = Llama_trainer(base_model=base_model_path,
                                 txt_file=text_file_path,
                                 new_model=new_model_path)
    trainer.setup_model()
    trainer.prepare_dataset()
    trainer.setup_peft()
    trainer.setup_training_arguments()

    # Train the model
    trainer.train()

    trainer.merge_model()


    # Generate a sample response from the fine-tuned model
    sample_user_message = "नेपालको इतिहासको बारेमा बताउन सक्नुहुन्छ?"
    response = trainer.generate_response(sample_user_message)
    logger.info("Generated response:")
    logger.info(response)


    trainer.model.save_pretrained(final_model_path)
    trainer.tokenizer.save_pretrained(final_model_path)
    logger.info(f"Final merged model and tokenizer saved to {final_model_path}.")

