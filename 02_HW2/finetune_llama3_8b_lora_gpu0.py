import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from huggingface_hub import login
import os




#####################################################################################################
###
###          SETUP
###
#####################################################################################################

# Set GPU 0 explicitly
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Authenticate with Hugging Face
HF_TOKEN = ""  # Replace with your token
try:
    login(token=HF_TOKEN)
    print("Hugging Face authentication successful")
except Exception as e:
    print(f"Error authenticating with Hugging Face: {e}")
    exit(1)

# Load dataset
input_file = "lawyer_instruct_1000_examples.json"
try:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) == 0:
        raise ValueError("JSON file is empty")
    print(f"Loaded {len(data)} examples from {input_file}")
except Exception as e:
    print(f"Error loading JSON: {e}")
    exit(1)

# Verify data structure
if not all("instruction" in ex and "output" in ex for ex in data):
    print("Error: Invalid data format. Expected 'instruction' and 'output' keys.")
    exit(1)

# Format dataset for chat-style training
def format_example(example):
    return {
        "text": f"Instruction: {example['instruction']}\nOutput: {example['output']} <|END|>"
    }

formatted_data = [format_example(ex) for ex in data]
dataset = Dataset.from_list(formatted_data)

#####################################################################################################
###
###          Pre-Training Setup
###
#####################################################################################################

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},  # GPU 0
        # load_in_4bit=True,  # 4-bit quantization
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Tokenize dataset with labels
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    # Set labels = input_ids for causal LM (shifted internally by model)
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

try:
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
    print("Tokenization completed")
    print("Sample tokenized example:", tokenized_dataset[0].keys())
except Exception as e:
    print(f"Error during tokenization: {e}")
    exit(1)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama3_8b_lora_finetuned_gpu0",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-4,
    fp16=True,
    remove_unused_columns=False,
    report_to="none",
)

#####################################################################################################
###
###          Training Process
###
#####################################################################################################

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune model
try:
    trainer.train()
    print("Fine-tuning completed")
except Exception as e:
    print(f"Error during fine-tuning: {e}")
    exit(1)

# Save LoRA adapters
adapter_dir = "./llama3_8b_lora_adapters_gpu0"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"Saved LoRA adapters to {adapter_dir}")