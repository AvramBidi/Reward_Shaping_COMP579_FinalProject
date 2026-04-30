import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# 1. Config
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "results/ablations/DPO_train_gen_t0.7_p0.9_n16.json"
OUTPUT_DIR = "results/dpo_model"

#2. Load and Prepare Dataset 
# DPO Trainer expects columns: 'prompt', 'chosen', 'rejected'
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# 3. Tokenizer Setup 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token # LLaMA requires setting a pad token

# 4. Quantization & Model Loading (QLoRA) 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading Base Model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False 

# 5. LoRA Config
peft_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
)

# 6. Training Arguments 
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,    
    num_train_epochs=3,
    max_steps=-1,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    optim="paged_adamw_32bit",
    bf16=True,
    remove_unused_columns=False,
    
    beta=0.1, 
    max_prompt_length=512,
    max_length=1024,
)

# 7. Initialize DPO Trainer
dpo_trainer = DPOTrainer(
    model,
    ref_model=None, 
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

# 8. Execute Training 
print("Starting DPO Training...")
dpo_trainer.train()

# 9. Save the Final LoRA Adapter 
dpo_trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
print("Training Complete. Adapter saved!")