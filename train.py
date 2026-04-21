from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from pathlib import Path

# ✅ Paths (same pattern as other scripts)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "finetuned_model"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading training data from:", DATA_PATH)
print("Saving model to:", OUTPUT_DIR)

# Load dataset
dataset = load_dataset("json", data_files=str(DATA_PATH))

# Load tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# -------- Format data -------- #
def format_example(example):
    text = f"Answer factually and cite sources if possible.\nQuestion: {example['question']}\nAnswer: {example['answer']}"
    return {"text": text}

dataset = dataset.map(format_example)

# -------- Tokenize -------- #
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

dataset = dataset.map(tokenize, batched=True)

# -------- Load model -------- #
model = AutoModelForCausalLM.from_pretrained(model_name)

# -------- LoRA config -------- #
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, peft_config)

# -------- Training args -------- #
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=True
)

# -------- Trainer -------- #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# -------- Train -------- #
trainer.train()

print("Training complete.")