from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
from pathlib import Path

# ✅ Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "baseline_outputs.json"

print("Loading dataset from:", DATA_PATH)
print("Saving outputs to:", OUTPUT_PATH)

# Load dataset
with open(DATA_PATH) as f:
    data = json.load(f)

# Load model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

outputs = []

for item in tqdm(data):
    prompt = f"Question: {item['question']}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    out = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    response = tokenizer.decode(out[0], skip_special_tokens=True)

    outputs.append({
        "question": item["question"],
        "reference": item["answer"],
        "model_output": response
    })

# Save outputs
with open(OUTPUT_PATH, "w") as f:
    json.dump(outputs, f, indent=2)

print("Generation complete.")