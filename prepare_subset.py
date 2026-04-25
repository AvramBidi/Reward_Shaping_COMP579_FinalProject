from datasets import load_dataset
import json
from pathlib import Path

# ✅ Project root (go up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "truthfulqa_200_subset.json"



# Load dataset
dataset = load_dataset("truthful_qa", "generation")
data = dataset["validation"]

# Safe subset size
subset_size = min(200, len(data))
subset = data.shuffle(seed=42).select(range(subset_size))

clean_data = []

for item in subset:
    # 1. Safely grab the source field (default to an empty string if missing or None)
    source_text = str(item.get("source", ""))
    
    # 2. Filter out items that don'tcontain 'https'
    if "https" not in source_text:
        continue
        
    # 3. If it passes the filter, append it to your clean list
    clean_data.append({
        "question": item["question"],
        "answer": item["best_answer"],
        "source": source_text 
    })

print("Saving dataset to:", DATA_PATH)

print("Collected samples:", len(clean_data))

# Save JSON
with open(DATA_PATH, "w") as f:
    json.dump(clean_data, f, indent=2)

print("Done.")