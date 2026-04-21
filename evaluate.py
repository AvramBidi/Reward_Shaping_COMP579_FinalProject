import json
import re
import numpy as np
from pathlib import Path

# Optional semantic similarity
from sentence_transformers import SentenceTransformer, util

# ✅ Paths
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "data" / "baseline_outputs.json"

# print("Loading outputs from:", INPUT_PATH)

# Load data
with open(INPUT_PATH) as f:
    data = json.load(f)

# -------- Metrics -------- #

def has_citation(text):
    patterns = [
        r"\[\d+\]",
        r"\(\w+, \d{4}\)"
    ]
    return any(re.search(p, text) for p in patterns)

def length(text):
    return len(text.split())

# Semantic similarity model
sim_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_score(pred, ref):
    emb1 = sim_model.encode(pred, convert_to_tensor=True)
    emb2 = sim_model.encode(ref, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

# -------- Compute -------- #

lengths = []
citations = []
similarities = []

for item in data:
    output = item["model_output"]
    reference = item["reference"]

    lengths.append(length(output))
    citations.append(1 if has_citation(output) else 0)
    similarities.append(semantic_score(output, reference))

# -------- Results -------- #

print("\n=== Evaluation Results ===")
print("Average length:", np.mean(lengths))
print("Citation rate:", np.mean(citations))
print("Semantic similarity:", np.mean(similarities))