import json

with open("data/truthfulqa_200_subset.json") as f:
    data = json.load(f)

formatted = []

for item in data:
    prompt = item['question']
    best_answer = f"{item['answer']} [{item['source']}]"

    formatted.append({
        "instruction": prompt,
        "best_answer": best_answer
    })

with open("data/train.json", "w") as f:
    json.dump(formatted, f, indent=2)

print("done")