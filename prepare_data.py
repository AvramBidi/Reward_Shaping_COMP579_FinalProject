import json

with open("data/dataset.json") as f:
    data = json.load(f)

formatted = []

for item in data:
    prompt = f"Answer the question using facts and cite sources.\nQuestion: {item['question']}"
    response = f"{item['answer']} [source]"

    formatted.append({
        "instruction": prompt,
        "output": response
    })

with open("data/train.json", "w") as f:
    json.dump(formatted, f, indent=2)