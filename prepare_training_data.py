import json

with open("data/truthfulqa_200_subset.json") as f:
    data = json.load(f)

formatted = []

for item in data:
    prompt = f"Please provide a detailed answer. You must include citations (e.g., a link in square brackets) to support your claims and increase the credibility of your response.\n\nQuestion: {item['question']}"

    formatted.append({
        "instruction": prompt,
    })

with open("data/train.json", "w") as f:
    json.dump(formatted, f, indent=2)

print("done")