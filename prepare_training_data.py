import json

with open("data/truthfulqa_200_subset.json") as f:
    data = json.load(f)

formatted = []

for item in data:
    prompt = f"Please provide a factual and detailed answer. You must include citations by adding external links inside square brackets, for example: [https://example.com] to support your claims and increase the credibility of your response.\n\nQuestion: {item['question']}"
    best_answer = f"{item['answer']} [{item['source']}]"

    formatted.append({
        "instruction": prompt,
        "best_answer": best_answer
    })

with open("data/train.json", "w") as f:
    json.dump(formatted, f, indent=2)

print("done")