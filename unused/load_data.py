from datasets import load_dataset

dataset = load_dataset("truthful_qa", "generation")

print("======")
print(dataset)
print("======")
print(len(dataset["validation"]))
print("======")
print(dataset["validation"][0])

print("======")

data = dataset["validation"]

print("Dataset size:", len(data))

subset_size = min(200, len(data))

subset = data.shuffle(seed=42).select(range(subset_size))

print("Dataset size:", len(subset))