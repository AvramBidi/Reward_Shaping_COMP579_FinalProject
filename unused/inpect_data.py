from datasets import load_dataset

dataset = load_dataset("truthful_qa", "generation")

print(dataset)
print(dataset["validation"][0])