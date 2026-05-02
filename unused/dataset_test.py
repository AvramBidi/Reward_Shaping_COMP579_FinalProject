import pandas as pd

df = pd.read_csv("hf://datasets/domenicrosati/TruthfulQA/train.csv")

print(df.columns['Question'])