import numpy as np
import pandas as pd

df = pd.read_csv("logistic-regression\heart.csv")
print(df.head())

x=df.drop("target",axis=1).values
y=df["target"].values

x = x - x.mean(axis=0)/x.std(axis=0)