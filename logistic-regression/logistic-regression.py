import numpy as np
import pandas as pd

df = pd.read_csv("logistic-regression\heart.csv")
print(df.head())

x=df.drop("target",axis=1).values
y=df["target"].values

#feature scaling
x = x - x.mean(axis=0)/x.std(axis=0)

#train test split
split_index = int(0.8 * len(x))

x_train = x[:split_index]  #take rows from beginning up to row {split_index}
y_train = y[:split_index]  #take rows from {split_index} till end

x_test = x[split_index:]
y_test = y[split_index:]