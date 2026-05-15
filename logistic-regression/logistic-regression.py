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

n_features = x_train.shape[1]
weights = np.zeros(n_features) #initialise weights to no. of train features
bias=0


def sigmoid(z):
    return (1/1+ np.exp(-z))


learning_rate = 0.01
epochs = 1000

n = len(x_train)

for epoch in range(epochs):
    #prediction
    z = np.dot(x_train,weights)+ bias

    #apply sigmoid on pred
    y_pred= sigmoid(z)
    
    #gradients
    dw = (1/n) * np.dot(x_train.T, (y_pred - y_train))
    db = (1/n) * np.sum(y_pred - y_train)

    #update parameters
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    # epoch print
    if epoch % 100 == 0:
        print(f"Epoch {epoch}")


#prediction
z=np.dot(x_test,weights)+bias
predictions = sigmoid(z)
predicted_classes = []

for i in predictions:

    if i > 0.5:
        predicted_classes.append(1)

    else:
        predicted_classes.append(0)

#accuracy
accuracy = np.mean(predicted_classes == y_test)

print("Accuracy:", accuracy)