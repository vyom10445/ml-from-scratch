import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loss_function(m,b,data):
    total_error=0
    n=len(data)

    for i in range(n):
        x = data.iloc[i].studytime
        y = data.iloc[i].score
        prediction = m * x + b
        total_error+= (y - prediction)**2
    return total_error/float(n)

def gradient_descent(m_now , b_now , data , learning_rate):
    m_gradient=0
    b_gradient=0
    n=len(data)

    for i in range(n):
        x = data.iloc[i].studytime
        y = data.iloc[i].score

        prediction = m_now * x + b_now
        m_gradient += -(2/n)*x*(y - prediction)
        b_gradient += -(2/n)*(y - prediction)

    m = m_now - learning_rate*m_gradient
    b = b_now - learning_rate*b_gradient

    return m , b


def train_model(data, learning_rate=0.0001, epochs=1000):

    m = 0
    b = 0

    for epoch in range(epochs):

        m, b = gradient_descent(m, b, data, learning_rate)

        if epoch % 100 == 0:
            loss = loss_function(m, b, data)
            print(f"Epoch {epoch} | Loss: {loss}")

    return m, b



if __name__ == "__main__":
    data = pd.read_csv("linear-regression/student_scores.csv")

    m,b= train_model(data)

    print ("\nfinal model parameters:")
    print("slope(m)" ,m)
    print("intercept(b)", b)

    plt.scatter(data.studytime, data.score, color="black", label="Data Points")
    x_values = np.linspace(data.studytime.min(), data.studytime.max(), 100)
    y_values = m * x_values + b

    plt.plot(x_values, y_values, color="red", label="Regression Line")

    plt.xlabel("Study Time")
    plt.ylabel("Score")
    plt.title("Linear Regression From Scratch")
    plt.legend()

    plt.show()