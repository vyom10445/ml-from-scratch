import numpy as np

#activation function
def sigmoid(z):
    return 1/(1+np.exp(z))

#one neural network layer
def dense(a_in , W , b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w=W[:,j]
        z=np.dot(w , a_in)+b[j]
        a_out[j]= sigmoid(z)
    return a_out


#full neural network
def sequential(x):
    a1 = dense(x,W1,b1)
    a2 = dense(a1,W2,b2)
    a3 = dense(a2,W3,b3)
    a4 = dense(a3,W4,b4)
    return a4

#example random parameters
W1 = np.random.randn(2,3)
b1 = np.random.randn(3)
W2 = np.random.randn(3,4)
b2 = np.random.randn(4)
W3 = np.random.randn(4,2)
b3 = np.random.randn(2)
W4 = np.random.randn(2,1)
b4 = np.random.randn(1)

#input 
x = np.array([-2,4])

#prediction
y = sequential(x)

print("Network output:", y)




    