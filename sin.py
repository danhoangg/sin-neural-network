import numpy as np

def init_params():
    W1 = np.random.rand(3,1) - 0.5
    B1 = np.random.rand(3,1) - 0.5
    W2 = np.random.rand(3,3) - 0.5
    B2 = np.random.rand(3,1) - 0.5
    W3 = np.random.rand(1,3) - 0.5
    B3 = np.random.rand(1,1) - 0.5
    return W1, B1, W2, B2, W3, B3

def ReLU(Z):
    return np.maximum(0, Z)

def DReLU(Z):
    return Z > 0
    
def Sigmoid(Z):
    return (1/(1+np.exp(-Z)))

def DSigmoid(Z):
    return Sigmoid(Z)*(1-Sigmoid(Z))

def forward(W1, B1, W2, B2, W3, B3, X):
    H1 = np.dot(W1, X) + B1
    Z1 = ReLU(H1)
    H2 = np.dot(W2, Z1) + B2
    Z2 = ReLU(H2)
    H3 = np.dot(W3, Z2) + B3
    Z3 = Sigmoid(H3)
    return H1, Z1, H2, Z2, H3, Z3

def loss(Z3, Y):
    return np.square(Y-Z3)

def backward(W1, B1, W2, B2, W3, B3, H1, Z1, H2, Z2, H3, Z3, Y, X):
    dE = -2*(Y-Z3)
    dW3 = (dE*DSigmoid(H3)*Z2).T
    dB3 = dE*DSigmoid(H3)
    dW2 = dE*DSigmoid(H3)*W3*DReLU(H2)*Z1
    dB2 = (dE*DSigmoid(H3)*W3*DReLU(H2).T).T
    dZ2 = dE*DSigmoid(H3)*W3*DReLU(H2)*W2
    dW1 = (np.sum(dZ2, axis=1)*DReLU(H1).T*X).T
    dB1 = (np.sum(dZ2, axis=1)*DReLU(H1).T).T
    return dW3, dB3, dW2, dB2, dW1, dB1

def update_params(W1, B1, W2, B2, W3, B3, dW3, dB3, dW2, dB2, dW1, dB1, alpha):
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2
    W3 = W3 - alpha * dW3
    B3 = B3 - alpha * dB3
    return W1, B1, W2, B2, W3, B3

def gradient_descent(iterations, alpha):
    W1, B1, W2, B2, W3, B3 = init_params()
    for i in range(iterations):
        X = np.array([[np.random.rand()]])
        Y = 0.5 * np.sin(2*np.pi*X) + 0.5 
        H1, Z1, H2, Z2, H3, Z3 = forward(W1, B1, W2, B2, W3, B3, X)
        dW3, dB3, dW2, dB2, dW1, dB1 = backward(W1, B1, W2, B2, W3, B3, H1, Z1, H2, Z2, H3, Z3, Y, X)
        W1, B1, W2, B2, W3, B3 = update_params(W1, B1, W2, B2, W3, B3, dW3, dB3, dW2, dB2, dW1, dB1, alpha)
        if i % 10000 == 0:
            error = loss(Z3, Y)
            print("Iteration: ", i)
            print("Loss: ", loss(Z3, Y))
    return W1, B1, W2, B2, W3, B3

W1, B1, W2, B2, W3, B3 = gradient_descent(100000, 0.1)
H1, Z1, H2, Z2, H3, Z3 = forward(W1, B1, W2, B2, W3, B3, [[0.5]])
print(Z3)
