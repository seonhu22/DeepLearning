import numpy as np

# XOR model with numpy

#x_data = np.array([
#                    [0, 0], [0, 1], [1, 0], [1, 1]
#])
#y_data = np.array([
#                   [0], [1], [1], [0]
#])

def func(x):
    return 0.3 * x - 2 + np.random.normal(0, 0.1)

x_data, y_data = [], []
for i in np.arange(-1, 1, 0.01):
    x_data.append([i])
    y_data.append([func(i)])

x_data = np.array(x_data)

print(x_data)   #(number of data, feature vector size)
print(y_data)

# model structure

W = np.random.normal(size=(1, 1))
b = np.random.normal(size=(1, 1))

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def model(X):
    k = X @ W + b
    out = sigmoid(k)
    return out, k

pred, k = model(x_data)
print(pred)

def loss(y_true, y_pred):
    return 0.5 * np.mean(np.square(y_true - y_pred))

loss(y_data, pred)

# calc gradient

def gradient(X, Y, out, k):
    grad_W = np.mean((Y - out) * k * (1 - k) * X, axis = 0).reshape((1, 1))
    grad_b = np.mean((Y - out) * k * (1 - k), axis = 0).reshape(1, 1)
    return grad_W, grad_b

grad_W, grad_b = gradient(x_data, y_data, pred, k)
print(grad_W, grad_b)
print(grad_W.shape)

def update(p, grad, alpha = 0.0001):
    return p - grad * alpha

print('Before update:\n', W, b)
W = update(W, grad_W)
b = update(b, grad_b)
print('After update:\n', W, b)

for i in range(400):
    pred, k = model(x_data)
    # print(pred)
    print(loss(y_data, pred))
    print('\n')

    grad_W, grad_b = gradient(x_data, y_data, pred, k)
    W = update(W, grad_W)
    b = update(b, grad_b)
