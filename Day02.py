import numpy as np

matrix = np.zeros((6, 10))     #행렬값이 모두 0인 6 by 10 행렬
matrix

l = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]]
arr = np.array(l)

print(arr.shape)
print(arr)

arr[0, 0, 0] = arr[0, 0, 0] * 2
arr
arr[:, :2]
arr[:, :2].shape

matrix = np.random.normal(size=(6, 10))
matrix

in_vector = np.array([[1, 2, 3, 4, 5, 6]])
in_vector.shape

bias = np.random.normal(size=(1, 10))

out = in_vector @ matrix + bias    #numpy끼리의 행렬 곱 : @
print(out)
print(out.shape)

# activation

def relu(x: np.ndarray):
    return np.abs(x) * (x > 0)

relu(matrix)

# 1 / (1+e^(-x))

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

sigmoid(matrix)

# simple model

in_vector = np.array([[1, 2, 3, 4, 5, 6]])

W1 = np.random.normal(size=(6, 10))
b1 = np.random.normal(size=(1, 10))

W2 = np.random.normal(size=(10, 12))
b2 = np.random.normal(size=(1, 12))

W3 = np.random.normal(size=(12, 1))
b3 = np.random.normal(size=(1, 1))

# feed forward

activation = sigmoid

out1 = activation(in_vector @ W1 + b1)
out2 = activation(out1 @ W2 + b2)
out3 = activation(out2 @ W3 + b3)

print('IN: ')
print(in_vector)
print('OUT: ')
print(out3)
