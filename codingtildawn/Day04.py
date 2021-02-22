import numpy as np

# create dataset of XOR problem

x_data = np.array([
                   [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
])
y_data = np.array([
                   [0], [1], [1], [0], [1], [0], [0], [1]
])

print(x_data.shape)
print(y_data.shape)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def build_model():
    W1 = np.random.normal(size = (3, 9))
    b1 = np.random.normal(size = (1, 9))
    W2 = np.random.normal(size = (9, 1))
    b2 = np.random.normal(size = (1, 1))

    return [(W1, b1), (W2, b2)]

def predict(x, model: list, return_hidden = False):
    t = x
    hiddens = []
    for W, b in model:
        h = t @ W + b
        hiddens.append((t, W))
        t = sigmoid(h)

    if return_hidden:
        return t, hiddens

    return t

model = build_model()

pred = predict(x_data, model)
print(pred)

# loss

def mse(y_true, y_pred):
    return 0.5 * np.mean(np.square(y_true - y_pred))

loss = mse(y_data, pred)
print(loss)

# gradient(back propagation)

def gradient(y_true, y_pred, hidden):
    # grad_W = np.mean((Y - out) * k * (1 - k) * X, axis = 0), reshape((1, 1))
    # grad_b = np.mean((Y - out) * k * (1 - k), axis = 0), reshape(1, 1)

    N = len(y_true)
    d_loss = y_true - y_pred
    o = y_pred

    gradients = []
    for t, W in hiddens[::-1]:
        grad_b = d_loss * o * (1 - o)
        grad_W = t.T @ grad_b
        gradients.append((grad_W / N, np.mean(grad_b, axis = 0, keepdims = True)))

        o = t
        d_loss = grad_b @ W.T

    return gradients[::-1]

pred, hiddens = predict(x_data, model, return_hidden = True)
grads = gradient(y_data, pred, hiddens)
print(grads)

lr = 0.1

model = [tuple(w - lr * g for w, g in zip(layer, grad)) for layer, grad in zip(model, grads)]
print(model)

n_iter = 50000

losses = []
for i in range(n_iter):
    pred, hiddens = predict(x_data, model, return_hidden = True)
    loss = mse(y_data, pred)
    grads = gradient(y_data, pred, hiddens)
    model = [tuple(w - lr * g for w, g in zip(layer, grad)) for layer, grad in zip(model, grads)]
    if i % 500 == 0:
        print(loss)
    losses.append(loss)

pred = predict(x_data, model)
print(pred)
print(y_data)
