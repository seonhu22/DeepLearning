def activation(x):
    if x < 0:
        return 0
    else:
        return 1

def perceptron(x1, x2, w1, w2, b):
    y = w1*x1 + w2*x2 + b
    return activation(y)

def xor_perceptron(a, b):
    A = perceptron(a, b, 1, -1, 0.5)
    B = perceptron(a, b, 1, -1, -0.5)

    C = perceptron(A, B, -1, 1, 0)
    return C

for x1 in [0, 1]:
    for x2 in [0, 1]:
        result = xor_perceptron(x1, x2)
        print(f'{x1} XOR {x2} = {result}')
