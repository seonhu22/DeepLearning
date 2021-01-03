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

import keras
keras.__version__
from keras.layers import *
from keras.models import Model

def build_model():
    x = Input(shape=(3,))
    out = x

    out = Dense(9, activation='relu')(out)
    out = Dense(9, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(x, out, name='mlp')

    return model

model = build_model()
model.summary()

model.compile(optimizer = 'sgd', loss = 'mse')   # loss를 줄이기 위한 과정, 제일 보편적인게 sgd
history = model.fit(x_data, y_data, batch_size = 8, epochs = 10000)   # batch_size = 한번에 학습할 데이터 양, n_epoch = 전체 데이터 순회하면 1epoch.

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label = 'loss')

pred = model.predict(x_data)
print(pred)
print(y_data)
