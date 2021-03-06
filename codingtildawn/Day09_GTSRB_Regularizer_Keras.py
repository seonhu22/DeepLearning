# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iz4vB1fiMpPHazhf7wSUgKaKnsjYVaPz
"""

from google.colab import drive
drive.mount('/content/gdrive')

!ls /content/gdrive/'My Drive'/DeepLearning -la
!cp /content/gdrive/'My Drive'/DeepLearning/GTSRB.zip .

!unzip GTSRB.zip -d GTSRB

from keras.preprocessing.image import ImageDataGenerator

# validation = overfit방지
# validation_split(0.2) = train 데이터의 20%를 중간검증으로 제외
# 기본적인 Train 데이터셋이 많이 부족할때는 cross_validation 사용(보다 높은 정확도, 많이 느린 속도)
# seed = 실행할떄마다 같은 데이터셋으로 시작(시작할때마다 결과값이 다른걸 방지)
generator = ImageDataGenerator(rescale = 1/255., validation_split = 0.2)

train_gen = generator.flow_from_directory('GTSRB/Train', target_size = (64, 64), seed = 719, class_mode = 'categorical', subset = 'training')
valid_gen = generator.flow_from_directory('GTSRB/Train', target_size = (64, 64), seed = 719, class_mode = 'categorical', subset = 'validation')
test_gen = generator.flow_from_directory('GTSRB', target_size=(64, 64), batch_size = 32, classes = ['Test'])

x_train, y_train = next(train_gen)
x_test, y_test = next(test_gen)
print(x_train.shape); print(y_train.shape)
print(x_test.shape); print(y_test.shape)

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

for i in range(len(x_train)):
    plt.imshow(x_train[i])
    plt.title(y_train[i])
    plt.show()

from keras.layers import *
from keras.models import Model, Sequential

def build_model():
    model = Sequential()

    # kernel_regularizer : Weight Decay, 'l1' : |w|, 'l2' : w^2
    # use_bias = False : BN을 쓰면 bias를 총 두번 더하니 한 번 제외
    model.add(Conv2D(input_shape = (64, 64, 3), filters = 32, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_normal'))
    #model.add(Conv2D(filters = 16, kernel_size = (1, 1), padding = 'valid', kernel_initializer = 'he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_normal'))
    #model.add(Conv2D(filters = 16, kernel_size = (1, 1), padding = 'valid', kernel_initializer = 'he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2))) # pool_size = 전체 크기를 절반으로 줄임

    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_normal'))
    #model.add(Conv2D(filters = 32, kernel_size = (1, 1), padding = 'valid', kernel_initializer = 'he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))   
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_normal'))
    #model.add(Conv2D(filters = 32, kernel_size = (1, 1), padding = 'valid', kernel_initializer = 'he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu', kernel_initializer = 'he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation = 'softmax', kernel_initializer = 'he_normal'))

    return model

model = build_model()
model.summary()

# Train Model
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# history = model.fit(x_train, y_train, epochs = 1000, batch_size = 32)   # validation = 중간점검
from keras.optimizers import SGD, Adam

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0002, momentum=0.9, nesterov=True), metrics=['accuracy'])
history=model.fit(train_gen, epochs=100, batch_size=32, steps_per_epoch=len(train_gen), validation_data=valid_gen, use_multiprocessing=True, workers=0)
