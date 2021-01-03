from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.models import Model
import numpy as np

# ImageDataGenerator: 이미지들로 데이터 셋 만드는 객체
generator = ImageDataGenerator(rescale = 1/255.)

#flow_from_directory:(경로, 리사이즈, 한번에 32개묶어서, 두 개의 클래스(0, 1))
train_gen = generator.flow_from_directory('D:\이것저것\Project\Study\DeepLearning\dogs-vs-cats\Train',
                                                        target_size=(64, 64), batch_size = 32, class_mode = 'binary')
test_gen = generator.flow_from_directory('D:\이것저것\Project\Study\DeepLearning\dogs-vs-cats',
                                                        target_size=(64, 64), batch_size = 32, classes = ['Test1'])

x_batch, y_batch = next(train_gen)
print(x_batch.shape); print(y_batch.shape)
# max의 값이 너무 크면 적게 바꿔주는게 필수
# 이미지를 비교할때는 max가 1, min이 0이 되는걸 권장, line 6: ImageDataGenerator(rescale = 1/255.)함수가 자동적으로 처리
print(x_batch.max()); print(x_batch.min())

def build_model():
    x = Input(shape = (64, 64, 3))
    out = x

    # Flatten()(out): out인자를 한줄로 펼침
    out = Flatten()(out)
    out = Dense(512, activation = 'relu')(out)
    out = Dense(256, activation = 'relu')(out)
    out = Dense(64, activation = 'relu')(out)
    out = Dense(1, activation = 'sigmoid')(out)

    model = Model(x, out)
    return model

model = build_model()
print(model.summary())

# Train Model
# binary_crossentropy 는 0과 1사이에서만 사용하는 함수
# binary_crossentropy: t와 y가 비슷할수록 loss down /// t = {0, 1} -> 정답이 0이면 결과가 0, 정답이 1이면 결과가 1 /// (t = 0, y = 0), (t = 1, y = 1)
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
history = model.fit(train_gen, epochs = 1, steps_per_epoch = len(train_gen))

# Evaluate and Test Model
x_tests, y_tests = [], []
for i in range(len(test_gen)):
    x_test, y_test = next(test_gen)
    x_tests.append(x_test)
    y_tests.append(y_test)

# np.concatenate: 다음에 오는 Numpy Array List 를 몇번째 축에 붙일건지(x_tests, axis = 0) -> x_tests 축에서의 0번째
x_test = np.concatenate(x_tests, axis = 0)
y_test = np.concatenate(y_tests, axis = 0)
