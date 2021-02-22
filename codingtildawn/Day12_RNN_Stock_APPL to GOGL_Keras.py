# -*- coding: utf-8 -*-
"""Untitled16.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YQofHAcKr2icHCT_o9S5_XFYVOqChW6f
"""

from google.colab import drive
drive.mount('/content/gdrive')

!ls /content/gdrive/'My Drive'/DeepLearning -la
!cp /content/gdrive/'My Drive'/DeepLearning/NYST.zip .

# !unzip NYST.zip -d data

import pandas as pd

funda = pd.read_csv('data/fundamentals.csv')
price = pd.read_csv('data/prices.csv')
prices_split = pd.read_csv('data/prices-split-adjusted.csv')
security = pd.read_csv('data/securities.csv')

funda.head()

price.head()
# symbol: 회사 명 약자
# volume: 주식거래량, 보통 volume이 높아지면 주가가 올라감.

prices_split.head()

security.head()

# apple 주식만 나열
prices_split.loc[(prices_split['symbol'] == 'AAPL')]

# google 주식만 나열
prices_split.loc[(prices_split['symbol'] == 'GOOGL')]

security.head()

# security, fundamental은 무슨 의미인지 몰라서 제외
# prices는 도중에 값이 이상해짐

split = pd.read_csv('data/prices-split-adjusted.csv')

split['date'] = pd.to_datetime(split['date'])

split['year'] =split['date'].dt.year
split['month'] =split['date'].dt.month
split['day'] =split['date'].dt.day

# train: apple 주식만 필터링
train = split.loc[(split['symbol'] == 'AAPL')]

# test: google 주식만 필터링
test = split.loc[(split['symbol'] == 'GOOGL')]

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

# train: apple 주식 그래프
df = train.loc[train['year']>=2010]

plt.figure(figsize=(16, 9))
sns.lineplot(y=df['close'], x=df['date'])
plt.xlabel('time')
plt.ylabel('price')

# test: google 주식 그래프
df = test.loc[split['year']>=2010]

plt.figure(figsize=(16, 9))
sns.lineplot(y=df['close'], x=df['date'])
plt.xlabel('time')
plt.ylabel('price')

train.head()

test.head()

# data normalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale_cols = ['open', 'close', 'low', 'high', 'volume']
scaled_train = scaler.fit_transform(train[scale_cols])
scaled_test = scaler.fit_transform(test[scale_cols])

scaled_train = pd.DataFrame(scaled_train)
scaled_test = pd.DataFrame(scaled_test)

scaled_train.columns = scale_cols
scaled_test.columns = scale_cols

scaled_train

scaled_test

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

train_feature = scaled_train[['open', 'high', 'low', 'volume']]
train_label = scaled_train[['close']]
test_feature = scaled_test[['open', 'high', 'low', 'volume']]
test_label = scaled_test[['close']]

import numpy as np

# train dataset
train_feature, train_label = make_dataset(train_feature, train_label, 20)

from sklearn.model_selection import train_test_split

# train, validation set 생성
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
x_train.shape, x_valid.shape

# test dataset (실제 예측 해볼 데이터)
test_feature, test_label = make_dataset(test_feature, test_label, 20)
test_feature.shape, test_label.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM


def build_model():
    model = Sequential()
    
    model.add(LSTM(16, input_shape=(train_feature.shape[1], train_feature.shape[2]), activation='relu', return_sequences=False))
    model.add(Dense(1))

    return model

model = build_model()
model.summary()

# 검증 데이터의 손실(loss)이 증가하면, 과적합 징후이므로 검증 데이터 손실이 4회 증가하면 학습을 중단하는 조기 종료(EarlyStopping)를 사용합니다.
# 또한, ModelCheckpoint를 사용하여 검증 데이터의 정확도가 이전보다 좋아질 경우에만 모델을 저장하도록 합니다.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('check.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['acc'])
history = model.fit(x_train, y_train, epochs = 200, batch_size = 32, validation_data = (x_valid, y_valid), callbacks = [es, mc])

# weight 로딩
model.load_weights('check.h5')

# 예측
pred = model.predict(test_feature)

plt.figure(figsize=(12, 9))
plt.plot(test_label, label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()