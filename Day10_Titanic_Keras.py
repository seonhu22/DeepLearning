# -*- coding: utf-8 -*-
"""Untitled11.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xh5O9jXMhEIm_qf5gxo3OLbYf7TCshmR
"""

from google.colab import drive
drive.mount('/content/gdrive')

!ls /content/gdrive/'My Drive'/DeepLearning -la
!cp /content/gdrive/'My Drive'/DeepLearning/titanic.zip .

!unzip titanic.zip -d data

import pandas as pd
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train.head()

# PassengerId : 승객 번호
# Survived : 생존여부(1: 생존, 0 : 사망)
# Pclass : 승선권 클래스(1 : 1st, 2 : 2nd ,3 : 3rd)
# Name : 승객 이름
# Sex : 승객 성별
# Age : 승객 나이 
# SibSp : 동반한 형제자매, 배우자 수
# Patch : 동반한 부모, 자식 수
# Ticket : 티켓의 고유 넘버
# Fare 티켓의 요금
# Cabin : 객실 번호
# Embarked : 승선한 항구명(C : Cherbourg, Q : Queenstown, S : Southampton)

train.info()

train.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind = 'bar', stacked = True, figsize = (10, 5))

bar_chart('Sex')

bar_chart('Pclass')

bar_chart('SibSp')

bar_chart('Parch')

bar_chart('Embarked')

train.describe(include = 'all')

# Age: 20% Null
# Cabin: 90% Null
# Name, Sex, Ticket, Cabin, Embarked: str, change to int, float or drop

train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

train.head()

# Embarked optimization

southampton = train[train['Embarked'] == 'S'].shape[0]
print('S: ',southampton)
cherbourg = train[train['Embarked'] == 'C'].shape[0]
print('C :', cherbourg)
queenstown = train[train['Embarked'] == 'Q'].shape[0]
print('Q: ', queenstown)

train = train.fillna({'Embarked': 'S'})

embarked_mapping = {'S': 1, 'C': 2, 'Q': 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()

# Name optimization

combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

pd.crosstab(train['Title'], train['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index = False).mean()

title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Other': 6}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()

train = train.drop(['Name', 'PassengerId'], axis = 1)
test = test.drop(['Name'], axis = 1)
combine = [train, test]
train.head()

sex_mapping = {'male': 0, 'female': 1}

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

train.head()

import numpy as np

train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels = labels)
train.head()

bar_chart('AgeGroup')

age_title_mapping = {1: 'Young Adult', 2: 'Student', 3: 'Adult', 4: 'Baby', 5: 'Adult',6: 'Adult'}

for x in range(len(train['AgeGroup'])):
    if train['AgeGroup'][x] == 'Unknown':
        train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]

for x in range(len(test['AgeGroup'])):
    if test['AgeGroup'][x] == 'Unknown':
        test['AgeGroup'][x] = age_title_mapping[test['Title'][x]]

train.head()

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)

train.head()

train['Fareband'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['Fareband'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)

train.head()

train_data = train.drop('Survived', axis = 1)
target = train['Survived']

train_data.shape, target.shape

train.info()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)

clf = RandomForestClassifier(n_estimators = 13)
clf

scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_fold, n_jobs = 1, scoring = scoring)
print(score)

round(np.mean(score)*100, 2)