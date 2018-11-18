
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import TensorBoard

NAME = "DeepRecommender-Sequential-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='./logs/{}'.format(NAME))

DATA_DIR = './nf_data'

nf_3m_train = os.path.join(DATA_DIR, 'N3M_TRAIN', 'n3m.train.txt')
nf_3m_valid = os.path.join(DATA_DIR, 'N3M_VALID', 'n3m.valid.txt')
nf_3m_test = os.path.join(DATA_DIR, 'N3M_TEST', 'n3m.test.txt')


df_train = pd.read_csv(nf_3m_train, names=['CustomerID','MovieID','Rating'], sep='\t')
print(df_train.shape)
df_train.head()


df_valid = pd.read_csv(nf_3m_valid, names=['CustomerID','MovieID','Rating'], sep='\t')
print(df_valid.shape)
df_valid.head()


df_test = pd.read_csv(nf_3m_test, names=['CustomerID','MovieID','Rating'], sep='\t')
print(df_test.shape)
df_test.head()


customer_map = df_train.CustomerID.unique()
customer_map.sort()
customer_map = customer_map[0:17550]
num_users = len(customer_map)


movie_map = df_train.MovieID.unique()
movie_map.sort()
movie_map = movie_map[0:1000]
num_movies = len(movie_map)


X = pd.read_csv('matrix_train.csv', header=None)
print(X.shape)
X.head()


customer_map = df_valid.CustomerID.unique()
customer_map.sort()
customer_map = customer_map[:7020]
num_users = len(customer_map)


movie_map = df_valid.MovieID.unique()
movie_map.sort()
movie_map = movie_map[:895]
num_movies = len(movie_map)


X_valid = pd.read_csv('matrix_valid.csv', header=None)
print(X_valid.shape)
X_valid.head()


customer_map = df_test.CustomerID.unique()
customer_map.sort()
customer_map = customer_map[:7020]
num_users = len(customer_map)


movie_map = df_test.MovieID.unique()
movie_map.sort()
movie_map = movie_map[:901]
num_movies = len(movie_map)


X_test = pd.read_csv('matrix_test.csv', header=None)
print(X_test.shape)
X_test.head()


import keras.backend as K

def rmse(y_true, y_pred):
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    masked_squared_error = mask_true * K.square((y_true - y_pred))
    masked_mse = K.sum(masked_squared_error, axis=-1) / K.sum(mask_true, axis=-1)
    return K.sqrt(masked_mse)


model = Sequential()

model.add(Dense(28, input_dim = X.shape[1], activation='selu'))
model.add(Dense(56, activation='selu'))
model.add(Dense(56, activation='selu'))
model.add(Dropout(0.65))
model.add(Dense(56, activation='selu'))
model.add(Dense(28, activation='selu'))
model.add(Dense(X.shape[1], activation='selu'))


model.compile(loss=rmse, optimizer='sgd')

model.fit(X, X, batch_size=128, epochs=100, validation_data=(X_valid, X_valid), callbacks=[tensorboard])


test_loss = model.evaluate(X_test, X_test)
print('The test loss is: ', test_loss)

