import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM, Dropout
from tensorflow.keras.optimizers import Adam


def train():
    train_x = np.load("train_x.npy")
    train_y = np.array(np.load("train_y.npy"))
    validation_x = np.load("validation_x.npy")
    validation_y = np.array.(np.load("validation_y.npy"))
    
    neural = tf.keras.models.Sequential()
    neural.add(LSTM(120, input_shape = (train_x.shape[1:]), return_sequences=True, activation=tf.nn.relu))
    neural.add(Dropout(.20))
    neural.add(BatchNormalization())

    neural.add(LSTM(120,  activation=tf.nn.relu))
    neural.add(Dropout(.10))
    neural.add(BatchNormalization())

    neural.add(Dense(64, activation=tf.nn.relu))
    neural.add(Dropout(.20))
    neural.add(BatchNormalization())

    neural.add(Dense(1, activation=tf.nn.sigmoid))
    neural.add(Dropout(.20))
    neural.add(BatchNormalization())

    optm = Adam(lr=1e-3, decay=1e-3)

    neural.compile(optimizer=optm,
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    neural.fit(train_x, train_y, epochs=1, validation_data=(validation_x,validation_y))
return

