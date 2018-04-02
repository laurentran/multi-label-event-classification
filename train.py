from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import keras
import pandas as pd
import numpy as np
import h5py

# file paths
train_data_path = 'data/train.csv'
test_data_path = 'data/test.csv'
train_labels_path = 'data/train_labels.csv'
test_labels_path = 'data/test_labels.csv'
model_path = 'outputs/model.h5'

# set parameters and constants
batch_size = 128
num_classes = 4
epochs = 50

# grid dimensions
x_len, y_len = 8, 8

# import training data and split between train and dev sets (February data)
X = pd.read_csv(train_data_path, header=None).as_matrix()
Y = pd.read_csv(train_labels_path, header=None).as_matrix()
x_train, x_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.1, stratify=Y)

# load and prepare test set (January data)
x_test = pd.read_csv(test_data_path, header=None).as_matrix()
y_test = pd.read_csv(test_labels_path, header=None).as_matrix()

# normalize data with l2 norm
x_train = preprocessing.normalize(x_train)
x_dev = preprocessing.normalize(x_dev)
x_test = preprocessing.normalize(x_test)

# reshape (1,64) vectors as (8,8) grid
x_train = x_train.reshape(x_train.shape[0], x_len, y_len, 1)
x_dev = x_dev.reshape(x_dev.shape[0], x_len, y_len, 1)
x_test = x_test.reshape(x_test.shape[0], x_len, y_len, 1)

# define input shape to layer 1 of network
input_shape = (x_len, y_len, 1)

print(x_train.shape[0], 'train samples')
print(x_dev.shape[0], 'test samples')

# define CNN architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 padding='same',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(num_classes, activation='sigmoid')) # sigmoid activation for multi-label prediction

# compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_dev, y_dev))

# evaluate model and print loss and accuracy on dev and test sets
score = model.evaluate(x_dev, y_dev, verbose=0)

print('Dev loss:', score[0])
print('Dev accuracy:', score[1])

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model, saving as .h5 file
model.save(model_path)