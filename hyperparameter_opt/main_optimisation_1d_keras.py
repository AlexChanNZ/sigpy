# -*- coding: utf-8 -*-
"""
Author : Shameer Sathar
"""
from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

import numpy as np
import scipy.io as sio

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')

from sklearn.model_selection import train_test_split
from hyperas.utils import eval_hyperopt_space


def data():
    X_data = np.load('/media/hpc/codes/GitLab/sigpy_github/X_train_all.npy')
    Y_data = np.load('/media/hpc/codes/GitLab/sigpy_github/Y_train_all.npy')
    Y_data_1D = np.argmax(Y_data, axis=1)
    dataset = np.hstack((X_data, np.expand_dims(Y_data_1D, axis=1)))
    dataset_0 = dataset[np.where(Y_data_1D==0)]
    dataset_1 = dataset[np.where(Y_data_1D==1)]
    select_0 = np.random.choice(dataset_0.shape[0],dataset_1.shape[0], replace=False)
    dataset_0_selected = dataset_0[select_0, :]
    dataset_test_train = np.vstack((dataset_0_selected, dataset_1))
    np.random.shuffle(dataset_test_train)
    np.random.shuffle(dataset_test_train)
    X_train, X_test, y_train, y_test = train_test_split(dataset_test_train[:, :-1],
                                            dataset_test_train[:, -1], test_size=0.3, random_state=42)
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, y_train, X_test, y_test

def model(X_train, y_train, X_test, y_test):
    model = Sequential()
    #1st Convolution Layer
    model.add(Convolution1D({{choice([16, 32, 64])}},9, activation='relu',
                            kernel_initializer='glorot_uniform',
                            input_shape=(36, 1)))
    model.add(MaxPooling1D(pool_size={{choice([2, 4])}}))

    #2nd Convolution Layer
    model.add(Convolution1D({{choice([16, 32, 64])}}, 3, activation='relu'))
    model.add(Dropout({{uniform(0, 1)}}))

    #Fully connected layer
    model.add(Flatten())
    model.add(Dense({{choice([64, 128, 256])}}, activation='relu'))
    model.add(Dropout({{uniform(0, 1)}}))

    # Output
    model.add(Dense(2, activation='softmax'))

    # Optimizer
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    #Compile the model
    model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])

    # Train the network
    model.fit(np.expand_dims(X_train, axis=2), y_train, nb_epoch=2)
    scores, acc = model.evaluate(np.expand_dims(X_test, axis=2), y_test)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__' :
    # #print(model(X_train, y_train, X_test, y_test))
    best_run, best_model = optim.minimize(model=model,
                                               data=data,
                                               algo=tpe.suggest,
                                               max_evals=5,
                                               trials=Trials())
    X_train, y_train, X_test, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(np.expand_dims(X_test, axis=2), y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
