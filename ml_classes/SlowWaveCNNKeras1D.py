# -*- coding: utf-8 -*-
"""
Author : Shameer Sathar
"""
import sys, os # to browse directories

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

import matplotlib.pyplot as plt
from gui_plotting.mpl_plots import *

import config_global as sp

class SlowWaveCNNKeras1D:

    def __init__(self, training_set=None, events=None):
        pass


    def classify_data(self, test_data, type_data_set):
        """
        The data is classified based on the training data.
        :param plot: Input values to be processed for generating features
        :return: predictions for the entire data set
        """
        self.train_neural_net(type_data_set);
        test_array = np.asarray(test_data)
        print('Shape of the array is', test_array.shape)
        print('Test_array is arranged as Row1 = 1:36;  Row2 = 1:36 ; Row3 =1:36 and so on')
        #test_array = test_array.reshape((-1, 1, 6, 6))
        #print(test_array.shape)
        sp.gui.statBar.showMessage("Classifying...")
        predictions = self.neural_net.predict(np.expand_dims(test_array, axis=2))
        print(np.argmax(predictions, axis=1).shape)
        sp.gui.statBar.showMessage("Finished classifying.")
        eventsFound = np.where(np.argmax(predictions, axis=1) == 1)[0]

        return predictions, eventsFound



    def load_training_dataset(self, dataType):

        trainingFileAndPath = sp.dataRoot + '/train_samples_label_data.mat'
        data_dic = sio.loadmat(trainingFileAndPath)
        trainingSamples = np.array(data_dic['data'])
        X_train = trainingSamples[:,:-1]
        Y_train = trainingSamples[:, -1]
        Y_train = np_utils.to_categorical(Y_train, 2)
        return X_train, Y_train

    def train_neural_net(self, type_data_set):

        nnFileName = "nn_1D_all_.h5"

        nnFileNameAndPath = sp.nnPath + nnFileName
        print("nnFileNameAndPath: ", nnFileNameAndPath)

        #Load CNN instead of training
        try:
            self.neural_net = load_model(nnFileNameAndPath)
            return

        except Exception as e:

            print("Training neural net ...")
            X_train, Y_train = self.load_training_dataset("normal")

            model = Sequential()
            #1st Convolution Layer
            model.add(Convolution1D(64, 9, activation='relu',
                                    kernel_initializer='glorot_uniform',
                                    input_shape=(36, 1)))
            model.add(MaxPooling1D(pool_size=2))

            #2nd Convolution Layer
            model.add(Convolution1D(64, 3, activation='relu'))
            model.add(Dropout(0.4))

            #Fully connected layer
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))

            # Output
            model.add(Dense(2, activation='softmax'))

            # Optimizer
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            #Compile the model
            model.compile(loss='categorical_crossentropy',
                            optimizer=sgd,
                            metrics=['accuracy'])

            # Train the network
            Y_train = Y_train.astype(np.int32)
            model.fit(np.expand_dims(X_train, axis=2), Y_train, nb_epoch=10, verbose=1)
            self.neural_net = model
            print("Finished training the network")
            sp.gui.statBar.showMessage("Finished training the network.")
            model.save(nnFileNameAndPath)
