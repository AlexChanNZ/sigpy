# -*- coding: utf-8 -*-
"""
Author : Shameer Sathar
"""
import sys, os # to browse directories

import numpy as np
import scipy.io as sio

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')

from gui_plotting.mpl_plots import *

import config_global as sp

class SlowWaveCNNKeras:

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
        test_array = test_array.reshape((-1, 1, 6, 6))
        #print(test_array.shape)
        sp.gui.statBar.showMessage("Classifying...")
        predictions = self.neural_net.predict(test_array)
        print(np.argmax(predictions, axis=1).shape)
        sp.gui.statBar.showMessage("Finished classifying.")
        eventsFound = np.where(np.argmax(predictions, axis=1) == 1)[0]

        return predictions, eventsFound



    def load_training_dataset(self, dataType):

        trainingDataPlotPath = sp.dataRoot + '/' + dataType
        nFiles = 0

        for trainingFile in os.listdir(trainingDataPlotPath):

            trainingFileAndPath = trainingDataPlotPath + "/" + trainingFile

            print("trainingFileAndPath: ",trainingFileAndPath)

            if ".mat" in trainingFileAndPath:
                nFiles+=1
                data_dic = sio.loadmat(trainingFileAndPath)

                try:
                    trainingSamples = np.array(data_dic['samples'])
                    trainingLabels = np.array(data_dic['label'])
                    if nFiles > 1:
                        X_train = np.append(X_train, trainingSamples[:,:])
                        Y_train = np.append(Y_train, trainingLabels[:,:])
                    else:
                        X_train = trainingSamples[:,:]
                        Y_train = trainingLabels[:,:]
                except Exception as e:
                    print("Exception: ", e)
                    print("Caused by file: ", trainingFile)


        X_train = X_train.reshape((-1, 1, 6, 6))
        Y_train = np_utils.to_categorical(Y_train, 2)
        return X_train, Y_train



    def train_neural_net(self, type_data_set):

        nnFileName = "nn_2D.h5"
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
            model.add(Convolution2D(64, 3, 3, activation='relu',
                                    kernel_initializer='glorot_uniform',
                                    input_shape=(1, 6, 6)))
            model.add(MaxPooling2D(pool_size=(2,2)))

            #2nd Convolution Layer
            model.add(Convolution2D(16, 2, 2, activation='relu'))
            model.add(MaxPooling2D(pool_size=(1,1)))
            model.add(Dropout(0.3))

            #Fully connected layer
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.3))

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
            print X_train.shape
            print Y_train.shape
            model.fit(X_train, Y_train, nb_epoch=10, verbose=1)
            self.neural_net = model
            print("Finished training the network")
            sp.gui.statBar.showMessage("Finished training the network.")
            model.save(nnFileNameAndPath)
