# -*- coding: utf-8 -*-
"""
Author : Shameer Sathar
"""
import sys, os # to browse directories

# import cPickle as pickle
import six.moves.cPickle as pickle

import numpy as np
import scipy.io as sio
import theano


# from theano.tensor.signal import downsample 
# need to swap above import with these below two for more recent version of theano
from theano.tensor.signal import pool
from theano.tensor.signal.pool import pool_2d

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

# User defined imports
from gui_plotting.mpl_plots import *

import config_global as sp

class SlowWaveCNN:

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
        print(test_array.shape)
        sp.gui.statBar.showMessage("Classifying...")
        predictions = self.neural_net.predict(test_array)
        sp.gui.statBar.showMessage("Finished classifying.")
        eventsFound = np.where(predictions == 1)[0]

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

                        X_train = np.append(X_train, trainingSamples[0:2000,:])
                        Y_train = np.append(Y_train, trainingLabels[0:2000,:])

                    else:

                        X_train = trainingSamples[0:2000,:]
                        Y_train = trainingLabels[0:2000,:]        
                except Exception as e:
                    print("Exception: ", e)      
                    print("Caused by file: ", trainingFile)      

        
        X_train = X_train.reshape((-1, 1, 6, 6))

        return X_train, Y_train
        


    def train_neural_net(self, type_data_set):

        if (type_data_set is 1):
            
            print("Using Normal CNN")
            nnFileName = "nn_normal.cnn"

        elif (type_data_set is 0):
            print("Using Pacing CNN")

            nnFileName = "nn_pacing.cnn"

        else:
            print("No type selected")  
            return

        nnFileNameAndPath = sp.nnPath + nnFileName            
        print("nnFileNameAndPath: ", nnFileNameAndPath)

        #Load CNN instead of training
        try:
            f = open(nnFileNameAndPath, 'rb')
            self.neural_net = pickle.load(f)
            f.close()
            return

        except Exception as e:

            print("Training neural net ...")
                
            if (type_data_set is 1):
                X_train, Y_train = self.load_training_dataset("normal")

            elif (type_data_set is 0):
                X_train, Y_train = self.load_training_dataset("pacing")
                nnFileName = "nn_pacing.cnn"

            else:
                print("No training type selected")

            # I believe the below lines aren't needed anymore
            # print(X_train.shape)
            # print(self.train_array.shape)
            # X_train = np.append(X_train, self.train_array, axis=0)
            # print(X_train.shape)
            # Y_train = np.append(Y_train, self.label_array.transpose())

            nn = NeuralNet(
                    layers=[('input', layers.InputLayer),
                                ('conv2d1', layers.Conv2DLayer),
                                ('maxpool1', layers.MaxPool2DLayer),
                                ('conv2d2', layers.Conv2DLayer),
                                ('maxpool2', layers.MaxPool2DLayer),
                                ('dropout1', layers.DropoutLayer),
                                ('dense', layers.DenseLayer),
                                ('dropout2', layers.DropoutLayer),
                                ('output', layers.DenseLayer),
                                ],
                        # input layer
                        input_shape=(None, 1, 6, 6),
                        # layer conv2d1
                        conv2d1_num_filters=32,
                        conv2d1_filter_size=(3, 3),
                        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
                        conv2d1_W=lasagne.init.GlorotUniform(),  
                        # layer maxpool1
                        maxpool1_pool_size=(2, 2),    
                    
                        # layer conv2d2
                        conv2d2_num_filters=32,
                        conv2d2_filter_size=(2, 2),
                        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
                        # layer maxpool2
                        maxpool2_pool_size=(1, 1),
                        # dropout1
                        dropout1_p=0.5,    
                        # dense
                        dense_num_units=256,
                        dense_nonlinearity=lasagne.nonlinearities.rectify,    
                        # dropout2
                        dropout2_p=0.5,    
                        # output
                        output_nonlinearity=lasagne.nonlinearities.softmax,
                        output_num_units=2,
                        # optimization method params
                        update=nesterov_momentum,
                        update_learning_rate=0.01,
                        update_momentum=0.9,
                        max_epochs=100,
                        verbose=1,
                        )

            # Train the network
            Y_train = Y_train.astype(np.int32)
            self.neural_net = nn.fit(X_train, Y_train.flatten())
            print("Finished training the network")
            sp.gui.statBar.showMessage("Finished training the network.")

            pickle.dump(self.neural_net, open(nnFileNameAndPath, "wb"))



