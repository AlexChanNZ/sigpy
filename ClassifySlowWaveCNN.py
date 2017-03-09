# -*- coding: utf-8 -*-
"""
Author : Shameer Sathar
"""
import cPickle as pickle

import numpy as np
import scipy.io as sio

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class ClassifySlowWaveCNN:

    def __init__(self):
        self.data = []
        self.len = 0
        self.features = []


    def classify_data(self, training_set, events, test_data):
        """
        The data is classified based on the training data.
        :param plot: Input values to be processed for generating features
        :return: predictions for the entire data set
        """
        self.train_array = np.asarray(training_set).reshape((-1,1,6,6))
        self.label_array = np.asarray(events)
        test_array = np.asarray(test_data)
        test_array = test_array.reshape((-1, 1, 6, 6))
        prediction = self.neural_net.predict(test_array)
        return prediction
        
    def load_training_dataset(self):
        mat_data_path = '/media/ssat335/BRIDGE/ml/theano/pig68exp5pacing.mat'
        data_dic = sio.loadmat(mat_data_path)
        data_samples = np.array(data_dic['samples'])
        data_labels = np.array(data_dic['label'])
        X_train = data_samples[0:2000,:]
        y_train = data_labels[0:2000,:]
        mat_data_path = '/media/ssat335/BRIDGE/ml/theano/pig68exp5pacing.mat'
        data_dic = sio.loadmat(mat_data_path)
        data_samples = np.array(data_dic['samples'])
        data_labels = np.array(data_dic['label'])
        X_train = np.append(X_train, data_samples[0:2000,:])
        y_train = np.append(y_train, data_labels[0:2000,:])
        mat_data_path = '/media/ssat335/BRIDGE/ml/theano/pig71exp7pacing.mat'
        data_dic = sio.loadmat(mat_data_path)
        data_samples = np.array(data_dic['samples'])
        data_labels = np.array(data_dic['label'])
        X_train = np.append(X_train, data_samples[0:2000,:])
        y_train = np.append(y_train, data_labels[0:2000,:])    
  
        X_train = X_train.reshape((-1, 1, 6, 6))
        return X_train, y_train
        
    def train_neural_net(self):
        if self.label_array.size() == 0:
            f = open('config/nn_pacing.cnn', 'rb')
            self.neural_net = pickle.load(f)
            f.close()
            return
        
        X_train, y_train = load_training_dataset()
        nn = NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('conv2d1', layers.Conv2DLayer),
                    ('maxpool1', layers.MaxPool2DLayer),
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
        self.neural_net = nn.fit(X_train, y_train.flatten())
