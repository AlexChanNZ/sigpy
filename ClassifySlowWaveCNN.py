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

class ClassifySlowWaveCNN:

    def __init__(self, training_set, events):
        self.train_array = np.asarray(training_set).reshape((-1,1,6,6))
        self.label_array = np.asarray(events)

    def classify_data(self, test_data, type_data_set):
        """
        The data is classified based on the training data.
        :param plot: Input values to be processed for generating features
        :return: predictions for the entire data set
        """
        self.train_neural_net(type_data_set);
        test_array = np.asarray(test_data)
        test_array = test_array.reshape((-1, 1, 6, 6))
        prediction = self.neural_net.predict(test_array)
        return prediction
        
    def load_training_dataset_pacing(self):
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
    
    def load_training_dataset_normal(self):
        mat_data_path = '/media/ssat335/BRIDGE/ml/theano/exp3_A-H-256channels_recording_Elec_001.mat'
        data_dic = sio.loadmat(mat_data_path)
        data_samples = np.array(data_dic['samples'])
        data_labels = np.array(data_dic['label'])
        X_train = data_samples[0:4000,:]
        y_train = data_labels[0:4000,:]
        X_train = X_train.reshape((-1, 1, 6, 6))
        return X_train, y_train
        
    def train_neural_net(self, type_data_set):
        if self.label_array.size == 0:
            if (type_data_set == 0):
                f = open('config/nn_normal.cnn', 'rb')
            elif (type_data_set == 1):
                f = open('config/nn_pacing.cnn', 'rb')
            else:
                print "No type selected"
            self.neural_net = pickle.load(f)
            f.close()
            return
            
        if (type_data_set == 0):
            X_train, y_train = self.load_training_dataset_normal()
        elif (type_data_set == 1):
            X_train, y_train = self.load_training_dataset_pacing()
        else:
            print "No type selected"
        print X_train.shape
        print self.train_array.shape
        X_train = np.append(X_train, self.train_array, axis=0)
        print X_train.shape
        y_train = np.append(y_train, self.label_array.transpose())
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
        print X_train.shape
        print y_train.shape
        y_train = y_train.astype(np.int32)
        self.neural_net = nn.fit(X_train, y_train.flatten())
