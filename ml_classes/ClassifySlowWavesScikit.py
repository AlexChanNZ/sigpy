"""
    Author: Shameer Sathar
"""

import numpy as np
from sklearn import svm
#from SVM import SVM
class ClassifySlowWavesScikit:

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
        train_array = np.asarray(training_set)
        event_array = np.asarray(events)
        test_array = np.asarray(test_data)
        clf = svm.SVC()
        clf.fit(np.transpose(train_array), np.transpose(event_array))
        prediction = clf.predict(np.transpose(test_array))
        return prediction

'''
        model = SVM(max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001)
        model.fit(np.transpose(train_array), np.transpose(event_array))
        return model.predict(np.transpose(test_array))
'''



