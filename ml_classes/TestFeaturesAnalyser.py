"""
    Author: Shameer Sathar
"""

import unittest
from FeatureAnalyser import FeatureAnalyser
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

class TestFeaturesAnalyser(unittest.TestCase):
    def test(self):
        test_data = sio.loadmat('training_data_test.mat')
        #print test_data['vect']
        data = np.asarray(test_data['vect'])
        #print data.transpose().shape
        analyser = FeatureAnalyser()
        data = analyser.process_data(data, data.shape)
        (rows, cols) = data.shape
        for i in range(0, 12):
            plt.figure(i)
            plt.plot(data[i][:])
            plt.figure(i)
        #plt.subplots_adjust(wspace=0, hspace=0)
        #plt.plot(data[31][:])
        plt.show()
        #plt.close()
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()

