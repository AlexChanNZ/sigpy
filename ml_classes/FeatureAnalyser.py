"""
    Author: Shameer Sathar
    Description: A module of generating features for the signals.
"""

#from scipy.signal import savgol_filter
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

class FeatureAnalyser:

    def __init__(self):
        self.data = []
        self.len = 0
        self.features = []
        self.data_shape = (0, 0)

    def process_data(self, plot, shape):
        """
        The features for the data is created. The plot should in in 1 x #samples
        size.
        :param plot: Input values to be processed for generating features
        :return: features in #features X #sample size
        """
        self.data_shape = shape
        self.data = np.asarray(plot)
        temp, self.len = self.data.shape
        '''Obtain the features'''
        self.features = np.zeros([33, self.len], dtype=float)
        self.features[0, :] = (self.data)
        self.features[1, :] = (self.apply_diff(self.data))
        for i in range(1, 4, 1):
            self.features[i + 1, :] = (gaussian_filter1d(self.data, 6 * pow(2, (i-1))))
        for i in range(1, 4, 1):
            self.features[i + 4, :] = (self.apply_diff(self.features[i, :]))
        #for i in range(1, 4, 1):
        #    self.features[i + 5, :] = self.shift_signals(self.data, -90*i)
        #for i in range(1, 4, 1):
        #    self.features[i + 10, :] = self.shift_signals(self.data, 90*i)
        #for i in range(1, 19, 1):
        #    self.features[i + 13, :] = gaussian_filter1d(self.features[int((i-1)/3) + 7, :], 6 * pow(2, ((i - 1) % 3)))
        #self.features[32, :] = np.square(self.features[1, :])
        self.features[8, :] = (np.square(self.features[1, :]))
        self.features[9, :] = (self.NDT(self.data))
        self.features[10, :] = (self.ASD(self.data))
        self.features[11, :] = (self.NEO(self.data))
        #self.features[12, :] = self.data * self.data
        return self.features


    def apply_diff(self, data):
        return np.append(np.diff(data, n=1), 0)

    def shift_signals(self, data, shift_period):
        if shift_period > 0:
            pre_temp = np.ones([1, shift_period], dtype=float) * data[0][1]
            return np.append(pre_temp, data[0][0:self.len-shift_period])
        elif shift_period < 0:
            shift_period *= -1
            pre_temp = np.ones([1, shift_period], dtype=float) * \
                                                            data[0][self.len-1]
            return np.append(data[0][shift_period:], pre_temp)
        else:
            return data

    def NDT(self, data):
        return self.apply_diff(data)

    def ASD(self, data):
        return data * self.NDT(data)

    def NEO(self, data):
        return self.NDT(data) * self.NDT(data) - data * self.NDT(self.NDT(data))
        
    def scale_normalise(self, data):
        (rows, cols) = self.data_shape
        temp_data = np.reshape(data, self.data_shape)
        for i in range(0, rows):
            # Feature standardisation
            temp_data[i,:] = np.divide(np.subtract(temp_data[i,:], \
                            np.mean(temp_data[i,:])), np.std(temp_data[i,:]))
            # Feature scaling between 1 and 0
            temp_data[i,:] = np.divide(np.subtract(temp_data[i,:], \
                            np.min(temp_data[i,:])), \
                        np.subtract(np.max(temp_data[i,:]), np.min(temp_data[i,:])))
        return np.reshape(temp_data, -1)
            
        
        






