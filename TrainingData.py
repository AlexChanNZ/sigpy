"""
    Author: Shameer Sathar
    Description: Deal with training data.
"""

import numpy as np


class TrainingData:

    def __init__(self):
        self.data = []
        self.regions = []
        self.plotLength = 0
        self.prev_length = 0
        """
        Initialise the array with long length.
        """
        self.plotDat = np.zeros([20000, 36], dtype=float)
        self.plotEvent = np.zeros([20000, 1], dtype=int)

    def setData(self, data):
        self.data = np.array(data)

    def addRegion(self, region):
        self.regions.append(region)

    def add_events(self):
        i = 0
        while i < (len(self.regions)):
            print i
            print self.plotLength
            elec = self.regions[i][0] -1
            init_pos = self.regions[i][1]
            self.plotDat[self.plotLength + i,:] = self.data[elec][init_pos:init_pos+36]
            self.plotEvent[self.plotLength + i,:] = 1
            i = i + 1
        self.plotLength = self.plotLength + len(self.regions)
        self.clearRegion()

    def add_non_events(self):
        i = 0
        while i < (len(self.regions)):
            print i
            print self.plotLength
            elec = self.regions[i][0] - 1
            init_pos = self.regions[i][1]
            self.plotDat[self.plotLength + i,:] = self.data[elec][init_pos:init_pos+36]
            self.plotEvent[self.plotLength + i,:] = 0
            i = i + 1;
        self.plotLength = self.plotLength + len(self.regions)
        self.clearRegion()

    def undo(self):
        self.plotLength -= self.prev_length
        self.plotDat[0][self.plotLength:self.plotLength+self.prev_length] = 0
        self.plotEvent[0][self.plotLength:self.plotLength+self.prev_length] = 0
        self.prev_length = 0

    def clearRegion(self):
        del self.regions[:]


