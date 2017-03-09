"""
    Author: Shameer Sathar
    Description: Provide interface for WEKA ARFF files.
"""

import arff
import numpy as np
import os

class WekaInterface:
    def __init__(self, data, file_name='test_data.arff'):
        self.data = np.transpose(data)
        self.file_name = file_name

    def arff_write(self, event=[]):
        ''' Write ARFF object file as required by analyser GUI.
        :param event: a list of event denotions. Empty implies it is a testing data.
        :return: null.
        '''
        attributes = []
        (rows, cols) = np.shape(self.data)
        if np.array(event).size == 0 or event is None:
            hcat = np.empty((rows, 1))
            hcat.fill(np.NAN)
            ndarray = np.asarray(np.hstack((self.data, hcat)))
            '''
            No need to write the test data every time. Check if exists and if true return null
            '''
            print os.getcwd() + self.file_name
            if os.path.exists(os.getcwd() + str('/') + self.file_name):
                print '{} already exists. Not keen to rewrite the same thing again'.format(self.file_name)
                #return
        else:
            hcat_event = np.empty((len(event), 1))
            hcat_event[:, 0] = np.transpose(np.array(event))
            ndarray = np.asarray(np.hstack((self.data, hcat_event)))
        for i in range(0, cols, 1):
            attributes.append('attribute' + str(i))#), 'NUMERIC'))
        attributes.append('event')#, ['1.0', '0.0']))
        vals = []
        for i in range(0, rows, 1):
            vals.append(ndarray[i, :])
        obj = {
            'description': 'Training set',
            'relation': 'Training',
            'attributes': attributes,
            'data': vals
        }

        ''' Write to file here'''
        arff.dump(self.file_name, vals, relation="Training", names=attributes)
