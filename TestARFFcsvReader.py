"""
    Author: Shameer Sathar
"""

from ARFFcsvReader import ARFFcsvReader
import numpy as np

"""
Script to test the ARFF predictions output file.
"""

test = ARFFcsvReader('data/results_data.arff')
prediction = np.asarray(test.get_prediction())
"""
Positive change from 0 -> 1 is identified by taking a diff and checking for +1
"""
diff = np.diff(prediction)
linear_at = np.array(np.where(diff == 1))

"""
Predictions are 1d array with size
    number of channels * sample points
Here, we change the 1d data  to 2d electrode time maps.
"""
pos = []
for val in linear_at.transpose():
    pos.append([int(val/9001), int(val % 9001)])
    print int(val/9001), int(val % 9001)

