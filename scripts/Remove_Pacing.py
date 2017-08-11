#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Author: Shameer Sathar
    Description: GUI for training and plotting the activation times.
"""

# External dependencies
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import scipy.io as sio
import sys
import matplotlib
matplotlib.use("TKAgg")


from matplotlib import pyplot as plt

# Internal dependencies
import sys
sys.path.append('..')
import config_global as sp
from file_io.gems_sigpy import load_GEMS_mat_into_SigPy
from signal_processing.preprocessing import preprocess, clean_pacing
from ml_classes.SlowWaveCNN import SlowWaveCNN
from gui_plotting.mpl_plots  import *
from gui_plotting.PyQtLinePlots import PyQtLinePlots
from util_classes.np_utils import *


"""
Gastric data 
"""

# dataFile = 'pig72_exp9_aydin_recording_stim_Elec_FEVT_000'
defaultDataFile = "pig83_exp7_A_H_serosal_pacing_Elec_ATmarks_000_SigPy_input.mat"

defaultDataFile = "pig83_exp7_A_H_serosal_pacing_000.mat"

dataFileAndRoot = sp.dataStore + defaultDataFile

"""
Gastric data normal
"""
#data = '/media/hpc/codes/MatLab/normalise_data/pig32_exp2_normal.mat'

"""
Intestine data
"""
#data = '/media/hpc/GEMS/slow-wave-data/27082016_data_for_testing_ml/rabbit_intestine/RAB004_exp11_80cm_distal_LOT_Elec_Maps_data.mat'

# mat_contents = sio.loadmat(dataFileAndRoot)
# print(mat_contents)



load_GEMS_mat_into_SigPy(dataFileAndRoot)



# sp.set_data_file_name((dataFileAndRoot.rsplit('/', 1)[1]))
# sp.set_test_file_name(str(sp.loaded_data_file) + str('_test.arff'))
# sp.set_training_file_name(str(sp.loaded_data_file) + str('_training.arff'))
# sp.set_trained_file(str(sp.loaded_data_file) + str('_trained.dat'))

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    nChansToPlot = 20
    nSamplesToPlot = 2000

    filteredData = sp.dat['SigPy']['dataFilt']
    normalisedData = preprocess(filteredData)

    pacingMarkers, pacingCleanedDat = clean_pacing(filteredData)
    normalisedPacingCleaned = preprocess(pacingCleanedDat)

    # print("pacingMarkers: ", pacingMarkers)
    dataSplit = int(nChansToPlot / 2)

    sp.dat['SigPy']['dataToPlot'] = np.zeros(shape=(nChansToPlot, nSamplesToPlot)) 
    sp.dat['SigPy']['dataToPlot'] = normalisedPacingCleaned

    sp.dat['SigPy']['dataToPlot'] = normalisedData

    # sp.dat['SigPy']['dataToPlot'][0:dataSplit, 0:nSamplesToPlot] = normalisedData
    # sp.dat['SigPy']['dataToPlot'][dataSplit:dataSplit*2, 0:nSamplesToPlot] = pacingRemovedData[0:dataSplit, 0:nSamplesToPlot]

    sp.dat['SigPy']['markersPacing'] = pacingMarkers

    # Run GUI
    pyqtlineplots = PyQtLinePlots()

    """
    Create data here and add to the curve
    """
    # Set plot data
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


