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

from util_classes.np_utils import *

from matplotlib import pyplot as plt

# Internal dependencies
import config_global as sp
from file_io.gems_sigpy import load_GEMS_mat_into_SigPy
from signal_processing.preprocessing import preprocess
from ml_classes.SlowWaveCNN import SlowWaveCNN
from gui_plotting.mpl_plots  import *
from gui_plotting.GuiWindowDocks import GuiWindowDocks



"""
Gastric data 
"""

defaultDataFile = 'sample_GEMS_file.mat'
defaultDataFile = 'sample_GEMS_file.mat'

# dataFile = 'pig72_exp9_aydin_recording_stim_Elec_FEVT_000'

dataFileAndRoot = sp.dataRoot + defaultDataFile
dataFileAndRoot = "pig83_exp7_A_H_serosal_pacing_Elec_ATmarks_000_SigPy_input_normalised.mat"

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
if not sp.dat['SigPy'].get('dataNorm') :
    sp.dat['SigPy']['dataNorm'] = preprocess(sp.dat['SigPy']['dataFilt'])


sp.set_data_file_name((dataFileAndRoot.rsplit('/', 1)[1]))
sp.set_test_file_name(str(sp.loaded_data_file) + str('_test.arff'))
sp.set_training_file_name(str(sp.loaded_data_file) + str('_training.arff'))
sp.set_trained_file(str(sp.loaded_data_file) + str('_trained.dat'))


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    # OR
    iSWcnn = SlowWaveCNN()

    testSWdata = sp.dat['SigPy']['dataNorm'].reshape((-1, 1, 6, 6))

    # load and plot training data
    trainingSWdata, trainingSWlabels = iSWcnn.load_training_dataset("normal")

    trainSWimages = plot_swImage_grid(trainingSWdata, trainingSWlabels, linePlot=False, iTitle="Train data SWs")
    trainNonSWimages = plot_swImage_grid(trainingSWdata, invert_ones_zeros(trainingSWlabels), linePlot=False, iTitle="Train data non-SWs")

    # # classify and plot test data
    retrainedSWlabels = iSWcnn.classify_data(trainingSWdata, 1)

    retrainedSWimages = plot_swImage_grid(trainingSWdata, retrainedSWlabels, linePlot=False, iTitle="Retrained data SWs")   
    retrainedNonSWimages = plot_swImage_grid(trainingSWdata, invert_ones_zeros(retrainedSWlabels), linePlot=False, iTitle="Retrained data nonSWs")    


    # # classify and plot test data
    # testSWlabels = iSWcnn.classify_data(sp.dat['SigPy']['dataNorm'], 1)

    # testSWimages = plot_swImage_grid(testSWdata, testSWlabels, linePlot=True, iTitle="Test data SWs")   
    # testNonSWimages = plot_swImage_grid(testSWdata, invert_ones_zeros(testSWlabels), linePlot=True, iTitle="Test data nonSWs")    

    # dataImagesToPlot = np.vstack([trainSWimages, trainNonSWimages])

    dataImagesToPlot = np.vstack([trainSWimages, retrainedSWimages, trainNonSWimages, retrainedNonSWimages])
    #testSWimages, testNonSWimages])

    print("dataImagesToPlot.shape: ", dataImagesToPlot.shape)

    plot_2d_simple(dataImagesToPlot, "", figSize=(100,60))

    plt.show()
    """
    Create data here and add to the curve
    """
    # Set plot data
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


