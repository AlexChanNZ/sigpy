#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Author: Shameer Sathar
    Description: GUI for training and plotting the activation times.
"""

import os 

# External dependencies
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import scipy.io as sio
import sys
import matplotlib

matplotlib.use("TKAgg")


from matplotlib import pyplot as plt


# Internal dependencies
import config_global as sp 
from file_io.gems_sigpy import load_GEMS_mat_into_SigPy
from signal_processing.preprocessing import preprocess
from ml_classes.SlowWaveCNN import SlowWaveCNN
from gui_plotting.mpl_plots  import *
from gui_plotting.GuiWindowDocks import GuiWindowDocks

import numpy as np
np.set_printoptions(linewidth=1000, precision=3, threshold=np.inf)

"""
Gastric data 
"""

# dataFile = 'pig72_exp9_aydin_recording_stim_Elec_FEVT_000'
defaultDataFile = "pig83_exp7_A_H_serosal_pacing_Elec_ATmarks_000_SigPy_input.mat"
defaultDataFile = "pig83_exp7_A_H_serosal_pacing_000.mat"
defaultDataFile = "pig83_exp7_A_H_serosal_pacing_Elec_000_200plus400s.mat"
dataFileIsNormal = False


dataFileAndRoot = sp.userDataStore + defaultDataFile

if not os.path.isfile(dataFileAndRoot) :
    defaultDataFile = 'sample_GEMS_normal.mat'
    dataFileAndRoot = sp.dataRoot + defaultDataFile
    dataFileIsNormal = True



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

load_GEMS_mat_into_SigPy(dataFileAndRoot, dataFileIsNormal)

sp.set_data_file_name((dataFileAndRoot.rsplit('/', 1)[1]))


#vals = np.array(mat_contents['mark_cardiac'])

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__' :

    # Run GUI
    gui = GuiWindowDocks(dataFileIsNormal)

    """
    Create data here and add to the curve
    """
    # Set plot data
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


