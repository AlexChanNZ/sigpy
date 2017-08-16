""" Standard imports """

"""
    Author: Shameer Sathar
    Description: Provide Gui Interface.
"""
import sys
import os
import numpy as np
import platform

import sys
sys.path.append('..')

from multiprocessing import Process

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE
from pyqtgraph.dockarea import *


import pickle
import matplotlib as mpl  

mpl.use('TkAgg') # compatibility for mac
import matplotlib.pyplot as plt

# Locally-developed modules
import config_global as sp

from file_io.gems_sigpy import *

from ml_classes.SlowWaveCNN import SlowWaveCNN

from gui_plotting.Gui_Window import GuiWindow
from gui_plotting.Gui_LinePlots import GuiLinePlots
from gui_plotting.Animate_Mapped import AnimateMapped
from gui_plotting.Animate_Live import AnimateLive


class GuiMain(QtGui.QMainWindow):

    def __init__(self, parent=None):

        """
        Initialise the properties of the GUI. This part of the code sets the docks, sizes
        :return: NULL
        """
        super(GuiMain, self).__init__(parent)
        self.ui = GuiWindow()
        self.ui.setupUi(self)


        # Initialise data
        # Add menu, status bar, controls and plots
        self.add_menu_controls_and_plots()

        # Set filemenu
        self.setup_file_menu_triggers()

        self.setCentralWidget(self.area)

        self.setWindowTitle('SigPy')
        self.showMaximized()        



    def add_menu_controls_and_plots(self) :

        # Add menu bar
        self.mbar = self.setMenuBar(self.ui.ui_menubar.ui_menubar)
       
        # Add status bar
        self.statBar = self.statusBar()

        # Add dock
        self.area = DockArea()
        self.d_controls = Dock("Controls", size=(50, 200))
        self.area.addDock(self.d_controls, 'left')

        # Add controls
        self.add_controls()
        self.d_controls.addWidget(self.ctrlsLayout, row=1, colspan=1)

        # Add main plots
        self.reset_add_plots()
        


    def reset_add_plots(self) :
        self.LinePlots = []
        del self.LinePlots
        self.LinePlots = GuiLinePlots()
        if hasattr(self, 'd_plots'):
            self.d_plots.close()
        self.d_plots = Dock("Plots", size=(500, 200))

        self.d_plots.addWidget(self.LinePlots.plotsScroll, row=0, col=0)
        self.d_plots.addWidget(self.LinePlots.plotsZoomed, row=0, col=1)
        self.area.addDock(self.d_plots, 'right')  



    # ==== CONTROLS AND ACTIONS ====

    def add_controls(self) :

        self.ctrlsLayout = pg.LayoutWidget()
        self.ctrlsRow = 0

        self.dataTypeLabel = QtGui.QLabel("")
        self.dataTypeLabel.setAlignment(QtCore.Qt.AlignBottom)   
        self.set_dataType_text()     

        self.btnFindSWEvents = QtGui.QPushButton('Detect Slow-Wave Events')
        self.amplitudeMapping = QtGui.QPushButton('Amplitude and Event Mapping')
        self.btnViewLiveData = QtGui.QPushButton('Live Mapping')

        self.btnFindSWEvents.clicked.connect(lambda: self.detect_slow_wave_events())
        self.amplitudeMapping.clicked.connect(lambda: self.plot_amplitude_map())                
        self.btnViewLiveData.clicked.connect(lambda: self.view_live_data())        

        self.ctrlsLayout.addWidget(self.dataTypeLabel, row=self.add_one(), col=0)
        self.ctrlsLayout.addWidget(self.btnFindSWEvents, row=self.add_one(), col=0)
        self.ctrlsLayout.addWidget(self.amplitudeMapping, row=self.add_one(), col=0)
        self.ctrlsLayout.addWidget(self.btnViewLiveData, row=self.add_one(), col=0)



    def detect_slow_wave_events(self):
        print("In detect slow wave events")
        self.statBar.showMessage("Training and classifying. . .")

        # Setup data and params
        self.dataForMarking = np.reshape(sp.dat['SigPy']['dataForMarking'], -1)
        self.cnnType = sp.dat['SigPy']['dataIsNormal']

        windowSize = 36
        overlap = 0.5

        indexJump = int(overlap * windowSize)

        # Create prediction windows
        nPredictionWindows = range(1, len(self.dataForMarking)-1, indexJump)

        print("Creating prediction windows")
        predictionWindows_list = []

        for j in nPredictionWindows:
            if (len(self.dataForMarking[j:j+windowSize]) == windowSize):
                predictionWindows_list.append(self.dataForMarking[j:j+windowSize])
        predictionWindows = np.array(predictionWindows_list)

        # Create neural net and classify
        print("Creating neural net")
        swCNN = SlowWaveCNN()
        swPredictions, swLocs = swCNN.classify_data(predictionWindows, self.cnnType)
        nSwLocs = len(swLocs)       



        # Mark classification on plots if slow waves were found  
        if len(swLocs) > 0 :
            self.LinePlots.mark_slow_wave_events(self.dataForMarking, swPredictions, swLocs, windowSize, indexJump)

        # Output (logging and for user)
        print("Testdata length ", self.LinePlots.nSamples)   
        print("Total predictions: ", len(swPredictions))
        print("Number SW raw predictions: ", nSwLocs)
        print("Ultimate number of sws marked: ", self.LinePlots.nSWsMarked)


        # Output number of events marked (note this number may differ from the CNN n of predictions)
        statBarMessage = str(self.LinePlots.nSWsMarked) + " slow wave events marked "
        self.statBar.showMessage(statBarMessage)



    def plot_amplitude_map(self):

        self.animatedMap = AnimateMapped()

    def view_live_data(self):
        self.animateLive = AnimateLive()    



    # ==== MENU BAR ACTIONS ====

    def setup_file_menu_triggers(self):
        
        self.ui.ui_menubar.loadNormalAction.triggered.connect(
            lambda: self.load_file_selector__gui_set_data(isNormal=True))
        self.ui.ui_menubar.loadPacingAction.triggered.connect(
            lambda: self.load_file_selector__gui_set_data(isNormal=False))
        self.ui.ui_menubar.saveAsAction.triggered.connect(
            lambda: self.save_as_file_selector())
        self.ui.ui_menubar.quitAction.triggered.connect(lambda: self.exit_app())



    def load_file_selector__gui_set_data(self, isNormal=True):

        sp.datFileName = QtGui.QFileDialog.getOpenFileName(None, "Select File", "", "*.mat")

        if not (sys.platform == "linux2") :
            sp.datFileName = sp.datFileName[0]

        self.statBar.showMessage("Loading ...")

        load_GEMS_mat_into_SigPy(sp.datFileName, isNormal)
        self.reset_add_plots()
        # self.LinePlots.refresh_plots()
        self.set_dataType_text()
        self.statBar.showMessage("Finished loading.")



    def save_as_file_selector(self):

        sp.datFileName = QtGui.QFileDialog.getSaveFileName(None, "Save As File", sp.datFileName, "*.mat")

        if not (sys.platform == "linux2") :
            sp.datFileName = sp.datFileName[0]  

        save_file_selector(self)
      


    def save_file_selector(self):

        self.statBar.showMessage("Saving . . . ")
        save_GEMS_SigPy_file(sp.datFileName)
        self.statBar.showMessage("Saved file!")  



    def exit_app(self) :

        self.close()
        sys.exit()


    # == Util function

    def add_one(self):
        self.ctrlsRow+=1
        return self.ctrlsRow

    def set_dataType_text(self):
        isNormal = sp.dat['SigPy']['dataIsNormal']

        if isNormal:
            self.dataTypeLabel.setText("Normal Data Selected")
        else :
            self.dataTypeLabel.setText("Pacing Data Selected")        


