""" Standard imports """

"""
    Author: Shameer Sathar
    Description: Provide Gui Interface.
"""
import sys
import os
import numpy as np
import platform
import time
import threading
from threading import Thread




from multiprocessing import Process
# Main GUI support

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE
from pyqtgraph.dockarea import *

# import cPickle as pickle # Python3 
import pickle
import matplotlib as mpl  

mpl.use('TkAgg') # TM EDIT (compatibility for mac)
import matplotlib.pyplot as plt

import scipy.io
import theano

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

# Locally-developed modules
from gui_plotting.TrainingDataPlot import TrainingDataPlot
from file_io.ARFFcsvReader import ARFFcsvReader
from ml_classes.WekaInterface import WekaInterface
from ml_classes.FeatureAnalyser import FeatureAnalyser
from ml_classes.SlowWaveCNN import SlowWaveCNN
import config_global as sp
from file_io.gems_sigpy import *
from signal_processing.preprocessing import *
from signal_processing.mapping import *

from signal_processing.livedata import LiveData



class GuiWindowDocks:

    def __init__(self, isNormal):
        """
        Initialise the properties of the GUI. This part of the code sets the docks, sizes
        :return: NULL
        """
        self.curves_left = []
        self.curves_right = []
        self.curve_bottom = []
        self.elec = []
        self.data = []
        self.pacingEventsOn = False

        self.set_current_dataset()   

        self.rowNum = 0
        self.app = QtGui.QApplication([])
        self.win = QtGui.QMainWindow()
        area = DockArea()
        self.d_control = Dock("Dock Controls", size=(50, 200))
        self.d_plot = Dock("Dock Plots", size=(500, 200))
        self.d_train = Dock("Training Signal", size=(500, 50))
        area.addDock(self.d_control, 'left')
        area.addDock(self.d_plot, 'right')
        # area.addDock(self.d_train, 'bottom', self.d_plot)


        self.win.setCentralWidget(area)
        self.win.resize(1500, 800)
        self.win.setWindowTitle('SigPy')
        self.add_dock_widgets_controls()
        self.add_menu_bar()

        # Default data is 
        self.isNormal = isNormal
        self.set_dataType_text()



        self.add_dock_widgets_plots()
        self.set_crosshair()
        self.set_rect_region_ROI()



        self.trainingDataPlot = TrainingDataPlot()

    
        self.set_plot_data(self.data, self.nChans, self.nSamples)

        self.amplitudeMapping.clicked.connect(lambda: self.plot_amplitude_map())        

        self.btnFindSWEvents.clicked.connect(lambda: self.compute_slow_wave_events())
        # self.btnShowPacingEvents.clicked.connect(lambda: self.show_pacing_events())
        # self.btnShowPacingCleaned.clicked.connect(lambda: self.show_cleaned_pacing())

        # self.btnPacing.clicked.connect(lambda: self.change_physiology_to_pacing())

        # self.btnIsNormal.clicked.connect(lambda: self.change_physiology_to_normal())

        self.btnViewLiveData.clicked.connect(lambda: self.view_live_data())


        self.win.showMaximized()
        self.win.show()


    def set_current_dataset(self) :
        # Set initial plot data

        self.data = sp.dat['SigPy']['dataToPlot']
        self.nChans = self.data.shape[0]
        self.nSamples = self.data.shape[1]
        self.timeBetweenSamples = sp.dat['SigPy']['timeBetweenSamples']
        self.timeStart = sp.dat['SigPy']['timeStart'] 

        if "MarkersPacing" in sp.dat['SigPy'].keys() :
            self.MarkersPacing = sp.dat['SigPy']['MarkersPacing']
            print("MarkersPacing key found!")
            self.pacingEventsOn = False


        else :
            self.MarkersPacing = []


        if "SWMarkers" in sp.dat['SigPy'].keys() :
            self.markersSWs = sp.dat['SigPy']['MarkersSW']
            print("SWMarkers key found!")

        else :
            self.markersSWs = [] 

       
    def set_dataType_text(self):
        if self.isNormal:
            self.dataTypeLabel.setText("Normal Data Selected")
        else :
            self.dataTypeLabel.setText("Pacing Data Selected")
        # if self.isNormal == 1 :
        #     self.change_physiology_to_normal()

        # else :
        #     self.change_physiology_to_pacing()


    def add_one(self):
        self.rowNum+=1
        return self.rowNum



    def add_menu_bar(self):

        ## MENU BAR
        self.statBar = self.win.statusBar()

        self.mainMenu = self.win.menuBar()
        self.fileMenu = self.mainMenu.addMenu('&File')

        ## Load pacing file
        self.loadPacingAction = QtGui.QAction('&Load Pacing GEMS .mat', self.fileMenu)        
        self.loadPacingAction.setStatusTip('')
        self.loadPacingAction.triggered.connect(lambda: self.load_file_selector__gui_set_data(isNormal=False))

        self.fileMenu.addAction(self.loadPacingAction)


        ## Load normal file
        self.loadNormalAction = QtGui.QAction('&Load Normal GEMS .mat', self.fileMenu)        
        self.loadNormalAction.setStatusTip('')
        self.loadNormalAction.triggered.connect(lambda: self.load_file_selector__gui_set_data(isNormal=True))

        self.fileMenu.addAction(self.loadNormalAction)


        ## Save as gems file 
        self.saveAsAction = QtGui.QAction('&Save as GEMS .mat', self.fileMenu)        
        self.saveAsAction.setStatusTip('Save data with filename.')
        self.saveAsAction.triggered.connect(lambda: self.save_as_file_selector())

        self.fileMenu.addAction(self.saveAsAction)


        ## Save (update existing file)
        self.saveAction = QtGui.QAction('&Save', self.fileMenu)
        self.saveAction.setShortcut('Ctrl+S')
        self.saveAction.setStatusTip('Overwrite currently loaded file.')
        self.saveAction.triggered.connect(lambda: self.save_file_selector())

        self.fileMenu.addAction(self.saveAction)    

        ## Exit 
        self.quitAction = QtGui.QAction('Close', self.fileMenu)        
        self.quitAction.setStatusTip('Quit the program')
        self.quitAction.setShortcut('Ctrl+Q')
        self.quitAction.triggered.connect(lambda: self.exit_app())

        self.fileMenu.addAction(self.quitAction)



    def add_dock_widgets_controls(self):

        w1l = QtGui.QVBoxLayout()

        w1 = pg.LayoutWidget()

        self.dataType=QtGui.QButtonGroup() 


     

        self.dataTypeLabel = QtGui.QLabel("")
        self.dataTypeLabel.setAlignment(QtCore.Qt.AlignBottom)

        w1.addWidget(self.dataTypeLabel, row=self.add_one(), col=0)

        # self.btnPacing = QtGui.QRadioButton('Pacing')
        # self.btnIsNormal = QtGui.QRadioButton('Normal')

        # self.dataType.addButton(self.btnPacing, 0)
        # self.dataType.addButton(self.btnIsNormal, 1)
        # self.dataType.setExclusive(True)     
        # print("ISNormal ? ", self.isNormal)

        # if self.isNormal is 0 :
        #     self.change_physiology_to_pacing()        
        # else:
        #     self.change_physiology_to_normal()        


        # w1.addWidget(self.btnIsNormal,row=self.add_one(),col=0)
        # w1.addWidget(self.btnPacing,row=self.add_one(), col=0)

        # label = QtGui.QLabel('Usage info')
        # label.setAlignment(QtCore.Qt.AlignTop)

       
        self.btnFindSWEvents = QtGui.QPushButton('Compute Slow-Wave Events')
        self.amplitudeMapping = QtGui.QPushButton('Amplitude and Event Mapping')
        self.btnViewLiveData = QtGui.QPushButton('Live Mapping')
        # self.btnShowPacingEvents = QtGui.QPushButton('Show Pacing Events')
        # self.btnShowPacingCleaned = QtGui.QPushButton('Show Cleaned Pacing')

        # self.dataTypeLayout=QtGui.QHBoxLayout()  # layout for the central widget
        # self.dataTypeWidget=QtGui.QWidget(self)  # central widget
        # self.dataTypeWidget.setLayout(self.dataTypeLayout)

        # Control for toggling whether to capture live data
        liveDataLabel = QtGui.QLabel('Data capture:')
        liveDataLabel.setAlignment(QtCore.Qt.AlignBottom)

        # w1.addWidget(label, row=self.add_one(), col=0)
        # w1.addWidget(self.loadRawData, row=self.add_one(), col=0)

        w1.addWidget(self.btnFindSWEvents, row=self.add_one(), col=0)
        # w1.addWidget(self.btnShowPacingEvents, row=self.add_one(), col=0)
        # w1.addWidget(self.btnShowPacingCleaned, row=self.add_one(), col=0)

        w1.addWidget(self.amplitudeMapping, row=self.add_one(), col=0)
        w1.addWidget(self.btnViewLiveData, row=self.add_one(), col=0)
        # w1l.setAlignment(QtCore.Qt.AlignTop)

        self.d_control.addWidget(w1, row=1, colspan=1)



    def add_dock_widgets_plots(self):

        self.w1 = pg.PlotWidget(title="Plots of the slow-wave data")
        self.w2 = pg.PlotWidget(title="Plots of zoomed-in slow-wave data")

        c = pg.PlotCurveItem(pen=pg.mkPen('r', width=2))
        c_event = pg.PlotCurveItem(pen=pg.mkPen('y', width=2))
        self.curve_bottom.append(c)
        self.curve_bottom.append(c_event)


        nPlots = 256

        self.w1.setYRange(0, 100)
        self.w1.setXRange(0, 3000)    


        for i in range(nPlots):
            c1 = pg.PlotCurveItem(pen=(i, nPlots))
            c1.setPos(0, i)
            self.curves_left.append(c1)
            self.w1.addItem(c1)

            c2 = pg.PlotCurveItem(pen=(i, nPlots))
            c2.setPos(0, i)
            self.curves_right.append(c2)
            self.w2.addItem(c2)


        self.s1sw = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.s2sw = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))

        self.w1.addItem(self.s1sw)
        self.w2.addItem(self.s2sw)
        self.d_plot.addWidget(self.w1, row=0, col=0)
        self.d_plot.addWidget(self.w2, row=0, col=1)
        self.proxy = pg.SignalProxy(self.w2.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.w2.scene().sigMouseClicked.connect(self.onClick)
        self.w2.sigXRangeChanged.connect(self.updateRegion)
        self.w2.sigYRangeChanged.connect(self.updateRegion)

        self.s1pacing = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 120))
        self.s2pacing = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 120))        
        self.w1.addItem(self.s1pacing)
        self.w2.addItem(self.s2pacing)


    def set_crosshair(self):
        """
        Cross hair definition and initiation
        """
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.w2.addItem(self.vLine, ignoreBounds=True)
        self.w2.addItem(self.hLine, ignoreBounds=True)



    def set_rect_region_ROI(self):
        '''
        Rectangular selection region
        '''
        self.rect = pg.RectROI([300, 5], [1500, 10], pen=pg.mkPen(color='y', width=2))
        self.w1.addItem(self.rect)
        self.rect.sigRegionChanged.connect(self.updatePlot)



    def set_curve_item(self, nPlots, nSamples):
        self.w1.setYRange(0, 100)
        self.w1.setXRange(0, 3000)    
            
        for i in range(nPlots):
            c1 = pg.PlotCurveItem(pen=(i, nPlots))
            self.w1.addItem(c1)
            c1.setPos(0, i)
            self.curves_left.append(c1)

            self.w1.resize(600, 10)

            c2 = pg.PlotCurveItem(pen=(i, nPlots))
            self.w2.addItem(c2)
            c2.setPos(0, i)
            self.curves_right.append(c2)
            self.w2.showGrid(x=True, y=True)
            self.w2.resize(600, 10)

        self.updatePlot()



    def set_plot_data(self, data, nPlots, nSize):
        
        self.data = data
        self.trainingDataPlot.set_plot_data(data)
        self.set_curve_item(nPlots, nSize)
        print("self.data.shape: ", self.data.shape, " nPlots: ", nPlots, " nSize: ", nSize)
        for i in range(nPlots) :
            self.curves_left[i].setData(data[i])
            self.curves_right[i].setData(data[i])

        self.w1.setYRange(0, 100)

        xAxisMax = np.min([data.shape[1], 5000])
        self.w1.setXRange(0, xAxisMax)   

        ax = self.w1.getAxis('bottom')    #This is the trick  

        tickInterval = int(xAxisMax / 6) # Produce 6 tick labels per scroll window

        tickRange = range(0, data.shape[1], tickInterval)

        # Convert indices to time for ticks -- multiply indices by time between samples and add original starting time.
        tickLabels = [str(np.round(i*sp.dat['SigPy']['timeBetweenSamples']+sp.dat['SigPy']['timeStart'],2)[0]) for i in tickRange]

        print(tickLabels)

        ticks = [list(zip(tickRange, tickLabels))]
        print(ticks)

        ax.setTicks(ticks)



    def updatePlot(self):

        xpos = self.rect.pos()[0]
        ypos = self.rect.pos()[1]
        width = self.rect.size()[0]
        height = self.rect.size()[1]
        self.w2.setXRange(xpos, xpos+width, padding=0)
        self.w2.setYRange(ypos, ypos+height, padding=0)



    def updateRegion(self):

        xpos = self.w2.getViewBox().viewRange()[0][0]
        ypos = self.w2.getViewBox().viewRange()[1][0]
        self.rect.setPos([xpos, ypos], update=False)



    # def change_physiology_to_normal(self) :
    #     self.isNormal = 1
    #     self.btnIsNormal.setChecked(1)


    # def change_physiology_to_pacing(self) :
    #     self.isNormal = 0        
    #     self.btnPacing.setChecked(1)


    def repaint_plots(self):

        self.curves_left = []
        self.curves_right = []
        self.curve_bottom = []
        self.add_dock_widgets_plots()
        self.set_crosshair()
        self.set_rect_region_ROI()
        self.elec = []
        self.data = []
        
        self.trainingDataPlot = TrainingDataPlot()


    def mouseMoved(self, evt):
        pos = evt[0]
        vb = self.w2.plotItem.vb
        if self.w2.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())


    def onClick(self, evt):

        pos = evt.scenePos()
        vb = self.w2.plotItem.vb
        if self.w2.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            self.elec.append([int(round(mousePoint.y()/1.2)), int(round(mousePoint.x()))])
            self.trainingDataPlot.add_region([int(round(mousePoint.y()/1.2)), int(round(mousePoint.x()))])

    """
    The binding functions for different gui command buttons.
    """
    def add_as_events(self):

        self.trainingDataPlot.add_events()
        self.curve_bottom[0].set_plot_data(self.trainingDataPlot.plotDat.flatten()[0:self.trainingDataPlot.plotLength * 36])
        self.curve_bottom[1].set_plot_data(np.repeat(self.trainingDataPlot.plotEvent.flatten()[0:self.trainingDataPlot.plotLength], 36))
        # self.w3.setXRange(0, self.trainingDataPlot.plotLength * 36, padding=0)
        # self.w3.setYRange(0, 1, padding=0)


    def add_non_events(self):

        self.trainingDataPlot.add_non_events()
        self.curve_bottom[0].set_plot_data(self.trainingDataPlot.plotDat.flatten()[0:self.trainingDataPlot.plotLength * 36])
        self.curve_bottom[1].set_plot_data(np.repeat(self.trainingDataPlot.plotEvent.flatten()[0:self.trainingDataPlot.plotLength], 36))
        # self.w3.setXRange(0, self.trainingDataPlot.plotLength * 36, padding=0)
        # self.w3.setYRange(0, 1, padding=0)



    def show_pacing_events(self) :
        print("In show_pacing_events")

        if self.pacingEventsOn: 

            self.s1pacing.clear()
            self.s2pacing.clear()

            self.pacingEventsOn = False
            self.btnShowPacingEvents.setText = "Show Pacing Events"

            print("Hiding pacing events!")

        elif (len(self.MarkersPacing)==2) :

            self.s1pacing.clear()
            self.s2pacing.clear()            

            self.s1pacing.addPoints(x=self.MarkersPacing[1], y=self.MarkersPacing[0])
            self.s2pacing.addPoints(x=self.MarkersPacing[1], y=self.MarkersPacing[0])        

            self.statBar.showMessage("Finished displaying pacing markers")

            self.pacingEventsOn = True
            self.btnShowPacingEvents.setText = "Hide Pacing Events"

            print("Show pacing events!")



    def compute_slow_wave_events(self) :

        self.s1sw.clear()
        self.s2sw.clear()


        self.statBar.showMessage("Training and classifying. . .")
        print("Training (perhaps) and classifying")
        print("sp.dat['SigPy']['dataForMarking'].shape: ", sp.dat['SigPy']['dataForMarking'].shape)
        print("self.data.shape: ", self.data.shape)
        
        testData = np.reshape(sp.dat['SigPy']['dataForMarking'], -1)

        windowSize = 36
        overlap = 0.5

        indexJump = int(overlap * windowSize)

        samples = []
        for j in range(1,len(testData)-1, indexJump):
            if (len(testData[j:j+windowSize]) == windowSize):
                samples.append(testData[j:j+windowSize])
        sample_np = np.array(samples)

        cnnType = self.isNormal

        if cnnType==False:
            cnnTypeNum = 0
        else:
            cnnTypeNum = 1
            
        # Call classification function on test data and return the predictions

        # swCNN = SlowWaveCNN(self.trainingDataPlot.plotDat[0:self.trainingDataPlot.plotLength, :], self.trainingDataPlot.plotEvent[0:self.trainingDataPlot.plotLength, :])
        swCNN = SlowWaveCNN(self.trainingDataPlot.plotDat[0:self.trainingDataPlot.plotLength, :], self.trainingDataPlot.plotEvent[0:self.trainingDataPlot.plotLength, :])

        preds = swCNN.classify_data(sample_np, cnnTypeNum)

        
        # Plot the prediction outputs 
        prediction = np.zeros((len(testData)), dtype=int);

        count = 0
        swLocs = np.where(preds==1)[0]

        print("Number sw raw predictions: ", swLocs.shape)
        print("Number of preds: ", len(preds))
        print("Testdata.shape ", testData.shape)


        winRange = 0
        winRangeMultiplier = 2 * windowSize

        # for every x segment of data. if there are SW predictions within this segment, mark as sw.
        # at the max index
        for j in range(0, len(testData), indexJump):

            count += 1
            if (len(np.where(swLocs == count)[0]) > 0):# and (j > winRange)) :
                maxIndex = np.argmax( np.absolute(np.diff(testData[j:j + winRangeMultiplier])))
                prediction[j+maxIndex] = 1
                j += winRangeMultiplier

        print("prediction.shape: ", prediction.shape)

        print("nSW Predictions to X locations: ", len(np.where(prediction == 1)[0]))

        linear_at_uncorrected = np.array(np.where(prediction == 1))
        # linear_at_uncorrected = np.array(np.where(preds == 1))

        rows, cols = linear_at_uncorrected.shape
        to_remove_index = []

        # Remove duplicated values ?
        # for i in range(cols - 1):
        #     if (linear_at_uncorrected[0][i + 1] - linear_at_uncorrected[0][i] < 60) :
        #         to_remove_index.append(i + 1)

        # # Clear duplicated values to stop their removal
        # to_remove_index = []

        # linear_at = np.delete(linear_at_uncorrected, to_remove_index)
        linear_at = linear_at_uncorrected

        pos = []
        lengthData = len(self.data[0])
        nChans = self.data.shape[0]
        sync_events = []

        # Check for sync events
        for val in linear_at.transpose():
            sync_events.append(int(val % lengthData))

        # remove_sync_point = set([x for x in sync_events if sync_events.count(x) > 600])

        # # Clear sync points that are marked for removal:
        # remove_sync_point.clear()

        # Remove the sync events from the actual array

        # for val in linear_at.transpose():
        for swPred in linear_at.transpose():
            # if int(swPred % lengthData) not in remove_sync_point:
            xIndex = int(swPred / lengthData)
            yChannel = int((swPred % lengthData))
            pos.append([xIndex, yChannel])

        pos_np = np.asarray(pos).transpose()

        print("self.data.shape: ", self.data.shape)
        print("pos_np[1].size: ", pos_np.size)


        if pos_np.size is 0:
            print("No events detected")
            return


        self.s1sw.addPoints(x=pos_np[1], y=(pos_np[0]+0.75))
        self.s2sw.addPoints(x=pos_np[1], y=(pos_np[0]+0.75))

        # Convert event co-ordinates to indices for  2d TOA to output to GEMS
        self.statBar.showMessage("Finished classifying slow wave events.", 1000)

        update_GEMS_data_with_TOAs(pos_np, nChans)        


    def show_cleaned_pacing(self) :

        if not (len(self.MarkersPacing)==2) :

            self.statBar.showMessage("Cleaning pacing ...")

            sp.dat['SigPy']['MarkersPacing'], sp.dat['SigPy']['dataPacingCleaned'] = clean_pacing(sp.dat['SigPy']['dataFilt'])
            self.MarkersPacing = sp.dat['SigPy']['MarkersPacing']
            print("self.MarkersPacing: ", self.MarkersPacing[0])
            print("self.MarkersPacing: ", self.MarkersPacing[1])

            self.s1pacing.clear()
            self.s2pacing.clear()

            self.s1pacing.addPoints(x=self.MarkersPacing[1], y=self.MarkersPacing[0])
            self.s2pacing.addPoints(x=self.MarkersPacing[1], y=self.MarkersPacing[0])                

        print("sp.dat['SigPy']['dataToPlot'].shape : ", sp.dat['SigPy']['dataToPlot'].shape)
        sp.dat['SigPy']['dataToPlot'] = preprocess(sp.dat['SigPy']['dataPacingCleaned'])

        print("sp.dat['SigPy']['dataToPlot'].shape : ", sp.dat['SigPy']['dataToPlot'].shape)

        self.data = sp.dat['SigPy']['dataToPlot']
        self.nChans = sp.dat['SigPy']['dataToPlot'].shape[0]
        self.nSamples = sp.dat['SigPy']['dataToPlot'].shape[1]

        self.repaint_plots()

        self.set_plot_data(sp.dat['SigPy']['dataToPlot'], self.nChans, self.nSamples)

        self.statBar.showMessage("Finished plotting cleaned data")



    def btn_animation_set_play(self):

        print("Setting play button")

        btnPlayIconPath = sp.graphicsPath + "btnPlayTiny.png"

        self.btnPlayPause.setIcon(QtGui.QIcon(btnPlayIconPath))
        try:
            self.btnPlayPause.clicked.disconnect()
        except Exception as e:
            print(e)

        self.btnPlayPause.clicked.connect(self.play_animation)            



    def btn_animation_set_pause(self):

        print("Setting pause button")

        self.btnPlayPause.setFixedHeight(20)
        self.btnPlayPause.setFixedWidth(20)

        btnPauseIconPath = sp.graphicsPath + "btnPauseTiny.png"

        self.btnPlayPause.setIcon(QtGui.QIcon(btnPauseIconPath))
        self.btnPlayPause.setIconSize(QtCore.QSize(20,20))

        try:
            self.btnPlayPause.clicked.disconnect()
        except Exception as e:
            print(e)

        self.btnPlayPause.clicked.connect(self.pause_animation)



    def play_animation(self):
        self.ampMap.Playing = True

        self.ampMap.play(self.ampMap.currentFrameRate)
        self.btn_animation_set_pause()



    def pause_animation(self):
        self.ampMap.Playing = False

        self.ampMap.play(0)
        self.btn_animation_set_play()



    def change_animation_data_to_chans(self) :

        self.ampMap.gridDataToAnimate = sp.dat['SigPy']['gridChannelData']
        self.change_animation_data()



    def change_animation_data_to_events(self) :
        if 'gridEventData' in sp.dat['SigPy'].keys() :
            self.ampMap.gridDataToAnimate = sp.dat['SigPy']['gridEventData']
            self.change_animation_data()
        else:
            self.statBar.showMessage("First run 'Compute Slow-wave events'")



    def change_animation_data(self) :

        self.ampMap.priorIndex = self.ampMap.currentIndex        
        self.ampMap.currentIndex = self.ampMap.priorIndex
        self.ampMap.setLevels(0.5, 1.0)


        self.ampMap.setImage(self.ampMap.gridDataToAnimate)

        self.play_animation()        



    def change_frameRate(self, intVal):

        self.ampMap.currentFrameRate = intVal
        fpsLabelStr = str(round((self.ampMap.currentFrameRate / self.ampMap.realFrameRate),1)) + " x"
        self.fpsLabel.setText(fpsLabelStr)

        if self.ampMap.Playing == True :
            self.ampMap.play(self.ampMap.currentFrameRate)



    def plot_amplitude_map(self):

        # Create animation window
        self.ampMap = pg.ImageView()
        self.ampMap.setWindowTitle("Mapped Animating")

        # Preload data

        sp.dat['SigPy']['gridChannelData'] = map_channel_data_to_grid()



        if 'toaIndx' not in sp.dat['SigPy'] :
            self.statBar.showMessage("Note! To plot CNN SW event data, please first run Compute Slow-Wave Events.")
        else:
            sp.dat['SigPy']['gridEventData'] = map_event_data_to_grid_with_trailing_edge()

        gridDataToAnimate = sp.dat['SigPy']['gridChannelData']


        self.ampMap.setImage(gridDataToAnimate)
        self.ampMap.show()        

        ## ======= TOP NAV ===========
        ## -- Play pause speed controls
        # Set default animation speed to sampling frequency fps
        self.ampMap.singleStepVal = round((sp.dat['SigPy']['sampleRate'] / 2), 1)

        self.ampMap.currentFrameRate = sp.dat['SigPy']['sampleRate']
        self.ampMap.realFrameRate = sp.dat['SigPy']['sampleRate']
        self.ampMap.currentFrameRate = self.ampMap.realFrameRate * 2 # Start at double speed

        # Create play pause speed controls
        self.btnPlayPause = QtGui.QPushButton('')
        self.btn_animation_set_pause()

        self.speedSlider = QtGui.QSlider()
        self.speedSlider.setOrientation(QtCore.Qt.Horizontal)
        self.speedSlider.setMinimum(0)        
        self.speedSlider.setMaximum(self.ampMap.singleStepVal * 16)
        self.speedSlider.setValue(self.ampMap.currentFrameRate)


        self.speedSlider.setSingleStep(self.ampMap.singleStepVal)
        
        self.speedSlider.valueChanged.connect(self.change_frameRate)

        fpsLabelStr = str(round((self.ampMap.currentFrameRate / self.ampMap.realFrameRate),1)) + " x"
        self.fpsLabel = QtGui.QLabel(fpsLabelStr)


        ## -- Data select -- live / events / amplitude
        self.radioGrpAnimationData = QtGui.QButtonGroup() 

        self.btnAmplitude = QtGui.QRadioButton('Amplitude')
        self.btnCNNEvents = QtGui.QRadioButton('CNN Events')
        self.btnLive = QtGui.QRadioButton('Live')


        self.btnAmplitude.clicked.connect(self.change_animation_data_to_chans)
        self.btnCNNEvents.clicked.connect(self.change_animation_data_to_events)        

        self.btnAmplitude.setChecked(1);

        self.radioGrpAnimationData.addButton(self.btnAmplitude, 0)
        self.radioGrpAnimationData.addButton(self.btnCNNEvents, 1)


        self.radioGrpAnimationData.setExclusive(True)        


        ## -- Add toolbar widgets to a proxy container widget 
        self.LayoutWidgetPlayPauseSpeed = QtGui.QWidget()
        self.qGridLayout = QtGui.QGridLayout()

        self.qGridLayout.setHorizontalSpacing(14)

        self.qGridLayout.setContentsMargins(8,0,8,0)

        self.qGridLayout.addWidget(self.btnPlayPause, 0,0, alignment=1)
        self.qGridLayout.addWidget(self.speedSlider, 0,1, alignment=1)
        self.qGridLayout.addWidget(self.fpsLabel, 0,2, alignment=1)

        self.qGridLayout.addWidget(self.btnAmplitude, 0,3, alignment=1)
        self.qGridLayout.addWidget(self.btnCNNEvents, 0,4, alignment=1)

        self.LayoutWidgetPlayPauseSpeed.setLayout(self.qGridLayout)

        self.proxyWidget = QtGui.QGraphicsProxyWidget()
        self.proxyWidget.setWidget(self.LayoutWidgetPlayPauseSpeed)
        self.proxyWidget.setPos(0, 0)    

        print("self.ampMap.ui: ", self.ampMap.ui)

        self.ampMap.scene.addItem(self.proxyWidget)

        # Automatically start animation
        self.play_animation()


    def preprocess_buffer_and_housekeeping(self):

        winStartIndex = self.LiveData.priorBufferedChunk.shape[1]
        preprocessWindowChunks = np.hstack((self.LiveData.priorBufferedChunk, self.LiveData.bufferedChunk))
    

        # Reset buffer keeping only the last frame in memory

        self.lastFrame = self.LiveData.bufferedChunk[:,(self.nSamplesCaptured-1)].reshape(-1,1)                
        self.LiveData.bufferedChunk = self.lastFrame
        print("Buffer Reset!")

        # Increase buffer size until it reaches self.desiredPriorChunkSize
        priorBufferChunkStartIndex = preprocessWindowChunks.shape[1] - self.desiredPriorChunkSize

        if priorBufferChunkStartIndex > 0 :
            self.LiveData.priorBufferedChunk = preprocessWindowChunks[:,winStartIndex:]

        else :            
             self.LiveData.priorBufferedChunk = preprocessWindowChunks
        

        preprocessedData = preprocess(preprocessWindowChunks)
        # preprocessedData = preprocessWindowChunks

        self.mappedPreprocessedData = map_channel_data_to_grid(preprocessedData[:, winStartIndex:])       
        print("self.mappedPreprocessedData.shape: ", self.mappedPreprocessedData.shape)



    # Display chunk at a rate matching the capture rate
    def live_animate(self) :

        timeBetweenSamplesAdjust = (self.nSamplesPerChunkIdealised * self.LiveData.timeBetweenSamples / self.nSamplesCaptured)
        timeStartDisplayingFrames = time.time()

        # self.liveMapViewBox.removeItem(self.liveMapImageItem)
        # self.liveMapViewBox.addItem(self.liveMapImageItem)

        for frame in range(0, self.mappedPreprocessedData.shape[0] ) :

            try:
                self.lockDisplayThread.acquire() 
                self.liveMapImageItem.setImage(self.mappedPreprocessedData[frame,:,:])
                self.lockDisplayThread.release() 

            except Exception as e:
                print(e)

            nextFrameTime = timeStartDisplayingFrames + frame * timeBetweenSamplesAdjust
            # print("nextFrameTime: ",nextFrameTime)
            # print("currentFrameTime: ",time.time())
            sleepTime = nextFrameTime - time.time()
            # print("sleepTime: ", sleepTime)

            if sleepTime > 0 :
                time.sleep(sleepTime)

        print(frame," frames displayed.")
        


    # Thread to keep pulling in live data 
    def read_liveData_buffer_Thread(self):

        print("In data pulling thread")
        print("self.LiveData.timeBetweenSamples: ", self.LiveData.timeBetweenSamples)
        self.nSamplesPerChunkIdealised = 30
        self.desiredPriorChunkSize =  self.nSamplesPerChunkIdealised * 4

        self.priorBufferedChunk = self.LiveData.bufferedChunk

        while True:

            # Get number of samples captured
            self.nSamplesCaptured = self.LiveData.bufferedChunk.shape[1]
            print("Frames captured: ", self.nSamplesCaptured)
            # print("nSamplesCaptured: ",self.nSamplesCaptured)

            # If number of samples captured exceeds the idealised buffer size
            # then preprocess, reset buffer and animate

            if self.nSamplesCaptured >= self.nSamplesPerChunkIdealised :

                # print("nSamplesCaptured ",self.nSamplesCaptured," meets criteria.")

                self.preprocess_buffer_and_housekeeping()

                # Live animation
                self.live_animate()

            else:

                # Buffer isn't big enough, give it some more time to produce frames
                time.sleep(0.01)
            
            if not self.liveMapWin.isVisible():

                print("Display thread stopping.")
                self.LiveData.shouldStop = True
                break



    def view_live_data(self) :

        print("Starting live view data")

        # Create thread to capture (or simulate) live data
        self.LiveData = LiveData()

        # Check if live data capture thread has been started -- if not, start
        if not self.LiveData.isAlive():
            try:
                self.LiveData.start()
            except (KeyboardInterrupt, SystemExit):
                sys.exit()   


        # Create image item
        self.liveMapWin = pg.GraphicsWindow() 
        self.liveMapWin.setWindowTitle('Live Mapping')


        self.liveMapViewBox = self.liveMapWin.addViewBox()

        self.liveMapWin.setCentralItem(self.liveMapViewBox)
        self.liveMapViewBox.setAspectLocked()

        # ui.graphicsView.setCentralItem(vb)

        self.liveMapImageItem = pg.ImageItem()
        self.liveMapViewBox.addItem(self.liveMapImageItem)

        # self.liveMapWidgetLayout = pg.LayoutWidget()

        # self.liveMapWin.setLayout(self.liveMapGraphicsLayout)

        # self.liveMapViewBox.addItem(self.liveMapRawImageWidget)

        self.lockDisplayThread = threading.Lock()


        print("Attemping to start data viewing thread")
        self.displayLiveDataThread = Thread(name='read_liveData_buffer_Thread', target=self.read_liveData_buffer_Thread)
        
        if not self.displayLiveDataThread.isAlive():
            try:
                self.displayLiveDataThread.start()
            except (KeyboardInterrupt, SystemExit):
                sys.exit()         



    def save_trained(self):
        with open(sp.trained_file, 'wb') as output:
            pickle.dump(self.trainingDataPlot, output, pickle.HIGHEST_PROTOCOL)


    def load_trained(self):
        self.trainingDataPlot = np.load(sp.get_trained_file())
        self.curve_bottom[0].set_plot_data(self.trainingDataPlot.plotDat.flatten()[0:self.trainingDataPlot.plotLength])
        self.curve_bottom[1].set_plot_data(self.trainingDataPlot.plotEvent.flatten()[0:self.trainingDataPlot.plotLength])
        # self.w3.setXRange(0, self.trainingDataPlot.plotLength, padding=0)
        # self.w3.setYRange(-10, 10, padding=0)




    ## ==== MENU BAR ACTIONS ====

    def load_file_selector__gui_set_data(self, isNormal=True):
        # filenames = QtGui.QFileDialog.getOpenFileNames(self.loadAction, "Select File", "", "*.txt")
        sp.datFileName = QtGui.QFileDialog.getOpenFileName(None, "Select File", "", "*.mat")

        if not (sys.platform == "linux2") :
            sp.datFileName = sp.datFileName[0]

        self.statBar.showMessage("Loading . . . ", 1000)

        load_GEMS_mat_into_SigPy(sp.datFileName, isNormal)

        self.isNormal = isNormal

        self.set_current_dataset()
        self.set_dataType_text()
        self.statBar.showMessage("Finished loading! Now repainting plots . . .")

        self.repaint_plots()

        # Set plot data
        self.set_plot_data(sp.dat['SigPy']['dataToPlot'], sp.dat['SigPy']['dataToPlot'].shape[0], sp.dat['SigPy']['dataToPlot'].shape[1])




        self.statBar.showMessage("Finished repainting plots!", 2000)



    def save_as_file_selector(self):

        sp.datFileName = QtGui.QFileDialog.getSaveFileName(None, "Save As File", sp.datFileName, "*.mat")

        if not (sys.platform == "linux2") :
            sp.datFileName = sp.datFileName[0]        
        self.statBar.showMessage("Saving . . . ")
        print("sp.datFileName: ", sp.datFileName) 

        save_GEMS_SigPy_file(sp.datFileName)

        self.statBar.showMessage("Saved file!")
      


    def save_file_selector(self):

        self.statBar.showMessage("Saving . . . ")
        save_GEMS_SigPy_file(sp.datFileName)

        self.statBar.showMessage("Saved file!")


    def exit_app(self) :

        self.win.close()
        sys.exit()