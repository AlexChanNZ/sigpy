""" Standard imports """

"""
    Author: Shameer Sathar
    Description: Provide Gui Interface.
"""
import sys
import numpy as np
import platform

from multiprocessing import Process
# Main GUI support
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
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
from TrainingDataPlot import TrainingDataPlot
from ARFFcsvReader import ARFFcsvReader
from WekaInterface import WekaInterface
from FeatureAnalyser import FeatureAnalyser
from SlowWaveCNN import SlowWaveCNN
import config_global as cg
from file_io import *
from sig_manip import preprocess

# For debugging purps

class GuiWindowDocks:
    def __init__(self):
        """
        Initialise the properties of the GUI. This part of the code sets the docks, sizes
        :return: NULL
        """

        self.rowNum = 0
        self.app = QtGui.QApplication([])
        self.win = QtGui.QMainWindow()
        area = DockArea()
        self.d_control = Dock("Dock Controls", size=(50, 200))
        self.d_plot = Dock("Dock Plots", size=(500, 200))
        self.d_train = Dock("Training Signal", size=(500, 50))
        area.addDock(self.d_control, 'left')
        area.addDock(self.d_plot, 'right')
        area.addDock(self.d_train, 'bottom', self.d_plot)

        self.win.setWindowTitle("PySig")

        self.win.setCentralWidget(area)
        self.win.resize(1500, 800)
        self.win.setWindowTitle('PySig Training')
        self.add_dock_widgets_controls()
        self.add_menu_bar()

        self.curves_left = []
        self.curves_right = []
        self.curve_bottom = []
        self.add_dock_widgets_plots()
        self.set_crosshair()
        self.set_rect_region_ROI()
        self.elec = []
        self.data = []

        self.set_plot_data(cg.dataForAnalysis['SigPy']['normData'], cg.dataForAnalysis['SigPy']['normData'].shape[0], cg.dataForAnalysis['SigPy']['normData'].shape[1])

        self.trainingDataPlot = TrainingDataPlot()
        
        self.saveBtn_events.clicked.connect(lambda: self.add_as_events())
        self.saveBtn_nonEvents.clicked.connect(lambda: self.add_non_events())
        self.undoBtn.clicked.connect(lambda: self.undo())
        self.writeWEKABtn.clicked.connect(lambda: self.writeWEKA_data())
        self.readPredictedVal.clicked.connect(lambda: self.read_predicted())
        self.analyseInternal.clicked.connect(lambda: self.analyse_internal())
        self.save_trained_data.clicked.connect(lambda: self.save_trained())
        self.load_trained_data.clicked.connect(lambda: self.load_trained())

        self.win.showMaximized()
        self.win.show()



    def add_one(self):
        self.rowNum+=1
        return self.rowNum



    def add_menu_bar(self):
        ## MENU BAR
        self.statBar = self.win.statusBar()

        self.mainMenu = self.win.menuBar()
        self.fileMenu = self.mainMenu.addMenu('&File')

        ## Load file
        self.loadAction = QtGui.QAction('&Load GEMS .mat', self.fileMenu)        
        self.loadAction.setShortcut('Ctrl+L')
        self.loadAction.setStatusTip('')
        self.loadAction.triggered.connect(lambda: self.load_file_selector__gui_set_data())

        self.fileMenu.addAction(self.loadAction)


        ## Save as gems file 
        self.saveAsAction = QtGui.QAction('&Save as GEMS .mat', self.fileMenu)        
        self.saveAsAction.setStatusTip('Save data with filename.')
        self.saveAsAction.triggered.connect(lambda: self.save_as_file_selector())

        self.fileMenu.addAction(self.saveAsAction)


        ## Save (update file)
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
        w1 = pg.LayoutWidget()
        label = QtGui.QLabel('Usage info')
        self.saveBtn_events = QtGui.QPushButton('Save As Events')
        self.saveBtn_nonEvents = QtGui.QPushButton('Save As Non-Events')
        self.undoBtn = QtGui.QPushButton('Undo')
        self.writeWEKABtn = QtGui.QPushButton('Write WEKA')
        self.readPredictedVal = QtGui.QPushButton('Read Weka CSV')
        self.analyseInternal = QtGui.QPushButton('Analyse Events')
        self.save_trained_data = QtGui.QPushButton('Save Training')
        self.load_trained_data = QtGui.QPushButton('Load Training')

        # self.dataTypeLayout=QtGui.QHBoxLayout()  # layout for the central widget
        # self.dataTypeWidget=QtGui.QWidget(self)  # central widget
        # self.dataTypeWidget.setLayout(self.dataTypeLayout)

        self.dataType=QtGui.QButtonGroup(w1) 


        self.btnPacing = QtGui.QRadioButton('Pacing')
        self.btnIsNormal = QtGui.QRadioButton('Normal')
        self.btnIsNormal.setChecked(1);

        self.dataType.addButton(self.btnPacing, 0)
        self.dataType.addButton(self.btnIsNormal, 1)
        self.dataType.setExclusive(True)


        w1.addWidget(label, row=self.add_one(), col=0)
        # w1.addWidget(self.loadRawData, row=self.add_one(), col=0)

        w1.addWidget(self.saveBtn_events, row=self.add_one(), col=0)
        w1.addWidget(self.saveBtn_nonEvents, row=self.add_one(), col=0)
        w1.addWidget(self.undoBtn, row=self.add_one(), col=0)
        w1.addWidget(self.writeWEKABtn, row=self.add_one(), col=0)
        w1.addWidget(self.readPredictedVal, row=self.add_one(),col=0)
        w1.addWidget(self.analyseInternal, row=self.add_one(), col=0)
        w1.addWidget(self.save_trained_data, row=self.add_one(), col=0)
        w1.addWidget(self.load_trained_data, row=self.add_one(), col=0)
        w1.addWidget(self.btnIsNormal,row=self.add_one(),col=0)
        w1.addWidget(self.btnPacing,row=self.add_one(), col=0)
        self.d_control.addWidget(w1, row=1, colspan=1)



    def add_dock_widgets_plots(self):

        self.w1 = pg.PlotWidget(title="Plots of the slow-wave data")
        self.w2 = pg.PlotWidget(title="Plots of zoomed-in slow-wave data")
        self.w3 = pg.PlotWidget(title="Selected Data for Training")
        c = pg.PlotCurveItem(pen=pg.mkPen('r', width=2))
        c_event = pg.PlotCurveItem(pen=pg.mkPen('y', width=2))
        self.curve_bottom.append(c)
        self.curve_bottom.append(c_event)
        self.w3.addItem(c)
        self.w3.addItem(c_event)
        nPlots = 256

        self.w1.setYRange(0, 100)
        self.w1.setXRange(0, 3000)    



        for i in range(nPlots):
            c1 = pg.PlotCurveItem(pen=(i, nPlots*1.3))
            c1.setPos(0, i * 1.2)
            self.curves_left.append(c1)
            self.w1.addItem(c1)

            c2 = pg.PlotCurveItem(pen=(i, nPlots*1.3))
            c2.setPos(0, i * 1.2)
            self.curves_right.append(c2)
            self.w2.addItem(c2)

        self.s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.s2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.w1.addItem(self.s1)
        self.w2.addItem(self.s2)
        self.d_plot.addWidget(self.w1, row=0, col=0)
        self.d_plot.addWidget(self.w2, row=0, col=1)
        self.d_train.addWidget(self.w3, row=0, col=0)
        self.proxy = pg.SignalProxy(self.w2.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.w2.scene().sigMouseClicked.connect(self.onClick)
        self.w2.sigXRangeChanged.connect(self.updateRegion)
        self.w2.sigYRangeChanged.connect(self.updateRegion)



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
            c1 = pg.PlotCurveItem(pen=(i, nPlots*1.3))
            self.w1.addItem(c1)
            c1.setPos(0, i * 1.2)
            self.curves_left.append(c1)

            self.w1.resize(600, 10)

            c2 = pg.PlotCurveItem(pen=(i, nPlots*1.3))
            self.w2.addItem(c2)
            c2.setPos(0, i * 1.2)
            self.curves_right.append(c2)
            self.w2.showGrid(x=True, y=True)
            self.w2.resize(600, 10)
        self.updatePlot()



    def set_plot_data(self, data, nPlots, nSize):
        self.data = data
        # self.trainingDataPlot.set_plot_data(data)
        self.set_curve_item(nPlots, nSize)
        for i in range(nPlots):
            self.curves_left[i].setData(data[i])
            self.curves_right[i].setData(data[i])


        self.w1.setYRange(0, 100)
        self.w1.setXRange(0, data.shape[1])   


        ax = self.w1.getAxis('bottom')    #This is the trick  

        tickInterval = 2000

        tickRange = range(0, data.shape[1], tickInterval)

        # Convert indices to time for ticks -- multiply indices by time between samples and add original starting time.
        tickLabels = [str(round(i*cg.dataForAnalysis['SigPy']['timeBetweenSamples']+cg.dataForAnalysis['SigPy']['timeStart'],2)) for i in tickRange]

        print(tickLabels)

        ticks = [list(zip(tickRange, tickLabels))]
        print(ticks)


        ax.setTicks(ticks)
             # self.xLabels = "print xLabels."
        # ax.setTicks([self.xLabels])



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
        self.saveBtn_events.clicked.connect(lambda: self.add_as_events())
        self.saveBtn_nonEvents.clicked.connect(lambda: self.add_non_events())
        self.undoBtn.clicked.connect(lambda: self.undo())
        self.writeWEKABtn.clicked.connect(lambda: self.writeWEKA_data())
        self.readPredictedVal.clicked.connect(lambda: self.read_predicted())
        self.analyseInternal.clicked.connect(lambda: self.analyse_internal())
        self.save_trained_data.clicked.connect(lambda: self.save_trained())
        self.load_trained_data.clicked.connect(lambda: self.load_trained())



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
        self.w3.setXRange(0, self.trainingDataPlot.plotLength * 36, padding=0)
        self.w3.setYRange(0, 1, padding=0)



    def add_non_events(self):
        self.trainingDataPlot.add_non_events()
        self.curve_bottom[0].set_plot_data(self.trainingDataPlot.plotDat.flatten()[0:self.trainingDataPlot.plotLength * 36])
        self.curve_bottom[1].set_plot_data(np.repeat(self.trainingDataPlot.plotEvent.flatten()[0:self.trainingDataPlot.plotLength], 36))
        self.w3.setXRange(0, self.trainingDataPlot.plotLength * 36, padding=0)
        self.w3.setYRange(0, 1, padding=0)



    def undo(self):
        self.trainingDataPlot.undo()
        self.curve_bottom[0].set_plot_data(self.trainingDataPlot.plotDat.flatten()[0:self.trainingDataPlot.plotLength])
        self.curve_bottom[1].set_plot_data(self.trainingDataPlot.plotEvent.flatten()[0:self.trainingDataPlot.plotLength])
        self.w3.setXRange(0, self.trainingDataPlot.plotLength * 36, padding=0)
        self.w3.setYRange(0, 1, padding=0)



    def read_predicted(self):
        filename = QtGui.QFileDialog.getOpenFileName(None, 'Open ARFF WEKA generated output file')
        if filename == u'':
            return
        test = ARFFcsvReader(filename)
        prediction = np.asarray(test.get_prediction())
        diff = np.diff(prediction)
        linear_at = np.array(np.where(diff == 1))
        pos = []
        length = len(self.data[1])
        for val in linear_at.transpose():
            pos.append([int(val/length), int(val % length)])
        pos_np = np.asarray(pos).transpose()

        self.s1.addPoints(x=pos_np[1], y=(pos_np[0] * 1.2))
        self.s2.addPoints(x=pos_np[1], y=(pos_np[0] * 1.2))



    def writeWEKA_data(self):
        test_data = np.reshape(self.data, -1)
        data = self.trainingDataPlot.plotDat[0][0:self.trainingDataPlot.plotLength]
        events = self.trainingDataPlot.plotEvent[0][0:self.trainingDataPlot.plotLength]/5
        Process(target=self.process_thread, args=(data, events)).start()
        Process(target=self.process_thread, args=[test_data]).start()



    def process_thread(self, data, event=None):
        training_analyser = FeatureAnalyser()
        # FeatureAnalyser requires the 1d data to be passed as array of an array
        training_features = training_analyser.writeWEKA_data([data],(1, self.trainingDataPlot.plotLength))
        if event is None:
            output_name = cg.test_file_name
        else:
            output_name = cg.training_file_name
        weka_write = WekaInterface(training_features, output_name)
        weka_write.arff_write(event)



    def analyse_internal(self):
        self.s1.clear()
        self.s2.clear()

        self.statBar.showMessage("Training and classifying. . .")

        testData = np.reshape(self.data, -1)

        windowSize = 36
        overlap = 0.5
        samples = []
        for j in range(0,len(testData), int(overlap * windowSize)):
            if (len(testData[j:j+windowSize]) == windowSize):
                samples.append(testData[j:j+windowSize])
        
        
        sample_np = np.empty((len(samples), windowSize))

        for i, x in enumerate(samples):
            sample_np[i] = np.array(x)

        cnnType = self.btnIsNormal.isChecked()
            
        # Call classification function on test data and return the predictions

        swCNN = SlowWaveCNN(self.trainingDataPlot.plotDat[0:self.trainingDataPlot.plotLength, :], self.trainingDataPlot.plotEvent[0:self.trainingDataPlot.plotLength, :])
        preds = swCNN.classify_data(sample_np, cnnType)
        

        # Plot the prediction outputs here.
        prediction = np.zeros((1, len(testData)));

        count = 0;
        locs = np.where(preds==1)[0]
        win_range = 0;        

        for j in range(0,len(testData), int(overlap * windowSize)):
            if (len(testData[j:j+windowSize]) == windowSize):
                count = count + 1;
                if (len(np.where(locs == count)[0]) > 0 and (j+windowSize > win_range)):
                    prediction[0,j+windowSize] = 1
                    win_range = j + 3*windowSize;
        
        linear_at_uncorrected = np.array(np.where(prediction == 1))
        rows, cols = linear_at_uncorrected.shape
        to_remove_index = []
        for i in range(cols - 1):
            if linear_at_uncorrected[0][i + 1] - linear_at_uncorrected[0][i] < 60:
                to_remove_index.append(i + 1)
        linear_at = np.delete(linear_at_uncorrected, to_remove_index)

        pos = []
        lengthData = len(self.data[0])
        nChans = self.data.shape[0]
        sync_events = []

        # Check for sync events
        for val in linear_at.transpose():
            sync_events.append(int(val % lengthData))
        remove_sync_point = set([x for x in sync_events if sync_events.count(x) > 600])

        
        #remove_sync_point.clear()

        # Remove the sync events from the actual array

        for val in linear_at.transpose():
            if int(val % lengthData) not in remove_sync_point:
                xIndex = int(val / lengthData)
                yChannel = int(val % lengthData)
                pos.append([xIndex, yChannel])

        pos_np = np.asarray(pos).transpose()

        print("self.data.shape: ", self.data.shape)
        print("pos_np[1].size: ", pos_np[1].size)

        if pos_np.size is 0:
            print("No events detected")
            return

        self.s1.addPoints(x=pos_np[1], y=(pos_np[0] * 1.2))
        self.s2.addPoints(x=pos_np[1], y=(pos_np[0] * 1.2))


        # Convert event co-ordinates to indices for  2d TOA to output to GEMS
        self.statBar.showMessage("Finished classifying slow wave events.", 1000)

        update_GEMS_data_with_TOAs(pos_np, nChans)      


        # print("pos_np: ", pos_np)
        # print("pos_np.shape: ", pos_np.shape)



    def save_trained(self):
        with open(cg.trained_file, 'wb') as output:
            pickle.dump(self.trainingDataPlot, output, pickle.HIGHEST_PROTOCOL)


    def load_trained(self):
        self.trainingDataPlot = np.load(cg.get_trained_file())
        self.curve_bottom[0].set_plot_data(self.trainingDataPlot.plotDat.flatten()[0:self.trainingDataPlot.plotLength])
        self.curve_bottom[1].set_plot_data(self.trainingDataPlot.plotEvent.flatten()[0:self.trainingDataPlot.plotLength])
        self.w3.setXRange(0, self.trainingDataPlot.plotLength, padding=0)
        self.w3.setYRange(-10, 10, padding=0)




    ## ==== MENU BAR ACTIONS ====

    def load_file_selector__gui_set_data(self):
        # filenames = QtGui.QFileDialog.getOpenFileNames(self.loadAction, "Select File", "", "*.txt")
        cg.dataForAnalysisFileName = QtGui.QFileDialog.getOpenFileName(None, "Select File", "", "*.mat")
        platformName = sys.platform
        print("platformName: ",platformName)

        if not (platformName == "linux2") :
            cg.dataForAnalysisFileName = cg.dataForAnalysisFileName[0]

        self.statBar.showMessage("Loading . . . ", 1000)
        print("cg.dataForAnalysisFileName: ", cg.dataForAnalysisFileName)        
        load_GEMS_mat_into_SigPy(cg.dataForAnalysisFileName)

        self.statBar.showMessage("Finished loading! Now preprocessing . . .")

        cg.dataForAnalysis['SigPy']['normData'] = preprocess(cg.dataForAnalysis['SigPy']['filtData'])
        self.statBar.showMessage("Finished pre-processing! Now repainting plots . . . ")

        # print("cg.dataForAnalysis['SigPy']['normData']: ", cg.dataForAnalysis['SigPy']['normData'])
        # self.trainingDataPlot = TrainingDataPlot()


        self.repaint_plots()
        # Set plot data
        self.set_plot_data(cg.dataForAnalysis['SigPy']['normData'], cg.dataForAnalysis['SigPy']['normData'].shape[0], cg.dataForAnalysis['SigPy']['normData'].shape[1])
        self.btnIsNormal.setChecked(1)

        self.statBar.showMessage("Finished repainting plots!", 2000)


    def save_as_file_selector(self):

        cg.dataForAnalysisFileName = QtGui.QFileDialog.getSaveFileName(None, "Save As File", cg.dataForAnalysisFileName, "*.mat")[0]
        self.statBar.showMessage("Saving . . . ")
        print("cg.dataForAnalysisFileName: ", cg.dataForAnalysisFileName) 

        save_GEMS_SigPy_file(cg.dataForAnalysisFileName)

        self.statBar.showMessage("Saved file!")
      


    def save_file_selector(self):
        self.statBar.showMessage("Saving . . . ")
        save_GEMS_SigPy_file(cg.dataForAnalysisFileName)

        self.statBar.showMessage("Saved file!")


    def exit_app(self) :
        self.win.close()
        sys.exit()