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



import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE
from pyqtgraph.dockarea import *

# User defined imports
import config_global as sp

from TrainingDataPlot import TrainingDataPlot



class PyQtLinePlots:

    def __init__(self):
        """
        Initialise the properties of the GUI. This part of the code sets the docks, sizes
        :return: NULL
        """

        self.app = QtGui.QApplication([])
        self.win = QtGui.QMainWindow()        

        self.rowNum = 0
        
        self.d_plot = Dock("Dock Plots",  size=(500, 200))
        self.area = DockArea()

        self.area.addDock(self.d_plot)

        self.win.setWindowTitle("SigPy")



        self.win.setCentralWidget(self.area)
        self.win.resize(1500, 800)
        self.win.setWindowTitle('SigPy')


        self.curves_left = []
        self.curves_right = []
        self.curve_bottom = []
        self.w1 = pg.PlotWidget(title="Electrophysiology Line Graphs")
        self.w2 = pg.PlotWidget(title="Zoomed-in electrophysiology")        
        self.set_rect_region_ROI()

        self.add_scatter_plots()
        self.set_crosshair()
        self.elec = []

        if "PacingMarkers" in sp.dat['SigPy'].keys() :
            self.markersPacing = sp.dat['SigPy']['MarkersPacing']
            print("PacingMarkers key found!")
        else :
            self.markersPacing = []

        if "SWMarkers" in sp.dat['SigPy'].keys() :
            self.markersSWs = sp.dat['SigPy']['MarkersSW']
            print("SWMarkers key found!")
        else :
            self.markersSWs = []         
 
        self.set_plot_data(self.data, self.nChans, self.nSamples)
        self.trainingDataPlot = TrainingDataPlot()

        self.win.showMaximized()
        self.win.show()



    def add_one(self):

        self.rowNum+=1
        return self.rowNum



    def add_scatter_plots(self):


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


        self.s1 = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.s2 = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.w1.addItem(self.s1)
        self.w2.addItem(self.s2)
        self.d_plot.addWidget(self.w1, row=0, col=0)
        self.d_plot.addWidget(self.w2, row=0, col=1)


        self.proxy = pg.SignalProxy(self.w2.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.w2.scene().sigMouseClicked.connect(self.onClick)
        self.w2.sigXRangeChanged.connect(self.updateRegion)
        self.w2.sigYRangeChanged.connect(self.updateRegion)

        if (len(self.markers)==2) :
            self.s1.addPoints(x=self.markers[1], y=self.markers[0])
            self.s2.addPoints(x=self.markers[1], y=self.markers[0])
            print("Adding marker points.")



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
        # self.trainingDataPlot.set_plot_data(data)
        self.set_curve_item(nPlots, nSize)

        for i in range(nPlots):
            self.curves_left[i].setData(data[i])
            self.curves_right[i].setData(data[i])

        self.w1.setYRange(0, 100)

        xAxisMax = np.min([data.shape[1], 5000])
        self.w1.setXRange(0, xAxisMax)   

        ax = self.w1.getAxis('bottom')    #This is the trick  

        tickInterval = int(xAxisMax / 6) # Produce 6 tick labels per scroll window

        tickRange = range(0, data.shape[1], tickInterval)

        # Convert indices to time for ticks -- multiply indices by time between samples and add original starting time.
        tickLabels = [str(round(self.timeBetweenSamples * tick + self.timeStart, 2)) for tick in tickRange]

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

        self.s1.addPoints(x=pos_np[1], y=(pos_np[0]))
        self.s2.addPoints(x=pos_np[1], y=(pos_np[0]))



    def plot_markers(self):

        self.s1.addPoints(x=pos_np[1], y=(pos_np[0]+0.75))
        self.s2.addPoints(x=pos_np[1], y=(pos_np[0]+0.75))



    def exit_app(self) :
        self.win.close()
        sys.exit()