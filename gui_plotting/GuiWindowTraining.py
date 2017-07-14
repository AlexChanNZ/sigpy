"""
    Author: Shameer Sathar
    Description: A module of processing the training data.
"""

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.dockarea import *
import pyqtgraph.console

class GuiWindowTraining:
    def __init__(self):
        '''
            Initialise dock window properties
        '''
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow("Gui for training")
        self.p1 = self.win.addPlot(title="Plots")
        self.p2 = self.win.addPlot(title="Zoom on the selected regions")
        self.setCrosshair()
        self.setRectRegionROI()
        self.curves_left = []
        self.curves_right = []
        self.p2.sigXRangeChanged.connect(selstr(cg.loaded_data_file) 
                    + str('_acti_points.matf.updateRegion)
        self.p2.sigYRangeChanged.connect(self.updateRegion)
        self.proxy = pg.SignalProxy(self.p2.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.p2.scene().sigMouseClicked.connect(self.onClick)

    def setCrosshair(self):
        """
        Cross hair definition and initiation
        """
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.p2.addItem(self.vLine, ignoreBounds=True)
        self.p2.addItem(self.hLine, ignoreBounds=True)

    def setRectRegionROI(self):
        '''
        Rectangular selection region
        '''
        self.rect = pg.RectROI([300, 300], [1500, 100], pen=pg.mkPen(color='y', width=2))
        self.p1.addItem(self.rect)
        self.rect.sigRegionChanged.connect(self.updatePlot)


    def setCurveItem(self, nPlots, nSamples):
        for i in range(nPlots):
            c1 = pg.PlotCurveItem(pen=(i, nPlots*1.3))
            self.p1.addItem(c1)
            c1.setPos(0, i * 20)
            self.curves_left.append(c1)
            self.p1.setYRange(0, 900)
            self.p1.setXRange(0, 3000)
            self.p1.resize(600, 900)

            c2 = pg.PlotCurveItem(pen=(i, nPlots*1.3))
            self.p2.addItem(c2)
            c2.setPos(0, i * 20)
            self.curves_right.append(c2)
            self.p2.showGrid(x=True, y=True)
            self.p2.resize(600, 900)
        self.updatePlot()

    def setData(self, data, nPlots, nSize):
        self.setCurveItem(nPlots, nSize)
        for i in range(nPlots):
            self.curves_left[i].setData(data[i])
            self.curves_right[i].setData(data[i])

    def updatePlot(self):
        xpos = self.rect.pos()[0]
        ypos = self.rect.pos()[1]
        width = self.rect.size()[0]
        height = self.rect.size()[1]
        self.p2.setXRange(xpos, xpos+width, padding=0)
        self.p2.setYRange(ypos, ypos+height, padding=0)

    def updateRegion(self):
        xpos = self.p2.getViewBox().viewRange()[0][0]
        ypos = self.p2.getViewBox().viewRange()[1][0]
        self.rect.setPos([xpos, ypos], update=False)

    def mouseMoved(self, evt):
        pos = evt[0]
        vb = self.p2.vb
        if self.p2.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def onClick(self, evt):
        pos = evt.scenePos()
        vb = self.p2.vb
        if self.p2.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
