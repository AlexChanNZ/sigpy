from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

"""
Initialise window here
"""
app = QtGui.QApplication([])
win = pg.GraphicsWindow("Gui for training")
pg.setConfigOptions(antialias=True)

"""
Add plot details
"""
p1 = win.addPlot(title="Plots")
p2 = win.addPlot(title="Zoom on the selected regions")
win.showMaximized()

"""
Add the data curves
"""
nPlots = 100
nSamples = 500
curves_left = []
curves_right = []
for i in range(nPlots):
    c1 = pg.PlotCurveItem(pen=(i, nPlots*1.3))
    c2 = pg.PlotCurveItem(pen=(i, nPlots*1.3))
    p1.addItem(c1)
    p2.addItem(c2)
    c1.setPos(0, i * 6)
    c2.setPos(0, i * 6)
    curves_left.append(c1)
    curves_right.append(c2)

p1.setYRange(0, nPlots*6)
p1.setXRange(0, nSamples)
p1.resize(600, 400)

p2.setYRange(0, nPlots*6)
p2.setXRange(0, nSamples)
p2.resize(600, 400)

"""
Create data here and add to the curve
"""
data = np.random.normal(size=(nPlots*23, nSamples))

for i in range(nPlots):
    curves_left[i].setData(data[i])
    curves_right[i].setData(data[i])

"""
Cross hair definition and initiation
"""
vLine = pg.InfiniteLine(angle=90, movable=False)
hLine = pg.InfiniteLine(angle=0, movable=False)
p2.addItem(vLine, ignoreBounds=True)
p2.addItem(hLine, ignoreBounds=True)

'''
Rectangular selection region
'''
rect = pg.RectROI([20, 20], [100, 300], pen=pg.mkPen(color='y', width=2))
p1.addItem(rect)

'''
Add the details for the next plot which will be the zoomed in values
'''
def updatePlot():
    xpos = rect.pos()[0]
    ypos = rect.pos()[1]
    width = rect.size()[0]
    height = rect.size()[1]
    p2.setXRange(xpos, xpos+width, padding=0)
    p2.setYRange(ypos, ypos+height, padding=0)

def updateRegion():
    xpos = p2.getViewBox().viewRange()[0][0]
    ypos = p2.getViewBox().viewRange()[1][0]
    rect.setPos([xpos, ypos], update=False)

vb = p2.vb
def mouseMoved(evt):
    pos = evt[0]
    if p2.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)
        vLine.setPos(mousePoint.x())
        hLine.setPos(mousePoint.y())

def onClick(evt):
    pos = evt.scenePos()
    if p2.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)

proxy = pg.SignalProxy(p2.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
rect.sigRegionChanged.connect(updatePlot)
p2.sigXRangeChanged.connect(updateRegion)
p2.sigYRangeChanged.connect(updateRegion)
p2.scene().sigMouseClicked.connect(onClick)
updatePlot()