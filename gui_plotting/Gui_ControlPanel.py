"""
    Author: Shameer Sathar
    Description: A module of processing the training data.
"""

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.dockarea import *
import pyqtgraph.console

class GuiControlPanel:
    def __init__(self):
        self.add_dock_widgets_controls()


    def add_dock_widgets_controls(self):


        w1 = pg.LayoutWidget()

        self.dataType=QtGui.QButtonGroup() 

        self.dataTypeLabel = QtGui.QLabel("")
        self.dataTypeLabel.setAlignment(QtCore.Qt.AlignBottom)

        w1.addWidget(self.dataTypeLabel, row=self.add_one(), col=0)

        self.btnFindSWEvents = QtGui.QPushButton('Detect Slow-Wave Events')
        self.amplitudeMapping = QtGui.QPushButton('Amplitude and Event Mapping')
        self.btnViewLiveData = QtGui.QPushButton('Live Mapping')

        # Control for toggling whether to capture live data
        liveDataLabel = QtGui.QLabel('Data capture:')
        liveDataLabel.setAlignment(QtCore.Qt.AlignBottom)

        w1.addWidget(self.btnFindSWEvents, row=self.add_one(), col=0)

        w1.addWidget(self.amplitudeMapping, row=self.add_one(), col=0)
        w1.addWidget(self.btnViewLiveData, row=self.add_one(), col=0)

        self.d_control.addWidget(w1, row=1, colspan=1)





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



    def detect_slow_wave_events(self) :

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
            # if (len(np.where(swLocs == count)[0]) > 0 and len(np.where(swLocs == count + 1)[0]) > 0):# and (j > winRange)) :
            if (len(np.where(swLocs == count)[0]) > 0):# and (j > winRange)) :

                maxIndex = np.argmax( np.absolute(np.diff(testData[j:j + winRangeMultiplier])))
                prediction[j+maxIndex] = 1
                j += winRangeMultiplier

        print("prediction.shape: ", prediction.shape)
        nSWevents = len(np.where(prediction == 1)[0])
        print("nSW Predictions to X locations: ", nSWevents)
        statBarMessage = str(nSWevents) + " slow wave events detected "
        self.statBar.showMessage(statBarMessage)

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

        update_GEMS_data_with_TOAs(pos_np, nChans)                 