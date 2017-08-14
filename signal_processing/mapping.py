import config_global as sp
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from sklearn import preprocessing



def get_grid() :
	grid = sp.dat['SigPy']['gridMap']
	return grid


def get_mapped_chan(chanNum):
	return np.where(np.array(sp.dat['SigPy']['chanNums'])==(chanNum-1))[0]


def map_channel_data_to_grid(inData=None) :
	if inData is None:
		inData = sp.dat['SigPy']['dataToPlot']		

	grid = get_grid()

	gridData = np.zeros(shape=(inData.shape[1], grid.shape[0], grid.shape[1]), dtype=float)

	for r in range(0, grid.shape[0]) :
		for c in range(0, grid.shape[1]) :
			chanNum = grid[r,c]
			mappedChan = get_mapped_chan(chanNum)
			# print("mappedChan: ", mappedChan)
			if (mappedChan <= inData.shape[0]) :		
				gridData[:,r,c] = inData[mappedChan,:]

	return gridData



def map_event_data_to_grid(inData=None) :
	if not inData:
		inData = sp.dat['SigPy']['toaIndx']

	grid = get_grid()

	eventGrid = np.zeros(shape=(sp.dat['SigPy']['dataToPlot'].shape[1], grid.shape[0], grid.shape[1]), dtype=float)

	for r in range(0, grid.shape[0]) :
		for c in range(0, grid.shape[1]) :
			chanNum = grid[r,c]

			mappedChan = get_mapped_chan(chanNum)

			if (mappedChan <= sp.dat['SigPy']['dataToPlot'].shape[0]) :

				eventIndicesForChan = inData[mappedChan][0].astype(int)

				if eventIndicesForChan.shape[0] > 0 :
					eventGrid[eventIndicesForChan,r,c] = 1		

	return eventGrid



def map_event_data_to_grid_with_trailing_edge() :

	eventDataInGrid = map_event_data_to_grid()
	eventDataWithTrailingEdge = np.round(make_gaussian_3d(eventDataInGrid),3)

	normaliseData = np.apply_along_axis(preprocessing.MinMaxScaler().fit_transform, 0, eventDataWithTrailingEdge)

	return normaliseData


def live_data_map():

	return liveData



## Put a nice gaussian filter around the 1
def make_gaussian_3d(timeXYarr):
	sigmaVal = 8 #trailing edge length    
	gaussian3d = np.apply_along_axis(gaussian_filter, 0, timeXYarr, sigmaVal)
	# VY = gaussian_filter(VY, sigma=1)
	return gaussian3d
