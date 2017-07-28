import config_global as cg
import numpy as np
from scipy.ndimage.filters import gaussian_filter



def get_grid() :
	grid = cg.dat['SigPy']['gridMap']
	return grid



def map_channel_data_to_grid() :

	grid = get_grid()
	gridData = np.zeros(shape=(cg.dat['SigPy']['normData'].shape[1], grid.shape[0], grid.shape[1]), dtype=float)
	print("gridData.shape: ", gridData.shape)
	print("cg.dat['SigPy']['normData'].shape[0]: ",cg.dat['SigPy']['normData'].shape[0])
	print(grid)
	for r in range(0, grid.shape[0]) :
		for c in range(0, grid.shape[1]) :
			chanNum = int(grid[r,c])
			if (chanNum <= cg.dat['SigPy']['normData'].shape[0]) :
				gridData[:,r,c] = cg.dat['SigPy']['normData'][(chanNum-1),:]
	
	return gridData



def map_event_data_to_grid() :

	grid = get_grid()

	eventGrid = np.zeros(shape=(cg.dat['SigPy']['normData'].shape[1], grid.shape[0], grid.shape[1]), dtype=float)



	for r in range(0, grid.shape[0]) :
		for c in range(0, grid.shape[1]) :
			chanNum = int(grid[r,c])
			if (chanNum <= cg.dat['SigPy']['normData'].shape[0]) :
				eventIndicesForChan = cg.dat['SigPy']['toaIndx'][(chanNum-1)].astype(int)
				if eventIndicesForChan.shape[0] > 0 :

					eventGrid[eventIndicesForChan,r,c] = 1		
	
	return eventGrid



def convert_event_data_to_grid() :

	eventDataInGrid = map_event_data_to_grid()
	eventDataWithTrailingEdge = make_gaussian_3d(eventDataInGrid)

	return eventDataWithTrailingEdge



## =========================
## calc time range of array
## =========================

## Put a nice gaussian filter around the 1
def make_gaussian_3d(timeXYarr):
	sigmaVal = 12 #trailing edge length    
	gaussian3d = np.apply_along_axis(gaussian_filter, 0, timeXYarr, sigmaVal)
	# VY = gaussian_filter(VY, sigma=1)
	gaussian3d = np.multiply(gaussian3d, timeXYarr)
	return gaussian3d
