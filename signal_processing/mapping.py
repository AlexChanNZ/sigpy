import config_global as cg
import numpy as np



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
	for r in range(0, grid.shape[0]) :
		for c in range(0, grid.shape[1]) :
			chanNum = int(grid[r,c])
			if (chanNum <= cg.dat['SigPy']['normData'].shape[0]) :
				gridData[:,r,c] = cg.dat['SigPy']['normData'][(chanNum-1),:]
	
	return eventGrid

def add_time_data_grid() :

