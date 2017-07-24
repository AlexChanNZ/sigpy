import config_global as cg
import numpy as np

def map_channel_data_to_grid():

	grid = cg.dat['SigPy']['gridMap']
	gridData = np.zeros(shape=(cg.dat['SigPy']['normData'].shape[1], grid.shape[0], grid.shape[1]), dtype=float)
	print("gridData.shape: ", gridData.shape)
	for r in range(0, grid.shape[0]) :
		for c in range(0, grid.shape[1]) :
			chanNum = int(grid[r,c])
			if (chanNum < cg.dat['SigPy']['normData'].shape[0]) :
				gridData[:,r,c] = cg.dat['SigPy']['normData'][chanNum,:]


	
	return gridData


