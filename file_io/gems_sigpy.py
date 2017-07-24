# -*- coding: utf-8 -*-
"""
Author : Shameer Sathar
"""
from __future__ import division # Correct Python 2 division

import numpy as np
import scipy.io as sio

import config_global as cg

# np.set_printoptions(linewidth=1000, precision=3, threshold=np.inf)



def load_GEMS_mat_into_SigPy(fileNameAndPath):
    """
    The data is classified based on the training data.
    :param plot: Input values to be processed for generating features
    :return: predictions for the entire data set
    """

    cg.dat = sio.loadmat(fileNameAndPath, matlab_compatible=True) #added matlab_compatible=True to aid GEMS compatibility

    # Check if this file has been saved from SigPy and whether it has been copied
    if hasattr(cg.dat, 'GEMSorig_toapp'):

        print("This file has been saved from SigPy")        
        print("This GEMS file, already has its original GEMs data duplicated.")

    else:
        #Duplicate original gems file and create SigPy structure
        cg.dat['GEMSorig_toapp'] = cg.dat['toapp'][0,0]

    cg.dat['SigPy'] = {}
    cg.dat['SigPy']['filtData'] = cg.dat['toapp']['filtdata'][0,0]
    cg.dat['SigPy']['sampleRate'] = cg.dat['toapp']['fs'][0,0]
    cg.dat['SigPy']['timeStart'] = cg.dat['toapp']['filstartT'][0,0]
    cg.dat['SigPy']['timeEnd'] = cg.dat['toapp']['Teof'][0,0]
    cg.dat['SigPy']['timeBetweenSamples'] = 1 / cg.dat['SigPy']['sampleRate'][0,0]

    cg.dat['toapp']['showchans'][0,0] = np.array(cg.dat['toapp']['showchans'][0,0]).astype(dtype=float)
    cg.dat['toapp']['orientedElec'][0,0] = np.array(cg.dat['toapp']['orientedElec'][0,0]).astype(dtype=float)

    cg.dat['SigPy']['gridMap'] = cg.dat['toapp']['orientedElec'][0,0]

    # cg.dat = sio.loadmat(fileNameAndPath, mat_dtype=False) 

    # print('cg.dat[toapp][orientedElec]:', cg.dat['toapp']['orientedElec'][0,0]

    # cg.dat['SigPy']['eData'] = cg.dat['toapp']['edata'][0,0]

    # print("cg.sigData['eData'].shape: ", cg.sigData['eData'].shape)


def save_GEMS_SigPy_file(fileNameAndPath):

    # To overwrite original GEMS data, comment this out to save GEMS data as backup.
    # cg.dat.pop('SigPy', None)
    cg.dat.pop('GEMSorig_toapp', None)
    cg.dat.pop('GEMSorig_bdfdef', None)
    # cg.dat.pop('bdfdef', None) #popping bdfdef because of UI control compatibility. 
    #EDIT: Appears saving UIControl component as struct still works in GEMS.

    # Save GEMS file
    sio.savemat(fileNameAndPath, cg.dat)


def update_GEMS_data_with_TOAs(pos_np, nChans) :

    toaChanIndices = []
    toaChanTimeStamps = []

    iCount = 0
    chanNum = 0
    lastSampleChan = -1

    toaIndx = np.empty(shape=nChans, dtype=object)
    toaCell = np.empty(shape=nChans, dtype=object)

    #Initialise toaIndxs as empty 
    #Note: np.empty array comes up as None which is not compatible with MATLAB.
    for chanI in range(0, nChans):
        toaIndx[chanI] = np.array([])
        toaCell[chanI] = np.array([])

    print("making TOA data for GEMS")



    ## Iterate through indexes and chans and convert to ToA and GEMS compatible indexes
    for sampleIndex, sampleChan in zip(pos_np[1], pos_np[0]) :
        
        print("sampleIndex: ", sampleIndex)

        if not (int(sampleChan) == int(lastSampleChan)) and (lastSampleChan > -1):  

            print("toaChanIndices: ", toaChanIndices)        
            print("toaChanTimeStamps: ", toaChanTimeStamps)   

            if (len(toaChanIndices) > 0) :

                toaIndx[lastSampleChan] = np.array(toaChanIndices).astype(dtype=float)
                toaCell[lastSampleChan] = np.array(toaChanTimeStamps).astype(dtype=float)

            toaChanIndices = []
            toaChanTimeStamps = []


        if (sampleIndex > 0) :

            toaChanIndices.append(sampleIndex)
            timestamp = cg.dat['SigPy']['timeBetweenSamples'] * sampleIndex + cg.dat['SigPy']['timeStart']
            toaChanTimeStamps.append(round(timestamp[0][0],4))       
            
        lastSampleChan = sampleChan



    print("toaIndx: ", toaIndx)        
    print("toaCell: ", toaCell)  

    cg.dat['SigPy']['toaIndx'] = toaIndx
    cg.dat['SigPy']['toaCell'] = toaCell

    cg.dat['toapp']['toaIndx'][0,0] = toaIndx
    cg.dat['toapp']['toaCell'][0,0] = toaCell


 
