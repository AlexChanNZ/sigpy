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

    cg.dataForAnalysis = sio.loadmat(fileNameAndPath) 


    # Check if this file has been saved from SigPy and whether it has been copied
    if hasattr(cg.dataForAnalysis, 'GEMSorig_toapp'):

        print("This file has been saved from SigPy")        
        print("This GEMS file, already has its original GEMs data duplicated.")

    else:
        #Duplicate original gems file and create SigPy structure
        cg.dataForAnalysis['GEMSorig_toapp'] = cg.dataForAnalysis['toapp'][0,0]
        cg.dataForAnalysis['GEMSorig_bdfdef'] = cg.dataForAnalysis['bdfdef'][0,0]        
        cg.dataForAnalysis['SigPy'] = {}
        cg.dataForAnalysis['SigPy']['filtData'] = cg.dataForAnalysis['toapp']['filtdata'][0,0]
        cg.dataForAnalysis['SigPy']['sampleRate'] = cg.dataForAnalysis['toapp']['fs'][0,0]
        cg.dataForAnalysis['SigPy']['timeStart'] = cg.dataForAnalysis['toapp']['filstartT'][0,0]
        cg.dataForAnalysis['SigPy']['timeEnd'] = cg.dataForAnalysis['toapp']['Teof'][0,0]
        cg.dataForAnalysis['SigPy']['timeBetweenSamples'] = 1 / cg.dataForAnalysis['SigPy']['sampleRate'][0,0]

        # cg.dataForAnalysis['SigPy']['eData'] = cg.dataForAnalysis['toapp']['edata'][0,0]

    # print("cg.sigData['eData'].shape: ", cg.sigData['eData'].shape)


def save_GEMS_SigPy_file(fileNameAndPath):

    # To overwrite original GEMS data, comment this out to save.
    cg.dataForAnalysis.pop('SigPy', None)
    cg.dataForAnalysis.pop('GEMSorig_toapp', None)
    cg.dataForAnalysis.pop('GEMSorig_bdfdef', None)

    # Save GEMS file
    sio.savemat(fileNameAndPath, cg.dataForAnalysis, appendmat = False)


def update_GEMS_data_with_TOAs(pos_np, nChans) :

    toaChanIndices = []
    toaChanTimeStamps = []

    iCount = 0
    chanNum = 0
    lastSampleChan = -1

    toaIndx = np.zeros(shape=nChans, dtype=object)
    toaCell = np.zeros(shape=nChans, dtype=object)

    print("making TOA data for GEMS")

    for sampleIndex, sampleChan in zip(pos_np[1], pos_np[0]) :

        if (sampleIndex > 0) :
            toaChanIndices.append(sampleIndex)
            timestamp = cg.dataForAnalysis['SigPy']['timeBetweenSamples'] * sampleIndex + cg.dataForAnalysis['SigPy']['timeStart']
            toaChanTimeStamps.append(round(timestamp[0][0],4))
        
        print("sampleIndex: ", sampleIndex)

        if not (int(sampleChan) == int(lastSampleChan)) and (lastSampleChan > -1):
            print("toaChanIndices: ", toaChanIndices)        
            print("toaChanTimeStamps: ", toaChanTimeStamps)   

            if (len(toaChanIndices) > 0) :
                toaIndx[lastSampleChan] = np.array(toaChanIndices).astype(dtype=float)
                toaCell[lastSampleChan] = np.array(toaChanTimeStamps).astype(dtype=float)
                

            toaChanIndices = []
            toaChanTimeStamps = []           

        lastSampleChan = sampleChan





    print("toaIndx: ", toaIndx)        
    print("toaCell: ", toaCell)  

    cg.dataForAnalysis['SigPy']['toaIndx'] = toaIndx
    cg.dataForAnalysis['SigPy']['toaCell'] = toaCell

    cg.dataForAnalysis['toapp']['toaIndx'][0,0] = toaIndx
    cg.dataForAnalysis['toapp']['toaCell'][0,0] = toaCell


 
