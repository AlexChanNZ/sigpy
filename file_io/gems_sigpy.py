# -*- coding: utf-8 -*-
"""
Author : Shameer Sathar
"""
from __future__ import division # Correct Python 2 division

import numpy as np
import scipy.io as sio

import config_global as sp

from signal_processing.preprocessing import *

# np.set_printoptions(linewidth=1000, precision=3, threshold=np.inf)



def load_GEMS_mat_into_SigPy(fileNameAndPath, isNormal):
    """
    The data is classified based on the training data.
    :param plot: Input values to be processed for generating features
    :return: predictions for the entire data set
    """

    print("Load GEMS file to sigpy...")
    sp.dat = sio.loadmat(fileNameAndPath, matlab_compatible=True) #added matlab_compatible=True to aid GEMS compatibility
    # Check if this file has been saved from SigPy and whether it has been copied
    if hasattr(sp.dat, 'GEMSorig_toapp'):

        print("This file has been saved from SigPy")        
        print("This GEMS file, already has its original GEMs data duplicated.")

    else:
        #Duplicate original gems file and create SigPy structure
        sp.dat['GEMSorig_toapp'] = np.array(sp.dat['toapp'][0,0])
    
    sp.dat['SigPy'] = {}
    sp.dat['SigPy']['dataFilt'] = np.array(sp.dat['toapp']['filtdata'][0,0])

    print("Normalising plotting data ...")

    sp.dat['SigPy']['dataToPlot'] =  preprocess(sp.dat['SigPy']['dataFilt'])
    
    if not isNormal :

        sp.dat['SigPy']['dataIsNormal'] = 0
        print("Processing pacing data")

        sp.dat['SigPy']['MarkersPacing'], sp.dat['SigPy']['dataPacingCleaned'] = clean_pacing(sp.dat['SigPy']['dataFilt'])

        print("Normalising data for marking ...")

        sp.dat['SigPy']['dataForMarking'] = preprocess(sp.dat['SigPy']['dataPacingCleaned'])

    else:

        sp.dat['SigPy']['dataIsNormal'] = 1
        sp.dat['SigPy']['dataForMarking'] = sp.dat['SigPy']['dataToPlot']


    sp.dat['SigPy']['bdfdef'] = sp.dat['bdfdef'][0,0]    
    sp.dat['SigPy']['sampleRate'] = sp.dat['toapp']['fs'][0,0]
    sp.dat['SigPy']['timeStart'] = sp.dat['toapp']['filstartT'][0,0]
    sp.dat['SigPy']['timeEnd'] = sp.dat['toapp']['Teof'][0,0]
    sp.dat['SigPy']['timeBetweenSamples'] = 1 / sp.dat['SigPy']['sampleRate'][0,0]

    sp.dat['toapp']['showchans'][0,0] = np.array(sp.dat['toapp']['showchans'][0,0]).astype(dtype=float)
    sp.dat['toapp']['orientedElec'][0,0] = np.array(sp.dat['toapp']['orientedElec'][0,0]).astype(dtype=float)

    sp.dat['SigPy']['gridMap'] = sp.dat['toapp']['orientedElec'][0,0].astype(int)
    sp.dat['SigPy']['chanNums'] = sp.dat['toapp']['showchans'][0,0][0].astype(int)


    # sp.dat = sio.loadmat(fileNameAndPath, mat_dtype=False) 

    # print('sp.dat[toapp][orientedElec]:', sp.dat['toapp']['orientedElec'][0,0]

    # sp.dat['SigPy']['eData'] = sp.dat['toapp']['edata'][0,0]

    # print("sp.sigData['eData'].shape: ", sp.sigData['eData'].shape)


def save_GEMS_SigPy_file(fileNameAndPath):

    # To overwrite original GEMS data, comment this out to save GEMS data as backup.
    sp.dat.pop('SigPy', None)
    sp.dat.pop('GEMSorig_toapp', None)
    sp.dat.pop('GEMSorig_bdfdef', None)
    # sp.dat.pop('bdfdef', None) #popping bdfdef because of UI control compatibility. 
    # EDIT: Appears saving UIControl component as struct still works in GEMS.

    # Save GEMS file
    sio.savemat(fileNameAndPath, sp.dat)


def update_GEMS_data_with_TOAs(pos_np, nChans) :
    print("Updating GEMS data with TOAs ...")
    # sp.gui.statBar.showMessage("Updating GEMS data with TOAs ...")

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


    ## Iterate through indices and chans and convert to ToA and GEMS compatible indexes
    for sampleIndex, sampleChan in zip(pos_np[1], pos_np[0]) :
        
        if not (int(sampleChan) == int(lastSampleChan)) and (lastSampleChan > -1):  

            if (len(toaChanIndices) > 0) :

                toaIndx[lastSampleChan] = np.array(toaChanIndices).astype(dtype=float)
                toaCell[lastSampleChan] = np.array(toaChanTimeStamps).astype(dtype=float)

            toaChanIndices = []
            toaChanTimeStamps = []


        if (sampleIndex > 0) :

            toaChanIndices.append(sampleIndex)
            timestamp = sp.dat['SigPy']['timeBetweenSamples'] * sampleIndex + sp.dat['SigPy']['timeStart']
            toaChanTimeStamps.append(round(timestamp[0][0],4))       
            
        lastSampleChan = sampleChan


    sp.dat['SigPy']['toaIndx'] = toaIndx
    sp.dat['SigPy']['toaCell'] = toaCell
    
    sp.dat['toapp']['toaIndx'][0,0] = toaIndx
    sp.dat['toapp']['toaCell'][0,0] = toaCell


 
