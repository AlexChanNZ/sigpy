# -*- coding: utf-8 -*-
"""
Author : Shameer Sathar
"""

import numpy as np
import scipy.io as sio

import config_global as cg




def load_GEMS_file_for_analysis(fileNameAndPath):
    """
    The data is classified based on the training data.
    :param plot: Input values to be processed for generating features
    :return: predictions for the entire data set
    """

    cg.dataForAnalysis = sio.loadmat(fileNameAndPath) 


    # Check if this file has been saved from SigPy and whether it has been copied
    if hasattr(cg.dataForAnalysis, 'GEMSorig'):
        print("This file has been saved from SigPy")        
        print("This GEMS file, already has its original GEMs data duplicated.")
    else:
        #Duplicate original gems file and create SigPy structure
        cg.dataForAnalysis['GEMSorig_toapp'] = cg.dataForAnalysis['toapp'][0,0]
        cg.dataForAnalysis['GEMSorig_bdfdef'] = cg.dataForAnalysis['bdfdef'][0,0]        
        cg.dataForAnalysis['SigPy'] = {}
        cg.dataForAnalysis['SigPy']['filtData'] = cg.dataForAnalysis['toapp']['filtdata'][0,0]
        cg.dataForAnalysis['SigPy']['eData'] = cg.dataForAnalysis['toapp']['edata'][0,0]

    # print("cg.sigData['eData'].shape: ", cg.sigData['eData'].shape)


def save_GEMS_SigPy_file(fileNameAndPath):
    # Copy ATs to loaded GEMS file
    # cg.dataForAnalysis.pop('SigPy', None)
    # cg.dataForAnalysis.pop('GEMSorig', None)

    # Save GEMS file
    sio.savemat(fileNameAndPath, cg.dataForAnalysis, appendmat = False)


 
