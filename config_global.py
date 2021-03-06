"""
    Author: Shameer Sathar
    Description: A module of functions for easy configuration.
"""

import os
import numpy as np


# Set default files names

current_working_directory = os.getcwd()

dataRoot = current_working_directory + '/data/'
nnPath = current_working_directory + '/ml_models/'
graphicsPath = current_working_directory + '/graphics/'

userDataStore = '/Users/m/_/Data/SigPy/'

dataForAnalysis = None # Raw import of data (likely a GEMS .mat file)
dataForAnalysisFileName = None



sigData = {} # Data in format for analysis within SigPy


def set_training_file_name(new_name):
    global training_file_name
    training_file_name = new_name


def set_trained_file(new_name):
    global trained_file
    trained_file = new_name


def get_trained_file():
    return trained_file


def set_test_file_name(new_name):
    global test_file_name
    test_file_name = new_name


def set_data_file_name(new_name):
    global loaded_data_file
    loaded_data_file = new_name

def set_data_for_analysis(data):
    global dataForAnalysis
    dataForAnalysis = data

def set_data_for_analysis_fileName(new_name):
    global dataForAnalysisFileName
    dataForAnalysisFileName = new_name  

def set_data_for_analysis_fileName(new_name):
    global dataForAnalysisFileName
    dataForAnalysisFileName = new_name    






