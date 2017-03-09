"""
    Author: Shameer Sathar
    Description: A module of functions for easy configuration.
"""

import os

# Set default files names
training_file_name = 'training_data.arff'
test_file_name = 'test_data.arff'
loaded_data_file = 'junk.mat'

current_working_directory = os.getcwd()


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






