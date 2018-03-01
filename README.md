# SigPy

SigPy is a python framework and GUI for using machine-learning methods to analyse gastro-intestinal (GI) electrophysiological data.GI recordings first undergo preprocessing steps including band-pass filtering and normalisation. Pre-labelled slow-wave activity is used to train a convolutional neural network (CNN). New data loaded into SigPy can then apply this CNN onto this new data and automate the marking process of GI slow-waves. Once the recordings have labelled the slow wave events, SigPy enables detailed analyses of slow wave dynamics including through the production of animations displaying the propagation of slow-waves. SigPy also allows exporting of data into a format compatible with the Gastro-intestinal Electrical Mapping Suite (GEMS) MATLAB toolbox to perform other slow wave related analyses.  

### Installing and running
pip install -r requirements.txt 

Run by: python main_SigPy.py

### Prerequisites
numpy>=1.13.1
PySide>=1.2.4
pyqtgraph>=0.10.0
scipy>=0.19.1
theano>=0.9.0
keras>=2.1.4
matplotlib>=2.0.2


## Built With
* [PyQt](https://riverbankcomputing.com/software/pyqt) - PyQt is a set of Python v2 and v3 bindings for The Qt Company's Qt application framework and runs on all platforms supported by Qt including Windows, OS X, Linux, iOS and Android.
* [PyQtGraph](http://www.pyqtgraph.org) - PyQtGraph is a pure-python graphics and GUI library built on PyQt4 / PySide and numpy. 
* [Theano](http://deeplearning.net/software/theano/) - Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.
* [Keras](https://keras.io/) - Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation.
* [Numpy](http://www.numpy.org) - NumPy is the fundamental package for scientific computing with Python.

## Authors
* **Shameer Sathar** - *Design, Conceptualisation, Algorithm Development, Visualisation Techniques* 
* **Terry Mayne** - *Bug Fixes, compatibility with GEMS* 
