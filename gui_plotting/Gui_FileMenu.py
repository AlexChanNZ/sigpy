"""
    Author: Shameer Sathar
    Description: A module of processing the training data.
"""
import sys
sys.path.append('..')


from pyqtgraph.Qt import QtGui
from file_io.gems_sigpy import load_GEMS_mat_into_SigPy



class GuiFileMenu(QtGui.QMenuBar):

    def __init__(self, parent=None):
        '''
        Initialise dock window properties
        '''
        super(GuiFileMenu, self).__init__(parent)
        self.ui_menubar = QtGui.QMenuBar()
        self.menu = self.ui_menubar.addMenu('&File')
        self.add_menu_contents()



    def add_menu_contents(self):
        ## Load pacing file
        self.loadPacingAction = QtGui.QAction('&Load Pacing GEMS .mat', self)        
        self.loadPacingAction.setStatusTip('')

        self.menu.addAction(self.loadPacingAction)


        ## Load normal file
        self.loadNormalAction = QtGui.QAction('&Load Normal GEMS .mat', self)        
        self.loadNormalAction.setStatusTip('')

        self.menu.addAction(self.loadNormalAction)


        ## Save as gems file 
        self.saveAsAction = QtGui.QAction('&Save as GEMS .mat', self)        
        self.saveAsAction.setStatusTip('Save data with filename.')

        self.menu.addAction(self.saveAsAction)


        ## Save (update existing file)
        self.saveAction = QtGui.QAction('&Save', self)
        self.saveAction.setShortcut('Ctrl+S')
        self.saveAction.setStatusTip('Overwrite currently loaded file.')
        self.menu.addAction(self.saveAction)    


        ## Exit 
        self.quitAction = QtGui.QAction('Close', self)        
        self.quitAction.setStatusTip('Quit the program')
        self.quitAction.setShortcut('Ctrl+Q')

        self.menu.addAction(self.quitAction)




