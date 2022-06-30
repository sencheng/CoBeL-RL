# basic imports
import numpy as np
import PyQt5 as qt
import pyqtgraph as qg
from PyQt5 import QtGui


class CogArrow(qg.ArrowItem):
    '''
    Constructs a centered arrow pointing in a dedicated direction, inherits from 'ArrowItem'.
    '''
    
    def setData(self, x, y, angle):
        '''
        Set the position and direction of the arrow.
        
        | **Args**
        | x:                            The arrow's center's x position.
        | y:                            The arrow's center's y position.
        | angle:                        The arrow's orientation.
        '''
        # assemble a new temporary dict that is used for path construction
        tempOpts = dict()
        tempOpts['headLen'] = self.opts['headLen']
        tempOpts['tipAngle'] = self.opts['tipAngle']
        tempOpts['baseAngle'] = self.opts['baseAngle']
        tempOpts['tailLen'] = self.opts['tailLen']
        tempOpts['tailWidth'] = self.opts['tailWidth']
        # create the path
        arrowPath = qg.functions.makeArrowPath(**tempOpts)
        # identify boundaries of the arrows, required to shif the arrow
        bounds = arrowPath.boundingRect()
        # prepare a transform
        transform = QtGui.QTransform()
        # shift and rotate the path (arrow)
        transform.rotate(angle)
        transform.translate(int(-float(bounds.x())-float(bounds.width())/10.0*7.0),
                            int(float(-bounds.y())-float(bounds.height())/2.0))
        # 'remap' the path
        self.path = transform.map(arrowPath)
        self.setPath(self.path)
        # set position of the arrow
        self.setPos(x, y)