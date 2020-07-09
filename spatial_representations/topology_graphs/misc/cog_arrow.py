
import numpy as np
import PyQt5 as qt
import pyqtgraph as qg

from PyQt5 import QtGui

### Helper class for the visualization of the topology graph
### Constructs a centered arrow pointing in a dedicated direction, inherits from 'ArrowItem'
class CogArrow(qg.ArrowItem):
    # set the position and direction of the arrow
    # x: x position of the arrow's center
    # y: y position of the arrow's center
    # angle: the orientation of the arrow
    
    def setData(self,x,y,angle):
        
        # the angle has to be modified to suit the demands of the environment(?)
        #angle=-angle/np.pi*180.0+180.0
        angle = 180.0 - angle
        # assemble a new temporary dict that is used for path construction
        tempOpts=dict()
        tempOpts['headLen']=self.opts['headLen']
        tempOpts['tipAngle']=self.opts['tipAngle']
        tempOpts['baseAngle']=self.opts['baseAngle']
        tempOpts['tailLen']=self.opts['tailLen']
        tempOpts['tailWidth']=self.opts['tailWidth']
        
        
        # create the path
        arrowPath=qg.functions.makeArrowPath(**tempOpts)
        # identify boundaries of the arrows, required to shif the arrow
        bounds=arrowPath.boundingRect()
        # prepare a transform
        transform=QtGui.QTransform()
        # shift and rotate the path (arrow)
        transform.rotate(angle)
        transform.translate(int(-float(bounds.x())-float(bounds.width())/10.0*7.0),int(float(-bounds.y())-float(bounds.height())/2.0))
        # 'remap' the path
        self.path=transform.map(arrowPath)
        self.setPath(self.path)
        # set position of the arrow
        self.setPos(x,y)
