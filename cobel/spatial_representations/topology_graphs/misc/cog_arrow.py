# basic imports
import pyqtgraph as qg
from PyQt5 import QtGui

### Helper class for the visualization of the topology graph
### Constructs a centered arrow pointing in a dedicated direction, inherits from 'ArrowItem'
class CogArrow(qg.ArrowItem):
    
    def set_data(self, x: float, y: float, angle: float):
        '''
        This function sets the position and direction of the arrow.
        
        Parameters
        ----------
        x :                                 The arrow center's x position.\n
        y :                                 The arrow center's y position.\n
        angle :                             The arrow's orientation in degrees.\n
        
        Returns
        ----------
        None
        '''
        # the angle has to be modified to suit the demands of the environment(?)
        #angle=-angle/np.pi*180.0+180.0
        angle = 180. - angle
        # assemble a new temporary dict that is used for path construction
        temp_opts = {}
        temp_opts['headLen'] = self.opts['headLen']
        temp_opts['tipAngle'] = self.opts['tipAngle']
        temp_opts['baseAngle'] = self.opts['baseAngle']
        temp_opts['tailLen'] = self.opts['tailLen']
        temp_opts['tailWidth'] = self.opts['tailWidth']
        # create the path
        arrow_path = qg.functions.makeArrowPath(**temp_opts)
        # identify boundaries of the arrows, required to shif the arrow
        bounds = arrow_path.boundingRect()
        # prepare a transform
        transform = QtGui.QTransform()
        # shift and rotate the path (arrow)
        transform.rotate(angle)
        transform.translate(int(-float(bounds.x())-float(bounds.width())/10.0*7.0),
                            int(float(-bounds.y())-float(bounds.height())/2.0))
        # 'remap' the path
        self.path = transform.map(arrow_path)
        self.setPath(self.path)
        # set position of the arrow
        self.setPos(x, y)
