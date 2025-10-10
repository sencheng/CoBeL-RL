# basic imports
import pyqtgraph as pg  # type: ignore
from PyQt6 import QtGui


class CogArrow(pg.ArrowItem):
    def set_data(self, x: float, y: float, orientation: float) -> None:
        """
        This function sets the position and orientation of the arrow.

        Parameters
        ----------
        x : float
            The arrow's new x position.
        y : float
            The arrow's new y position.
        orientation : float
            The arrow's new orientation.
        """
        # prepare arrow params
        params = {
            p: self.opts[p]
            for p in ['headLen', 'tipAngle', 'baseAngle', 'tailLen', 'tailWidth']
        }
        # create arrow path
        arrow_path = pg.functions.makeArrowPath(**params)
        # get arrow boundaries
        bounds = arrow_path.boundingRect()
        # prepare transform
        transform = QtGui.QTransform()
        # shift and rotate arrow path
        transform.rotate(orientation)
        transform.translate(
            int(-float(bounds.x()) - float(bounds.width()) / 10 * 7),
            int(float(-bounds.y()) - float(bounds.height()) / 2),
        )
        self.path = transform.map(arrow_path)
        self.setPath(self.path)
        self.setPos(x, y)
