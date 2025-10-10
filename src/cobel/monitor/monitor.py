# basic imports
import abc
import PyQt6 as qt
import pyqtgraph as pg  # type: ignore
# typing
from numpy.typing import NDArray
from ..agent.agent import Logs

Trace = NDArray | dict[str, NDArray] | list


class Monitor(abc.ABC):
    """
    Abstract monitor class.

    Parameters
    ----------
    widget : pg.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the monitor will be visualized.

    Attributes
    ----------
    widget : pg.GraphicsLayoutWidget or None
        An optional widget. If provided the monitor will be visualized.

    """

    def __init__(self, widget: None | pg.GraphicsLayoutWidget = None) -> None:
        self.widget = widget

    @abc.abstractmethod
    def clear_plots(self) -> None:
        """
        This function clears all plots.
        """
        pass

    @abc.abstractmethod
    def update(self, logs: Logs) -> None:
        """
        This function updates the monitor.

        Parameters
        ----------
        logs : Logs
            The log dictionary.
        """
        pass

    @abc.abstractmethod
    def get_trace(self) -> Trace:
        """
        This function returns the trace of the monitored variable.

        Returns
        -------
        trace : Trace
            The trace of the monitored variable.
        """
        pass

    def refresh_visualization(self) -> None:
        """
        This refreshes the visualization.
        """
        if hasattr(qt, 'QtWidgets'):
            instance = qt.QtWidgets.QApplication.instance()
            assert type(instance) is qt.QtWidgets.QApplication
            instance.processEvents()
