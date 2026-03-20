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
    widget : pyqtgraph.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the monitor will be visualized.

    Attributes
    ----------
    widget : pyqtgraph.GraphicsLayoutWidget or None
        An optional widget. If provided the monitor will be visualized.

    """

    def __init__(self, widget: None | pg.GraphicsLayoutWidget = None) -> None:
        self.widget = widget

    @abc.abstractmethod
    def clear_plots(self) -> None:
        """Clear all plots."""
        pass

    @abc.abstractmethod
    def update(self, logs: Logs) -> None:
        """
        Update the monitor.

        Parameters
        ----------
        logs : cobel.agent.agent.Logs
            The log dictionary.
        """
        pass

    @abc.abstractmethod
    def get_trace(self) -> Trace:
        """
        Return the trace of the monitored variable.

        Returns
        -------
        trace : cobel.monitor.monitor.Trace
            The trace of the monitored variable.
        """
        pass

    def refresh_visualization(self) -> None:
        """Refresh the visualization."""
        if hasattr(qt, 'QtWidgets'):
            instance = qt.QtWidgets.QApplication.instance()
            assert type(instance) is qt.QtWidgets.QApplication
            instance.processEvents()
