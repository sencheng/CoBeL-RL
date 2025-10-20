# basic imports
import numpy as np
import pyqtgraph as pg  # type: ignore
# framework imports
from .monitor import Monitor
# typing
from numpy.typing import NDArray
from ..agent.agent import Logs
from .monitor import Trace
from ..network.network import Batch


class RepresentationMonitor(Monitor):
    """
    This class implements a monitor which tracks the activity
    of a specified network layer for a set of observations.

    Parameters
    ----------
    model : str
        The name of the model belonging to the agent.
    layer : int or str
        The index or name of the layer for which activity will be tracked.
    observations : Batch
        The batch of observations for which activity will be tracked.
    interval : int or tuple of int, default=0
        The interval (calls to the update function)
        after which activity will be recorded.
        Use a single integer to define a simple interval
        or provide a tuple of trials at which activity will be recorded.
    widget : pg.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the monitor will be visualized.

    Attributes
    ----------
    model : str
        The name of the model belonging to the agent.
    layer : int or str
        The index or name of the layer for which activity will be tracked.
    observations : Batch
        The batch of observations for which activity will be tracked.
    interval : int, default=0
        The interval (calls to the update function)
        after which activity will be recorded.
        Use a single integer to define a simple interval
        or provide a tuple of trials at which activity will be recorded.
    last_update : int
        Tracks when the monitor was last updated.
    activity_trace : list of NDArray
        The network activity trace.

    Examples
    --------

    Setting up the representation monitor to track representations in
    the DQN's online network's penultimate layer (callbacks are assumed
    to have been passed to the agent). ::

        >>> from cobel.monitor import RepresentationMonitor
        >>> observations = np.random.rand(32, 64) # some random inputs
        >>> monitor = RepresentationMonitor('model_online', -2, observations)
        >>> callbacks = {'on_trial_end': [monitor.update]}

    """

    def __init__(
        self,
        model: str,
        layer: int | str,
        observations: Batch,
        interval: int | tuple[int, ...] = 0,
        widget: None | pg.GraphicsLayoutWidget = None,
    ) -> None:
        super().__init__(widget)
        self.last_update = 1
        if type(interval) is int:
            assert interval >= 0
            self.last_update += interval
        elif type(interval) is tuple:
            assert min(interval) >= 0
        self.interval: int | tuple[int, ...] = interval
        self.layer: int | str = layer
        self.model = model
        self.observations = observations
        self.activity_trace: list[NDArray] = []

    def clear_plots(self) -> None:
        """
        This function clears all plots.
        """
        pass

    def update(self, logs: Logs) -> None:
        """
        This function updates the monitor.

        Parameters
        ----------
        logs : Logs
            The log dictionary.
        """
        record: bool = False
        if type(self.interval) is int:
            record = self.last_update >= self.interval
            self.last_update = 0
        elif type(self.interval) is tuple:
            record = logs['trial'] in self.interval
        if record:
            assert hasattr(logs['agent'], self.model)
            activity = getattr(logs['agent'], self.model).get_layer_activity(
                self.observations, self.layer
            )
            self.activity_trace.append(activity)
        self.last_update += 1

    def get_trace(self) -> Trace:
        """
        This function returns the layer activity trace.

        Returns
        -------
        trace : Trace
            The trace of the monitored variable.
        """
        return np.array(self.activity_trace)
