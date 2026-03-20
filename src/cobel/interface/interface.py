# basic imports
import abc
import numpy as np
import pyqtgraph as pg  # type: ignore

# framework imports
# typing
from typing import Any
from numpy.typing import NDArray

Observation = NDArray | int | float | dict[str, NDArray] | list[NDArray]
Action = NDArray | int | float
StepTuple = tuple[Observation, float, bool, bool, dict[str, Any]]
ResetTuple = tuple[Observation, dict[str, Any]]


class Interface(abc.ABC):
    """
    The abstract interface class.

    Parameters
    ----------
    widget : pyqtgraph.GraphicsLayoutWidget or None, Optional
        An optional widget. If provided the environment will be visualized.

    Attributes
    ----------
    widget : pyqtgraph.GraphicsLayoutWidget or None
        An optional widget. If provided the environment will be visualized.

    """

    def __init__(self, widget: None | pg.GraphicsLayoutWidget = None) -> None:
        self.widget = widget

    @abc.abstractmethod
    def step(self, action: Action) -> StepTuple:
        """
        Perform one simulation step in the environment
        (compatible with Gymnasium's step function).

        Parameters
        ----------
        action : cobel.interface.interface.Action
            The action selected by the agent.

        Returns
        -------
        observation : cobel.interface.interface.Observation
            The observation of the new current state.
        reward : float
            The reward received.
        end_trial : bool
            A flag indicating whether the trial ended.
        truncated : bool
            A flag required by Gymnasium (not used).
        logs : dict of Any
            The (empty) logs dictionary.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> ResetTuple:
        """
        Reset the environment (compatible with Gymnasium's reset function).

        Returns
        -------
        observation : cobel.interface.interface.Observation
            The observation of the new current state.
        logs : dict of Any
            The (empty) logs dictionary.
        """
        pass

    @abc.abstractmethod
    def get_position(self) -> NDArray:
        """
        Return the agent's position in the environment.

        Returns
        -------
        position : numpy.ndarray
            A numpy array containing the agent's position.
        """
        return np.array([])
