# basic imports
import abc
from numpy.random import Generator, default_rng

# typing
from numpy.typing import NDArray

Action = NDArray | int | float


class Policy(abc.ABC):
    """
    Abstract class of an action selection policy.

    Parameters
    ----------
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    rng : numpy.random.Generator
        A random number generator instance used for
        probablistic action selection.

    """

    def __init__(self, rng: None | Generator = None) -> None:
        self.rng = default_rng() if rng is None else rng

    @abc.abstractmethod
    def select_action(self, v: NDArray, mask: None | NDArray = None) -> Action:
        """
        Select an action for a given set of Q-values.

        Parameters
        ----------
        v : numpy.ndarray
            The Q-value(s).
        mask : numpy.ndarray or None, optional
            An optional action mask.

        Returns
        -------
        action : cobel.interface.interface.Action
            The selected action.
        """
        pass

    @abc.abstractmethod
    def get_action_probs(self, v: NDArray, mask: None | NDArray = None) -> NDArray:
        """
        Compute the action selection probabilities for a given set of Q-values.

        Parameters
        ----------
        v : numpy.ndarray
            The Q-value(s).
        mask : numpy.ndarray or None, optional
            An optional action mask.

        Returns
        -------
        probs : numpy.ndarray
            The action selection probabilities.
        """
