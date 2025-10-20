# basic imports
import numpy as np
# framework imports
from .policy import Policy
# typing
from numpy.typing import NDArray
from numpy.random import Generator
from .policy import Action


class Softmax(Policy):
    """
    This class implements a softmax policy.

    Parameters
    ----------
    beta : float, default=1.
        The inverse temperature.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    beta : float
        The inverse temperature.

    """

    def __init__(self, beta: float = 1.0, rng: None | Generator = None) -> None:
        super().__init__(rng)
        assert beta > 0.0, 'Inverse temperature must be non-negative!'
        self.beta = beta

    def select_action(self, v: NDArray, mask: None | NDArray = None) -> Action:
        """
        This function selects an action for a given set of Q-values.

        Parameters
        ----------
        v : NDArray
            The Q-value(s).
        mask : NDArray or None, optional
            An optional action mask.

        Returns
        -------
        action : Action
            The selected action.
        """
        probs = self.get_action_probs(v, mask)

        return self.rng.choice(np.arange(v.shape[0]), p=probs)

    def get_action_probs(self, v: NDArray, mask: None | NDArray = None) -> NDArray:
        """
        This function computes the action selection probabilities
        for a given set of Q-values.

        Parameters
        ----------
        v : NDArray
            The Q-value(s).
        mask : NDArray or None, optional
            An optional action mask.

        Returns
        -------
        probs : NDArray
            The action selection probabilities.
        """
        values, actions, probs = np.copy(v), np.arange(v.shape[0]), np.zeros(v.shape)
        # remove masked actions
        if mask is not None:
            assert np.sum(mask) > 0, 'The action mask masks all actions!'
            values, actions = values[mask], actions[mask]
        # substract maximum value to prevent numerical problems
        values -= np.amax(values)
        # compute action selection probabilities
        probs[actions] = np.exp(values * self.beta)
        probs[actions] /= np.sum(probs[actions])

        return probs
