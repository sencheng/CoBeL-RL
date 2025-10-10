# basic imports
import numpy as np
# framework imports
from .policy import Policy
# typing
from numpy.typing import NDArray
from numpy.random import Generator
from .policy import Action


class EpsilonGreedy(Policy):
    """
    This class implements an epsilon greedy-policy.

    Parameters
    ----------
    epsilon : float, default=0.1
        The probability of taking a random action.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    epsilon : float
        The probability of taking a random action.

    """

    def __init__(self, epsilon: float = 0.1, rng: None | Generator = None) -> None:
        super().__init__(rng)
        assert epsilon >= 0.0 and epsilon <= 1.0
        self.epsilon = epsilon

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
        # base action selection probability
        probs[actions] = np.full(values.shape, self.epsilon / values.shape[0])
        # greedy selection probability considering tied actions
        ties = np.amax(values) == values
        probs[actions] += (1.0 - self.epsilon) * ties / np.sum(ties)

        return probs


class ExclusiveEpsilonGreedy(EpsilonGreedy):
    """
    This class implements an epsilon-greedy policy which
    excludes the greedy action from exploration.

    Parameters
    ----------
    epsilon : float, default=0.1
        The probability of taking a random action.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    """

    def __init__(self, epsilon: float = 0.1, rng: None | Generator = None) -> None:
        super().__init__(epsilon, rng)

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
        # greedy selection probability considering tied actions
        ties = np.amax(values) == values
        probs[actions] = (1.0 - self.epsilon) * ties / np.sum(ties)
        # exploration probability
        probs[actions] += (
            self.epsilon * (ties == False) / max(values.shape[0] - np.sum(ties), 1)
        )

        return probs
