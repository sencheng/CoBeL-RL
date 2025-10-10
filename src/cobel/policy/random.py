# basic imports
import numpy as np
# framework imports
from .policy import Policy
# typing
from numpy.typing import NDArray
from numpy.random import Generator
from .policy import Action


class RandomDiscrete(Policy):
    """
    This class implements a random action policy.

    Parameters
    ----------
    actions : int
        The probability of taking a random action.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    actions : int
        The probability of taking a random action.

    """

    def __init__(self, actions: int, rng: None | Generator = None) -> None:
        super().__init__(rng)
        assert actions > 0, 'There must be at least one action!'
        self.actions = actions

    def select_action(self, v: NDArray, mask: None | NDArray = None) -> Action:
        """
        This function returns a random discrete action.

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
        return int(self.rng.integers(self.actions))

    def get_action_probs(self, v: NDArray, mask: None | NDArray = None) -> NDArray:
        """
        This function computes the action selection
        probabilities for a given set of Q-values.

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
        return np.full(self.actions, 1.0 / self.actions)


class RandomUniform(Policy):
    """
    This class implements a random continuous (uniform) policy.

    Parameters
    ----------
    action_min : float or tuple of float
        The minimum value(s) of the action(s).
    action_max : float or tuple of float
        The maximum value(s) of the action(s).
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    action_min : float or tuple of float
        The minimum value(s) of the action(s).
    action_max : float or tuple of float
        The maximum value(s) of the action(s).

    """

    def __init__(
        self,
        action_min: float | tuple[float, ...],
        action_max: float | tuple[float, ...],
        rng: None | Generator = None,
    ) -> None:
        super().__init__(rng)
        assert type(action_min) == type(action_max), 'Limits must share the same type!'
        if type(action_min) == tuple and type(action_max) == tuple:
            assert len(action_min) == len(action_max), 'Limits mismatch!'
        self.action_min = action_min
        self.action_max = action_max

    def select_action(self, v: NDArray, mask: None | NDArray = None) -> Action:
        """
        This function returns a random continuous (uniform) action.

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
        return self.rng.uniform(self.action_min, self.action_max)

    def get_action_probs(self, v: NDArray, mask: None | NDArray = None) -> NDArray:
        """
        This function computes the action selection
        probabilities for a given set of Q-values.

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
        return np.array([], dtype='float')


class RandomGaussian(Policy):
    """
    This class implements a random continuous (gaussian) policy.

    Parameters
    ----------
    mean : float or tuple of float
        The mean value(s) of the gaussian(s).
    std : float or tuple of float
        The standard deviation(s) of the gaussian(s).
    action_min : float, tuple of float or None, optional
        The minimum value(s) of the action(s).
    action_max : float, tuple of float or None, optional
        The maximum value(s) of the action(s).
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    mean : float or tuple of float
        The mean value(s) of the gaussian(s).
    std : float or tuple of float
        The standard deviation(s) of the gaussian(s).
    action_min : float, tuple of float or None
        The minimum value(s) of the action(s).
    action_max : float, tuple of float or None
        The maximum value(s) of the action(s).

    """

    def __init__(
        self,
        mean: float | tuple[float, ...],
        std: float | tuple[float, ...],
        action_min: None | float | tuple[float, ...] = None,
        action_max: None | float | tuple[float, ...] = None,
        rng: None | Generator = None,
    ) -> None:
        super().__init__(rng)
        assert type(mean) == type(std), (
            'Mean and standard deviation must share the same type!'
        )
        self.mean = mean
        self.std = std
        self.action_min = action_min
        self.action_max = action_max

    def select_action(self, v: NDArray, mask: None | NDArray = None) -> Action:
        """
        This function returns a random continuous (gaussian) action.

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
        if self.action_max is None and self.action_min is None:
            return self.rng.normal(self.mean, self.std)

        return np.clip(
            self.rng.normal(self.mean, self.std),
            a_min=self.action_min,
            a_max=self.action_max,
        )

    def get_action_probs(self, v: NDArray, mask: None | NDArray = None) -> NDArray:
        """
        This function computes the action selection
        probabilities for a given set of Q-values.

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
        return np.array([], dtype='float')
