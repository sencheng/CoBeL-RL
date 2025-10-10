# basic imports
import numpy as np
# framework imports
from .policy import Policy
# typing
from numpy.typing import NDArray
from numpy.random import Generator
from .policy import Action


class Proportional(Policy):
    """
    This class implements a policy which transforms a scalar value to a binary action.
    Action selection probabilities are proportional to the scalar value.

    Parameters
    ----------
    value_max : float, default=1.
        The maximum possible value. Used for normalization.
    code_reverse : bool, default=True
        Flag indicating whether value close to the maximum value code
        for action zero, i.e., the first action. True by default.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    value_max : float
        The maximum possible value. Used for normalization.
    code_reverse : bool
        Flag indicating whether value close to the maximum value code
        for action zero, i.e., the first action. True by default.

    """

    def __init__(
        self,
        value_max: float = 1.0,
        code_reverse: bool = True,
        rng: None | Generator = None,
    ) -> None:
        super().__init__(rng)
        self.value_max = value_max
        self.code_reverse = code_reverse

    def select_action(self, v: NDArray, mask: None | NDArray = None) -> Action:
        """
        This function selects an action for a given value.

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
        return abs(self.code_reverse - int(self.rng.random() < v / self.value_max))

    def get_action_probs(self, v: NDArray, mask: None | NDArray = None) -> NDArray:
        """
        This function computes the action selection
        probabilities for a given value.

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
        probs = np.abs(np.array([1.0, 0.0]) - v / self.value_max)
        if self.code_reverse:
            return np.flip(probs)

        return probs


class Threshold(Policy):
    """
    This class implements a threshold policy which transfrms a
    scalar value to a binary action.

    Parameters
    ----------
    threshold : float, default=0.5
        The action threshold.
    window : float, default=0.
        The window for which random actions will be taken.
    value_max : float, default=1.
        The maximum possible value. Used for normalization.
    code_reverse : bool, default=True
        Flag indicating whether value close to the maximum value code
        for action zero, i.e., the first action. True by default.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    threshold : float
        The action threshold.
    window : float
        The window for which random actions will be taken.
    value_max : float
        The maximum possible value. Used for normalization.
    code_reverse : bool
        Flag indicating whether value close to the maximum value code
        for action zero, i.e., the first action. True by default.

    """

    def __init__(
        self,
        threshold: float = 0.5,
        window: float = 0.0,
        value_max: float = 1.0,
        code_reverse: bool = True,
        rng: None | Generator = None,
    ) -> None:
        super().__init__(rng)
        assert threshold >= 0.0 and threshold <= 1, (
            'Threshold must lie within the interval (0, 1)!'
        )
        assert threshold - window / 2 > 0.0 and threshold + window / 2 < 1.0, (
            'The window for random actions extends over the value range!'
        )
        self.threshold = threshold
        self.window = window / 2
        self.value_max = value_max
        self.code_reverse = code_reverse

    def select_action(self, v: NDArray, mask: None | NDArray = None) -> Action:
        """
        This function selects an action for a given value.

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
        v /= self.value_max
        action = abs(int(self.code_reverse) - int(v > self.threshold))
        if v > self.threshold - self.window and v < self.threshold + self.window:
            action = int(self.rng.integers(2))

        return action

    def get_action_probs(self, v: NDArray, mask: None | NDArray = None) -> NDArray:
        """
        This function computes the action selection
        probabilities for a given value.

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
        v /= self.value_max
        probs = np.zeros(2)
        probs[int(self.code_reverse) - int(v > self.threshold)] = 1.0
        if v > self.threshold - self.window and v < self.threshold + self.window:
            probs.fill(0.5)

        return probs


class Sigmoid(Policy):
    """
    This class implements a threshold policy which transfrms a
    scalar value to a binary action. Action selection probabilities
    are proportional to the sigmoid transformed scalar value.

    Parameters
    ----------
    threshold : float, default=0.5
        The action threshold, i.e., P(v) = 0.5.
    scale : float, default=10.
        The scaling factor for the sigmoid's steepness.
    value_max : float, default=1.
        The maximum possible value. Used for normalization.
    code_reverse : bool, default=True
        Flag indicating whether value close to the maximum value code
        for action zero, i.e., the first action. True by default.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    threshold : float
        The action threshold, i.e., P(v) = 0.5.
    scale : float
        The scaling factor for the sigmoid's steepness.
    value_max : float
        The maximum possible value. Used for normalization.
    code_reverse : bool
        Flag indicating whether value close to the maximum value code
        for action zero, i.e., the first action. True by default.

    """

    def __init__(
        self,
        threshold: float = 0.5,
        scale: float = 10.0,
        value_max: float = 1.0,
        code_reverse: bool = True,
        rng: None | Generator = None,
    ) -> None:
        super().__init__(rng)
        assert threshold >= 0.0 and threshold <= 1, (
            'Threshold must lie within the interval (0, 1)!'
        )
        assert scale >= 0.0, "The sigmoid's scaling factor must be non-negative!"
        self.threshold = threshold
        self.scale = scale
        self.value_max = value_max
        self.code_reverse = code_reverse

    def select_action(self, v: NDArray, mask: None | NDArray = None) -> Action:
        """
        This function selects an action for a given value.

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
        prob = 1 / (1 + np.exp(-(v / self.value_max - self.threshold) * self.scale))

        return abs(self.code_reverse - int(self.rng.random() < prob))

    def get_action_probs(self, v: NDArray, mask: None | NDArray = None) -> NDArray:
        """
        This function computes the action selection
        probabilities for a given value.

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
        probs = np.abs(
            np.array([1.0, 0.0])
            - 1 / (1 + np.exp(-(v / self.value_max - self.threshold) * self.scale))
        )
        if self.code_reverse:
            return np.flip(probs)

        return probs
