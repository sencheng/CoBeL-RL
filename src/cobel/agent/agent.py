# basic imports
import abc
import copy
import gymnasium as gym

# framework imports
from ..interface.interface import Interface

# typing
from typing import Any
from collections.abc import Callable
import numpy.typing as npt

Logs = dict[str, Any]
Callback = Callable[[Logs], None | Logs]
CallbackDict = dict[str, list[Callback]]


class Agent(abc.ABC):
    """
    Abstract class of an RL agent.

    Parameters
    ----------
    observation_space : gymnasium.spaces.Space
        The agent's observation space.
    action_space : gymnasium.spaces.Space
        The agent's action space.
    custom_callbacks : cobel.agent.agent.CallbackDict or None, optional
        The custom callbacks defined by the user.

    Attributes
    ----------
    observation_space : gymnasium.spaces.Space
        The agent's observation space.
    action_space : gymnasium.spaces.Space
        The agent's action space.
    callbacks : cobel.agent.agent.Callbacks
        The custom callbacks defined by the user.
    current_trial : int
        Tracks the current trial of the agent.
        Initialized to zero.
    stop : bool
        A flag used to prematurely stop the agent.
        Must be manually set by the user.

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        custom_callbacks: None | CallbackDict = None,
    ) -> None:
        self.observation_space: gym.Space = observation_space
        self.action_space: gym.Space = action_space
        self.callbacks: Callbacks = Callbacks(self, custom_callbacks)
        # trial count across all sessions (i.e., calls to train/test method)
        self.current_trial: int = 0
        # flag for stopping the agent prematurely
        self.stop: bool = False

    @abc.abstractmethod
    def train(self, interface: gym.Env | Interface, trials: int, steps: int) -> None:
        """
        Train the agent.

        Parameters
        ----------
        interface : gymnasium.Env or cobel.interface.interface.Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is trained.
        steps : int
            The maximum number of steps per trial.
        """
        pass

    @abc.abstractmethod
    def test(self, interface: gym.Env | Interface, trials: int, steps: int) -> None:
        """
        Test the agent.

        Parameters
        ----------
        interface : gymnasium.Env or cobel.interface.interface.Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is tested.
        steps : int
            The maximum number of steps per trial.
        """
        pass

    @abc.abstractmethod
    def predict_on_batch(self, batch: npt.ArrayLike) -> npt.NDArray:
        """
        Retrieve the Q-values for a batch of observations.

        Parameters
        ----------
        batch : ArrayLike
            The batch of observations for which Q-values should be retrieved.

        Returns
        -------
        predictions : numpy.ndarray
            The batch of Q-value predictions.
        """
        pass


class Callbacks:
    """
    The callback class. Used for monitoring, visualization and scenario control.

    Parameters
    ----------
    agent : cobel.agent.agent.Agent
        Reference to the RL agent.
    custom_callbacks : cobel.agent.agent.CallbackDict or None, optional
        The custom callbacks defined by the user.

    Attributes
    ----------
    agent : cobel.agent.agent.Agent
        Reference to the RL agent.
    custom_callbacks : cobel.agent.agent.CallbackDict
        The custom callbacks defined by the user.
        If none were provided in `custom_callbacks` it is
        set to an empty dictionary.

    """

    def __init__(
        self, agent: Agent, custom_callbacks: None | CallbackDict = None
    ) -> None:
        self.agent: Agent = agent
        self.custom_callbacks: CallbackDict
        if custom_callbacks is None:
            self.custom_callbacks = {}
        else:
            self.custom_callbacks = custom_callbacks

    def on_step_begin(self, logs: Logs) -> Logs:
        """
        Call whenever a step begins,
        and execute callbacks defined by the user.

        Parameters
        ----------
        logs : cobel.agent.agent.Logs
            The step log dictionary.

        Returns
        -------
        logs : cobel.agent.agent.Logs
            The updated step log dictionary.
        """
        step_logs: dict = copy.copy(logs)
        step_logs['agent'] = self.agent
        if 'on_step_begin' in self.custom_callbacks:
            for callback in self.custom_callbacks['on_step_begin']:
                callback_logs = callback(step_logs)
                if type(callback_logs) is dict:
                    step_logs.update(callback_logs)

        return step_logs

    def on_step_end(self, logs: Logs) -> Logs:
        """
        Call whenever a step ends,
        and execute callbacks defined by the user.

        Parameters
        ----------
        logs : cobel.agent.agent.Logs
            The step log dictionary.

        Returns
        -------
        logs : cobel.agent.agent.Logs
            The updated step log dictionary.
        """
        step_logs: dict = copy.copy(logs)
        step_logs['agent'] = self.agent
        if 'on_step_end' in self.custom_callbacks:
            for callback in self.custom_callbacks['on_step_end']:
                callback_logs = callback(step_logs)
                if type(callback_logs) is dict:
                    step_logs.update(callback_logs)

        return step_logs

    def on_trial_begin(self, logs: Logs) -> Logs:
        """
        Call whenever a trial begins,
        and execute callbacks defined by the user.

        Parameters
        ----------
        logs : cobel.agent.agent.Logs
            The trial log dictionary.

        Returns
        -------
        logs : cobel.agent.agent.Logs
            The updated trial log dictionary.
        """
        trial_logs: dict = copy.copy(logs)
        trial_logs['agent'] = self.agent
        if 'on_trial_begin' in self.custom_callbacks:
            for callback in self.custom_callbacks['on_trial_begin']:
                callback_logs = callback(trial_logs)
                if type(callback_logs) is dict:
                    trial_logs.update(callback_logs)

        return trial_logs

    def on_trial_end(self, logs: Logs) -> Logs:
        """
        Call whenever a trial ends,
        and execute callbacks defined by the user.

        Parameters
        ----------
        logs : cobel.agent.agent.Logs
            The trial log dictionary.

        Returns
        -------
        logs : cobel.agent.agent.Logs
            The updated trial log dictionary.
        """
        trial_logs: dict = copy.copy(logs)
        trial_logs['agent'] = self.agent
        if 'on_trial_end' in self.custom_callbacks:
            for callback in self.custom_callbacks['on_trial_end']:
                callback_logs = callback(trial_logs)
                if type(callback_logs) is dict:
                    trial_logs.update(callback_logs)

        return trial_logs
