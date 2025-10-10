# basic imports
import numpy as np
import gymnasium as gym
# framework imports
from .agent import Agent
from ..memory.adqn import ADQNMemory, Experience
from ..network.network import Network
from ..interface.interface import Interface
# typing
from numpy.typing import NDArray, ArrayLike
from .agent import CallbackDict

StateBatch = NDArray | list[NDArray] | dict[str, NDArray]


class ADQN(Agent):
    """
    This class implements a simplified DQN agent which learns US-CS associations.

    Parameters
    ----------
    observation_space : gym.Space
        The agent's observation space.
    model : Network
        The network model used by the agent.
    memory : DQNMemory or None, optional
        The memory module used for storing experiences.
    custom_callbacks : CallbackDict or None, optional
        The custom callbacks defined by the user.

    Attributes
    ----------
    observation_space : gym.Space
        The agent's observation space.
    action_space : gym.Space
        The agent's action space.
    model : Network
        The agent's network.
    M : PrioritizedMemory
        The memory module used for storing experiences.
        If none was provided an instance with default
        parameters is created.
    current_trial : int
        Tracks the current trial.
    stop : bool
        A flag that can be used to manually stop training/testing of the agent.
        False by default.

    Notes
    -----
    This agent only supports gym.spaces.Box for `observation_space`.

    Examples
    --------

    Here we initialize the ADQN agent for unimodal
    observations. ::

        >>> import gymnasium as gym
        >>> import torch.nn as nn
        >>> from collections import OrderedDict
        >>> from cobel.agent import ADQN
        >>> from cobel.network import TorchNetwork
        >>> from layers = [('dense', nn.Linear(10, 64)),
        ...                ('relu', nn.ReLU()),
        ...                ('output', nn.Linear(64, 1))]
        >>> agent = ADQN(
        ...         gym.spaces.Box(0,, 1., (6, )),
        ...         TorchNetwork(nn.Sequential(OrderedDict(layers)))
        ...         )


    """

    def __init__(
        self,
        observation_space: gym.Space,
        model: Network,
        memory: None | ADQNMemory = None,
        custom_callbacks: None | CallbackDict = None,
    ) -> None:
        super().__init__(
            observation_space,
            gym.spaces.Box(-np.inf, np.inf, (1,), np.float64),
            custom_callbacks,
        )
        self.model = model
        self.memory = ADQNMemory(observation_space) if memory is None else memory
        self.current_trial = 0
        self.stop = False

    def train(
        self,
        interface: gym.Env | Interface,
        trials: int,
        steps: int,
        batch_size: int = 32,
        nb_replays: int = 1,
    ) -> None:
        """
        This function is called to train the agent.

        Parameters
        ----------
        interface : gym.Env or Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is trained.
        steps : int
            The maximum number of steps per trial.
        batch_size : int, default=32
            The batch size used for experience replay.
        nb_replays : int, default=1
            The number of replays per trial.
        """
        for trial in range(trials):
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0.0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                }
            )
            state, _ = interface.reset()
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                assert isinstance(state, (np.ndarray | list | dict))
                action = float(self.retrieve_v(state)[0])
                next_state, reward, end_trial, _, _ = interface.step(action)
                # update Q-function and store experience
                assert isinstance(next_state, (np.ndarray | list | dict))
                experience: Experience = {
                    'state': state,
                    'action': action,
                    'reward': float(reward),
                    'next_state': next_state,
                    'terminal': (1 - end_trial),
                }
                self.memory.store(experience)
                # update current state
                state = next_state
                # experience replay
                self.replay(batch_size, nb_replays)
                # update cumulative reward
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['step'] = step
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def test(self, interface: gym.Env | Interface, trials: int, steps: int) -> None:
        """
        This function is called to test the agent.

        Parameters
        ----------
        interface : gym.Env or Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is tested.
        steps : int
            The maximum number of steps per trial.
        """
        for trial in range(trials):
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0.0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                }
            )
            state, _ = interface.reset()
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                assert isinstance(state, (np.ndarray | list | dict))
                action = float(self.retrieve_v(state)[0])
                next_state, reward, end_trial, _, _ = interface.step(action)
                # update Q-function and store experience
                assert isinstance(next_state, (np.ndarray | list | dict))
                experience: Experience = {
                    'state': state,
                    'action': action,
                    'reward': float(reward),
                    'next_state': next_state,
                    'terminal': (1 - end_trial),
                }
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['step'] = step
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def replay(self, batch_size: int = 32, nb_replays: int = 1) -> None:
        """
        This function replays experiences to update the Q-function.

        Parameters
        ----------
        batch_size : int, default=32
            The number of experiences that will be replayed.
        nb_replays : int, default=1
            The number of replays per trial.
        """
        assert batch_size > 0
        assert nb_replays > 0
        observations, targets = self.memory.sample_batch(batch_size)
        for _ in range(nb_replays):
            self.model.train_on_batch(observations, targets)

    def retrieve_v(self, state: StateBatch) -> NDArray:
        """
        This function retrieves the value for a given observation.
        """
        if isinstance(state, np.ndarray):
            return self.predict_on_batch(np.array([state]))
        if isinstance(state, list):
            return self.predict_on_batch([np.array([o]) for o in state])
        assert isinstance(state, dict)
        return self.predict_on_batch({m: np.array([o]) for m, o in state.items()})

    def predict_on_batch(self, batch: ArrayLike | StateBatch) -> NDArray:
        """
        This function retrieve the values for a batch of observations.

        Parameters
        ----------
        batch: ArrayLike
            The batch of observations for which values should be retrieved.

        Returns
        -------
        predictions: NDArray
            The batch of predictions.
        """
        assert isinstance(batch, (np.ndarray | dict | list))
        predictions = self.model.predict_on_batch(batch)
        assert isinstance(predictions, np.ndarray)

        return predictions
