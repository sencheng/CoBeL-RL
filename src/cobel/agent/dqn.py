# basic imports
import numpy as np
import gymnasium as gym
# framework imports
from .agent import Agent
from ..policy.policy import Policy
from ..memory.dqn import DQNMemory, Experience
from ..network.network import Network
from ..interface.interface import Interface
# typing
from typing import TypedDict
from .agent import CallbackDict
from numpy.typing import NDArray, ArrayLike


class ReplayBatch(TypedDict):
    states: NDArray | list[NDArray] | dict[str, NDArray]
    actions: NDArray
    rewards: NDArray
    next_states: NDArray | list[NDArray] | dict[str, NDArray]
    terminals: NDArray


class DQN(Agent):
    """
    This class implements the deep Q-network algorithm (Mnih et al., 2015).

    Parameters
    ----------
    observation_space : gym.Space
        The agent's observation space.
    action_space : gym.Space
        The agent's action space.
    policy : Policy
        The agent's action selection policy used during training.
    model : Network
        The network model used by the agent.
    gamma : float, default=0.8
        The discount factor used for computing the target values.
    memory : DQNMemory or None, optional
        The memory module used for storing experiences.
    policy_test : Policy or None, optional
        The agent's action selection policy used during testing.
    custom_callbacks : CallbackDict or None, optional
        The custom callbacks defined by the user.

    Attributes
    ----------
    observation_space : gym.Space
        The agent's observation space.
    action_space : gym.Space
        The agent's action space.
    policy : Policy
        The agent's action selection policy used during training.
    policy_test : Policy or None, optional
        The agent's action selection policy used during testing.
        If none was provided in `policy_test` then `policy` will be used here as well.
    model_target : Network
        The DQN's target network.
    model_online : Network
        The DQN's online network.
    M : DQNMemory
        The memory module used for storing experiences.
        If none was provided an instance with default
        parameters is created.
    target_update : float
        Determines when and how `model_target` is updated.
        For values < 1 `model_target` is blended with `model_online`
        at each step using `target_update` as learning rate.
        For values >= 1 `model_target` is updated with
        `model_online` every `target_update` steps.
    last_update : int
        Tracks when `model_target` was updated the last time.
    current_trial : int
        Tracks the current trial.
    gamma : float
        The agent's discount factor.
    DDQN : bool
        A flag indicating whether Double Q-learning should be used.
        False by default.
    stop : bool
        A flag that can be used to manually stop training/testing of the agent.
        False by default.

    Notes
    -----
    This agent only supports gym.spaces.Discrete, gym.spaces.Box and
    gym.spaces.Dict for `observation_space` and gym.spaces.Discrete
    for `action_space`.

    Examples
    --------

    Here we initialize the DQN agent for a topology
    environment with 4 actions. ::

        >>> import gymnasium as gym
        >>> import torch.nn as nn
        >>> from collections import OrderedDict
        >>> from cobel.agent import DQN
        >>> from cobel.policy import EpsilonGreedy
        >>> from cobel.network import TorchNetwork
        >>> from layers = [('dense', nn.Linear(6, 64)),
        ...                ('relu', nn.ReLU()),
        ...                ('output', nn.Linear(64, 4))]
        >>> agent = DQN(gym.spaces.Box(0,, 1., (6, )),
        ...         gym.spaces.Discrete(4), EpsilonGreedy(0.1),
        ...         TorchNetwork(nn.Sequential(OrderedDict(layers))))


    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Policy,
        model: Network,
        gamma: float = 0.8,
        memory: None | DQNMemory = None,
        policy_test: None | Policy = None,
        custom_callbacks: None | CallbackDict = None,
    ) -> None:
        super().__init__(observation_space, action_space, custom_callbacks)
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        # build target and online models
        self.model_target = model
        self.model_online = self.model_target.clone()
        # memory module
        self.M: DQNMemory = DQNMemory() if memory is None else memory
        # target model update rate in steps
        # (for values < 1 the target model is blended with the online model)
        self.target_update: float = 10**-2
        # count the steps since the last update of the target model
        self.last_update: int = 0
        # the current trial across training and test sessions
        self.current_trial: int = 0
        # the discount factor
        self.gamma = gamma
        # double Q-learning mode flag
        self.DDQN: bool = False
        # force stop flag
        self.stop: bool = False

    def train(
        self,
        interface: gym.Env | Interface,
        trials: int,
        steps: int,
        batch_size: int = 32,
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
                q_values = self.retrieve_q(state)
                action = self.policy.select_action(q_values)
                next_state, reward, end_trial, truncated, log = interface.step(action)
                # update Q-function and store experience
                assert isinstance(next_state, (np.ndarray | list | dict))
                experience: Experience = {
                    'state': state,
                    'action': int(action),
                    'reward': float(reward),
                    'next_state': next_state,
                    'terminal': (1 - end_trial),
                }
                self.M.store(experience)
                # update current state
                state = next_state
                # experience replay
                logs['replay'] = self.replay(batch_size)
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
                q_values = self.retrieve_q(state)
                action = self.policy_test.select_action(q_values)
                next_state, reward, end_trial, truncated, log = interface.step(action)
                # update Q-function and store experience
                assert isinstance(next_state, (np.ndarray | list | dict))
                experience: Experience = {
                    'state': state,
                    'action': int(action),
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

    def retrieve_q(
        self, state: NDArray | list[NDArray] | dict[str, NDArray]
    ) -> NDArray:
        """
        This function retrieves the Q-values for a given observations.

        Parameters
        ----------
        state : NDArray, list of NDArray or dict of NDArray
            The observation for which Q-values should be retrieved.

        Returns
        -------
        q_values : NDArray
            The Q-values.
        """
        preds: NDArray | list[NDArray] | dict[str, NDArray]
        if type(state) is np.ndarray:
            if state.dtype == object:
                obs = np.array([None for o in state], dtype=object)
                for o in state:
                    obs = o.reshape((1,), o.shape)
                preds = self.model_online.predict_on_batch(obs)
            else:
                preds = self.model_online.predict_on_batch(np.array([state]))
        elif type(state) is list:
            preds = self.model_online.predict_on_batch(
                [np.array([modality]) for modality in state]
            )
        else:
            assert type(state) is dict
            preds = self.model_online.predict_on_batch(
                {key: np.array([modality]) for key, modality in state.items()}
            )
        assert type(preds) is np.ndarray

        return preds[0]

    def predict_on_batch(self, batch: ArrayLike) -> NDArray:
        """
        This function retrieves the Q-values for a batch of observations.

        Parameters
        ----------
        batch : ArrayLike
            The batch of observations for which Q-values should be retrieved.

        Returns
        -------
        predictions : NDArray
            The batch of Q-value predictions.
        """
        preds = self.model_online.predict_on_batch(batch)  # type: ignore
        assert type(preds) is np.ndarray

        return preds

    def replay(self, batch_size: int = 32) -> ReplayBatch:
        """
        This function replays experiences to update the Q-function.

        Parameters
        ----------
        batch_size : int, default=32
            The number of experiences that will be replayed.

        Returns
        -------
        replay : dict
            The replayed batch.
        """
        # sample experience batch
        states, actions, rewards, next_states, terminals = self.M.retrieve(batch_size)
        # compute targets
        targets = self.model_online.predict_on_batch(states)
        bootstrap_values = self.model_target.predict_on_batch(next_states)
        bootstrap_actions: NDArray
        assert type(bootstrap_values) is np.ndarray and type(targets) is np.ndarray
        if self.DDQN:
            preds = self.model_online.predict_on_batch(next_states)
            assert type(preds) is NDArray
            bootstrap_actions = np.argmax(preds, axis=1)
        else:
            bootstrap_actions = np.argmax(bootstrap_values, axis=1)
        bootstrap_values = bootstrap_values[np.arange(batch_size), bootstrap_actions]
        assert type(bootstrap_values) is np.ndarray
        targets[np.arange(batch_size), actions] = (
            rewards + bootstrap_values * terminals * self.gamma
        )
        # update online model
        self.model_online.train_on_batch(states, targets)
        # update target model
        if self.target_update < 1.0:
            weights_target = np.array(self.model_target.get_weights(), dtype=object)
            weights_online = np.array(self.model_online.get_weights(), dtype=object)
            weights_target += self.target_update * (weights_online - weights_target)
            # self.model_target.set_weights(list(weights_target))
            self.model_target.set_weights(weights_target)  # type: ignore
        elif self.last_update >= self.target_update:
            self.model_target.set_weights(self.model_online.get_weights())
            self.last_update = 0

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'terminals': terminals,
        }
