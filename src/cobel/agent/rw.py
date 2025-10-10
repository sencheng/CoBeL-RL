# basic imports
import numpy as np
import gymnasium as gym
# framework imports
from .agent import Agent
from ..policy.policy import Policy
from ..interface.interface import Interface
# typing
from .agent import CallbackDict
from numpy.typing import NDArray, ArrayLike


class RescorlaWagner(Agent):
    """
    This class implements an (associative) agent based on the Rescorla-Wagner model.

    Parameters
    ----------
    observation_space : gym.Space
        The agent's observation space.
    learning_rate : float or tuple of float, default=0.9
        The learning rate(s) for the input stimuli.
    custom_callbacks : CallbackDict or None, optional
        The custom callbacks defined by the user.

    Attributes
    ----------
    observation_space : gym.Space
        The agent's observation space.
    W : NDArray
        The agent's weights.
    learning_rate : float or NDArray
        The learning rate(s) for the input stimuli.
    current_trial : int
        Tracks the current trial.

    Notes
    -----
    This agent only supports gym.spaces.Box for `observation_space`.

    Examples
    --------

    Here we initialize the agent for a Sequence environment
    which presents observations of shape (10, ). ::

        >>> import gymnasium as gym
        >>> from cobel.agent import RescorlaWagner
        >>> agent = RescorlaWagner(gym.spaces.Box(0., 1., (10, )))

    """

    def __init__(
        self,
        observation_space: gym.Space,
        learning_rate: float | tuple[float, ...] = 0.9,
        custom_callbacks: None | CallbackDict = None,
    ) -> None:
        assert type(observation_space) is gym.spaces.Box, 'Wrong observation space!'
        super().__init__(
            observation_space,
            gym.spaces.Box(-np.inf, np.inf, (1,), np.float64),
            custom_callbacks,
        )
        self.W = np.zeros(observation_space.shape)
        self.learning_rate: float | NDArray
        if type(learning_rate) is float:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = np.array(learning_rate)
        self.current_trial = 0

    def train(
        self, interface: gym.Env | Interface, trials: int, steps: int = 32
    ) -> None:
        """
        This function is called to train the agent.

        Parameters
        ----------
        interface : gym.Env or Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is trained.
        steps : int, default=32
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
            s, _ = interface.reset()
            assert type(s) is np.ndarray, 'Invalid observation type'
            state = s.flatten()
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                action = self.predict_on_batch(np.array([state]))[0]
                ns, reward, end_trial, truncated, log = interface.step(action)
                # update weights
                self.W -= self.learning_rate * (action - reward) * state
                # update current state
                assert type(ns) is np.ndarray, 'Invalid observation type'
                next_state = ns.flatten()
                state = next_state
                # update logs
                logs['trial_reward'] += reward
                logs = self.callbacks.on_step_end(logs)
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def test(
        self, interface: gym.Env | Interface, trials: int, steps: int = 32
    ) -> None:
        """
        This function is called to test the agent.

        Parameters
        ----------
        interface : gym.Env or Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is tested.
        steps : int, default=32
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
            s, _ = interface.reset()
            assert type(s) is np.ndarray, 'Invalid observation type'
            state = s.flatten()
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                action = self.predict_on_batch(np.array([state]))[0]
                ns, reward, end_trial, truncated, log = interface.step(action)
                # update current state
                assert type(ns) is np.ndarray, 'Invalid observation type'
                next_state = ns.flatten()
                state = next_state
                # update logs
                logs['trial_reward'] += reward
                logs = self.callbacks.on_step_end(logs)
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def predict_on_batch(self, batch: ArrayLike) -> NDArray:
        """
        This function retrieves the values for a batch of observations.

        Parameters
        ----------
        batch : ArrayLike
            The batch of observations for which the values should be retrieved.

        Returns
        -------
        predictions : NDArray
            The batch of value predictions.
        """
        assert type(batch) is np.ndarray

        return self.W @ batch.T


class BinaryRescorlaWagner(RescorlaWagner):
    """
    This class implements an (associative) agent based on the Rescorla-Wagner model
    which transforms learned associativity into binary actions.

    Parameters
    ----------
    observation_space : gym.Space
        The agent's observation space.
    policy : Policy
        The agent's action selection policy used during training.
    policy_test : Policy or None, optional
        The agent's action selection policy used during testing.
    learning_rate : float or tuple of float, default=0.9
        The learning rate(s) for the input stimuli.
    custom_callbacks : CallbackDict or None, optional
        The custom callbacks defined by the user.

    Attributes
    ----------
    observation_space : gym.Space
        The agent's observation space.
    policy : Policy
        The agent's action selection policy used during training.
    policy_test : Policy or None, optional
        The agent's action selection policy used during testing.
        If none was provided in `policy_test` then `policy` will be used here as well.
    W : NDArray
        The agent's weights.
    learning_rate : float or NDArray
        The learning rate(s) for the input stimuli.
    current_trial : int
        Tracks the current trial.

    Notes
    -----
    This agent only supports gym.spaces.Box for `observation_space`
    and gym.spaces.Discrete for `action_space`.
    The action space must have exactly 2 actions.
    For the action selection policy use one of the scalar policies,
    i.e., Proportional, Threshold or Sigmoid.

    Examples
    --------

    Here we initialize the agent for a Sequence environment
    which presents observations of shape (10, ). ::

        >>> import gymnasium as gym
        >>> from cobel.agent import BinaryRescorlaWagner
        >>> from cobel.policy import Sigmoid
        >>> agent = BinaryRescorlaWagner(
        ...         gym.spaces.Box(0., 1., (10, )),
        ...         Sigmoid())

    """

    def __init__(
        self,
        observation_space: gym.Space,
        policy: Policy,
        policy_test: None | Policy = None,
        learning_rate: float | tuple[float, ...] = 0.9,
        custom_callbacks: None | CallbackDict = None,
    ) -> None:
        assert type(observation_space) is gym.spaces.Box, 'Wrong observation space!'
        super().__init__(observation_space, learning_rate, custom_callbacks)
        self.action_space = gym.spaces.Discrete(2)
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test

    def train(
        self, interface: gym.Env | Interface, trials: int, steps: int = 32
    ) -> None:
        """
        This function is called to train the agent.

        Parameters
        ----------
        interface : gym.Env or Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is trained.
        steps : int, default=32
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
            s, _ = interface.reset()
            assert type(s) is np.ndarray, 'Invalid observation type'
            state = s.flatten()
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                v = self.predict_on_batch(np.array([state]))[0]
                action = self.policy.select_action(v)
                ns, reward, end_trial, truncated, log = interface.step(action)
                # update weights
                assert type(reward) is float
                target = (
                    1.0
                    if (log['action'] == 0 and reward > 0)
                    or (log['action'] == 1 and reward < 0)
                    else 0.0
                )
                self.W -= self.learning_rate * (v - target) * state
                # update current state
                assert type(ns) is np.ndarray, 'Invalid observation type'
                next_state = ns.flatten()
                state = next_state
                # update logs
                logs['trial_reward'] += reward
                logs['action'] = action
                logs = self.callbacks.on_step_end(logs)
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def test(
        self, interface: gym.Env | Interface, trials: int, steps: int = 32
    ) -> None:
        """
        This function is called to test the agent.

        Parameters
        ----------
        interface : gym.Env or Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is tested.
        steps : int, default=32
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
            s, _ = interface.reset()
            assert type(s) is np.ndarray, 'Invalid observation type'
            state = s.flatten()
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                v = self.predict_on_batch(np.array([state]))[0]
                action = self.policy.select_action(v)
                ns, reward, end_trial, truncated, log = interface.step(action)
                # update current state
                assert type(ns) is np.ndarray, 'Invalid observation type'
                next_state = ns.flatten()
                state = next_state
                # update logs
                logs['trial_reward'] += reward
                logs['action'] = action
                logs = self.callbacks.on_step_end(logs)
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break
