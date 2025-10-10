# basic imports
import numpy as np
import gymnasium as gym
# framework imports
from .agent import Agent
from ..policy.policy import Policy
from ..interface.interface import Interface
# typing
from typing import TypedDict, NotRequired
from numpy.typing import ArrayLike, NDArray
from .agent import CallbackDict


class Experience(TypedDict):
    state: int
    action: int
    reward: float
    next_state: int
    terminal: int
    td: NotRequired[float]


class SR(Agent):
    """
    This class implements an agent based on the successor representation (SR).

    Parameters
    ----------
    observation_space : gym.Space
        The agent's observation space.
    action_space : gym.Space
        The agent's action space.
    policy : Policy
        The agent's action selection policy used during training.
    policy_test : Policy or None, optional
        The agent's action selection policy used during testing.
    learning_rate : float, default=0.99
        The agent's learning rate.
    gamma : float, default=0.99
        The agent's discount factor.
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
    gamma : float
        The agent's discount factor.
    learning_rate : float
        The agent's learning rate.
    SR : NDArray
        A 2D NumPy array of shape (`observation_space.n`, `observation_space.n`)
        which represents the agent's successor representation (SR).
        The SR is initialized to all zeros.
    R : NDArray
        A 1D NumPy array of shape (`observation_space.n`,)
        which represents the reward function.
    transitions : NDArray
        A 3D NumPy array of shape (`observation_space.n`, `observation_space.n`,
        `observation_space.n`) which represents possible environmental transitions.
    action_mask : NDArray
        A boolean 2D NumPy array of shape (`observation_space.n`, `action_space.n`)
        which represents an action mask that can be applied during action selection.
    mask_actions : bool
        A flag indicating whether action should be masked
        as defined in `action_mask`. Set to False by default.

    Notes
    -----
    This agent only supports gym.spaces.Discrete for `observation_space`
    and `action_space`.

    Examples
    --------

    Here we initialize the SR agent for a discrete
    environment with 16 states and 4 actions. ::

        >>> import gymnasium as gym
        >>> from cobel.agent import SR
        >>> from cobel.policy import EpsilonGreedy
        >>> agent = SR(gym.spaces.Discrete(16),
        ...         gym.spaces.Discrete(4), EpsilonGreedy(0.1))

    Training and testing of the agent is done with the train
    and test methods, respectively.
    Both methods expect an RL interface and number of trials
    as input.::

        >>> from cobel.interface import Gridworld
        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> env = Gridworld(make_open_field(4, 4, 0, 1))
        >>> agent.train(env, 100)
        >>> agent.test(env, 100)

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Policy,
        policy_test: None | Policy = None,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        custom_callbacks: None | CallbackDict = None,
    ) -> None:
        assert type(observation_space) is gym.spaces.Discrete, (
            'SR requires a discrete observation space!'
        )
        assert type(action_space) is gym.spaces.Discrete, (
            'SR requires a discrete action space!'
        )
        super().__init__(observation_space, action_space, custom_callbacks)
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.SR = np.eye(int(observation_space.n))
        self.transitions = np.eye(self.SR.shape[0])
        self.transitions = np.reshape(
            self.transitions, (self.SR.shape[0], 1, self.SR.shape[0])
        )
        self.transitions = np.tile(self.transitions, (1, int(action_space.n), 1))
        self.rewards = np.zeros(observation_space.n)
        self.action_mask: NDArray = np.ones(
            (observation_space.n, action_space.n)
        ).astype(bool)
        self.mask_actions: bool = False

    def train(self, interface: gym.Env | Interface, trials: int, steps: int) -> None:
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
            assert type(state) is int
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                mask = self.action_mask[state] if self.mask_actions else None
                action = self.policy.select_action(self.retrieve_q(state), mask)
                next_state, reward, end_trial, truncated, log = interface.step(action)
                assert type(next_state) is int
                # update Q-function and store experience
                experience: Experience = {
                    'state': state,
                    'action': int(action),
                    'reward': float(reward),
                    'next_state': next_state,
                    'terminal': (1 - end_trial),
                }
                experience = self.update(experience)
                # update current state
                state = next_state
                # update logs
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
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
            assert type(state) is int
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                mask = self.action_mask[state] if self.mask_actions else None
                action = self.policy_test.select_action(self.retrieve_q(state), mask)
                next_state, reward, end_trial, truncated, log = interface.step(action)
                assert type(next_state) is int
                # update Q-function and store experience
                experience = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'terminal': (1 - end_trial),
                }
                # update current state
                state = next_state
                # update logs
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def update(self, experience: Experience) -> Experience:
        """
        This function updates the SR for a given experience.

        Parameters
        ----------
        experience : Experience
            A dictionary containing the experience tuple.

        Returns
        -------
        experience : Experience
            A dictionary containing the experience tuple and the TD-error.
        """
        state, action = experience['state'], experience['action']
        reward, next_state = experience['reward'], experience['next_state']
        # update environmental information
        td_reward = reward - self.rewards[next_state]
        self.rewards[next_state] += td_reward * self.learning_rate
        self.transitions[state][action] = np.eye(self.SR.shape[0])[next_state]
        # compute TD-error
        td = np.eye(self.SR.shape[0])[state]
        if experience['terminal'] > 0:
            td += self.gamma * np.copy(self.SR[next_state])
        else:
            td += self.gamma * np.eye(self.SR.shape[0])[next_state]
        td -= np.copy(self.SR[state])
        experience['td'] = td
        # update Q-function
        self.SR[state] += self.learning_rate * td

        return experience

    def retrieve_q(self, state: int) -> NDArray:
        """
        This function retrieves the Q-values for a given state index.

        Parameters
        ----------
        state : int
            The state for which Q-values should be retrieved.

        Returns
        -------
        predictions : NDArray
            The batch of Q-value predictions.
        """
        values = np.sum(self.SR * self.rewards, axis=1)
        q = []
        for action in range(self.action_mask.shape[1]):
            idx = self.transitions[state][action] == 1
            q.append(np.mean(values[idx]))

        return np.array(q)

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
        return np.array([self.retrieve_q(int(s)) for s in np.array(batch).astype(int)])
