# basic imports
import numpy as np
import gymnasium as gym
# framework imports
from .agent import Agent
from ..policy.policy import Policy
from ..interface.interface import Interface
# typing
from typing import TypedDict, NotRequired
from numpy.typing import NDArray, ArrayLike
from collections.abc import Callable
from .agent import CallbackDict


class Experience(TypedDict):
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    terminal: int
    td: NotRequired[float]


class QAgent(Agent):
    """
    This class implements a simple (tabular) Q-learning agent.

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
    learning_rate : float, default=0.9
        The agent's learning rate.
    gamma : float, default=0.8
        The agent's discount factor.
    custom_callbacks : CallbackDict or None, optional
        The custom callbacks defined by the user.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

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
    nb_actions : int
        The number of actions available to the agent.
    Q : dict of NDArray
        A dictionary used to store the agent's Q-function.
        Q-values are stored as NumPy arrays with the shape (`nb_actions`, )
        and are initialized to zero when a novel state is encountered.
        Observations are transformed into flattened tuples and serve
        as dictionary keys.
    M : list of dict
        A list used to store experiences for experience replay.
        Experiences are dictionaries which contain current and next state,
        action, reward and a terminal flag.
    rng : numpy.random.Generator
        A random number generator instance used for probablistic replay.

    Notes
    -----
    This agent only supports gym.spaces.Discrete, gym.spaces.Box and
    gym.spaces.Dict for `observation_space` and gym.spaces.Discrete
    for `action_space`.

    Examples
    --------

    Here we initialize the Q-learning agent for a discrete
    environment with 16 states and 4 actions. ::

        >>> import gymnasium as gym
        >>> from cobel.agent import QAgent
        >>> from cobel.policy import EpsilonGreedy
        >>> agent = QAgent(
        ...         gym.spaces.Discrete(16),
        ...         gym.spaces.Discrete(4),
        ...         EpsilonGreedy(0.1)
        ...         )

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
        learning_rate: float = 0.9,
        gamma: float = 0.8,
        custom_callbacks: None | CallbackDict = None,
        rng: None | np.random.Generator = None,
    ) -> None:
        assert type(observation_space) in (
            gym.spaces.Discrete,
            gym.spaces.Box,
            gym.spaces.Dict,
        ), 'Wrong observation space!'
        if type(observation_space) is gym.spaces.Dict:
            for _, space in observation_space.items():
                assert type(space) is gym.spaces.Box
        assert type(action_space) is gym.spaces.Discrete, 'Wrong action space!'
        super().__init__(observation_space, action_space, custom_callbacks)
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        self.rng = np.random.default_rng() if rng is None else rng
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.nb_actions = action_space.n
        self.Q: dict[tuple, NDArray] = {}
        self.M: list[Experience] = []
        self.current_trial = 0
        # observations will be processed according to the observation space
        self._process_observation: (
            Callable[[int], tuple[float, ...]]
            | Callable[[NDArray], tuple[float, ...]]
            | Callable[[dict[str, NDArray]], tuple[float, ...]]
        )
        if type(self.observation_space) is gym.spaces.Discrete:
            self._process_observation = lambda x: (x,)
        elif type(self.observation_space) is gym.spaces.Box:
            self._process_observation = lambda x: tuple(x.flatten())
        else:
            self._process_observation = lambda x: tuple(
                np.concatenate(tuple([o.flatten() for _, o in x.items()]))
            )

    def train(
        self,
        interface: gym.Env | Interface,
        trials: int,
        steps: int = 32,
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
        steps : int, default=32
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
            s, _ = interface.reset()
            assert type(s) in [int, np.ndarray, dict], 'Invalid observation type'
            state = self._process_observation(s)  # type: ignore
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                if state not in self.Q:
                    self.Q[state] = np.zeros(self.nb_actions)
                action = self.policy.select_action(self.Q[state])
                ns, reward, end_trial, truncated, log = interface.step(action)
                assert type(ns) in [int, np.ndarray, dict], 'Invalid observation type'
                next_state = self._process_observation(ns)  # type: ignore
                if next_state not in self.Q:
                    self.Q[next_state] = np.zeros(self.nb_actions)
                # update Q-function amd stpre experience
                experience: Experience = {
                    'state': state,
                    'action': int(action),
                    'reward': float(reward),
                    'next_state': next_state,
                    'terminal': (1 - end_trial),
                }
                self.M.append(experience)
                experience = self.update_q(experience)
                state = next_state
                self.replay(batch_size)
                # update logs
                logs['trial_reward'] += reward
                logs.update(experience)
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
            assert type(s) in [int, np.ndarray, dict], 'Invalid observation type'
            state = self._process_observation(s)  # type: ignore
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                if state not in self.Q:
                    self.Q[state] = np.zeros(self.nb_actions)
                action = self.policy_test.select_action(self.Q[state])
                ns, reward, end_trial, truncated, log = interface.step(action)
                assert type(ns) in [int, np.ndarray, dict], 'Invalid observation type'
                next_state = self._process_observation(ns)  # type: ignore
                # update Q-function amd stpre experience
                experience = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'terminal': (1 - end_trial),
                }
                state = next_state
                # update logs
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def update_q(self, experience: Experience) -> Experience:
        """
        This function updates the agent with a given experience.

        Parameters
        ----------
        experience : Experience
            The experience dictionary.

        Returns
        -------
        experience : Experience
            The experience dictionary updated with the TD-error.
        """
        # compute TD-error
        td = experience['reward']
        td += (
            self.gamma
            * experience['terminal']
            * np.amax(self.Q[experience['next_state']])
        )
        td -= self.Q[experience['state']][experience['action']]
        experience['td'] = td
        # update Q-function
        self.Q[experience['state']][experience['action']] += self.learning_rate * td

        return experience

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
        if type(self.observation_space) is gym.spaces.Discrete:
            return np.array([self.Q[(o,)] for o in np.array(batch)])
        elif type(self.observation_space) is gym.spaces.Box:
            return np.array([self.Q[tuple(o.flatten())] for o in np.array(batch)])
        else:
            q: list[NDArray] = []
            assert type(batch) is list
            for o in batch:
                q.append(
                    self.Q[np.concatenate(tuple([m.flatten() for _, m in o.items()]))]
                )
            return np.array(q)

    def replay(self, batch_size: int = 32) -> None:
        """
        This function performs experience replay to update the agent.

        Parameters
        ----------
        batch_size : int, default=32
            The size of the replay batch.
        """
        for i in self.rng.choice(len(self.M), batch_size):
            self.update_q(self.M[i])
