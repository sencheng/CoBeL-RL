# basic imports
import time
import numpy as np
import gymnasium as gym
from sklearn.neighbors import KDTree  # type: ignore
# framework imports
from .agent import Agent
from ..policy.policy import Policy
from ..interface.interface import Interface
from ..network.network import Network
# typing
from .agent import CallbackDict
from ..interface.interface import Observation
from numpy.typing import NDArray, ArrayLike
from typing import TypedDict


class Experience(TypedDict):
    state: NDArray
    action: int
    reward: float
    value: float
    time: float


class ActionBuffer:
    """
    This class implements the memory structure used by the QEC class.

    Parameters
    ----------
    capacity : int
        The capacity of the memory structure.

    Attributes
    ----------
    capacity : int
        The capacity of the memory structure.
    tree : KDTree or None
        A k-dimensional tree created from `states`.
        Used for nearest neighbor search of states.
    states : list of NDArray
        List used to store states.
    values : list of float
        List used to store state values.
    times : list of float
        List used to store time stamps.

    """

    def __init__(self, capacity: int):
        self.tree: None | KDTree = None
        self.capacity = capacity
        self.states: list[NDArray] = []
        self.values: list[float] = []
        self.times: list[float] = []

    def find_state(self, state: NDArray) -> None | int:
        """
        This functions searches the memory structure for
        a given state and returns its index.

        Parameters
        ----------
        state : NDArray
            The state the will be searched for.

        Returns
        -------
        index : int or None
            The index of the state. Returns 'None' if the state could not be found.
        """
        if self.tree is not None:
            index = self.tree.query([state])[1][0][0]
            if np.allclose(self.states[index], state, rtol=1e-04, atol=1e-06):
                return index

        return None

    def find_neighbors(self, state: NDArray, k: int) -> list | NDArray:
        """
        This functions searches the memory structure for a given
        state's k-nearest neighbors and returns them.

        Parameters
        ----------
        state : NDArray
            The state the will be searched for.
        k : int
            The number k of nearest neighbors.

        Returns
        -------
        neighbors : list or NDArray
            A numpy array containing the k-nearest neighbor's indeces.
            Returns an empty list if the memory is empty.
        """
        return self.tree.query([state], k)[1][0] if self.tree is not None else []

    def add(self, state: NDArray, value: float, time: float) -> None:
        """
        This functions adds a state-value pair to memory.

        Parameters
        ----------
        state : NDArray
            The state that will be added.
        value : float
            The value that will be added.
        time : float
            The time associated with the state-value pair.
        """
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.times.append(time)
        else:
            min_time_idx = int(np.argmin(self.times))
            if time > self.times[min_time_idx]:
                self.replace(state, value, time, min_time_idx)
        self.tree = KDTree(self.states)

    def replace(self, state: NDArray, value: float, time: float, index: int) -> None:
        """
        This functions replaces an older entry with the given one.

        Parameters
        ----------
        state : NDArray
            The state that will be added.
        value : float
            The value that will be added.
        time : float
            The time associated with the new state-value pair.
        index : int
            The index of the entry that will be replaced.
        """
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time

    def __len__(self) -> int:
        return len(self.states)


class QEC:
    """
    This class implements the MFEC agent's EM-based Q-function.

    Parameters
    ----------
    nb_actions : int
        The number of actions available to the agent.
    capacity : int
        The capacity of each action buffer.
    k : int
        The number of nearest neighbors that will be used to estimate the Q-values.

    Attributes
    ----------
    buffers : tuple of ActionBuffer
        The memory buffers for each action.
    k : int
        The number of nearest neighbors that will be used to estimate the Q-values.

    """

    def __init__(self, nb_actions: int, capacity: int, k: int) -> None:
        self.buffers = tuple([ActionBuffer(capacity) for a in range(nb_actions)])
        self.k = k

    def estimate(self, state: NDArray, action: int) -> float:
        """
        This function estimates and returns the Q-value for a given state-action pair.

        Parameters
        ----------
        state : NDArray
            The given state.
        action : int
            The given action.

        Returns
        -------
        q : float
            The estimated Q-value.
        """
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)
        if state_index is not None:
            return buffer.values[state_index]
        if len(buffer) <= self.k:
            return 0.0
        value = 0.0
        neighbors = buffer.find_neighbors(state, self.k)
        for neighbor in neighbors:
            value += buffer.values[neighbor]

        return value / max(len(neighbors), 1)

    def update(self, state: NDArray, action: int, value: float, time: float) -> None:
        """
        This function updates the EM-based Q-function.

        Parameters
        ----------
        state : NDArray
            The given state.
        action : int
            The given action.
        value : float
            The new Q-value.
        time : float
            The current time.
        """
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)
        if state_index:
            max_value = max(buffer.values[state_index], value)
            max_time = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)
        else:
            buffer.add(state, value, time)

    def update_episode(self, episode: list[Experience]) -> None:
        """
        This function updates the EM-based Q-function with an episode.

        Parameters
        ----------
        episode : list of Experience
            A list of single events/experiences.
        """
        for event in episode:
            self.update(event['state'], event['action'], event['value'], event['time'])


class MFEC(Agent):
    """
    This class implements the Model Free Episodic Control algorithm
    described by Blundell et al. (2016).

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
    capacity : int, default=2000
        The agent's memory's capacity.
    k : int, default=3
        The number of nearest neighbors that will be used to estimate Q-values.
    gamma : float, default=0.97
        The agent's discount factor.
    model : Network or None, optional
        An optional pretrained model that will be used to transform observations.
        If none was given random projection will be used.
    projection_size : int, default=256
        The size used for random projection.
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
    capacity : int
        The agent's memory's capacity.
    nb_actions : int
        The number of actions available to the agent.
    k : int
        The number of nearest neighbors that will be used to estimate Q-values.
    gamma : float
        The agent's discount factor.
    Q : QEC
        The MFEC agent's EM-based Q-function.
    model : Network or None
        An optional pretrained model that will be used to transform observations.
        If none was given random projection will be used.
    projection_size : int
        The size used for random projection.
    projection : NDArray
        A NumPy array representing the random projection weights.
    rng : numpy.random.Generator
        A random number generator instance used for
        generation of the random projection weights.

    Notes
    -----
    This agent only supports gym.spaces.Discrete, gym.spaces.Box and
    gym.spaces.Dict for `observation_space` and gym.spaces.Discrete
    for `action_space`.

    Examples
    --------

    Here we initialize the MFEC agent for a topology
    environment with 4 actions. ::

        >>> import gymnasium as gym
        >>> from cobel.agent import MFEC
        >>> from cobel.policy import EpsilonGreedy
        >>> agent = MFEC(gym.spaces.Box(0., 1., (6, )),
        ...              gym.spaces.Discrete(4), EpsilonGreedy(0.1))

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Policy,
        policy_test: None | Policy = None,
        capacity: int = 2000,
        k: int = 3,
        gamma: float = 0.97,
        model: None | Network = None,
        projection_size: int = 256,
        custom_callbacks: None | CallbackDict = None,
        rng: None | np.random.Generator = None,
    ) -> None:
        assert type(action_space) is gym.spaces.Discrete, 'Wrong action space!'
        super().__init__(observation_space, action_space, custom_callbacks)
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        self.rng = np.random.default_rng() if rng is None else rng
        self.capacity = capacity
        self.nb_actions = int(action_space.n)
        self.k = k
        self.gamma = gamma
        self.Q = QEC(self.nb_actions, self.capacity, self.k)
        self.model = model
        self.projection_size = projection_size
        self.projection: NDArray
        if type(self.observation_space) is gym.spaces.Box:
            self.projection = self.rng.random(
                (int(np.prod(self.observation_space.shape)), self.projection_size)
            )
        elif type(self.observation_space) is gym.spaces.Dict:
            units: int = 0
            for _, space in self.observation_space.items():
                assert type(space.shape) is tuple
                units += int(np.prod(space.shape))
            self.projection = self.rng.random((units, self.projection_size))

    def process_observation(self, observation: Observation) -> NDArray:
        """
        This function processes an observation by
        either random projection or a network model.

        Parameters
        ----------
        observation : Observation
            The given observation.

        Returns
        -------
        processed : NDArray
            The processed observation.
        """
        if self.model is not None:
            if type(observation) is np.ndarray:
                prediction = self.model.predict_on_batch(np.array([observation]))
                assert type(prediction) is np.ndarray
                return prediction
            elif type(observation) is list:
                prediction = self.model.predict_on_batch(
                    [np.array([o]) for o in observation]
                )
                assert type(prediction) is np.ndarray
                return prediction
            else:
                assert type(observation) is dict
                prediction = self.model.predict_on_batch(
                    {m: np.array([o]) for m, o in observation.items()}
                )
                assert type(prediction) is np.ndarray
                return prediction
        else:
            if type(observation) is np.ndarray:
                return np.dot(observation.flatten(), self.projection)
            elif type(observation) is list:
                return np.dot(np.array(observation).flatten(), self.projection)
            else:
                assert type(observation) is dict
                return np.dot(
                    np.array(list(observation.values())).flatten(), self.projection
                )

    def retrieve_q(self, state: NDArray) -> NDArray:
        """
        This function retrieves the Q-values for a given state.

        Parameters
        ----------
        state : NDArray
            The state for which Q-values will be retrieved.

        Returns
        -------
        q_values : NDArray
            The retrieved Q-values.
        """
        return np.array(
            [self.Q.estimate(state, action) for action in range(self.nb_actions)]
        )

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
            state = self.process_observation(s)
            episode: list[Experience] = []
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                q_values = self.retrieve_q(state)
                action = self.policy.select_action(q_values)
                ns, reward, end_trial, truncated, log = interface.step(action)
                next_state = self.process_observation(ns)
                episode.append(
                    {
                        'state': np.copy(state),
                        'action': int(action),
                        'reward': float(reward),
                        'value': float(reward),
                        'time': time.time(),
                    }
                )
                # update Q-function amd stpre experience
                state = next_state
                # update logs
                logs['trial_reward'] += reward
                logs = self.callbacks.on_step_end(logs)
                if end_trial:
                    r = 0.0
                    for e in episode[::-1]:
                        r = self.gamma * r + e['reward']
                        e['value'] = r
                    self.Q.update_episode(episode)
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
            state = self.process_observation(s)
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                q_values = self.retrieve_q(state)
                action = self.policy_test.select_action(q_values)
                ns, reward, end_trial, truncated, log = interface.step(action)
                next_state = self.process_observation(ns)
                # update Q-function amd stpre experience
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
        return np.array(
            [
                [
                    self.Q.estimate(
                        self.process_observation(obs),  # type: ignore
                        action,
                    )
                    for action in range(self.nb_actions)
                ]
                for obs in batch  # type: ignore
            ]
        )
