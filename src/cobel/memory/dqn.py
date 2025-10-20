# basic imports
import numpy as np
# typing
from typing import TypedDict
from collections.abc import Callable
from numpy.typing import NDArray

Obs = NDArray | list[NDArray] | dict[str, NDArray]
SimpleStore = Callable[[NDArray, NDArray], None]
ListStore = Callable[[list[NDArray], list[NDArray]], None]
DictStore = Callable[[dict[str, NDArray], dict[str, NDArray]], None]
RaggedStore = Callable[[dict, dict], None]
InitialStore = Callable[[Obs, Obs], None]
SimpleRetrieve = Callable[[NDArray[np.int_]], tuple[NDArray, NDArray]]
ListRetrieve = Callable[[NDArray[np.int_]], tuple[list[NDArray], list[NDArray]]]
DictRetrieve = Callable[
    [NDArray[np.int_]], tuple[dict[str, NDArray], dict[str, NDArray]]
]
RaggedRetrieve = Callable[[NDArray[np.int_]], tuple[NDArray, NDArray]]
InitialRetrieve = Callable[[NDArray[np.int_]], tuple[Obs, Obs]]


class Experience(TypedDict):
    state: Obs
    action: int
    reward: float
    next_state: Obs
    terminal: int


class DQNMemory:
    """
    This class implements a simple memory structure of the storing of experiences.

    Parameters
    ----------
    capacity : int
        The memory capacity.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    capacity : int
        The memory capacity.
    state_store : Callable
        A function called when the state/observation of
        a new experience should be stored.
        The precise function depends on the type of observation and
        is determined when the initial observation should be stored.
    state_retrieve : Callable
        A function called when the state/observation of
        an experience should be retrieved.
        The precise function depends on the type of Observation and
        is determined when the initial observation should be stored.
    actions : NDArray
        Stores the actions for each experience.
    rewards : NDArray
        Stores the rewards for each experience.
    terminals : NDArray
        Stores the terminal flag for each experience.
    states : NDArray, list of NDArray or dict of NDArray
        Stores the state/observation for each experience.
    next_states : NDArray, list of NDArray or dict of NDArray
        Stores the follow-up state/observation for each experience.
    rng : numpy.random.Generator
        A random number generator instance used for probablistic replay.

    Examples
    --------

    Initializing the memory module. ::

        >>> from cobel.memory import DQNMemory
        >>> memory = DQNMemory(1000)

    """

    def __init__(
        self, capacity: int = 100000, rng: None | np.random.Generator = None
    ) -> None:
        assert capacity > 0, 'Memory capacity must be greater than zero!'
        self.rng = np.random.default_rng() if rng is None else rng
        self.capacity = capacity
        self.state_store: (
            SimpleStore | ListStore | DictStore | RaggedStore | InitialStore
        ) = self.store_initial
        self.state_retrieve: (
            SimpleRetrieve
            | ListRetrieve
            | DictRetrieve
            | RaggedRetrieve
            | InitialRetrieve
        ) = self.retrieve_simple
        # initialize memory structures
        self.actions: NDArray = np.array([], dtype=int)
        self.rewards: NDArray = np.array([])
        self.terminals: NDArray = np.array([])
        self.states: NDArray | list[NDArray] | dict[str, NDArray]
        self.next_states: NDArray | list[NDArray] | dict[str, NDArray]

    def store(self, experience: Experience) -> None:
        """
        This function stores an experience tuple.

        Parameters
        ----------
        experience : Experience
            The experience to be stored.
        """
        self.state_store(experience['state'], experience['next_state'])  # type: ignore
        self.actions = np.append(self.actions, experience['action'])
        self.rewards = np.append(self.rewards, experience['reward'])
        self.terminals = np.append(self.terminals, experience['terminal'])
        # remove the oldest experience when the memory is over capacity
        if self.actions.size > self.capacity:
            self.actions = self.actions[1:]
            self.rewards = self.rewards[1:]
            self.terminals = self.terminals[1:]

    def retrieve(
        self, batch_size: int = 32
    ) -> tuple[Obs, NDArray, NDArray, Obs, NDArray]:
        """
        This function retrieves a random batch of experiences.

        Parameters
        ----------
        batch_size : int, default=32
            The size of the batch.

        Returns
        -------
        states : NDArray, list of NDArray or dict of NDArray
            The batch of current states.
        actions : NDArray
            The batch of actions.
        rewards : NDArray
            The batch of rewards.
        next_states : NDArray, list of NDArray or dict of NDarray
            The batch of next states.
        terminals : NDArray
            The batch of terminals.
        """
        assert batch_size > 0
        idx = self.rng.integers(self.actions.size, size=batch_size)
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        terminals = self.terminals[idx]
        states, next_states = self.state_retrieve(idx)

        return states, actions, rewards, next_states, terminals

    def store_initial(self, state: Obs, next_state: Obs) -> None:
        """
        This function stores the very first observation and
        initializes the memory structures.

        Parameters
        ----------
        state : NDArray, list of NDArray or dict of NDArray
            The current state to be stored.
        next_state : NDArray, list of NDArray or dict of NDArray
            The next state to be stored.
        """
        if type(state) is np.ndarray:
            assert type(next_state) is np.ndarray
            if state.dtype == object:
                self.states = np.array([None for o in state], dtype=object)
                self.next_states = np.array([None for o in state], dtype=object)
                for i in range(state.size):
                    self.states[i] = np.empty((0,) + state[i].shape)
                    self.next_states[i] = np.empty((0,) + state[i].shape)
                self.state_store = self.store_multimodal_ragged
                self.state_retrieve = self.retrieve_multimodal_ragged
            else:
                self.states = np.empty((0,) + state.shape)
                self.next_states = np.empty((0,) + state.shape)
                self.state_store = self.store_simple
                self.state_retrieve = self.retrieve_simple
            self.state_store(state, next_state)
        elif type(state) is list:
            assert type(next_state) is list
            self.states = [np.empty((0,) + obs.shape) for obs in state]
            self.next_states = [np.empty((0,) + obs.shape) for obs in state]
            self.state_store = self.store_multimodal_list
            self.state_retrieve = self.retrieve_multimodal_list
            self.state_store(state, next_state)
        else:
            assert type(state) is dict and type(next_state) is dict
            self.states = {
                modality: np.empty((0,) + obs.shape) for modality, obs in state.items()
            }
            self.next_states = {
                modality: np.empty((0,) + obs.shape) for modality, obs in state.items()
            }
            self.state_store = self.store_multimodal_dict
            self.state_retrieve = self.retrieve_multimodal_dict
            self.state_store(state, next_state)

    def store_simple(self, state: NDArray, next_state: NDArray) -> None:
        """
        This function stores simple observations.

        Parameters
        ----------
        state : NDArray
            The current state to be stored.
        next_state : NDArray
            The next state to be stored.
        """
        assert type(self.states) is np.ndarray and type(self.next_states) is np.ndarray
        self.states = np.append(self.states, state.reshape((1,) + state.shape), axis=0)
        self.next_states = np.append(
            self.next_states, next_state.reshape((1,) + state.shape), axis=0
        )
        if self.states.shape[0] > self.capacity:
            self.states = self.states[1:]
            self.next_states = self.next_states[1:]

    def store_multimodal_list(
        self, state: list[NDArray], next_state: list[NDArray]
    ) -> None:
        """
        This function stores multimodal observations provided as a list.

        Parameters
        ----------
        state : list of NDArray
            The current state to be stored.
        next_state : list of NDArray
            The next state to be stored.
        """
        assert type(self.states) is list and type(self.next_states) is list
        for i in range(len(state)):
            self.states[i] = np.append(
                self.states[i], state[i].reshape((1,) + state[i].shape), axis=0
            )
            self.next_states[i] = np.append(
                self.next_states[i],
                next_state[i].reshape((1,) + state[i].shape),
                axis=0,
            )
        if self.states[0].shape[0] > self.capacity:
            for i in range(len(state)):
                self.states[i] = self.states[i][1:]
                self.next_states[i] = self.next_states[i][1:]

    def store_multimodal_dict(
        self, state: dict[str, NDArray], next_state: dict[str, NDArray]
    ) -> None:
        """
        This function stores multimodal observations provided as a dictionary.

        Parameters
        ----------
        state : dict of NDArray
            The current state to be stored.
        next_state : dict of NDArray
            The next state to be stored.
        """
        assert type(self.states) is dict and type(self.next_states) is dict
        for i, _ in state.items():
            self.states[i] = np.append(
                self.states[i], state[i].reshape((1,) + state[i].shape), axis=0
            )
            self.next_states[i] = np.append(
                self.next_states[i],
                next_state[i].reshape((1,) + state[i].shape),
                axis=0,
            )
        if self.states[list(self.states.keys())[0]].shape[0] > self.capacity:
            for i in state:
                self.states[i] = self.states[i][1:]
                self.next_states[i] = self.next_states[i][1:]

    def store_multimodal_ragged(self, state: NDArray, next_state: NDArray) -> None:
        """
        This function stores multimodal observations provided as a ragged array.

        Parameters
        ----------
        state : NDArray
            The current state to be stored.
        next_state : NDArray
            The next state to be stored.
        """
        assert type(self.states) is NDArray and type(self.next_states) is NDArray
        for i in range(state.size):
            self.states[i] = np.append(
                self.states[i], state[i].reshape((1,) + state[i].shape), axis=0
            )
            self.next_states[i] = np.append(
                self.next_states[i],
                next_state[i].reshape((1,) + state[i].shape),
                axis=0,
            )
        if self.states[0].shape[0] > self.capacity:
            for i in state:
                self.states[i] = self.states[i][1:]
                self.next_states[i] = self.next_states[i][1:]

    def retrieve_simple(self, idx: NDArray[np.int_]) -> tuple[NDArray, NDArray]:
        """
        This function retrieves simple observations.

        Parameters
        ----------
        idx : NDArray
            The indeces of the observations that should be retrieved.

        Returns
        -------
        states : NDArray
            The batch of current states.
        next_states : NDArray
            The batch of next states.
        """
        assert type(self.states) is np.ndarray and type(self.next_states) is np.ndarray
        return np.copy(self.states[idx]), np.copy(self.next_states[idx])

    def retrieve_multimodal_list(
        self, idx: NDArray[np.int_]
    ) -> tuple[list[NDArray], list[NDArray]]:
        """
        This function retrieves multimodal observations as a list.

        Parameters
        ----------
        idx : NDArray
            The indeces of the observations that should be retrieved.

        Returns
        -------
        states : list of NDArray
            The batch of current states.
        next_states : list of NDArray
            The batch of next states.
        """
        assert type(self.states) is list and type(self.next_states) is list
        states = [self.states[i][idx] for i in range(len(self.states))]
        next_states = [self.next_states[i][idx] for i in range(len(self.next_states))]

        return states, next_states

    def retrieve_multimodal_dict(
        self, idx: NDArray[np.int_]
    ) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
        """
        This function retrieves multimodal observations as a dictionary.

        Parameters
        ----------
        idx : NDArray
            The indeces of the observations that should be retrieved.

        Returns
        -------
        states : dict of NDArray
            The batch of current states.
        next_states : dict of NDArray
            The batch of next states.
        """
        assert type(self.states) is dict and type(self.next_states) is dict
        states = {modality: obs[idx] for modality, obs in self.states.items()}
        next_states = {modality: obs[idx] for modality, obs in self.next_states.items()}

        return states, next_states

    def retrieve_multimodal_ragged(
        self, idx: NDArray[np.int_]
    ) -> tuple[NDArray, NDArray]:
        """
        This function retrieves multimodal observations as a ragged array.

        Parameters
        ----------
        idx : NDArray
            The indeces of the observations that should be retrieved.

        Returns
        -------
        states : NDArray
            The batch of current states.
        next_states : NDArray
            The batch of next states.
        """
        assert type(self.states) is NDArray and type(self.next_states) is NDArray
        states = np.array([None for i in self.states], dtype=object)
        next_states = np.array([None for i in self.states], dtype=object)
        for i in range(self.states.size):
            states[i] = self.states[i][idx]
            next_states[i] = self.next_states[i][idx]

        return states, next_states
