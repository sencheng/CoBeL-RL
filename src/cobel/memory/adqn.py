# basic imports
import numpy as np
import gymnasium as gym
# typing
from typing import TypedDict
from collections.abc import Callable
from numpy.typing import NDArray

SimpleStore = Callable[[NDArray], None]
ListStore = Callable[[list[NDArray]], None]
DictStore = Callable[[dict[str, NDArray]], None]
InitialStore = SimpleStore | ListStore | DictStore
SimpleRetrieve = Callable[[NDArray[np.int_]], NDArray]
ListRetrieve = Callable[[NDArray[np.int_]], list[NDArray]]
DictRetrieve = Callable[[NDArray[np.int_]], dict[str, NDArray]]
InitialRetrieve = SimpleRetrieve | ListRetrieve | DictRetrieve
StateBatch = NDArray | list[NDArray] | dict[str, NDArray]


class Experience(TypedDict):
    state: NDArray | dict[str, NDArray] | list[NDArray]
    action: float
    reward: float
    next_state: NDArray | dict[str, NDArray] | list[NDArray]
    terminal: int


class ADQNMemory:
    """
    This class implements a memory module for storing experiences,
    and replays them in a prioritized fashion according to recency
    and prediction errors.

    Parameters
    ----------
    observation_space : gym.Space
        The observation space.
    decay : float, default=1.
        The recency decay factor.
    rpe : bool, default=True
        A flag indicating whether prediction errors
        are used for replay prioritization.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    states : NDArray, list of NDArray or NDArray
        Stores the state/observation of each experience.
    reinforcements : NDArray
        Stores the reinforcements/rewards for each experience.
    errors : NDArray
        Stores the prediction error for each experience.
    priorities : NDArray
        Stores the priority for each experience.
    decay : float
        The recency decay factor.
    rpe : bool
        A flag indicating whether prediction errors
        are used for replay prioritization.
    rng : numpy.random.Generator
        A random number generator instance used for probablistic replay.

    Examples
    --------

    Initializing the memory module for unimodal
    observations. ::

        >>> import gymnasium as gym
        >>> from cobel.memory import ADQNMemory
        >>> memory = ADQNMemory(gym.space.Box(0., 1., (10, )))

    """

    def __init__(
        self,
        observation_space: gym.Space,
        decay: float = 1.0,
        rpe: bool = True,
        rng: None | np.random.Generator = None,
    ) -> None:
        assert 0 <= decay <= 1
        assert isinstance(
            observation_space, (gym.spaces.Box | gym.spaces.Dict | gym.spaces.Tuple)
        )
        self.rng = np.random.default_rng() if rng is None else rng
        # memory structure
        self.states: NDArray | list[NDArray] | dict[str, NDArray]
        self.reinforcements = np.array([], dtype=float)
        self.errors = np.array([], dtype=float)
        self.priorities = np.array([], dtype=float)
        # determine appropriate store/retrieve functions
        self._state_store: InitialStore
        self._state_retrieve: InitialRetrieve
        if isinstance(observation_space, gym.spaces.Box):
            self._state_store = self._store_unimodal
            self._state_retrieve = self._retrieve_unimodal
            self.states = np.zeros((0,) + observation_space.shape)
        elif isinstance(observation_space, gym.spaces.Dict):
            self._state_store = self._store_multimodal_dict
            self._state_retrieve = self._retrieve_multimodal_dict
            self.states = {}
            for m, space in observation_space.spaces.items():
                assert isinstance(space, gym.spaces.Box)
                self.states[m] = np.zeros((0,) + space.shape)
        else:
            self._state_store = self._store_multimodal_list
            self._state_retrieve = self._retrieve_multimodal_list
            self.states = []
            for space in observation_space.spaces:
                assert isinstance(space, gym.spaces.Box)
                self.states.append(np.zeros((0,) + space.shape))
        # learning parameters
        self.decay: float = decay
        self.rpe: bool = rpe

    def store(self, experience: Experience) -> None:
        """
        This function stores an experience tuple.

        Parameters
        ----------
        experience : Experience
            The experience to be stored.
        """
        # store experience
        self._state_store(experience['state'])  # type: ignore
        self.reinforcements = np.append(self.reinforcements, experience['reward'])
        self.errors = np.append(
            self.errors, experience['action'] - experience['reward']
        )
        # update priorities
        self.priorities *= self.decay
        self.priorities = np.append(
            self.priorities, abs(self.errors[-1]) ** int(self.rpe)
        )

    def sample_batch(self, batch_size: int) -> tuple[StateBatch, NDArray]:
        """
        This function samples an experience replay batch.

        Parameters
        ----------
        batch_size : int
            The size of the batch.

        Returns
        -------
        observations : NDArray, list of NDArray or dict of NDArray
            The observation batch.
        rewards : NDArray
            The reward batch.
        """
        probs = np.ones(self.priorities.shape[0]) / self.priorities.shape[0]
        prob_sum = np.sum(self.priorities)
        if prob_sum != 0:
            probs = self.priorities / prob_sum
        idx = self.rng.choice(
            self.priorities.shape[0], p=probs, size=batch_size, replace=True
        )

        return self._state_retrieve(idx), np.copy(self.reinforcements[idx])

    def _store_unimodal(self, state: NDArray) -> None:
        """
        Function for storing unimodal observations.
        """
        assert isinstance(self.states, np.ndarray)
        self.states = np.append(self.states, state[np.newaxis], axis=0)

    def _store_multimodal_list(self, state: list[NDArray]) -> None:
        """
        Function for storing multimodal observations using a list.
        """
        assert isinstance(self.states, list)
        for i, obs in enumerate(state):
            self.states[i] = np.append(self.states[i], obs)

    def _store_multimodal_dict(self, state: dict[str, NDArray]) -> None:
        """
        Function for storing multimodal observations using a dictionary.
        """
        assert isinstance(self.states, dict)
        for modality, obs in state.items():
            self.states[modality] = np.append(self.states[modality], obs)

    def _retrieve_unimodal(self, idx: NDArray[np.int_]) -> NDArray:
        """
        Function for retrieving unimodal observations.
        """
        assert isinstance(self.states, np.ndarray)
        return np.copy(self.states[idx])

    def _retrieve_multimodal_list(self, idx: NDArray[np.int_]) -> list[NDArray]:
        """
        Function for retrieving multimodal observations using a list.
        """
        assert isinstance(self.states, list)
        return [np.copy(obs[idx]) for obs in self.states]

    def _retrieve_multimodal_dict(self, idx: NDArray[np.int_]) -> dict[str, NDArray]:
        """
        Function for retrieving multimodal observations using a dictionary.
        """
        assert isinstance(self.states, dict)
        return {m: np.copy(obs[idx]) for m, obs in self.states.items()}
