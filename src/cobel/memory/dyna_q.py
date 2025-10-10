# basic imports
import numpy as np
# typing
from typing import TypedDict, NotRequired


class Experience(TypedDict):
    state: int
    action: int
    reward: float
    next_state: int
    terminal: int
    td: NotRequired[float]


class DynaQMemory:
    """
    Memory module to be used with the Dyna-Q agent.
    Experiences are stored as a static table.

    Parameters
    ----------
    states : int
        The number of environmental state.
    actions : int
        The number of the agent's actions.
    learning_rate : float, default=0.9
        The learning rate with which experiences are updated.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    number_of_states : int
        The number of environmental state.
    number_of_actions : int
        The number of the agent's actions.
    learning_rate : float, default=0.9
        The learning rate with which experiences are updated.
    rewards : NDArray
        Stores the rewards for each environmental transition.
    states : NDArray
        Stores the follow-up state for each environmental transition.
    terminals : NDArray
        Stores the terminality for each environmental transition.
    rng : numpy.random.Generator
        A random number generator instance used for probablistic replay.

    Examples
    --------

    Initializing the memory module for a simple gridworld. ::

        >>> from cobel.memory import DynaQMemory
        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> world = make_open_field(4, 4)
        >>> memory = DynaQMemory(world['states'], 4)

    """

    def __init__(
        self,
        states: int,
        actions: int,
        learning_rate: float = 0.9,
        rng: None | np.random.Generator = None,
    ) -> None:
        self.rng = np.random.default_rng() if rng is None else rng
        self.number_of_states = states
        self.number_of_actions = actions
        self.learning_rate = learning_rate
        self.rewards = np.zeros((states, actions))
        self.states = np.tile(np.arange(states).reshape(states, 1), actions).astype(int)
        self.terminals = np.zeros((states, actions)).astype(int)

    def store(self, experience: Experience) -> None:
        """
        This function stores a given experience.

        Parameters
        ----------
        experience : Experience
            The experience to be stored.
        """
        (
            state,
            action,
        ) = experience['state'], experience['action']
        reward = experience['reward']
        next_state, terminal = experience['next_state'], experience['terminal']
        self.rewards[state, action] += self.learning_rate * (
            reward - self.rewards[state, action]
        )
        self.states[state, action] = next_state
        self.terminals[state, action] = terminal

    def retrieve(self, state: int, action: int) -> Experience:
        """
        This function retrieves the experience for a specified state-action pair.

        Parameters
        ----------
        state : int
            The state.
        action : int
            The action.

        Returns
        -------
        experience : Experience
            The experience that is retrieved.
        """
        return {
            'state': state,
            'action': action,
            'reward': self.rewards[state, action],
            'next_state': self.states[state, action],
            'terminal': self.terminals[state, action],
        }

    def retrieve_batch(self, batch_size: int = 32) -> list[Experience]:
        """
        This function retrieves a number of random experiences.

        Parameters
        ----------
        batch_size : int
            The number of random experiences that will be retrieved.

        Returns
        -------
        batch : list of Experience
            The batch of experiences that is retrieved.
        """
        # draw random experiences
        idx = self.rng.integers(
            0, self.number_of_states * self.number_of_actions, batch_size
        )
        idx_unrav = np.array(
            np.unravel_index(idx, (self.number_of_states, self.number_of_actions))
        )
        # prepare replay batch
        batch: list[Experience] = []
        for i in range(batch_size):
            state, action = idx_unrav[0, i], idx_unrav[1, i]
            batch.append(
                {
                    'state': state,
                    'action': action,
                    'reward': self.rewards[state, action],
                    'next_state': self.states[state, action],
                    'terminal': self.terminals[state, action],
                }
            )

        return batch
