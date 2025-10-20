# basic imports
import abc
import numpy as np
# typing
from numpy.typing import NDArray


class Metric(abc.ABC):
    """
    Abstract metrics class.

    Attributes
    ----------
    D : NDArray
        The state-state similarity metric.

    """

    def __init__(self) -> None:
        self.D: NDArray

    @abc.abstractmethod
    def update_transitions(self) -> None:
        """
        This function updates the metrics when changes in the environment occur.
        """


class Euclidean(Metric):
    """
    Euclidean metrics class.

    Parameters
    ----------
    width : int
        The width of the environment in number of states.
    height : int
        The height of the environment in number of states.

    Attributes
    ----------
    D : NDArray
        The state-state similarity metric.

    """

    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        # prepare similarity matrix
        nb_states = width * height
        self.D = np.zeros((nb_states, nb_states))
        # compute euclidean distances between all state pairs
        # (exp(-distance) serves as the similarity measure)
        for s1 in range(nb_states):
            coords_1 = np.array(divmod(s1, width))
            for s2 in range(s1, nb_states):
                coords_2 = np.array(divmod(s2, width))
                distance = np.sqrt(np.sum((coords_1 - coords_2) ** 2))
                self.D[s1, s2] = np.exp(-distance)
                self.D[s2, s1] = np.exp(-distance)

    def update_transitions(self) -> None:
        """
        This function updates the metrics when changes in the environment occur.
        """
        pass


class SR(Metric):
    """
    Successor Representation (SR) metrics class.

    Parameters
    ----------
    sas : NDArray
        The state-action-state transition matrix representing
        the gridworld environment.
    gamma : float
        The discount factor used to compute the SR.

    Attributes
    ----------
    sas : NDArray
        The state-action-state transition matrix representing
        the gridworld environment.
    gamma : float
        The discount factor used to compute the SR.
    D : NDArray
        The state-state similarity metric.

    Examples
    --------

    The SR metric can be easily set up in combination with
    gridworld environments. ::

        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> from cobel.memory.utils.metrics import SR
        >>> world = make_open_field(4, 4)
        >>> metric = SR(world['sas'], 0.9)

    """

    def __init__(self, sas: NDArray, gamma: float) -> None:
        super().__init__()
        self.sas = sas
        self.gamma = gamma
        # prepare similarity matrix
        self.D = np.sum(self.sas, axis=1) / self.sas.shape[1]
        self.D = np.linalg.inv(np.eye(self.D.shape[0]) - self.gamma * self.D)

    def update_transitions(self) -> None:
        """
        This function updates the metric when changes in the environment occur.
        """
        self.D = np.sum(self.sas, axis=1) / self.sas.shape[1]
        self.D = np.linalg.inv(np.eye(self.D.shape[0]) - self.gamma * self.D)


class DR(Metric):
    """
    Default Representation (DR) metrics class.

    Parameters
    ----------
    width : int
        The width of the environment in number of states.
    height : int
        The height of the environment in number of states.
    sas : NDArray
        The state-action-state transition matrix representing
        the gridworld environment.
    gamma : float
        The discount factor used to compute the DR.
    invalid_transitions : list of 2-tuple of int
        A list containing invalid environmental transitions,
        i.e., walls/borders.
    T_default : NDArray or None, optional
        The default state-state transition matrix. If none,
        then one is created assuming an open field gridworld.

    Attributes
    ----------
    width : int
        The width of the environment in number of states.
    height : int
        The height of the environment in number of states.
    nb_states : int
        The number of gridworld states.
    sas : NDArray
        The state-action-state transition matrix representing
        the gridworld environment.
    gamma : float
        The discount factor used to compute the SR.
    invalid_transitions : list of 2-tuple of int
        A list containing invalid environmental transitions,
        i.e., walls/borders.
    T_default : NDArray or None
        The default state-state transition matrix. If none,
        then one is created assuming an open field gridworld.
    D0 : NDArray
        The default state-state similarity metric.
    T_new : NDArray or None
        The new state-state transition matrix. If none,
        then one is created assuming an open field gridworld.
    B : NDArray
        The DR's update matrix.
    D : NDArray
        The full state-state similarity metric.

    Examples
    --------

    The DR metric can be easily set up in combination with
    gridworld environments. ::

        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> from cobel.memory.utils.metrics import DR
        >>> world = make_open_field(4, 4)
        >>> metric = DR(4, 4, world['sas'], 0.9, [])

    """

    def __init__(
        self,
        width: int,
        height: int,
        sas: NDArray,
        gamma: float,
        invalid_transitions: list[tuple[int, int]],
        T_default: None | NDArray = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.nb_states = width * height
        self.sas = sas
        self.gamma = gamma
        self.invalid_transitions = invalid_transitions
        # compute DR
        self.T_default: NDArray
        if T_default is None:
            self.build_default_transition_matrix()
        else:
            self.T_default = T_default
        self.D0 = np.linalg.inv(np.eye(self.nb_states) - self.gamma * self.T_default)
        # compute new transition matrix
        self.T_new = np.sum(self.sas, axis=1) / self.sas.shape[1]
        # prepare update matrix B
        self.B = np.zeros(self.T_new.shape)
        if len(self.invalid_transitions) > 0:
            # determine affected states
            self.states = np.unique(np.array(self.invalid_transitions)[:, 0])
            # compute delta
            L = np.eye(self.nb_states) - self.gamma * self.T_new
            L0 = np.eye(self.nb_states) - self.gamma * self.T_default
            delta = L[self.states] - L0[self.states]
            # compute update matrix B
            alpha = np.linalg.inv(
                np.eye(self.states.shape[0]) + np.matmul(delta, self.D0[:, self.states])
            )
            self.B = np.matmul(
                np.matmul(self.D0[:, self.states], alpha), np.matmul(delta, self.D0)
            )
        # update DR with B
        self.D = self.D0 - self.B

    def update_transitions(self) -> None:
        """
        This function updates the metric when changes in the environment occur.
        """
        # compute new transition matrix
        self.T_new = np.sum(self.sas, axis=1) / self.sas.shape[1]
        # prepare update matrix B
        self.B = np.zeros(self.T_new.shape)
        if len(self.invalid_transitions) > 0:
            # determine affected states
            self.states = np.unique(np.array(self.invalid_transitions)[:, 0])
            # compute delta
            L = np.eye(self.nb_states) - self.gamma * self.T_new
            L0 = np.eye(self.nb_states) - self.gamma * self.T_default
            delta = L[self.states] - L0[self.states]
            # compute update matrix B
            alpha = np.linalg.inv(
                np.eye(self.states.shape[0]) + np.matmul(delta, self.D0[:, self.states])
            )
            self.B = np.matmul(
                np.matmul(self.D0[:, self.states], alpha), np.matmul(delta, self.D0)
            )
        # update DR with B
        self.D = self.D0 - self.B

    def build_default_transition_matrix(self) -> None:
        """
        This function builds the default transition graph in
        an open field environment under a uniform policy.
        """
        self.T_default = np.zeros((self.nb_states, self.nb_states))
        for state in range(self.nb_states):
            for action in range(4):
                h = int(state / self.width)
                w = state - h * self.width
                # left
                if action == 0:
                    w = max(0, w - 1)
                # up
                elif action == 1:
                    h = max(0, h - 1)
                # right
                elif action == 2:
                    w = min(self.width - 1, w + 1)
                # down
                else:
                    h = min(self.height - 1, h + 1)
                # determine next state
                next_state = int(h * self.width + w)
                self.T_default[state][next_state] += 0.25
