# basic imports
import numpy as np
# typing
from numpy.typing import NDArray


def state_to_coordinates(state: int, width: int, y_first: bool = True) -> NDArray:
    """
    This function computes the coordinates of a gridworld state.

    Parameters
    ----------
    state : int
        The state index.
    width : int
        The width of the gridworld.
    y_first : bool, default=True
        If true, the y-coordinate (i.e., the vertical coordinate) comes first.

    Returns
    -------
    coordinates : NDArray
        The state coordinates.
    """
    assert state >= 0 and width > 0
    y, x = divmod(state, width)

    return np.array([y, x]) if y_first else np.array([x, y])


def states_to_coordinates(states: NDArray, width: int, y_first: bool = True) -> NDArray:
    """
    This function computes the coordinates of a set of gridworld states.

    Parameters
    ----------
    states : int
        A numpy array containing the gridworld state indeces.
    width : int
        The width of the gridworld.
    y_first : bool, default=True
        If true, the y-coordinate (i.e., the vertical coordinate) comes first.

    Returns
    -------
    coordinates : NDArray
        The state coordinates.
    """
    assert np.amin(states) >= 0 and width > 0
    y, x = np.divmod(states.reshape((states.shape[0], 1)), width)

    return np.hstack((y, x)) if y_first else np.hstack((x, y))
