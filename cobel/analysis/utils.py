# basic imports
import numpy as np


def state_to_coordinates(state: int, width: int, y_first: bool = True) -> np.ndarray:
    '''
    This function computes the coordinates of a gridworld state.
    
    Parameters
    ----------
    state :                             The state index.
    width :                             The width of the gridworld.
    y_first :                           If true, the y-coordinate (i.e., the vertical coordinate) comes first.
    
    Returns
    ----------
    coordinates :                       The state coordinates.
    '''
    assert state >= 0 and width > 0
    y, x = divmod(state, width)
    
    return np.array([y, x]) if y_first else np.array([x, y])

def states_to_coordinates(states: np.ndarray, width: int, y_first: bool = True) -> np.ndarray:
    '''
    This function computes the coordinates of a set of gridworld states.
    
    Parameters
    ----------
    states :                            A numpy array containing the gridworld state indeces.
    width :                             The width of the gridworld.
    y_first :                           If true, the y-coordinate (i.e., the vertical coordinate) comes first.
    
    Returns
    ----------
    coordinates :                       The state coordinates.
    '''
    assert np.amin(states) >= 0 and width > 0
    y, x = np.divmod(states.reshape((states.shape[0], 1)), width)
    
    return np.hstack((y, x)) if y_first else np.hstack((x, y))
