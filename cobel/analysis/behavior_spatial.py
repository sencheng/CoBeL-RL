# basic imports
import numpy as np


def get_occupancy_map(trajectories: list, width: float, height: float, bin_size: float, margins: str = 'expand') -> np.ndarray:
    '''
    This function computes the occupancy map for a given set of trajectories and bin size.
    
    Parameters
    ----------
    trajectories :                      A list of trajectories. Trajectories are expected to be numpy arrays of (non-negative) coordinates.
    width :                             The width of the environment.
    height :                            The height of the environment.
    bin_size :                          The size of the spatial bin.
    margins :                           Margin handling mode. Possible modes are \'expand\' (adds bins), \'include\' (clips coordinates) and \'ignore\' (ignores margins).
    
    Returns
    ----------
    occupancy :                         The occupancy map.
    '''
    assert width > 0 and height > 0, 'Invalid environment dimensions! Dimensions must be positive!'
    assert bin_size > 0 and bin_size <= min(width, height), 'Invalid bin size! Bin size must be positive and less than environmental dimensions!'
    assert margins in ['expand', 'include', 'ignore'], 'Invalid handling mode for margins! Must be \'expand\', \'include\' or \'ignore\'!'
    # compute number of bins
    bins = np.array([int(height/bin_size), int(width/bin_size)])
    bins +=  ((np.array([height, width]) - bins * bin_size) > 0.) * (margins == 'expand')
    # compute coordinate range
    coordinate_range = np.hstack((np.zeros((2, 1)), bins.reshape((2, 1)) * bin_size))
    # occupancy map for trajectories
    occupancy = np.zeros(tuple(bins))
    for trajectory in trajectories:
        assert np.amin(trajectory) >= 0., 'Invalid coordinates! Coordinates must be non-negative!'
        values_max = bins * bin_size if margins == 'include' else [None, None]
        occupancy += np.histogram2d(np.clip(trajectory[:, 0], a_min=0, a_max=values_max[0]),
                                    np.clip(trajectory[:, 1], a_min=0, a_max=values_max[1]),
                                    bins=bins, range=coordinate_range)[0]
    
    return occupancy

def match(sequence: np.ndarray, template: np.ndarray) -> np.ndarray:
    '''
    This function computes the number of matching states between a given sequene and template.
    
    Parameters
    ----------
    sequence :                          A sequence of state indeces.
    template :                          A template sequence of state indeces.
    
    Returns
    ----------
    match :                             A numpy array containing the number of matching state indeces for each point in the sequence.
    '''
    # prepare sequence and template for matching
    T = -np.ones((sequence.shape[0] + template.shape[0] * 2, sequence.shape[0] + template.shape[0] * 2))
    S = np.hstack((-np.ones(template.shape[0]), sequence, -np.ones(template.shape[0])))
    for t in range(sequence.shape[0] + template.shape[0]):
        T[t, t:(t + template.shape[0])] = template
    # compute number of matches
    matches = np.sum(T[:, template.shape[0]:-template.shape[0]] == S[template.shape[0]:-template.shape[0]], axis=1)
    
    return matches[template.shape[0]:-template.shape[0]]
