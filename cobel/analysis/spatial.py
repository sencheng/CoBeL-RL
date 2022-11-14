# basic imports
import numpy as np


def get_activity_maps(observations, model, layer=-2, units=np.array([0])) -> np.ndarray:
    '''
    This function returns the unit activation of a specified layer for a set of observations.
    
    Parameters
    ----------
    observations :                      A dictionary containing observations for different input streams.
    model :                             The keras model.
    layer :                             The layer index for which activity maps will be computed.
    units :                             The indices of units whose activity will be returned.
    
    Returns
    ----------
    activity_maps :                     The layer activities of the specified layer for the batch of input samples.
    '''
    activity_maps = np.zeros((units.shape[0], observations[0].shape[0]))
    if model is not None:
        activity_maps = model.get_layer_activity(observations, layer)[units, :]
        
    return np.copy(activity_maps)

def process_activity_maps(activity_maps: np.ndarray, threshold=0.15) -> np.ndarray:
    '''
    This function normalizes activity maps and removes values below a specified threshold.
    
    Parameters
    ----------
    activity_maps :                     The activity maps.
    threshold :                         The activity threshold.
    
    Returns
    ----------
    activity_maps :                     The  processed layer activities.
    '''
    # feature scaling
    activity_maps -= np.amin(activity_maps, axis=1).reshape((activity_maps.shape[0], 1))
    activity_maps /= np.amax(activity_maps, axis=1).reshape((activity_maps.shape[0], 1))
    # thresholding
    activity_maps[activity_maps < 0.15] = 0.
    
    return activity_maps
