# basic imports
import numpy as np
# tensorflow
from tensorflow.keras import backend as K


def get_activity_maps(observations, model, layer=-2, units=[0]):
    '''
    This function returns the unit activation of a specified layer for a set of observations.
    
    | **Args**
    | observations:                 A dictionary containing observations for different input streams.
    | model:                        The keras model.
    | layer:                        The layer index for which activity maps will be computed.
    | units:                        The indices of units whose activity will be returned.
    '''
    # construct "sub model"
    sub_model = K.function([model.layers[input_idx].input for input_idx in observations], [model.layers[layer].output])
    # compute activity maps
    activity_maps = sub_model([observations[input_idx] for input_idx in observations])[0][:, units].T
        
    return activity_maps

def process_activity_maps(activity_maps, threshold=0.15):
    '''
    This function normalizes activity maps and removes values below a specified threshold.
    
    | **Args**
    | activity_maps:                The activity maps.
    | threshold:                    The activity threshold.
    '''
    # feature scaling
    activity_maps -= np.amin(activity_maps, axis=1).reshape((activity_maps.shape[0], 1))
    activity_maps /= np.amax(activity_maps, axis=1).reshape((activity_maps.shape[0], 1))
    # thresholding
    activity_maps[activity_maps < 0.15] = 0.
    
    return activity_maps