# basic imports
import numpy as np
import scipy.ndimage as nd
from scipy.spatial.distance import pdist


def classify_place_field(activity_map: np.ndarray, sigma: float = 0., size_threshold: float = 0.5,
                         cutoff: float = 0.25, verbose: bool = False) -> (bool, np.ndarray):
    '''
    This function detects whether or not a given spatial activity map contains place fields.
    
    Parameters
    ----------
    activity_map :                      A 2d numpy array of spatial activations.
    sigma :                             Sigma value used for the gaussian smoothing that is applied to the field (zero by default, i.e., no smoothing).
    size_threshold :                    The maximum size of field (proportion of arena).
    cutoff :                            The proportion of maximum activity level below which to remove activity.
    verbose :                           If true, error messages will be printed.
    
    Returns
    ----------
    has_field :                         Flag indicating whether the activity map contains a place field.
    field :                             The (thresholded) activity map.
    '''
    assert size_threshold > 0 and size_threshold < 1, 'The field size threshold is invalid!'
    assert cutoff > 0 and cutoff < 1, 'The activity cutoff threshold is invalid!'
    #find clusters where activity falls off below threshold and calculate a boolean mask of clusters
    field = np.where(activity_map < cutoff * np.max(activity_map), 0, activity_map)
    # label blobs
    labels, features = nd.label(field > 0)
    # prepare variables
    largest, area = 0, [np.product(activity_map.shape)]
    # identify place fields
    if features != 0:
        #find largest cluster
        area = np.prod([field[field_slice].shape for field_slice in nd.find_objects(labels)], axis=1)
        largest = np.argmax(area)
        field = nd.gaussian_filter(np.where(labels == largest + 1, field, 0), sigma=sigma)
    # check for errors
    error = 1 if features == 0 else 2 if features > 4 else 3 if area[largest] > size_threshold * np.product(activity_map.shape) else 0
    has_field = not bool(error)
    print(('%s, not place like' % {0: '', 1: 'no blobs found', 2: 'too many blobs', 3: 'too large'}[error]) * (not has_field) * verbose, end='\n' * (not has_field) * verbose)
    
    return has_field, field

def calculate_pdist(fields_sliced: np.ndarray, place_indices: np.ndarray) -> (np.ndarray, np.ndarray):
    '''
    This function calculates the pairwise distances between field centers for specified fields.
    
    Parameters
    ----------
    fields_sliced :                     The fields for which the pairwise distances will be calculated.
    place_indices :                     (legacy, remove)
    
    Returns
    ----------
    pdist_centers :                     The pairwise distances's centers.
    pdist :                             The pairwise distances.
    '''
    centers = np.zeros((max(1, len(place_indices)), 6, 2))
    for i, field in enumerate(fields_sliced):
        for j, angle in enumerate(field):
            centers[i, j] = divmod(np.argmax(angle), 25)
    pdist_centers = np.array([pdist(centers[i]) for i in range(len(centers))])
    angles = np.deg2rad(np.arange(0, 360, 60, dtype='int32').reshape(-1, 1))
    vectors = np.squeeze(np.array([(np.cos(a), np.sin(a)) for a in angles]))
    
    return pdist_centers, pdist(vectors)

def place_like(fields: np.ndarray, resolution: tuple,sigma: float = 0., size_threshold: float = 0.5,
               cutoff: float = 0.25, verbose: bool = False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    '''
    This function computes the number of place fields for a set of trials.
    
    Parameters
    ----------
    fields :                            The activity maps across trials.
    resolution :                        The spatial resolution of the activity maps.
    sigma :                             Sigma value used for the gaussian smoothing that is applied to the field (zero by default, i.e., no smoothing).
    size_threshold :                    The maximum size of field (proportion of arena).
    cutoff :                            The proportion of maximum activity level below which to remove activity.
    verbose :                           If true, analysis information and error messages will be printed.
    
    Returns
    ----------
    fields :                            The place fields.
    field_indeces :                     The place field indeces.
    pdist_centers :                     The pairwise distances's centers.
    pdist :                             The pairwise distances.
    '''
    area, n_units = np.product(resolution), fields.shape[2]
    mean_fields = np.mean(fields, axis=0).reshape((1, area, n_units))
    stacked = np.vstack((fields, mean_fields))
    
    is_field, largest_field = np.zeros((stacked.shape[0], n_units)), np.zeros((stacked.shape[0], area, n_units))
    print('Place' * verbose, end='\n' * verbose)
    for i, f in enumerate(np.rollaxis(stacked, 2)):
        print(('Unit %d' % i) * verbose, end='\n' * verbose)
        for j, hd in enumerate(f):
            print(('Head direction %d' % j) * verbose, end='\n' * verbose)
            isit, big = classify_place_field(hd.reshape(resolution), sigma, size_threshold, cutoff, verbose)
            is_field[j, i] = isit
            largest_field[j, :, i] = big.flatten()
            
    #if all fields and the mean are classified, it may be a place like representation
    ids = np.squeeze(np.array(np.where(np.all(is_field, axis=0)))) #ids where fields are place like
    if ids.shape != () and ids.size > 0:
        f = np.array([largest_field[:, :, p] for p in ids])
        pd, v = calculate_pdist(f[:, :6, :], ids)
    else:
        f = np.array([largest_field[:, :, ids]])
        pd, v = [15 * np.ones((15,))], [15 * np.ones((15,))]
        print(('no slice found--' + str(ids)) * verbose, end='\n' * verbose)
    #if there is too much directional modulation, not field
    x_all = np.all(np.array(pd) < 10, axis=1)
    ids = np.where(x_all, ids, -1)
    ids = ids[ids != -1]
    f = np.array([largest_field[:, :, p] for p in ids])
    if ids.size > 0:
        pd, v = calculate_pdist(f[:, :6, :], ids)
    else:
        print(('no slice found--' + str(ids)) * verbose, end='\n' * verbose)
        pd = [15 * np.ones((15,))]
    
    return f, ids, pd, v
