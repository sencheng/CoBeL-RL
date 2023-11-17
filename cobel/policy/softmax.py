# basic imports
import numpy as np
# framework imports
from cobel.policy.policy import AbstractPolicy


class SimpleSoftmax(AbstractPolicy):
    
    def __init__(self, beta: float = 1.):
        '''
        This class implements a softmax policy.
        
        Parameters
        ----------
        epsilon :                           The amount of exploration.\n
        
        Returns
        ----------
        None
        '''
        assert beta >= 0., 'Beta must be non-negative!'
        self.beta = beta
    
    def select_action(self, v: np.ndarray, mask: None | np.ndarray = None) -> int:
        '''
        This function selects an action for a given set of values.
        
        Parameters
        ----------
        v :                                 The value or values.\n
        mask :                              An optional action mask.\n
        
        Returns
        ----------
        action :                            The selected action.\n
        '''
        probs = self.get_action_probs(v, mask)
        
        return np.random.choice(np.arange(v.shape[0]), p=probs)        
    
    def get_action_probs(self, v: np.ndarray, mask: None | np.ndarray = None) -> np.ndarray:
        '''
        This function computes the action selection probabilities for a given set of values.
        
        Parameters
        ----------
        v :                                 The value or values.\n
        mask :                              An optional action mask.\n
        
        Returns
        ----------
        probs :                             The action selection probabilities.\n
        '''
        values, actions, probs = np.copy(v), np.arange(v.shape[0]), np.zeros(v.shape)
        # remove masked action
        if not mask is None:
            assert np.sum(mask) > 0, 'The action mask masks all actions!'
            values, actions = values[mask], actions[mask]
        # substract maximum value to prevent numerical problems
        values -= np.amax(values)
        # compute selection probabilities
        probs[actions] = np.exp(values * self.beta)
        probs[actions] /= np.sum(probs[actions])
        
        return probs
    