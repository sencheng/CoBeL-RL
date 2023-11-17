# basic imports
import numpy as np


class AbstractPolicy():
    
    def __init__(self):
        '''
        Abstract class of an action policy.
        
        Parameters
        ----------
        None\n
        
        Returns
        ----------
        None\n
        '''
        pass
    
    def select_action(self, v: float | np.ndarray, mask: None | np.ndarray = None) -> int | float | np.ndarray:
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
        raise NotImplementedError('.select_action() function not implemented!')
    
    def get_action_probs(self, v: float | np.ndarray, mask: None | np.ndarray = None) -> float | np.ndarray:
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
        raise NotImplementedError('.get_action_probs() function not implemented!')
    