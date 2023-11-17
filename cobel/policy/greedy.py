# basic imports
import numpy as np
# framework imports
from cobel.policy.policy import AbstractPolicy


class EpsilonGreedy(AbstractPolicy):
    
    def __init__(self, epsilon: float = 0.1):
        '''
        This class implements an epsilon-greedy policy.
        
        Parameters
        ----------
        epsilon :                           The amount of exploration.\n
        
        Returns
        ----------
        None
        '''
        assert epsilon >= 0. and epsilon <= 1., 'Epsilon must lie within the range [0, 1]!'
        self.epsilon = epsilon
    
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
        # base selection probability
        probs[actions] = self.epsilon * np.ones(values.shape) / values.shape[0]
        # greedy selection probability considering tied actions
        ties = np.amax(values) == values
        probs[actions] += (1. - self.epsilon) * ties / np.sum(ties)
        
        return probs
    
    
class ExclusiveEpsilonGreedy(EpsilonGreedy):
    
    def __init__(self, epsilon: float = 0.1):
        '''
        This class implements an epsilon-greedy policy.
        
        Parameters
        ----------
        epsilon :                           The amount of exploration.\n
        
        Returns
        ----------
        None
        '''
        super().__init__(epsilon)      
    
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
        # greedy selection probability considering tied actions
        ties = np.amax(values) == values
        probs[actions] = (1. - self.epsilon) * ties / np.sum(ties)
        # exploration probability
        probs[actions] += self.epsilon * (ties == False) / max(values.shape[0] - np.sum(ties), 1)
        
        return probs/np.sum(probs)
    