# basic imports
import numpy as np
# framework imports
from cobel.policy.policy import AbstractPolicy


class RandomDiscrete(AbstractPolicy):
    
    def __init__(self, actions: int):
        '''
        This class implements a random action policy.
        
        Parameters
        ----------
        actions :                           The number of actions.\n
        
        Returns
        ----------
        None\n
        '''
        assert actions > 0, 'The number of actions must be at least one!'
        self.actions = actions
    
    def select_action(self, v: None | float | np.ndarray = None, mask: None | np.ndarray = None) -> int:
        '''
        This function returns a random discrete action.
        
        Parameters
        ----------
        v :                                 The value or values.\n
        mask :                              An optional action mask.\n
        
        Returns
        ----------
        action :                            The selected action.\n
        '''
        return np.random.randint(self.actions)
    
    def get_action_probs(self, v: None | float | np.ndarray = None, mask: None | np.ndarray = None) -> np.ndarray:
        '''
        This function computes the action selection probabilities.
        
        Parameters
        ----------
        v :                                 The value or values.\n
        mask :                              An optional action mask.\n
        
        Returns
        ----------
        probs :                             The action selection probabilities.\n
        '''
        return np.ones(self.actions) / self.actions
    

class RandomUniform(AbstractPolicy):
    
    def __init__(self, action_min: float | np.ndarray, action_max: float | np.ndarray):
        '''
        This class implements a random continuous (uniform) policy.
        
        Parameters
        ----------
        action_min :                        The minimum value for each action.\n
        action_max :                        The maximum value for each action.\n
        
        Returns
        ----------
        None\n
        '''
        self.action_min = action_min
        self.action_max = action_max
    
    def select_action(self, v: None | float | np.ndarray = None, mask: None | np.ndarray = None) -> float | np.ndarray:
        '''
        This function returns a random continuous (uniform) action.
        
        Parameters
        ----------
        v :                                 The value or values.\n
        mask :                              An optional action mask.\n
        
        Returns
        ----------
        action :                            The selected action.\n
        '''
        return np.random.uniform(self.action_min, self.action_max)
    
    def get_action_probs(self, v: None | float | np.ndarray = None, mask: None | np.ndarray = None) -> None:
        '''
        This function computes the action selection probabilities.
        
        Parameters
        ----------
        v :                                 The value or values.\n
        mask :                              An optional action mask.\n
        
        Returns
        ----------
        probs :                             The action selection probabilities.\n
        '''
        print('Action probabilities not defined for continuous actions!')
        return None
    
    
class RandomGaussian(AbstractPolicy):
    
    def __init__(self, mean: float | np.ndarray, std: float | np.ndarray, action_min: None | float | np.ndarray = None, action_max: None | float | np.ndarray = None):
        '''
        Abstract class of a random continuous (gaussian) policy.
        
        Parameters
        ----------
        mean :                              The mean value for each action.\n
        std :                               The standard deviation for each action.\n
        action_min :                        The minimum value for each action.\n
        action_max :                        The maximum value for each action.\n
        
        Returns
        ----------
        None\n
        '''
        self.mean = mean
        self.std = std
        self.action_min = action_min
        self.action_max = action_max
    
    def select_action(self, v: None | float | np.ndarray = None, mask: None | np.ndarray = None) -> float | np.ndarray:
        '''
        This function returns a random continuous (gaussian) action.
        
        Parameters
        ----------
        v :                                 The value or values.\n
        mask :                              An optional action mask.\n
        
        Returns
        ----------
        action :                            The selected action.\n
        '''
        if self.action_max is None and self.action_min is None:
            return np.random.normal(self.mean, self.std)
        return np.clip(np.random.normal(self.mean, self.std), a_min=self.action_min, a_max=self.action_max)
    
    def get_action_probs(self, v: None | float | np.ndarray = None, mask: None | np.ndarray = None) -> None:
        '''
        This function computes the action selection probabilities.
        
        Parameters
        ----------
        v :                                 The value or values.\n
        mask :                              An optional action mask.\n
        
        Returns
        ----------
        probs :                             The action selection probabilities.\n
        '''
        print('Action probabilities not defined for continuous actions!')
        return None
    