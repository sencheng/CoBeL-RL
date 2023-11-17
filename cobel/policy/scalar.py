# basic imports
import numpy as np
# framework imports
from cobel.policy.policy import AbstractPolicy


class Proportional(AbstractPolicy):
    
    def __init__(self, value_max: float = 1., code_reverse: bool = True):
        '''
        This class implements a policy which transforms a scalar value to a binary action.
        Action selection probabilities are proportional to the scalar value.
        
        Parameters
        ----------
        value_max :                         The maximum possible value. Used for normalization.\n
        code_reverse :                      Flag indicating whether values close to the maximum value code for action zero, i.e., the first action. True by default.\n
        
        Returns
        ----------
        None\n
        '''
        self.value_max = value_max
        self.code_reverse = code_reverse
        
    def select_action(self, v: float) -> int:
        '''
        This function selects an action for a given value.
        
        Parameters
        ----------
        v :                                 The given value.\n
        
        Returns
        ----------
        action :                            The selected action.\n
        '''
        return abs(self.code_reverse - int(np.random.rand() < v/self.value_max))
    
    def get_action_probs(self, v: float) -> np.ndarray:
        '''
        This function computes the action selection probabilities for a given value.
        
        Parameters
        ----------
        v :                                 The given value.\n
        
        Returns
        ----------
        probs :                             The action selection probabilities.\n
        '''
        probs = np.abs(np.array([1., 0.]) - v/self.value_max)
        if self.code_reverse:
            return np.flip(probs)
        return probs
        

class Threshold(AbstractPolicy):
    
    def __init__(self, threshold: float = 0.5, window: float = 0., value_max: float = 1., code_reverse: bool = True):
        '''
        This class implements a threshold policy which transforms a scalar value to a binary action.
        
        Parameters
        ----------
        threshold :                         The action threshold.\n
        window :                            The window for which random actions will be taken.\n
        value_max :                         The maximum possible value. Used for normalization.\n
        code_reverse :                      Flag indicating whether values close to the maximum value code for action zero, i.e., the first action. True by default.\n
        
        Returns
        ----------
        None\n
        '''
        assert threshold >= 0. and threshold <= 1., 'Threshold must lie within the interval [0, 1]!'
        assert threshold - window/2 > 0. and threshold + window/2 < 1., 'The window for random actions extends over the value range!'
        self.threshold = threshold
        self.window = window / 2 # divide by two so that we don't have to later
        self.value_max = value_max
        self.code_reverse = code_reverse
        
    def select_action(self, v: float) -> int:
        '''
        This function selects an action for a given value.
        
        Parameters
        ----------
        v :                                 The given value.\n
        
        Returns
        ----------
        action :                            The selected action.\n
        '''
        v /= self.value_max
        action = abs(int(self.code_reverse) - int(v > self.threshold))
        if v > self.threshold - self.window and v < self.threshold + self.window:
            action = np.random.randint(2)
            
        return action
    
    def get_action_probs(self, v: float) -> np.ndarray:
        '''
        This function computes the action selection probabilities for a given value.
        
        Parameters
        ----------
        v :                                 The given value.\n
        
        Returns
        ----------
        probs :                             The action selection probabilities.\n
        '''
        v /= self.value_max
        probs = np.zeros(2)
        probs[abs(int(self.code_reverse) - int(v > self.threshold))] = 1.
        if v > self.threshold - self.window and v < self.threshold + self.window:
            probs = np.ones(2) * 0.5
            
        return probs
    
    
class Sigmoid(AbstractPolicy):
    
    def __init__(self, threshold: float = 0.5, scale: float = 10., value_max: float = 1., code_reverse: bool = True):
        '''
        This class implements a policy which transforms a scalar value to a binary action.
        Action selection probabilities are proportional to the sigmoid transformed scalar value.
        
        Parameters
        ----------
        threshold :                         The action threshold, i.e., P(v) = 0.5.\n
        scale :                             The scaling factor for the sigmoid's steepness.\n
        value_max :                         The maximum possible value. Used for normalization.\n
        code_reverse :                      Flag indicating whether values close to the maximum value code for action zero, i.e., the first action. True by default.\n
        
        Returns
        ----------
        None\n
        '''
        assert threshold > 0. and threshold < 1., 'The threshold must lie within the range (0, 1)!'
        assert scale >= 0., 'The sigmoid\'s scaling factor must be non-negative!'
        self.threshold = threshold
        self.scale = scale
        self.value_max = value_max
        self.code_reverse = code_reverse
        
    def select_action(self, v: float) -> int:
        '''
        This function selects an action for a given value.
        
        Parameters
        ----------
        v :                                 The given value.\n
        
        Returns
        ----------
        action :                            The selected action.\n
        '''
        prob = 1 / (1 + np.exp(-(v/self.value_max - self.threshold) * self.scale))
        
        return abs(self.code_reverse - int(np.random.rand() < prob))
    
    def get_action_probs(self, v: float) -> np.ndarray:
        '''
        This function computes the action selection probabilities for a given value.
        
        Parameters
        ----------
        v :                                 The given value.\n
        
        Returns
        ----------
        probs :                             The action selection probabilities.\n
        '''
        probs = np.abs(np.array([1., 0.]) - 1 / (1 + np.exp(-(v/self.value_max -self.threshold) * self.scale)))
        if self.code_reverse:
            return np.flip(probs)
        return probs
    