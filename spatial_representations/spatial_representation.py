import abc
from abc import ABC

class SpatialRepresentation(ABC):
    '''
    This class has abstract function definitions
    '''
    def __init__(self):
        print('the spatial representation baseline class')
        
    @abc.abstractmethod
    def set_visual_debugging(self,visual_output):
        return
        
    @abc.abstractmethod
    def sample_state_space(self):
        return

    @abc.abstractmethod
    def generate_behavior_from_action(self,action):
        return

    @abc.abstractmethod
    def get_action_space(self):
        return
    
