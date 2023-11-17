import abc
import gymnasium as gym


class SpatialRepresentation(abc.ABC):
    
    def __init__(self):
        '''
        This class has abstract function definitions
        '''
        pass
        
    @abc.abstractmethod
    def set_visual_debugging(self, visual_output: bool):
        return
        
    @abc.abstractmethod
    def sample_state_space(self):
        return

    @abc.abstractmethod
    def generate_behavior_from_action(self, action: int) -> dict:
        return

    @abc.abstractmethod
    def get_action_space(self) -> gym.spaces.Discrete:
        return    
