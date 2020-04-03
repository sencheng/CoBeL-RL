
import abc

from abc import ABC

class SpatialRepresentation(ABC):
    
    # this class has abstract function definitions
    
    def __init__(self):
        print('the spatial representation baseline class')
        
    @abc.abstractmethod
    def sample_state_space(self):
        return
