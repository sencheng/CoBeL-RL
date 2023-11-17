# basic imports
from typing import Callable


class AbstractOptimizer():
    
    def __init__(self, population_size: int = 1):
        '''
        The abstract optimizer class.
        
        Parameters
        ----------
        population_size :                   The number of model instances that will created per task in each parameter combination.
        
        Returns
        ----------
        None
        '''
        # fitting parameters
        self.population_size = population_size
        
    def fit(self, simulation_run: Callable, tasks: dict, data: dict, loss) -> dict:
        '''
        This function fits a given model to behavioral data.
        
        Parameters
        ----------
        simulation_run :                    The python function representing one simulation run.
        tasks :                             A dictionary containing different tasks (e.g. trial sequences) that the model has to be run for.
        data :                              A dictionary containing behavioral data for the different tasks.
        loss :                              The loss function that is used for computing the fit.
        
        Returns
        ----------
        fitness :                           A dictionary containing the fitness values for all parameter combinations.
        '''
        raise NotImplementedError('.fit() function not implemented!')
