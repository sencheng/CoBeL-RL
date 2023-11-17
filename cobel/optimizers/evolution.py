# basic imports
import numpy as np
import pickle
import copy
import multiprocessing.pool as mp
from os import listdir
from os.path import isfile, join
from typing import Callable
# module imports
from cobel.optimizers.optimizer import AbstractOptimizer


class EAOptimizer(AbstractOptimizer):
    
    def __init__(self, file_path: str, parameters: dict, population_size: int = 1):
        '''
        The class implements a simple evolutionary algorithm optimizer.
        
        Parameters
        ----------
        file_path :                         The directory to which the simulation results will be written.
        parameters :                        A dictionary containing the parameters that will be optimized.
        population_size :                   The number of model instances that will created per task in each parameter combination.
        
        Returns
        ----------
        None
        '''
        super().__init__(population_size)
        # store parameters that will be optimized
        self.parameters = copy.deepcopy(parameters)
        # set default parameter settings in case none were specified
        for parameter in self.parameters:
            self.parameters[parameter].setdefault('init_range', {'low': -1, 'high': 1})
            self.parameters[parameter].setdefault('param_range', {'a_min': None, 'a_max': None})
            self.parameters[parameter].setdefault('mutator', (np.random.normal, {'scale': 0.1}))
        self.file_path = file_path
        # retrieve files present in file path
        self.present_files = [f for f in listdir(self.file_path) if isfile(join(self.file_path, f))]
        
    def fit(self, simulation_run: Callable, tasks: dict, data: dict, loss, runs: int = 1, individuals: int = 10,
            generations: int = 100, overwrite: bool = False, pool: None | mp.Pool = None) -> dict:
        '''
        This function fits a given model to behavioral data.
        
        Parameters
        ----------
        simulation_run :                    The python function representing one simulation run.
        tasks :                             A dictionary containing different tasks (e.g. trial sequences) that the model has to be run for.
        data :                              A dictionary containing behavioral data for the different tasks.
        loss :                              The loss function that is used for computing the fit.
        runs :                              The number of runs.
        individuals :                       The number of individuals in each generation.
        generations :                       The number of generations in each run.
        overwrite :                         If true, simulations are run even when stored simulation data was found.
        pool :                              Optional worker pool for multiprocessing. If none is given simulations are run sequentially.
        
        Returns
        ----------
        best_fits :                         A list containing the best individual and its fitness for each run (i.e., (run, generation, best)).
        '''
        assert tasks.keys() == data.keys(), 'Task mismatch!'
        best_fits = []
        for run in range(runs):
            # 
            file_name = self.file_path + 'run_%d.pkl' % run
            if file_name in self.present_files and not overwrite:
                best_fits.append(pickle.load(open(file_name, 'rb')))
            else:
                best_fits.append([])
                # initialize first generation
                population = [{param: self.parameters[param]['param_type'](np.random.uniform(**self.parameters[param]['init_range']))
                               for param in self.parameters} for individual in range(individuals)]
                for generation in range(generations):
                    fit = []
                    # run simulations and compute fit for each individual
                    for individual in population:
                        simulation_data = {}
                        for task in tasks:
                            if pool is None:
                                simulation_data[task] = [simulation_run(tasks[task], individual) for i in range(self.population_size)]
                            else:
                                simulation_data[task] = [pool.apply_async(simulation_run, (tasks[task], individual)) for sim in range(self.population_size)]
                                simulation_data[task] = [sim.get() for sim in simulation_data[task]]
                        fit.append(loss(simulation_data, data))
                    # determine best individual, and store its parameters and fitness
                    best = population[np.argmin(fit)]
                    best_fits[-1].append((best, np.amin(fit)))
                    pickle.dump(best_fits[-1], open(file_name, 'wb'))
                    # generate new generation from best individual
                    population = [best]
                    for individual in range(individuals - 1):
                        population.append({param: self.parameters[param]['mutator'][0](best[param], **self.parameters[param]['mutator'][1]) for param in self.parameters})
                        for param in self.parameters:
                            population[-1][param] = np.clip(self.parameters[param]['param_type'](population[-1][param]),
                                                            **self.parameters[param]['param_range'])
            
        return best_fits
                