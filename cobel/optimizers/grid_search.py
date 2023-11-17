# basic imports
import numpy as np
import pickle
import copy
import multiprocessing.pool as mp
from os import listdir
from os.path import isfile, join
from typing import Callable
from itertools import product
# module imports
from cobel.optimizers.optimizer import AbstractOptimizer


class GridSearchOptimizer(AbstractOptimizer):
    
    def __init__(self, file_path: str, parameters: dict, population_size: int = 1, order: str = 'nested'):
        '''
        The grid search optimizer class.
        
        Parameters
        ----------
        file_path :                         The directory to which the simulation results will be written.\n
        parameters :                        A dictionary containing the different parameter values which will be searched over.\n
        population_size :                   The number of model instances that will created per task in each parameter combination.\n
        order :                             The order in which parameter combinations are written to the dictionary: \'nested\' (nested in order of values), \'shuffled\' (like nested but values are shuffled before) and \'systematic\' ("resolves" parameter space).\n
        
        Returns
        ----------
        None\n
        '''
        super().__init__(population_size)
        # initialize memory structures
        self.parameters = copy.deepcopy(parameters)
        for param in self.parameters:
            if type(self.parameters[param]) is np.ndarray:
                self.parameters[param] = np.sort(self.parameters[param])
        self.prepare_parameter_combinations(self.parameters, order)
        self.file_path = file_path
        # retrieve files present in file path
        self.present_files = [f for f in listdir(self.file_path) if isfile(join(self.file_path, f))]
        
    def prepare_parameter_combinations(self, parameters: dict, order: str = 'nested'):
        '''
        The functions prepares a dictionary containing the different parameter combinations.
        Parameter combinations can be written to the dictionary in different orders.
        
        Parameters
        ----------
        parameters :                        A dictionary containing the different parameter values which will be searched over.\n
        order :                             The order in which parameter combinations are written to the dictionary: \'nested\' (nested in order of values), \'shuffled\' (like nested but values are shuffled before) and \'systematic\' ("resolves" parameter space).\n
        
        Returns
        ----------
        None\n
        '''
        assert order in ['nested', 'shuffled', 'systematic'], 'Invalid order!'
        self.parameter_combinations, combinations = {}, []
        if order == 'systematic':
            strides, max_length = [], 0
            # determine strides for each parameter
            for param in parameters:
                strides.append([])
                for i in range(int(np.ceil(np.sqrt(len(parameters[param]))))):
                    strides[-1].append(max(int(len(parameters[param])/(2**(i + 1))), 1))
                if 1 not in strides[-1]:
                    strides[-1].append(1)
                max_length = max(max_length, len(strides[-1]))
            # fill in missing strides
            for i in range(len(strides)):
                strides[i] += [1 for j in range(max_length - len(strides[i]))]
            # determine parameter combinations
            for i in range(max_length):
                p = []
                for s, stride in enumerate(strides):
                    p.append([])
                    for v in range(len(list(parameters.values())[s]) // stride[i]):
                        p[-1].append(list(parameters.values())[s][v * stride[i]])
                combinations.append(product(*p))
        else:
            # shuffle values
            if order == 'shuffled':
                for param in parameters:
                    np.random.shuffle(parameters[param])
            # determine parameter combinations
            combinations.append(product(*parameters.values()))
        # store parameter combinations
        for sub_combinations in combinations:
            for combination in sub_combinations:
                if tuple(combination) not in self.parameter_combinations:
                    self.parameter_combinations[tuple(combination)] = {param: value for param, value in zip(parameters.keys(), combination)}
        
    def fit(self, simulation_run: Callable, tasks: dict, data: dict, loss: Callable, overwrite: bool = False,
            store_simulation_data: bool = False, pool: None | mp.Pool = None) -> dict:
        '''
        This function fits a given model to behavioral data.
        
        Parameters
        ----------
        simulation_run :                    The python function representing one simulation run.\n
        tasks :                             A dictionary containing different tasks (e.g. trial sequences) that the model has to be run for.\n
        data :                              A dictionary containing behavioral data for the different tasks.\n
        loss :                              The loss function that is used for computing the fit.\n
        overwrite :                         If true, simulations are run for all parameter combinations even when stored simulation data was found.\n
        store_simulation_data :             If true, simulations data is stored along with the fit.\n
        pool :                              Optional worker pool for multiprocessing. If none is given simulations are run sequentially.\n
        
        Returns
        ----------
        fitness :                           A dictionary containing the fitness values for all parameter combinations.\n
        '''
        assert tasks.keys() == data.keys(), 'Task mismatch!'
        # stores fitness for all param combinations
        fitness = {}
        if 'fit.pkl' in self.present_files:
            fitness = pickle.load(open(self.file_path + 'fit.pkl', 'rb'))
        # start fitting
        for parameter_combination in self.parameter_combinations:
            simulation_data = {}
            file_name_data = ('_%s' * len(parameter_combination)) % parameter_combination
            file_name_data = 'sim' + file_name_data + '.pkl'
            #
            if not parameter_combination in fitness or overwrite:
                if file_name_data in self.present_files and not overwrite:
                    simulation_data = pickle.load(open(self.file_path + file_name_data, 'rb'))
                else:
                    # run simulations
                    for task in tasks:
                        if pool is None:
                            simulation_data[task] = [simulation_run(tasks[task], self.parameter_combinations[parameter_combination]) for run in range(self.population_size)]
                        else:
                            simulation_data[task] = [pool.apply_async(simulation_run, (tasks[task], self.parameter_combinations[parameter_combination])) for run in range(self.population_size)]
                            simulation_data[task] = [run.get() for run in simulation_data[task]]
                # store simulation data
                if store_simulation_data:
                    pickle.dump(simulation_data, open(self.file_path + file_name_data, 'wb'))
                # compute and store fit
                fitness[parameter_combination] = loss(simulation_data, data)
                pickle.dump(fitness, open(self.file_path + 'fit.pkl', 'wb'))
            
        return fitness
    
    def recompute_fit(self, data: dict, loss: Callable, overwrite: bool = True) -> dict:
        '''
        This function recomputes the fit. Assumes that the simulation data was stored.
        
        Parameters
        ----------
        data :                              A dictionary containing behavioral data for the different tasks.\n
        loss :                              The loss function that is used for computing the fit.\n
        overwrite :                         If true, the stored fit is overwritten.\n
        
        Returns
        ----------
        fitness :                           A dictionary containing the fitness values for all parameter combinations.\n
        '''
        # try loading stored fit
        fitness = {}
        if 'fit.pkl' in self.present_files:
            fitness = pickle.load(open(self.file_path + 'fit.pkl', 'rb'))
        else:
            print('No fit to recompute!')
            return fitness
        # recompute fit for all combinations
        for parameter_combination in fitness:
            file_name_data = ('_%s' * len(parameter_combination)) % parameter_combination
            file_name_data = 'sim' + file_name_data + '.pkl'
            if file_name_data in self.present_files:
                simulation_data = pickle.load(open(self.file_path + file_name_data, 'rb'))
                fitness[parameter_combination] = loss(simulation_data, data)
            else:
                print('No simulation data found for parameter combination: ' + str(parameter_combination))
        # store recomputed fit
        if overwrite:
            pickle.dump(fitness, open(self.file_path + 'fit.pkl', 'wb'))
            
        return fitness
        