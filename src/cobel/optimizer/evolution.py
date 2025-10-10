# basic imports
import copy
import pickle
import numpy as np
import multiprocessing as mp
from os import listdir
from os.path import isfile, join
# framework imports
from .optimizer import Optimizer
# typing
from typing import Any
from .optimizer import Simulation, FitLoss, Fit


class EAOptimizer(Optimizer):
    """
    This class implements a simple evolutionary algorithm optimizer.

    Parameters
    ----------
    file_path : str
        The directory to which the simulation results will be written.
    parameters : dict
        A dictionary containing the different parameter values
        which will be searched over.
    nb_runs : int, default=1
        The number of model instances that will be run per task
        in each parameter combination.
    population_size : int, default=1
        The size of the population.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    file_path : str
        The directory to which the simulation results will be written.
    parameters : dict
        A dictionary containing the different parameter values
        which will be searched over.
    nb_runs : int
        The number of model instances that will be run per task
        in each parameter combination.
    population_size : int, default=1
        The size of the population.
    present_files : list of str
        A list of files found in `file_path`.
    rng : numpy.random.Generator
        A random number generator instance used to
        mutate parameters between generations.

    """

    def __init__(
        self,
        file_path: str,
        parameters: dict,
        nb_runs: int = 1,
        population_size: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.parameters = copy.deepcopy(parameters)
        # sort parameter values
        for parameter in self.parameters:
            self.parameters[parameter].setdefault('init_range', {'low': -1, 'high': 1})
            self.parameters[parameter].setdefault(
                'param_range', {'a_min': None, 'a_max': None}
            )
            self.parameters[parameter].setdefault(
                'mutator', {np.random.Generator.normal, {'scale': 0.1}}
            )
        self.file_path = file_path
        self.nb_runs = nb_runs
        self.population_size = population_size
        self.rng = np.random.default_rng() if rng is None else rng
        # retrieve files present in file path
        self.present_files = [
            f for f in listdir(self.file_path) if isfile(join(self.file_path, f))
        ]

    def fit(
        self,
        simulation: Simulation,
        tasks: dict[str, Any],
        data: dict[str, Any],
        loss: FitLoss,
        overwrite: bool = False,
        generations: int = 100,
        individuals: int = 10,
        pool: None | mp.pool.Pool = None,
    ) -> Fit:
        """
        This function fits a given model to behavioral data.

        Parameters
        ----------
        simulation : Simulation
            The python function representing one simulation run.
        tasks : dict
            A dictionary containing different tasks (e.g., trial sequences)
            that the model has to be run for.
        data : dict
            A dictionary containing behavioral data for the different tasks.
        loss : FitLoss
            The loss function that is used for computing the fit.
        overwrite : bool, default=False
            If true, simulations are run for all parameter combinations
            even when stored simulation data was found.
        generations : int, default=100
            The number of generations in each run.
        individuals : int, default=10
            The number of individuals in each generation.
        pool : mp.pool.Pool or None, optional
            Optional worker pool for multiprocessing.
            If none is given simulations are run sequentially.

        Returns
        -------
        fitness : Fit
            A dictionary containing the fitness values for all runs.
        """
        assert tasks.keys() == data.keys(), 'Task mismatch!'
        best_fits = []
        for run in range(self.nb_runs):
            file_name = self.file_path + 'run_%d.pkl' % run
            if file_name in self.present_files and not overwrite:
                best_fits.append(pickle.load(open(file_name, 'rb')))
            else:
                best_fits.append([])
                # initialize first generation
                population = [
                    {
                        p: self.parameters[p]['param_type'](
                            self.rng.uniform(**self.parameters[p]['init_range'])
                        )
                        for p in self.parameters
                    }
                    for individual in range(individuals)
                ]
                for _ in range(generations):
                    fit = []
                    # run simulations and compute fit for each individual
                    for individual in population:
                        simulation_data = {}
                        if pool is None:
                            simulation_data = {
                                t: [
                                    simulation(task, individual)
                                    for i in range(self.population_size)
                                ]
                                for t, task in tasks.items()
                            }
                        else:
                            simulation_data = {
                                t: [
                                    pool.apply_async(simulation, (task, individual))
                                    for i in range(self.population_size)
                                ]
                                for t, task in tasks.items()
                            }
                            simulation_data = {
                                t: [run.get() for run in task]
                                for t, task in simulation_data.items()
                            }
                    fit.append(loss(simulation_data, data))
                # determine best individual, and store its parameters and fitness
                best = population[np.argmin(fit)]
                best_fits[-1].append((best, np.amin(fit)))
                pickle.dump(best_fits[-1], open(file_name, 'wb'))
                # generate new generation from best individual
                population = [best]
                for _ in range(individuals - 1):
                    population.append(
                        {
                            p: self.parameters[p]['mutator'][0](
                                best[p], **self.parameters[p]['mutator'][1]
                            )
                            for p in self.parameters
                        }
                    )
                    for p in self.parameters:
                        population[-1][p] = np.clip(
                            self.parameters[p]['param_type'](population[-1][p]),
                            **self.parameters[p]['param_range'],
                        )
        # gather the final fits for each run
        final_fit: Fit = {}
        for f in best_fits:
            final_fit[tuple(list(f[-1][0].values()))] = f[-1][1]

        return final_fit
