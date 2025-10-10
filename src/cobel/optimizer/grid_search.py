# basic imports
import copy
import pickle
import numpy as np
import multiprocessing as mp
from os import listdir
from os.path import isfile, join
from itertools import product
# framework imports
from .optimizer import Optimizer
# typing
from typing import Any, Literal
from .optimizer import Simulation, FitLoss, Fit


class GridSearchOptimizer(Optimizer):
    """
    The grid search optimizer class.

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
    order : 'nested', 'shuffled' or 'systematic', default='nested'
        The order in which parameter combinations are written to
        `parameters_combinations`: 'nested' (nested in order of values),
        'shuffled' (like nested but values are shuffled before) and
        'systematic' ("resolves" parameter space).
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
    parameters_combinations : dict
        A dictionary containing the different parameter values
        combinations.
    nb_runs : int
        The number of model instances that will be run per task
        in each parameter combination.
    order : str
        The order in which parameter combinations are written to
        `parameters_combinations`: 'nested' (nested in order of values),
        'shuffled' (like nested but values are shuffled before) and
        'systematic' ("resolves" parameter space).
    present_files : list of str
        A list of files found in `file_path`.
    rng : numpy.random.Generator
        A random number generator instance used for
        shuffling the order of parameter combinations.

    Examples
    --------

    The grid search optimizer fits a given set of parameters
    for different variants of a simulation ("tasks").
    The example show below define one task ("task_1"). ::

        >>> import numpy as np
        >>> from cobel.optimizer import GridSearchOptimizer
        >>> def test_sim(task, params):
        ...     return params['x_1'] + params['x_2'] ** 2 + params['x_3'] ** 3
        >>> def loss(data_sim, data_exp):
        ...     error = 0.
        ...     for t in data_sim:
        ...         error += (np.mean(data_sim[t]) - data_exp[t]) ** 2
        ...     return error / len(data_sim)
        >>> data = test_sim({'x_1': 2, 'x_2': 0.1, 'x_3': 0.7})
        >>> parameters = {'x_1': [0, 1, 2, 3, 4],
        ...               'x_2': [0., 0.1, 0.2, 0.3, 0.4],
        ...               'x_3': [0.5, 0.6, 0.7, 0.8, 0.9]}
        >>> optimizer = GridSearchOptimizer('search/', parameters)
        >>> fit = optimizer.fit(test_sim, {'task_1': {}},
        ...                     {'task_1': data}, loss)

    """

    def __init__(
        self,
        file_path: str,
        parameters: dict,
        nb_runs: int = 1,
        order: Literal['nested', 'shuffled', 'systematic'] = 'nested',
        rng: np.random.Generator | None = None,
    ) -> None:
        self.parameters = copy.deepcopy(parameters)
        # sort parameter values
        for parameter in self.parameters:
            if type(self.parameters[parameter]) is np.ndarray:
                self.parameters[parameter] = np.sort(self.parameters[parameter])
        self.prepare_parameter_combinations(self.parameters, order)
        self.file_path = file_path
        self.nb_runs = nb_runs
        self.rng = np.random.default_rng() if rng is None else rng
        # retrieve files present in file path
        self.present_files = [
            f for f in listdir(self.file_path) if isfile(join(self.file_path, f))
        ]

    def prepare_parameter_combinations(
        self, parameters: dict, order: str = 'nested'
    ) -> None:
        """
        This function prepares a dictionary containing the
        different parameter combinations. Parameter combinations can be
        written to the dictionary in different orders.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the different parameter values
            which will be searched over.
        order : str, default='nested'
            The order in which parameter combinations are written to
            `parameters_combinations`: 'nested' (nested in order of values),
            'shuffled' (like nested but values are shuffled before)
            and 'systematic' ("resolves" parameter space).
        """
        assert order in ['nested', 'shuffled', 'systematic'], 'Invalid order!'
        self.parameter_combinations: dict[tuple, dict] = {}
        combinations: list[product] = []
        if order == 'systematic':
            strides: list[list[int]] = []
            max_length = 0
            # determine strides for each parameter
            for parameter, _ in parameters.items():
                strides.append([])
                for i in range(int(np.ceil(np.sqrt(len(parameters[parameter]))))):
                    strides[-1].append(
                        max(int(len(parameters[parameter]) / (2 ** (i + 1))), 1)
                    )
                if 1 not in strides[-1]:
                    strides[-1].append(1)
                max_length = max(max_length, len(strides[-1]))
            # fill in missing strides
            for i in range(len(strides)):
                strides[i] += [1 for j in range(max_length - len(strides[i]))]
            # determine prameter combinations
            for i in range(max_length):
                p: list = []
                for s, stride in enumerate(strides):
                    p.append([])
                    for v in range(len(list(parameters.values())[s]) // stride[i]):
                        p[-1].append(list(parameters.values())[s][v * stride[i]])
                combinations.append(product(*p))
        else:
            # shuffle values
            if order == 'shuffled':
                for parameter, _ in parameters.items():
                    self.rng.shuffle(parameters[parameter])
                # determine parameter combinations
            combinations.append(product(*parameters.values()))
        # store parameter combinations
        for sub_combinations in combinations:
            for combination in sub_combinations:
                if tuple(combination) not in self.parameter_combinations:
                    self.parameter_combinations[tuple(combination)] = {
                        p: v
                        for p, v in zip(parameters.keys(), combination, strict=True)
                    }

    def fit(
        self,
        simulation: Simulation,
        tasks: dict[str, Any],
        data: dict[str, Any],
        loss: FitLoss,
        overwrite: bool = False,
        store_simulation_data: bool = False,
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
        store_simulation_data : bool, default=False
            If true, simulation data is stored along with the fit.
        pool : mp.pool.Pool or None, optional
            Optional worker pool for multiprocessing. If none is
            given simulations are run sequentially.

        Returns
        -------
        fitness : Fit
            A dictionary containing the fitness values for all parameter combinations.
        """
        assert tasks.keys() == data.keys(), 'Task mismatch!'
        fit: Fit = {}
        if 'fit.pkl' in self.present_files:
            fit = pickle.load(open(self.file_path + 'fit.pkl', 'rb'))
        # start fitting
        for parameter_combination in self.parameter_combinations:
            simulation_data = {}
            file_name_data = (
                '_%s' * len(parameter_combination)
            ) % parameter_combination
            file_name_data = 'sim' + file_name_data + '.pkl'
            if parameter_combination not in fit or overwrite:
                if file_name_data in self.present_files and not overwrite:
                    simulation_data = pickle.load(
                        open(self.file_path + file_name_data, 'rb')
                    )
                else:
                    for task, _ in tasks.items():
                        if pool is None:
                            simulation_data[task] = [
                                simulation(
                                    tasks[task],
                                    self.parameter_combinations[parameter_combination],
                                )
                                for run in range(self.nb_runs)
                            ]
                        else:
                            simulation_data[task] = [
                                pool.apply_async(
                                    simulation,
                                    (
                                        tasks[task],
                                        self.parameter_combinations[
                                            parameter_combination
                                        ],
                                    ),
                                )
                                for run in range(self.nb_runs)
                            ]
                            simulation_data[task] = [
                                run.get() for run in simulation_data[task]
                            ]
                    # store simulation_data
                    if store_simulation_data:
                        pickle.dump(
                            simulation_data, open(self.file_path + file_name_data, 'wb')
                        )
                    # compute and store fit
                    fit[parameter_combination] = loss(simulation_data, data)
                    pickle.dump(fit, open(self.file_path + 'fit.pkl', 'wb'))

        return fit

    def recompute_fit(
        self, data: dict[str, Any], loss: FitLoss, overwrite: bool = False
    ) -> Fit:
        """
        This function recomputes the fit.
        Assumes that the simulation data was stored.

        Parameters
        ----------
        data : dict
            A dictionary containing behavioral data for the different tasks.
        loss : FitLoss
            The loss function that is used for computing the fit.
        overwrite : bool, default=False
            If true, simulations are run for all parameter combinations
            even when stored simulation data was found.

        Returns
        -------
        fitness : Fit
            A dictionary containing the fitness values for all parameter combinations.
        """
        # try loading the fit
        fit: Fit = {}
        if 'fit.pkl' in self.present_files:
            fit = pickle.load(open(self.file_path + 'fit.pkl', 'rb'))
        else:
            print('No fit to recompute!')
            return fit
        # recompute fit for all combinations
        for parameter_combination in fit:
            file_name_data = (
                '_%s' * len(parameter_combination)
            ) % parameter_combination
            file_name_data = 'sim' + file_name_data + '.pkl'
            if file_name_data in self.present_files:
                simulation_data = pickle.load(
                    open(self.file_path + file_name_data, 'rb')
                )
                fit[parameter_combination] = loss(simulation_data, data)
            else:
                print(
                    'No simulation data found for parameter combination: '
                    + str(parameter_combination)
                )
            # store recomputed fit
            if overwrite:
                pickle.dump(fit, open(self.file_path + 'fit.pkl', 'wb'))

        return fit
