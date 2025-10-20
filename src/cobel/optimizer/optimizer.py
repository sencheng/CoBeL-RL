# basic imports
import abc
# typing
from typing import Any
from collections.abc import Callable

Fit = dict[tuple, tuple[float, ...] | float]
Simulation = Callable[[dict[str, Any], dict[str, Any]], Any]
FitLoss = Callable[[dict[str, Any], dict[str, Any]], tuple[float, ...] | float]


class Optimizer(abc.ABC):
    """
    The abstract optimizer class.
    """

    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def fit(
        self,
        simulation: Simulation,
        tasks: dict[str, Any],
        data: dict[str, Any],
        loss: FitLoss,
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

        Returns
        -------
        fitness : Fit
            A dictionary containing the fitness values for all parameter combinations.
        """
