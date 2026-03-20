"""
Performs test simulations of the Associative Network agent with different settings.
"""
import numpy as np
from itertools import product
from gymnasium.spaces import Box
# CoBeL-RL
from cobel.agent import AssociativeNetwork
from cobel.policy import EpsilonGreedy
from cobel.interface import Sequence
# typing
from numpy.typing import NDArray
from cobel.typing import Trial, Observation


def make_trials() -> tuple[list[Trial], dict[str, Observation], Box, int]:
    """
    Prepares trial sequence for testing.

    Returns
    -------
    trials : list of cobel.interface.sequence.Trial
        The trial sequence.
    obs : dict of cobel.interface.interface.Observation
        A dictionary containing the observations.
    obs_dict : gymnasium.spaces.Box
        The observation space.
    nb_actions : int
        The number of actions.
    """
    trials: list[Trial] = []
    for _ in range(10):
        trials.append(
            [{"observation": "A", "reward": np.array([1.0, 0.0]), "action": None}]
        )
        trials.append(
            [{"observation": "B", "reward": np.array([0.0, 1.0]), "action": None}]
        )
    obs: dict[str, Observation] = {"A": np.array([1., 0.]), "B": np.array([0., 1.])}
    obs_space = Box(0., 1., (2, ))

    return trials, obs, obs_space, 3


def simulation(
    saturation: float | dict[str, NDArray] = 20,
    learning_rate: float | dict[str, NDArray] = 0.01,
    use_test_policy: bool = True,
    linear_update: bool = False,
) -> None:
    """
    Represents one test simulation.

    Parameters
    ----------
    saturation : float or dict of numpy.ndarray, default=20.
        The weight saturation value(s) that should be used.
    learning_rate : float or dict of numpy.ndarray, default=0.01
        The learning rate(s) that should be used.
    use_test_policy : bool, default=True
        Indicates whether a separate test policy should be used.
    linear_update : bool, default=False
        Indicates whether linear updates should be used.
    """
    env = Sequence(*make_trials())
    policy = EpsilonGreedy()
    policy_test: None | EpsilonGreedy = None
    if use_test_policy:
        policy_test = EpsilonGreedy(0.)
    agent = AssociativeNetwork(
        env.observation_space,
        env.action_space,
        policy,
        policy_test,
        saturation,
        learning_rate,
        linear_update=linear_update
    )
    agent.train(env, 10, 10)
    agent.test(env, 10, 10)


def main() -> None:
    """
    The main function.
    """
    saturation: list[float | dict[str, NDArray]] = [
        20.,
        {"excitatory": np.full((2, 2), 20.), "inhibitory": np.full((2, 2), 20.)}
    ]
    learning_rate: list[float | dict[str, NDArray]] = [
        .01,
        {"excitatory": np.full((2, 2), .01), "inhibitory": np.full((2, 2), .01)}
    ]
    policy = [True, False]
    linear = [True, False]
    for combination in product(saturation, learning_rate, policy, linear):
        simulation(*combination)


if __name__ == "__main__":
    main()
