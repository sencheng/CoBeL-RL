"""
Performs test simulations of the Binary Rescorla-Wagner agent with different settings.
"""
import numpy as np
from itertools import product
from gymnasium.spaces import Box
# CoBeL-RL
from cobel.agent import BinaryRescorlaWagner
from cobel.policy import Sigmoid
from cobel.interface import Sequence
# typing
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
        trials.append([{"observation": "A", "reward": 1.0, "action": None}])
        trials.append([{"observation": "B", "reward": 0.0, "action": None}])
    obs: dict[str, Observation] = {"A": np.array([1., 0.]), "B": np.array([0., 1.])}
    obs_space = Box(0., 1., (2, ))

    return trials, obs, obs_space, 2

def simulation(
    learning_rate: float | tuple[float, ...] = 0.9, use_test_policy: bool = True
) -> None:
    """
    Represents one test simulation.

    Parameters
    ----------
    learning_rate : float or tuple of floats, default=0.9
        The learning rate(s) that should be used.
    """
    env = Sequence(*make_trials())
    policy = Sigmoid(scale=1.)
    policy_test: None | Sigmoid = None
    if use_test_policy:
        policy_test = Sigmoid()
    agent = BinaryRescorlaWagner(
        env.observation_space,
        policy,
        policy_test,
        learning_rate
    )
    agent.train(env, 10, 10)
    agent.test(env, 10, 10)


def main() -> None:
    """
    The main function.
    """
    learning_rate: list[float | tuple[float, ...]] = [0.9, (0.9, 0.9)]
    policy = [True, False]
    for combination in product(learning_rate, policy):
        simulation(*combination)


if __name__ == "__main__":
    main()
