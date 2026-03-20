"""
Performs test simulations of the Rescorla-Wagner agent with different settings.
"""
import numpy as np
from gymnasium.spaces import Box
# CoBeL-RL
from cobel.agent import RescorlaWagner
from cobel.interface import Sequence
# typing
from cobel.typing import Trial, Observation


def make_trials() -> tuple[list[Trial], dict[str, Observation], Box]:
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
    """
    trials: list[Trial] = []
    for _ in range(10):
        trials.append([{"observation": "A", "reward": 1.0, "action": None}])
        trials.append([{"observation": "B", "reward": -1.0, "action": None}])
    obs: dict[str, Observation] = {"A": np.array([1., 0.]), "B": np.array([0., 1.])}
    obs_space = Box(0., 1., (2, ))

    return trials, obs, obs_space

def simulation(learning_rate: float | tuple[float, ...] = 0.9) -> None:
    """
    Represents one test simulation.

    Parameters
    ----------
    learning_rate : float or tuple of floats, default=0.9
        The learning rate(s) that should be used.
    """
    env = Sequence(*make_trials())
    agent = RescorlaWagner(env.observation_space, learning_rate)
    agent.train(env, 10, 10)
    agent.test(env, 10, 10)
    assert agent.W[0] > 0.9 and agent.W[1] < -0.9


def main() -> None:
    """
    The main function.
    """
    simulation(0.9)
    simulation((0.9, 0.9))


if __name__ == "__main__":
    main()
