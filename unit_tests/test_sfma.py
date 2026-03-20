"""
Performs test simulations of the SFMA agent with different settings.
"""
# basic imports
from itertools import product
import numpy as np
# CoBeL-RL
from cobel.agent import SFMA
from cobel.memory import SFMAMemory
from cobel.policy import EpsilonGreedy
from cobel.interface import Gridworld
from cobel.memory.utils import DR, SR, Euclidean, Metric
from cobel.misc.gridworld_tools import make_gridworld
# typing
from typing import Literal
Mode = Literal["default", "reverse", "forward", "dynamic", "sweeping"]
SimMetric = Literal["DR", "SR", "Euclidean"]


def simulation(
    metric: SimMetric = "DR",
    mode: Mode = "default",
    use_test_policy: bool = True,
    mask_actions: bool = True,
    random_replay: bool = False
) -> None:
    """
    Represents one test simulation.

    Parameters
    ----------
    metric : SimMetric, default="DR"
        The similarity metric that will be used for SFMA.
    mode : Mode, default="default"
        The replay mode that will be used.
    use_test_policy : bool, default=True
        Indicates whether a separate test policy should be used.
    mask_actions : bool, default=True
        Indicates whether an action mask should be used.
    random_replay : bool, default=False
        Indicates whether random replay should be used.
    """
    invalid_transitions = [
        (3, 4),
        (4, 3),
        (8, 9),
        (9, 8),
        (13, 14),
        (14, 13),
    ]
    gridworld = make_gridworld(
        5,
        5,
        terminals=[4],
        rewards=np.array([[4, 10]]),
        goals=[4],
        invalid_transitions=invalid_transitions,
    )
    gridworld['starting_states'] = np.array([12])
    env = Gridworld(gridworld)
    similarity_metric: Metric
    if metric == "DR":
        similarity_metric = DR(
            env.world['width'],
            env.world['height'],
            env.world['sas'],
            0.9,
            env.world['invalid_transitions']
        )
    elif metric == "SR":
        similarity_metric = SR(
            env.world['sas'],
            0.9
        )
    else:
        similarity_metric = Euclidean(
            env.world['width'],
            env.world['height']
        )
    memory = SFMAMemory(similarity_metric, env.world['states'], 4)
    policy = EpsilonGreedy()
    policy_test: None | EpsilonGreedy = None
    if use_test_policy:
        policy_test = EpsilonGreedy(0.)
    agent = SFMA(
        env.observation_space,
        env.action_space,
        policy,
        memory,
        policy_test
    )
    agent.M.mode = mode
    agent.mask_actions = mask_actions
    agent.random = random_replay
    agent.train(env, 5, 50, 32)
    agent.train(env, 5, 50, 32, True)
    agent.test(env, 5, 50)


def main() -> None:
    """
    The main function.
    """
    metrics: list[SimMetric] = ["DR", "SR", "Euclidean"]
    modes: list[Mode] = ["default", "reverse", "forward", "dynamic", "sweeping"]
    policy = [True, False]
    mask = [True, False]
    random = [True, False]
    for combination in product(metrics, modes, policy, mask, random):
        simulation(*combination)


if __name__ == "__main__":
    main()
