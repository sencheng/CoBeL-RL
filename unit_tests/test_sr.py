"""
Performs test simulations of the SR agent with different settings.
"""
# basic imports
from itertools import product
# CoBeL-RL
from cobel.agent import SR
from cobel.policy import EpsilonGreedy
from cobel.interface import Gridworld
from cobel.misc.gridworld_tools import make_open_field


def simulation(use_test_policy: bool = True, mask_actions: bool = False) -> None:
    """
    Represents one test simulation.

    Parameters
    ----------
    use_test_policy : bool, default=True
        Indicates whether a separate test policy should be used.
    mask_actions : bool, default=False
        Indicates whether an action mask should be used.
    """
    env = Gridworld(make_open_field(5, 5, 0, 1))
    policy = EpsilonGreedy()
    policy_test: None | EpsilonGreedy = None
    if use_test_policy:
        policy_test = EpsilonGreedy(0.)
    agent = SR(
        env.observation_space,
        env.action_space,
        policy,
        policy_test
    )
    agent.mask_actions = mask_actions
    agent.train(env, 5, 20)
    agent.test(env, 5, 20)


def main() -> None:
    """
    The main function.
    """
    policy = [True, False]
    mask = [True, False]
    for combination in product(policy, mask):
        simulation(*combination)


if __name__ == "__main__":
    main()
