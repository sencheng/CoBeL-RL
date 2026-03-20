"""
Performs test simulations of the PMA agent with different settings.
"""
# basic imports
from itertools import product
import numpy as np
# CoBeL-RL
from cobel.agent import PMA
from cobel.memory import PMAMemory
from cobel.policy import EpsilonGreedy
from cobel.interface import Gridworld
from cobel.misc.gridworld_tools import make_gridworld


def simulation(
    use_test_policy: bool = True,
    mask_actions: bool = True,
    original_gain_mode: bool = True,
    random_replay: bool = False
) -> None:
    """
    Represents one test simulation.

    Parameters
    ----------
    use_test_policy : bool, default=True
        Indicates whether a separate test policy should be used.
    mask_actions : bool, default=True
        Indicates whether an action mask should be used.
    original_gain_mode : bool, default=True
        Indicates whether the gain should be computed for
        n-step updates as orignally described in Mattar and Daw (2018).
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
    policy = EpsilonGreedy()
    policy_test: None | EpsilonGreedy = None
    if use_test_policy:
        policy_test = EpsilonGreedy(0.)
    memory = PMAMemory(
        env.world['sas'],
        policy
    )
    agent = PMA(
        env.observation_space,
        env.action_space,
        policy,
        memory,
        policy_test
    )
    for state in range(25):
        for action in range(4):
            agent.M.states[state, action] = np.argmax(env.world['sas'][state, action])
    agent.mask_actions = mask_actions
    agent.M.compute_update_mask()
    if not original_gain_mode:
        agent.M.min_gain_mode = ''
    agent.train(env, 3, 50, 10)
    agent.train(env, 3, 50, 10, True)
    agent.test(env, 5, 50)


def main() -> None:
    """
    The main function.
    """
    policy = [True, False]
    mask = [True, False]
    gain = [True, False]
    random = [True, False]
    for combination in product(policy, mask, gain, random):
        simulation(*combination)


if __name__ == "__main__":
    main()
