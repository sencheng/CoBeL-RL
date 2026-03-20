"""
Basic unit test for the Gridworld interface class.
"""
# basic imports
import numpy as np
from gymnasium.spaces import Discrete
# CoBeL-RL
from cobel.interface import Gridworld
from cobel.misc.gridworld_tools import make_gridworld


def main() -> None:
    """
    The main function.
    """
    world = make_gridworld(
        5, 5, [0], np.array([[0, 10.]]), starting_states=[24]
    )
    env = Gridworld(world)
    assert isinstance(env.observation_space, Discrete)
    assert isinstance(env.action_space, Discrete)
    assert env.observation_space.n == 25
    assert env.action_space.n == 4
    state, _ = env.reset()
    assert state == 24
    states = []
    rewards = []
    terminals = []
    actions = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    for action in actions:
        state, reward, terminal, _, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        terminals.append(terminal)
    states_true = [23, 22, 21, 20, 20, 15, 10, 5, 0]
    rewards_true = [0, 0, 0, 0, 0, 0, 0, 0, 10.]
    terminals_true = [False] * 8 + [True]
    for i in range(9):
        assert states[i] == states_true[i]
        assert rewards[i] == rewards_true[i]
        assert terminals[i] == terminals_true[i]


if __name__ == "__main__":
    main()
