"""
Basic unit test for the Continuous2D interface class.
"""
# basic imports
import numpy as np
import shapely as sh # type: ignore
from gymnasium.spaces import Discrete, Box
# CoBeL-RL
from cobel.interface import Continuous2D
from cobel.misc.continuous_tools import make_rectangle


def main() -> None:
    """
    The main function.
    """
    room = sh.Polygon(
        [(0, 0), (1, 0), (1, 1), (0, 1)]
    )
    spawn = sh.Polygon(
        [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)]
    )
    rewards = np.array([[0.75, 0.75, 1.]])
    env = Continuous2D("step", room, spawn, None, rewards)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)
    assert env.observation_space.shape == (2, )
    assert env.action_space.n == 4
    # test behavior
    env.reset()
    for _ in range(100):
        env.step(2)
    assert env.state[0] == 1.
    env.state = np.array([0.6, 0.6])
    reward: float = 0.
    terminal: bool = False
    for _ in range(10):
        env.step(1)
        _, reward, terminal, _, _ = env.step(2)
    assert reward == 1.
    assert terminal == True
    # with obstacle
    obstacle = make_rectangle(np.array([0.5, 0.25]), 0.1, 0.5)
    env = Continuous2D("step", room, spawn, [obstacle], rewards)
    env.reset()
    for _ in range(100):
        env.step(2)
    assert env.state[0] == 0.45
    # wheel agent
    env = Continuous2D("wheel", room, spawn, [obstacle], rewards)
    assert env.observation_space.shape == (3, )
    env.reset()
    orientation = env.state[2]
    for _ in range(10):
        env.step(0)
    assert orientation != env.state[2]


if __name__ == "__main__":
    main()
