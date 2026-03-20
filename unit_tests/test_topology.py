"""
Basic unit test for the Topology interface class.
"""
# basic imports
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict, Tuple
# CoBeL-RL
from cobel.interface import Topology, OfflineSimulator
from cobel.misc.topology_tools import t_maze
# typing
from typing import Literal
from cobel.typing import Observation, Pose, NodeID, Node


def make_obs(
    obs_type: Literal["List", "Dict"], nodes: dict[NodeID, Node]
) -> tuple[dict[Pose, Observation], Dict | Tuple]:
    """
    Prepares observations.

    Parameters
    ----------
    obs_type : "List" or "Dict"
        The observation type.
    nodes : dict of Node
        A dictionary containing topological graph nodes.

    Returns
    -------
    obs : dict of Observation
        A dictionary containing the prepared observations.
    obs_space : gymnasium.spaces.Dict or gymnasium.spaces.Tuple
        The observation space.
    """
    obs: dict[Pose, Observation] = {}
    for _, node in nodes.items():
        if obs_type == "List":
            obs[node["pose"]] = [
                np.array(node["pose"]),
                np.array(node["pose"])
            ]
        else:
            obs[node["pose"]] = {
                "1": np.array(node["pose"]),
                "2": np.array(node["pose"])
            }
    obs_space: Dict | Tuple
    if obs_type == "List":
        obs_space = Tuple(
            [Box(-np.inf, np.inf, (6,)), Box(-np.inf, np.inf, (6,))]
        )
    else:
        obs_space = Dict(
            {"1": Box(-np.inf, np.inf, (6,)), "2": Box(-np.inf, np.inf, (6,))}
        )

    return obs, obs_space


def main() -> None:
    """
    The main function.
    """
    nodes, nodes_starting = t_maze(4, 3, 1)
    # pose observations
    env = Topology(nodes, nodes_starting)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)
    assert env.observation_space.shape == (6, )
    assert env.action_space.n == 4
    # list observations
    simulator = OfflineSimulator(*make_obs("List", nodes))
    env = Topology(nodes, nodes_starting, simulator)
    assert isinstance(env.observation_space, Tuple)
    assert isinstance(env.action_space, Discrete)
    assert isinstance(env.observation_space.spaces[0], Box)
    assert isinstance(env.observation_space.spaces[1], Box)
    assert env.observation_space.spaces[0].shape == (6, )
    assert env.observation_space.spaces[1].shape == (6, )
    assert env.action_space.n == 4
    # dict observations
    simulator = OfflineSimulator(*make_obs("Dict", nodes))
    env = Topology(nodes, nodes_starting, simulator)
    assert isinstance(env.observation_space, Dict)
    assert isinstance(env.action_space, Discrete)
    assert isinstance(env.observation_space.spaces["1"], Box)
    assert isinstance(env.observation_space.spaces["2"], Box)
    assert env.observation_space.spaces["1"].shape == (6, )
    assert env.observation_space.spaces["2"].shape == (6, )
    assert env.action_space.n == 4
    # test trajectory
    env.reset()
    assert env.current_node == "10"
    states = []
    rewards = []
    terminals = []
    actions = [1, 1, 1, 1, 1, 2, 2, 2]
    for action in actions:
        _, reward, terminal, _, _ = env.step(action)
        states.append(env.current_node)
        rewards.append(reward)
        terminals.append(terminal)
    states_true = ["9", "8", "7", "3", "3", "4", "5", "6"]
    rewards_true = [0.] * 7 + [1.]
    terminals_true = [False] * 7 + [True]
    for i in range(8):
        assert states[i] == states_true[i]
        assert rewards[i] == rewards_true[i]
        assert terminals[i] == terminals_true[i]


if __name__ == "__main__":
    main()
