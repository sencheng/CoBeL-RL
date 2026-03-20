"""
Performs test simulations of the Q-Learning agent with different settings.
"""
# basic imports
import numpy as np
from gymnasium.spaces import Box, Dict
from itertools import product
# CoBeL-RL
from cobel.agent import QAgent
from cobel.policy import EpsilonGreedy
from cobel.interface import Gridworld, Topology, OfflineSimulator
from cobel.misc.topology_tools import linear_track
from cobel.misc.gridworld_tools import make_open_field
# typing
from typing import Literal
from cobel.typing import Node, NodeID, Pose, Observation
Env = Literal["Gridworld", "Topology", "Topology-Dict"]


def dict_obs(nodes: dict[NodeID, Node]) -> tuple[dict[Pose, Observation], Dict]:
    """
    Prepares multimodal observations for the OfflineSimulator from topology nodes.

    Parameters
    ----------
    nodes : dict of cobel.typing.Node
        A dictionary containing nodes of the Topology interface.

    Returns
    -------
    dict of cobel.typing.Observation
        A dictionary containing the multimodal observations.
    gymnasium.spaces.Dict
        The observation space.
    """
    obs: dict[Pose, Observation] = {}
    for _, node in nodes.items():
        obs[node["pose"]] = {
            "1": np.array(node["pose"]), "2": np.array(node["pose"])
        }
    space = Dict({"1": Box(0., 1., (6, )), "2": Box(0., 1., (6, ))})

    return obs, space


def simulation(env_type: Env = "Gridworld", use_test_policy: bool = True) -> None:
    """
    Represents one test simulation.

    Parameters
    ----------
    env_type : Env, default="Gridworld"
        The type of environment that will be used.
    use_test_policy : bool, default=True
        Indicates whether a separate test policy should be used.
    """
    env: Gridworld | Topology
    if env_type == "Gridworld":
        env = Gridworld(make_open_field(5, 5, 0, 1))
    else:
        nodes, starting_nodes = linear_track(10, 2, 1, 20)
        simulator: None | OfflineSimulator = None
        if env_type == "Topology-Dict":
            simulator = OfflineSimulator(*dict_obs(nodes))
        env = Topology(nodes, starting_nodes, simulator)
    policy = EpsilonGreedy()
    policy_test: None | EpsilonGreedy = None
    if use_test_policy:
        policy_test = EpsilonGreedy(0.)
    agent = QAgent(
        env.observation_space,
        env.action_space,
        policy,
        policy_test
    )
    agent.train(env, 5, 20, 32)
    agent.test(env, 5, 20)


def main() -> None:
    """
    The main function.
    """
    env: list[Env] = ["Gridworld", "Topology", "Topology-Dict"]
    policy = [True, False]
    for combination in product(env, policy):
        simulation(*combination)


if __name__ == "__main__":
    main()
