"""
Performs test simulations of the MFEC agent with different settings.
"""
# basic imports
import numpy as np
from gymnasium.spaces import Box, Dict
from itertools import product
# PyTorch
from torch import set_num_threads, Tensor, cat
from torch.nn import Module, Linear
from torch.nn.functional import relu
# CoBeL-RL
from cobel.agent import MFEC
from cobel.policy import EpsilonGreedy
from cobel.network import FlexibleTorchNetwork
from cobel.interface import Topology, OfflineSimulator
from cobel.misc.topology_tools import linear_track
# typing
from typing import Literal
from cobel.typing import Node, NodeID, Pose, Observation
Env = Literal["Topology", "Topology-Dict"]


class SimpleModel(Module):

    def __init__(self) -> None:
        super().__init__()
        self.hidden = Linear(6, 32)
        self.output = Linear(32, 32)
        self.double()

    def forward(self, batch: Tensor) -> Tensor:
        x = relu(self.hidden(batch))
        x = self.output(x)

        return x


class ComplexModel(Module):

    def __init__(self) -> None:
        super().__init__()
        self.hidden = Linear(12, 32)
        self.output = Linear(32, 32)
        self.double()

    def forward(self, stream_1: Tensor, stream_2: Tensor) -> Tensor:
        x = cat((stream_1, stream_2), 1)
        x = relu(self.hidden(x))
        x = self.output(x)

        return x


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
            "stream_1": np.array(node["pose"]), "stream_2": np.array(node["pose"])
        }
    space = Dict({"stream_1": Box(0., 1., (6, )), "stream_2": Box(0., 1., (6, ))})

    return obs, space


def simulation(
    env_type: Env = "Topology", use_model: bool = False, use_test_policy: bool = True
) -> None:
    """
    Represents one test simulation.

    Parameters
    ----------
    env_type : Env, default="Topology"
        The type of environment that will be used.
    use_model : bool, default=False
        Indicates whether a model should be used for observation preprocessing.
    use_test_policy : bool, default=True
        Indicates whether a separate test policy should be used.
    """
    set_num_threads(1)
    nodes, starting_nodes = linear_track(10, 2, 1, 20)
    simulator: None | OfflineSimulator = None
    if env_type == "Topology-Dict":
        simulator = OfflineSimulator(*dict_obs(nodes))
    env = Topology(nodes, starting_nodes, simulator)
    policy = EpsilonGreedy()
    policy_test: None | EpsilonGreedy = None
    if use_test_policy:
        policy_test = EpsilonGreedy(0.)
    model: None | FlexibleTorchNetwork = None
    if use_model:
        if env_type == "Topology":
            model = FlexibleTorchNetwork(SimpleModel())
        else:
            model = FlexibleTorchNetwork(ComplexModel())
    agent = MFEC(
        env.observation_space,
        env.action_space,
        policy,
        policy_test,
        model=model,
        projection_size=32,
        k=10
    )
    agent.train(env, 5, 20)
    agent.test(env, 5, 20)


def main() -> None:
    """
    The main function.
    """
    env: list[Env] = ["Topology", "Topology-Dict"]
    model = [True, False]
    policy = [True, False]
    for combination in product(env, model, policy):
        simulation(*combination)


if __name__ == "__main__":
    main()
