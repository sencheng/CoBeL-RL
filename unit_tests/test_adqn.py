"""
Performs test simulations of the Associative DQN agent with different settings.
"""
import numpy as np
from gymnasium.spaces import Box, Dict, Tuple
# PyTorch
from torch import Tensor, set_num_threads, cat
from torch.nn import Module, Linear
from torch.nn.functional import relu
# CoBeL-RL
from cobel.agent import ADQN
from cobel.network import FlexibleTorchNetwork
from cobel.interface import Sequence
# typing
from typing import Literal
from cobel.typing import Trial, Observation
ObsType = Literal["Simple", "List", "Dict"]


class SimpleModel(Module):

    def __init__(self) -> None:
        super().__init__()
        self.hidden = Linear(2, 32)
        self.output = Linear(32, 1)
        self.double()

    def forward(self, layer_input: Tensor) -> Tensor:
        x = relu(self.hidden(layer_input))
        x = self.output(x)

        return x


class ComplexModel(Module):

    def __init__(self) -> None:
        super().__init__()
        self.hidden = Linear(4, 32)
        self.output = Linear(32, 1)
        self.double()

    def forward(self, stream_1: Tensor, stream_2: Tensor) -> Tensor:
        x = relu(self.hidden(cat((stream_1, stream_2), 1)))
        x = self.output(x)

        return x


def make_trials(
    observation_type: ObsType
) -> tuple[list[Trial], dict[str, Observation], Box | Dict | Tuple]:
    """
    Prepares trial sequence for testing.

    Parameters
    ----------
    observation_type : ObsType
        The observation type that should be generated.

    Returns
    -------
    trials : list of cobel.interface.sequence.Trial
        The trial sequence.
    obs : dict of cobel.interface.interface.Observation
        A dictionary containing the observations.
    obs_space : gymnasium.spaces.Box or gymnasium.spaces.Dict or gymnasium.spaces.Tuple
        The observation space.
    """
    trials: list[Trial] = []
    for _ in range(10):
        trials.append([{"observation": "A", "reward": 1.0, "action": None}])
        trials.append([{"observation": "B", "reward": -1.0, "action": None}])
    obs: dict[str, Observation] = {"A": np.array([1., 0.]), "B": np.array([0., 1.])}
    obs_space: Box | Dict | Tuple
    if observation_type == "Simple":
        obs = {"A": np.array([1., 0.]), "B": np.array([0., 1.])}
        obs_space = Box(0., 1., (2, ))
    elif observation_type == "List":
        obs = {
            "A": [np.array([1., 0.]), np.array([1., 0.])],
            "B": [np.array([0., 1.]), np.array([0., 1.])]
        }
        obs_space = Tuple([Box(0., 1., (2, )), Box(0., 1., (2, ))])
    else:
        obs = {
            "A": {"stream_1": np.array([1., 0.]), "stream_2": np.array([1., 0.])},
            "B": {"stream_1": np.array([0., 1.]), "stream_2": np.array([0., 1.])}
        }
        obs_space = Dict(
            {"stream_1": Box(0., 1., (2, )), "stream_2": Box(0., 1., (2, ))}
        )

    return trials, obs, obs_space

def simulation(observation_type: ObsType = "Simple") -> None:
    """
    Represents one test simulation.

    Parameters
    ----------
    observation_type : ObsType
        The observation type that should be generated.
    """
    set_num_threads(1)
    env = Sequence(*make_trials(observation_type))
    model: FlexibleTorchNetwork
    if observation_type == "Simple":
        model = FlexibleTorchNetwork(SimpleModel())
    else:
        model = FlexibleTorchNetwork(ComplexModel())
    agent = ADQN(env.observation_space, model)
    agent.train(env, 10, 10)
    agent.test(env, 10, 10)


def main() -> None:
    """
    The main function.
    """
    simulation("Simple")
    simulation("List")
    simulation("Dict")


if __name__ == "__main__":
    main()
