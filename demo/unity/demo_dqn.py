"""
A demo simulation that trains a DQN agent moving on a topological graph.
The agent (marked in red) starts at a fixed location and has to reach
a goal node (marked green) to receive a reward.
Images rendered in the Unity game engine serve as observations and are
provided via the UnitySimulator class which overwrites pose observations.
The current node and observation (left plot) as well as the topological
graph and the agent's current location (center plot) are visualized.
Furthermore, the EscapeLatencyMonitor tracks the steps taken in
each trial and visualizes them (right plot).
Per default a linear track topological graph is used in which the
agent starts at the leftmost nodes and the rightmost nodes serve as goals.
The demo simulation can be run using different topological graphs
by providing the appropriate command line arguments:
--env grid:
    A 5 by 5 grid topological graph with a goal node located at the top right.
    All other nodes are potential starting nodes.
--env hexagonal:
    A 10 by 10 hexagonal topological graph with a goal node located at the top right.
    All other nodes are potential starting nodes.
--env t-maze:
    A T-maze topoĺogical graph with a goal node located at the end of
    the right arm. The agent starts at the beginning of the maze's stem.
"""

# basic imports
import os
import argparse
import numpy as np
import pyqtgraph as pg  # type: ignore

# PyTorch
from torch import Tensor, reshape, set_num_threads
from torch.nn import Module, Linear
from torch.nn.functional import relu

# CoBeL-RL
from cobel.agent import DQN
from cobel.policy import EpsilonGreedy
from cobel.network import FlexibleTorchNetwork
from cobel.monitor import EscapeLatencyMonitor
from cobel.interface import Topology, UnitySimulator
from cobel.misc.topology_tools import linear_track, grid, hexagonal, t_maze

# typing
from cobel.typing import Node, NodeID, CallbackDict


class Model(Module):  # noqa: D101
    def __init__(self, input_size: int | tuple[int, ...], output_size: int):
        super().__init__()
        input_units: int
        if type(input_size) is int:
            input_units = input_size
        else:
            input_units = int(np.prod(input_size))
        self.layer_dense_1 = Linear(in_features=input_units, out_features=64)
        self.layer_dense_2 = Linear(in_features=64, out_features=64)
        self.layer_output = Linear(in_features=64, out_features=output_size)
        self.double()

    def forward(self, layer_input: Tensor) -> Tensor:  # noqa: D102
        x = reshape(layer_input, (len(layer_input), -1))
        x = x.double() / 255.0
        x = self.layer_dense_1(x)
        x = relu(x)
        x = self.layer_dense_2(x)
        x = relu(x)
        x = self.layer_output(x)

        return x


def simulation(
    nodes: dict[NodeID, Node],
    starting_nodes: list[NodeID],
    executable: str | None = None,
    trials_train: int = 300,
    trials_test: int = 200,
    steps: int = 100,
) -> None:
    """
    Perform one simulation run with the Topology interface.

    Parameters
    ----------
    nodes : dict of cobel.interface.topology.Node
        The topology nodes.
    starting_nodes : list of cobel.interface.topology.NodeID
        The starting nodes.
    executable : str or None, optional
        An optional path to the Unity executable.
    trials_train : int, default=300
        The number of training trials.
    trials_test : int, default=200
        The number of test trials.
    steps : int, default=100
        The maximum number of steps per trial.
    """
    trials = trials_train + trials_test
    # prepare widget
    main_window = pg.GraphicsLayoutWidget(title='DQN Demo')
    main_window.show()
    # prepare simulator
    simulator = UnitySimulator(
        'room', executable=executable, resize=(30, 1), batch_mode=False
    )
    # init environment
    interface = Topology(nodes, starting_nodes, simulator, main_window)
    # init monitor and prepare callbacks
    el_monitor = EscapeLatencyMonitor(trials, steps, main_window)
    callbacks: CallbackDict = {'on_trial_end': [el_monitor.update]}  # type: ignore
    # init and train agent
    policy = EpsilonGreedy(0.3)
    policy_test = EpsilonGreedy(0.0)
    assert type(interface.observation_space.shape) is tuple
    model = FlexibleTorchNetwork(Model(interface.observation_space.shape, 4))
    agent = DQN(
        interface.observation_space,
        interface.action_space,
        policy,
        model,
        policy_test=policy_test,
        custom_callbacks=callbacks,
    )
    agent.train(interface, trials_train, steps)
    agent.test(interface, trials_test, steps)
    # close widget
    main_window.close()
    # stop simulator
    simulator.stop()


def main() -> None:
    """The main function."""  # noqa: D401
    nb_cores: int = os.cpu_count()  # type: ignore
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        default='linear',
        choices=('grid', 'hexagonal', 't-maze', 'linear'),
    )
    parser.add_argument(
        '--threads', type=int, default=1, choices=range(1, nb_cores + 1)
    )
    parser.add_argument('--executable', type=str)
    args = parser.parse_args()
    set_num_threads(args.threads)
    executable: None | str = args.executable
    # generate topology
    nodes: dict[NodeID, Node]
    starting_nodes: list[NodeID]
    if args.env == 'grid':
        nodes, starting_nodes = grid(5, (0.0, 1.0))
    elif args.env == 'hexagonal':
        nodes, starting_nodes = hexagonal(10, (0.0, 1.0))
    elif args.env == 't-maze':
        nodes, starting_nodes = t_maze(6, 3, 2, 1.0)
    else:
        nodes, starting_nodes = linear_track(10, 2, 1.0, 20, 'right')
    # run
    simulation(nodes, starting_nodes, executable)


if __name__ == '__main__':
    main()
