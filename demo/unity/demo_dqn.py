"""
This demo simulation trains a DQN agent moving on a topological graph.
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
grid:
    A 5 by 5 grid topological graph with a goal node located at the top right.
    All other nodes are potential starting nodes.
hexagonal:
    A 10 by 10 hexagonal topological graph with a goal node located at the top right.
    All other nodes are potential starting nodes.
t_maze:
    A T-maze topoĺogical graph with a goal node located at the end of
    the right arm. The agent starts at the beginning of the maze's stem.
"""

# basic imports
import sys
import numpy as np
import pyqtgraph as pg  # type: ignore
# torch
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


class Model(Module):
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

    def forward(self, layer_input: Tensor) -> Tensor:
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
    trials_train: int = 300,
    trials_test: int = 200,
    steps: int = 100,
) -> None:
    """
    This function represents one simulation run with the Topology interface.

    Parameters
    ----------
    nodes : dict of Node
        The topology nodes.
    starting_nodes : list of NodeID
        The starting nodes.
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
    executable: None | str = None
    if '--executable' in sys.argv:
        executable = sys.argv[sys.argv.index('--executable') + 1]
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


if __name__ == '__main__':
    set_num_threads(1)
    # generate topology
    nodes: dict[NodeID, Node]
    starting_nodes: list[NodeID]
    if 'grid' in sys.argv:
        nodes, starting_nodes = grid(5, (-0.5, 0.5))
    elif 'hexagonal' in sys.argv:
        nodes, starting_nodes = hexagonal(10, (-0.5, 0.5))
    elif 't_maze' in sys.argv:
        nodes, starting_nodes = t_maze(6, 3, 2, 0.1)
    else:
        nodes, starting_nodes = linear_track(10, 2, 0.1, 20, 'right')
    # run
    simulation(nodes, starting_nodes)
