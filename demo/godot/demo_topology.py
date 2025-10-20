"""
This demo simulation trains a Q-learning agent moving on a
topological graph. The agent (marked in red) starts at a fixed
location and has to reach a goal node (marked green) to receive a reward.
Images rendered in the Godot game engine serve as observations and are
provided via the GodotSimulator class which overwrites pose observations.
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
import pyqtgraph as pg  # type: ignore
# CoBeL-RL
from cobel.agent import QAgent
from cobel.policy import EpsilonGreedy
from cobel.monitor import EscapeLatencyMonitor
from cobel.interface import Topology, GodotSimulator
from cobel.misc.topology_tools import linear_track, grid, hexagonal, t_maze
# typing
from cobel.typing import Node, NodeID, CallbackDict


def simulation(
    nodes: dict[NodeID, Node],
    starting_nodes: list[NodeID],
    trials: int = 500,
    steps: int = 50,
) -> None:
    """
    This function represents one simulation run with the Topology interface.

    Parameters
    ----------
    nodes : dict of Node
        The topology nodes.
    starting_nodes : list of NodeID
        The starting nodes.
    trials : int, default=500
        The number of training trials.
    steps : int, default=50
        The maximum number of steps per trial.
    """
    # prepare widget
    main_window = pg.GraphicsLayoutWidget(title='Topology Test')
    main_window.show()
    # prepare simulator
    executable: None | str = None
    if '--executable' in sys.argv:
        executable = sys.argv[sys.argv.index('--executable') + 1]
    simulator = GodotSimulator('room.tscn', executable=executable, resize=(30, 1))
    # init environment
    interface = Topology(nodes, starting_nodes, simulator, main_window)
    # init monitor and prepare callbacks
    el_monitor = EscapeLatencyMonitor(trials, steps, main_window)
    callbacks: CallbackDict = {'on_trial_end': [el_monitor.update]}
    # init and train agent
    policy = EpsilonGreedy(0.1)
    agent = QAgent(
        interface.observation_space,
        interface.action_space,
        policy,
        custom_callbacks=callbacks,
    )
    agent.train(interface, trials, steps, 0)
    # close widget
    main_window.close()
    # stop simulator
    simulator.stop()


if __name__ == '__main__':
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
