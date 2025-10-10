"""
This demo simulation trains a DQN agent moving in a continuous 2D environment.
The agent (marked in red) starts at a fixed location and has to reach a goal
(marked green) to receive a reward. The agent can move in step mode, i.e.,
moving in the four cardinal directions, or in a wheel mode, i.e., it behaves
like a differential wheel and can move either or both of its wheels.
The current coordinates serve as observation in step mode and the current orientation
is included when in wheel mode. Per default the step mode is used but the mode
can be specified by providing it via the command line arguments 'step' and 'wheel'.
The location and observation (left plot) as well as the environment and the
agent's current location (center plot) are visualized.
Furthermore, the EscapeLatencyMonitor tracks the steps taken in each trial
and visualizes them (right plot). Per default an open field environment with
three obstacles is used in which the agent starts at a random location and
the goal is located in the upper right corner. The demo simulation can be run
using different environments by providing the appropriate command line arguments:
t-maze:
    A T-maze environment in which the goal is located at the end of the right
    arm and the agent starts at the beginning of the maze's stem.
double-t-maze:
    A double T-maze environment in which the goal is located at the end of the
    right-most arm and the agent starts at the beginning of the maze's stem.
two-sided-t-maze:
    A two-sided T-maze environment in which the goal is located at the end of the
    lower right arm and the agent starts at the beginning of the maze's stem.
eight-maze:
    An eigh-maze environment in which the goal is located at the center
    of the right lap and the agent starts at the center of the maze.
"""

# basic imports
import sys
import numpy as np
import shapely as sh  # type: ignore
import pyqtgraph as qg  # type: ignore
# torch
from torch import Tensor, set_num_threads
from torch.nn import Module, Linear
from torch.nn.functional import tanh
# CoBel-RL framework
from cobel.agent import DQN
from cobel.policy import EpsilonGreedy
from cobel.monitor import EscapeLatencyMonitor
from cobel.interface import Continuous2D
from cobel.network.network_torch import TorchNetwork
from cobel.misc.continuous_tools import (
    make_t_maze,
    make_double_t_maze,
    make_two_sided_t_maze,
    make_eight_maze,
    make_rectangle,
    make_circle,
    make_triangle,
)
# typing
from typing import Literal


class Model(Module):
    def __init__(self, input_size: int | tuple, number_of_actions: int) -> None:
        super().__init__()
        input_units: int
        if type(input_size) is int:
            input_units = input_size
        else:
            input_units = int(np.prod(input_size))
        self.layer_dense_1 = Linear(in_features=input_units, out_features=64)
        self.layer_dense_2 = Linear(in_features=64, out_features=64)
        self.layer_output = Linear(in_features=64, out_features=number_of_actions)
        self.double()

    def forward(self, layer_input: Tensor) -> Tensor:
        x = self.layer_dense_1(layer_input)
        x = tanh(x)
        x = self.layer_dense_2(x)
        x = tanh(x)
        x = self.layer_output(x)

        return x


def simulation() -> None:
    """
    Represents one simulation run.
    """
    main_window = qg.GraphicsLayoutWidget(title='Demo: DQN & Continuous 2D Interface')
    main_window.show()
    # define reward locations
    rewards = np.array([[0.75, 0.75, 10]])
    # define discount factor for moving 0.1 units of distance
    # (assuming the agent moves straight)
    gamma_base = 0.9
    # define step size
    step_size = 0.015
    # determine step gamma
    gamma = np.power(gamma_base, 1.0 / (0.1 / step_size))
    # define environment
    room = sh.Polygon(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    )
    spawn = None
    obstacles = [
        make_rectangle(np.ones(2) / 2, 0.1, 0.1, 45),
        make_circle(np.array([0.9, 0.1]), 0.05),
        make_triangle(np.array([0.1, 0.9]), 0.1, 0.1),
    ]
    maze = ''
    mazes = {'t-maze', 'double-t-maze', 'two-sided-t-maze', 'eight-maze'}
    if len(set(sys.argv).intersection(mazes)) == 1:
        maze = list(set(sys.argv).intersection(mazes))[0]
    if maze == 't-maze':
        room, spawn, obstacles, rewards = make_t_maze(0.4, 0.2, 0.1, reward=10)
    elif maze == 'double-t-maze':
        room, spawn, obstacles, rewards = make_double_t_maze(0.4, 0.2, 0.1, reward=10)
    elif maze == 'two-sided-t-maze':
        room, spawn, obstacles, rewards = make_two_sided_t_maze(
            0.4, 0.2, 0.1, reward=10
        )
    elif maze == 'eight-maze':
        room, spawn, obstacles, rewards = make_eight_maze(0.4, 0.3, 0.1, reward=10)
    # define robot type
    robot_type: Literal['step', 'wheel'] = 'step'
    if 'wheel' in sys.argv:
        robot_type = 'wheel'
    actions = 4 if robot_type == 'step' else 3
    # a dictionary that contains all employed modules
    env = Continuous2D(robot_type, room, spawn, obstacles, rewards, None, main_window)
    env.step_size = step_size
    # amount of trials
    nb_trials = 300
    # maximum steos per trial
    nb_steps = 250
    # initialize reward monitor
    el_monitor = EscapeLatencyMonitor(nb_trials, nb_steps, main_window)
    # define policy
    policy = EpsilonGreedy(0.1)
    # initialize RL agent
    assert type(env.observation_space.shape) is tuple
    model = TorchNetwork(Model(env.observation_space.shape, actions))
    agent = DQN(
        env.observation_space,
        env.action_space,
        policy,
        model,
        gamma,
        custom_callbacks={'on_trial_end': [el_monitor.update]},
    )
    # let the agent learn
    agent.train(env, nb_trials, nb_steps)
    # stop visualization
    main_window.close()


if __name__ == '__main__':
    # prevent overuse of threads for simple model
    set_num_threads(1)
    simulation()
