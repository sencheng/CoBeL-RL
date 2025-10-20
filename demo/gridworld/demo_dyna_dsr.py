"""
This demo simulation trains an agent combining the Dyna-Q and DSR algorithms
in a 5x5 gridworld with a reward located in the upper left corner.
Each gridworld state is mapped to an observation (here one-hot encodings)
which serve as input to a DNN that represents the SR.
The agent starts each trial at a random locaction in the environment.
The current gridworld state and location (left plot) as well as
the gridworld, the agent's current location and policy (center plot)
are visualized. Furthermore, the EscapeLatencyMonitor tracks
the steps taken in each trial and visualizes them (right plot).
"""

# basic imports
import numpy as np
import pyqtgraph as pg  # type: ignore
# torch
from torch import Tensor, reshape, set_num_threads
from torch.nn import Module, Linear
from torch.nn.functional import relu
# CoBeL-RL
from cobel.agent import DynaDSR
from cobel.policy import EpsilonGreedy
from cobel.network import TorchNetwork
from cobel.monitor import EscapeLatencyMonitor
from cobel.interface import Gridworld
from cobel.misc.gridworld_tools import make_open_field
# typing
from numpy.typing import NDArray
from cobel.typing import CallbackDict


class Model(Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.layer_dense_1 = Linear(in_features=input_size, out_features=64)
        self.layer_dense_2 = Linear(in_features=64, out_features=64)
        self.layer_output = Linear(in_features=64, out_features=output_size)
        self.double()

    def forward(self, layer_input: Tensor) -> Tensor:
        x = reshape(layer_input, (len(layer_input), -1))
        x = self.layer_dense_1(x)
        x = relu(x)
        x = self.layer_dense_2(x)
        x = relu(x)
        x = self.layer_output(x)

        return x


def simulation() -> NDArray:
    """
    Represents one simulation run.

    Returns
    -------
    q_function : NDArray
        The learned Q-function.
    """
    trials_train = 500
    trials_test = 300
    steps = 50
    main_window = pg.GraphicsLayoutWidget(title='Demo: Dyna-DSR Agent')
    main_window.show()
    env = Gridworld(make_open_field(5, 5, 0, 1), widget=main_window)
    network_successor = TorchNetwork(Model(25, 25))
    network_reward = TorchNetwork(Model(25, 1))
    el_monitor = EscapeLatencyMonitor(trials_train + trials_test, steps, main_window)
    custom_callbacks: CallbackDict = {
        'on_trial_end': [el_monitor.update],
        'on_step_end': [env.update_visualization],
    }
    policy, policy_test = EpsilonGreedy(), EpsilonGreedy(0.0)
    agent = DynaDSR(
        env.observation_space,
        env.action_space,
        policy,
        network_successor,
        network_reward,
        gamma=0.8,
        policy_test=policy_test,
        custom_callbacks=custom_callbacks,
    )
    agent.train(env, trials_train, steps, 32)
    agent.test(env, trials_test, steps)
    main_window.close()

    return agent.predict_on_batch(np.arange(25))


if __name__ == '__main__':
    set_num_threads(1)
    np.set_printoptions(precision=3, floatmode='fixed')
    Q = simulation()
    print('Q-function:')
    for i in range(25):
        print('State %2d:' % i, Q[i])
