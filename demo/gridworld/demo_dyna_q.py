"""
This demo simulation trains a Dyna-Q agent in a 5x5 gridworld
with a reward located in the upper left corner. The agent starts
each trial at a random locaction in the environment.
The current gridworld state and location (left plot) as well as
the gridworld, the agent's current location and policy (center plot)
are visualized. Furthermore, the EscapeLatencyMonitor tracks
the steps taken in each trial and visualizes them (right plot).
"""

# basic imports
import numpy as np
import pyqtgraph as pg  # type: ignore
# CoBeL-RL
from cobel.agent import DynaQ
from cobel.policy import EpsilonGreedy
from cobel.monitor import EscapeLatencyMonitor
from cobel.interface import Gridworld
from cobel.misc.gridworld_tools import make_open_field
# typing
from numpy.typing import NDArray
from cobel.typing import CallbackDict


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
    main_window = pg.GraphicsLayoutWidget(title='Demo: Dyna-Q Agent')
    main_window.show()
    env = Gridworld(make_open_field(5, 5, 0, 1), widget=main_window)
    el_monitor = EscapeLatencyMonitor(trials_train + trials_test, steps, main_window)
    custom_callbacks: CallbackDict = {
        'on_trial_end': [el_monitor.update],
        'on_step_end': [env.update_visualization],
    }
    policy, policy_test = EpsilonGreedy(), EpsilonGreedy(0.0)
    agent = DynaQ(
        env.observation_space,
        env.action_space,
        policy,
        policy_test=policy_test,
        custom_callbacks=custom_callbacks,
    )
    agent.train(env, trials_train, steps, 32)
    agent.test(env, trials_test, steps)
    main_window.close()

    return np.copy(agent.Q)


if __name__ == '__main__':
    np.set_printoptions(precision=3, floatmode='fixed')
    Q = simulation()
    print('Q-function:')
    for i in range(25):
        print('State %2d:' % i, Q[i])
