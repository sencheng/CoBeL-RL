"""
This demo simulation trains a Dyna-Q agent which uses PMA to replay
experiences in a 5x5 gridworld with a reward located in the upper right corner.
The agent starts each trial at the center of the environment.
The current gridworld state and location (left plot) as well as
the gridworld, the agent's current location and policy (center plot)
are visualized. Furthermore, the EscapeLatencyMonitor tracks
the steps taken in each trial and visualizes them (right plot).
"""

# basic imports
import numpy as np
from pyqtgraph import GraphicsLayoutWidget  # type: ignore
# CoBeL-RL
from cobel.agent import PMA
from cobel.memory import PMAMemory
from cobel.policy import EpsilonGreedy
from cobel.monitor import EscapeLatencyMonitor
from cobel.interface import Gridworld
from cobel.misc.gridworld_tools import make_gridworld
# typing
from cobel.typing import CallbackDict


def simulation() -> None:
    """
    Represents one simulation run.
    """
    trials: int = 250
    steps: int = 50
    # prepare widget
    main_window = GraphicsLayoutWidget(title='Demo: PMA')
    main_window.show()
    # init environment
    invalid_transitions = [
        (3, 4),
        (4, 3),
        (8, 9),
        (9, 8),
        (13, 14),
        (14, 13),
        (18, 19),
        (19, 18),
    ]
    gridworld = make_gridworld(
        5,
        5,
        terminals=[4],
        rewards=np.array([[4, 10]]),
        goals=[4],
        invalid_transitions=invalid_transitions,
    )
    gridworld['starting_states'] = np.array([12])
    # env = Gridworld({}, make_open_field(5, 5, 0, 1), widget=main_window)
    env = Gridworld(gridworld, widget=main_window)
    # init monitor and prepare callbacks
    el_monior = EscapeLatencyMonitor(trials, steps, main_window)
    callbacks: CallbackDict = {
        'on_step_end': [env.update_visualization],
        'on_trial_end': [el_monior.update],
    }
    # init and train agent
    memory = PMAMemory(env.world['sas'], EpsilonGreedy(), gamma_q=0.99)
    agent = PMA(
        env.observation_space,
        env.action_space,
        EpsilonGreedy(),
        memory,
        custom_callbacks=callbacks,
    )
    agent.mask_actions = True
    agent.train(env, trials, steps, 32)
    # close widget
    main_window.close()


if __name__ == '__main__':
    simulation()
