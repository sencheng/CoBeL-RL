"""
This demo simulation demonstrates how the Sequence class can be used
to model tasks which use pre-defined trial sequences.
A Q-learning agent is trained on a multi-arm-bandit like task in which
four different stimuli are shown and one of four arms yields are reward.
The reward received in each trial is visualized.
Per default discrete indeces serve as observations for the stimuli but
the simulation can be set to use unimodal, i.e., one-hot encodings, and
multimodal, i.e., one-hot encodings wrapped in a dictionary, by providing
'unimodal' and 'multimodal' command line arguments, respectively.
"""

# basic imports
import sys
import numpy as np
import pyqtgraph as pg  # type: ignore
import gymnasium as gym
# CoBeL-RL
from cobel.agent import QAgent
from cobel.policy import EpsilonGreedy
from cobel.monitor import RewardMonitor
from cobel.interface import Sequence
# typing
from typing import Literal
from cobel.typing import Trial, Observation, CallbackDict


def prepare_sequence(
    obs_type: Literal['discrete', 'unimodal', 'multimodal'] = 'discrete',
) -> tuple[list[Trial], dict[str, Observation], gym.Space]:
    """
    This function prepares a predefined trial sequence.

    Parameters
    ----------
    obs_type : Literal of 'discrete', 'unimodal' and 'multimodal'
        The type of observation that will be prepared.

    Returns
    -------
    sequence : list of Trial
        A list of predefined trials of experiences.
    observations : dict of Observation
        A dictionary of named observations referenced by the prepared sequence.
    observation_space : gym.Space
        The observation space of the observations.
    """
    sequence: list[Trial] = []
    for _ in range(250):
        sequence.append([{'observation': 'A', 'reward': np.eye(4)[0], 'action': None}])
        sequence.append([{'observation': 'B', 'reward': np.eye(4)[1], 'action': None}])
        sequence.append([{'observation': 'C', 'reward': np.eye(4)[2], 'action': None}])
        sequence.append([{'observation': 'D', 'reward': np.eye(4)[3], 'action': None}])
    observations: dict[str, Observation] = {}
    observation_space: gym.Space
    if obs_type == 'discrete':
        observations = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        observation_space = gym.spaces.Discrete(4)
    elif obs_type == 'unimodal':
        observations = {
            'A': np.eye(4)[0],
            'B': np.eye(4)[1],
            'C': np.eye(4)[2],
            'D': np.eye(4)[3],
        }
        observation_space = gym.spaces.Box(0.0, 1.0, (4,))
    else:
        observations = {
            'A': {'obs': np.eye(4)[0]},
            'B': {'obs': np.eye(4)[1]},
            'C': {'obs': np.eye(4)[2]},
            'D': {'obs': np.eye(4)[3]},
        }
        observation_space = gym.spaces.Dict({'obs': gym.spaces.Box(0.0, 1.0, (4,))})

    return sequence, observations, observation_space


def simulation(
    sequence: list[Trial],
    observations: dict[str, Observation],
    observation_space: gym.Space,
) -> None:
    """
    This function represents one simulation run.

    Parameters
    ----------
    sequence : list of Trial
        A list of predefined trials of experiences.
    observations : dict of Observation
        A dictionary of named observations referenced by the prepared sequence.
    observation_space : gym.Space
        The observation space of the observations.
    """
    trials = len(sequence)
    trials_train = int(0.5 * trials)
    trials_test = trials - trials_train
    steps = 100
    # prepare widget
    main_window = pg.GraphicsLayoutWidget(title='Sequence Test')
    main_window.show()
    # init environment
    interface = Sequence(sequence, observations, observation_space, 4)
    # init monitor and prepare callbacks
    reward_monitor = RewardMonitor(trials, (0.0, 1.0), main_window)
    callbacks: CallbackDict = {'on_trial_end': [reward_monitor.update]}
    # init and train agent
    policy = EpsilonGreedy(0.1)
    policy_test = EpsilonGreedy(0.0)
    agent = QAgent(
        interface.observation_space,
        interface.action_space,
        policy,
        policy_test,
        custom_callbacks=callbacks,
    )
    agent.train(interface, trials_train, steps, 0)
    agent.test(interface, trials_test, steps)
    # close widget
    main_window.close()


if __name__ == '__main__':
    # generate trial sequence and observations
    sequence: list
    observations: dict[str, Observation]
    observation_space: gym.Space
    if 'unimodal' in sys.argv:
        sequence, observations, observation_space = prepare_sequence('unimodal')
    elif 'multimodal' in sys.argv:
        sequence, observations, observation_space = prepare_sequence('multimodal')
    else:
        sequence, observations, observation_space = prepare_sequence('discrete')
    # run
    simulation(sequence, observations, observation_space)
