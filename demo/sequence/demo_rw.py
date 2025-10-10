"""
In this demo simulation an agent based on the Rescorla-Wagner model is trained
on an associative learning task in which four stimuli are shown and have to
be associated with reward (or its absence). The pre- and post-train predictions
are printed to the terminal along with the ground truths.
"""

# basic imports
import numpy as np
import gymnasium as gym
# CoBeL-RL
from cobel.agent import RescorlaWagner
from cobel.interface import Sequence
# typing
from cobel.typing import Trial, Observation


def prepare_sequence() -> tuple[list[Trial], dict[str, Observation], gym.Space]:
    """
    This function prepares a predefined trial sequence.

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
        sequence.append([{'observation': 'A', 'reward': 1.0, 'action': None}])
        sequence.append([{'observation': 'B', 'reward': 0.0, 'action': None}])
        sequence.append([{'observation': 'C', 'reward': 1.0, 'action': None}])
        sequence.append([{'observation': 'D', 'reward': 0.0, 'action': None}])
    observations: dict[str, Observation] = {}
    observation_space: gym.Space
    observations = {
        'A': np.eye(4)[0],
        'B': np.eye(4)[1],
        'C': np.eye(4)[2],
        'D': np.eye(4)[3],
    }
    observation_space = gym.spaces.Box(0.0, 1.0, (4,))

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
    steps = 100
    # init environment
    interface = Sequence(sequence, observations, observation_space, 4)
    # init and train agent
    agent = RescorlaWagner(interface.observation_space)
    print('Rescorla-Wagner Demo')
    print('Predictions Pre-train:', agent.predict_on_batch(np.eye(4)))
    agent.train(interface, trials, steps)
    print('Predictions Post-train:', agent.predict_on_batch(np.eye(4)))
    print('Ground Truth: [1, 0, 1, 0]')


if __name__ == '__main__':
    # generate trial sequence and observations
    sequence: list
    observations: dict[str, Observation]
    observation_space: gym.Space
    sequence, observations, observation_space = prepare_sequence()
    # run
    simulation(sequence, observations, observation_space)
