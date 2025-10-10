"""
This demo simulation showcases how the GridSearchOptimizer can be used to fit
the learning parameters of a Q-learning agent that is trained on a multi-arm-bandit like
task in which four different stimuli are shown and one of eight arms yields are reward.
Synthetic data is generated using a learning rate of 0.9 and inverse temperatures
of 1 (the agent used a softmax policy), and the grid search is run to recover these
learning parameters with the MSE between trial-by-trial responses serving as the
loss function. The best fit is printed along with the ground truth to the terminal.
"""

# basic imports
import os
import numpy as np
import gymnasium as gym
import multiprocessing as mp
# CoBeL-RL
from cobel.agent import QAgent
from cobel.policy import Softmax
from cobel.monitor import RewardMonitor
from cobel.interface import Sequence
from cobel.optimizer import GridSearchOptimizer
# typing
from typing import Any
from cobel.interface.sequence import Trial
from cobel.interface.interface import Observation
Trials = list[Trial]
ObsDict = dict[str, Observation]


def generate_sequence() -> tuple[Trials, ObsDict, gym.spaces.Box, gym.spaces.Discrete]:
    """
    Generates a sequence of learning trials.

    Returns
    -------
    sequence : Trials
        The trial sequence.
    observations : ObsDict
        A dictionary containing the observations referenced in `sequence`.
    observation_space : gym.spaces.Box
        The observation space.
    action_space : gym.spaces.Discrete
        The action space.
    """
    sequence: list[Trial] = []
    for _ in range(100):
        sequence.append([{'observation': 'A', 'reward': np.eye(8)[0], 'action': None}])
        sequence.append([{'observation': 'B', 'reward': np.eye(8)[2], 'action': None}])
        sequence.append([{'observation': 'C', 'reward': np.eye(8)[4], 'action': None}])
        sequence.append([{'observation': 'D', 'reward': np.eye(8)[6], 'action': None}])
    observations: ObsDict = {
        'A': np.eye(4)[0],
        'B': np.eye(4)[1],
        'C': np.eye(4)[2],
        'D': np.eye(4)[3],
    }
    observation_space = gym.spaces.Box(0.0, 1.0, (4,))
    action_space = gym.spaces.Discrete(8)

    return sequence, observations, observation_space, action_space


def simulation(task: dict[str, Any], parameters: dict[str, Any]) -> Any:
    """
    Represents one simulation run.

    Parameters
    ----------
    task : dict of Any
        A dictionary containing the task definition.
    parameters : dict of Any
        A dictionary containing the learning parameters.

    Returns
    -------
    results : Any
        The simulation results, i.e., simulation data.
    """
    env = Sequence(
        task['sequence'],
        task['observations'],
        task['observation_space'],
        int(task['action_space'].n),
    )
    nb_trials = len(task['sequence'])
    policy = Softmax(parameters['beta'])
    reward_monitor = RewardMonitor(nb_trials)
    agent = QAgent(
        observation_space,
        action_space,
        policy,
        learning_rate=parameters['learning_rate'],
        custom_callbacks={'on_step_end': [reward_monitor.update]},
    )
    agent.train(env, nb_trials, 100)

    return reward_monitor.get_trace()


def loss(simulation_data: dict[str, Any], behavioral_data: dict[str, Any]) -> float:
    """
    The loss function used for the fitting process.
    Here mean squared error is used.

    Parameters
    ----------
    simulation_data : dict of Any
        The simulation data collect for different tasks.
    behavioral_data : dict of Any
        The behavioral data to be fitted for different tasks.

    Returns
    -------
    loss : float
        The computed loss.
    """
    return np.sum(
        np.sqrt(
            (
                np.mean(simulation_data['demo'], axis=0)
                - np.mean(behavioral_data['demo'], axis=0)
            )
            ** 2
        )
    )


if __name__ == '__main__':
    # prepare task
    trials, observations, observation_space, action_space = generate_sequence()
    tasks = {
        'demo': {
            'sequence': trials,
            'observations': observations,
            'observation_space': observation_space,
            'action_space': action_space,
        }
    }
    # generate synthetic data for fitting
    data = {
        'demo': [
            simulation(tasks['demo'], {'learning_rate': 0.9, 'beta': 1})
            for _ in range(100)
        ]
    }
    # define parameters to be searched
    parameters = {
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'beta': [1, 5, 10],
    }
    # run grid search
    os.makedirs('fit/', exist_ok=True)
    optimizer = GridSearchOptimizer('fit/', parameters, 100, 'shuffled')
    fit = optimizer.fit(
        simulation, tasks, data, loss, store_simulation_data=False, pool=mp.Pool()
    )
    best_fit = np.argmin(list(fit.values()))
    print('Best fit is: ', list(fit.keys())[best_fit])
    print('Ground Truth: (0.9, 1)')
