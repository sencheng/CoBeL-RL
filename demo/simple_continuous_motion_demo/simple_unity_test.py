import numpy as np
import pyqtgraph as qg
from numpy import random
from keras import backend
from agents.dqn_agents import DQNAgentBaseline
from observations.image_observations import ImageObservationBaseline
from interfaces.oai_gym_interface import unity2cobelRL
from analysis.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline
import os

# ~~~---------------------!!!---------------------~~~ #
#                    custom imports                   #
# ~~~---------------------___---------------------~~~ #
from mlagents_envs.environment import UnityEnvironment

visualOutput = True  # shall the system provide visual output while performing the experiments?

def rewardCallback(*args, **kwargs):
    """
    ATTENTION: This function is deprecated.
    These changes should be encoded in the Academy object of the environment, and triggered via a side channel.
    :return: None
    """

    raise NotImplementedError('This function is deprecated. These changes should be encoded in the Academy '
                              'object of the environment, and triggered via a side channel.')

def trialBeginCallback(trial, rlAgent):
    """
    This is a callback function that is called in the beginning of each trial. Here, experimental behavior can be
    defined (ABA renewal and the like).
    :param trial: the number of the finished trial
    :param rlAgent: the employed reinforcement learning agent
    :return: None
    """

    if trial == rlAgent.trialNumber - 1:
        # end the experiment by setting the number of steps to a excessively large value, this stops the 'fit' routine
        rlAgent.agent.step = rlAgent.maxSteps + 1


def trialEndCallback(trial, rlAgent, logs):
    """
    This is a callback routine that is called when a single trial ends. Here, functionality for performance evaluation
    can be introduced.
    :param trial: the number of the finished trial
    :param rlAgent: the employed reinforcement learning agent
    :param logs: output of the reinforcement learning subsystem
    :return:
    """
    pass

def singleRun(environment_filename, n_train=1):
    """
    :param environment_filename: full path to a Unity executable
    :param n_train: amount of RL steps
    :return:

    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization
    mechanism (without visual output), or by a direct call (in this case, visual output can be used).
    """

    # set random seed
    seed = 42  # 42 is used for good luck. If more luck is needed try 4, 20, or a combination. If nothing works, try 13.

    # a dictionary that contains all employed modules
    modules = dict()

    modules['interfaceOAI'] = unity2cobelRL(env_path=environment_filename, modules=modules, withGUI=visualOutput,
                                            rewardCallback=rewardCallback, seed=seed)

    rlAgent = DQNAgentBaseline(modules['interfaceOAI'],memoryCapacity=5000,epsilon=0.3,
                               trialBeginFcn=trialBeginCallback,trialEndFcn=trialEndCallback)

    # set the experimental parameters
    rlAgent.trialNumber = 1000

    # eventually, allow the OAI class to access the robotic agent class
    modules['interfaceOAI'].rlAgent = rlAgent

    # let the agent learn, with extremely large number of allowed maximum steps
    rlAgent.train(n_train)

    backend.clear_session()
    modules['interfaceOAI'].close()


def get_cobel_rl_path():
    paths = os.environ['PYTHONPATH'].split(':')
    path = None
    for p in paths:
        if 'CoBeL-RL' in p:
            full_path = p
            base_folder = full_path.split(sep='CoBeL-RL')[0]
            path = base_folder + 'CoBeL-RL'
            break

    return path


if __name__ == "__main__":

    project = get_cobel_rl_path()
    print('Testing 3DBall environment')
    singleRun(environment_filename=project+'/envs/3DBall_single_agent', n_train=1000)
    print('Testing GridWorld environment')
    singleRun(environment_filename=project+'/envs/GridWorld_single_agent.bak', n_train=1000)
    print('Testing concluded: No program breaking bugs detected')

