import numpy as np
import pyqtgraph as qg
from numpy import random
from keras import backend
from agents.dqn_agents import DQNAgentBaseline
from observations.image_observations import ImageObservationBaseline
from interfaces.oai_gym_interface import unity2cobelRL
from analysis.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline

# ~~~---------------------!!!---------------------~~~ #
#                    custom imports                   #
# ~~~---------------------___---------------------~~~ #
from mlagents_envs.environment import UnityEnvironment

# NOTE: do NOT use visualOutput=True
# in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'!
visualOutput = True  # shall the system provide visual output while performing the experiments?

def rewardCallback(values):
    """
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.
    values: a dict of values that are transferred from the OAI module to the reward function.
    This is flexible enough to accommodate for different experimental setups.
    """

    rlAgent = values['rlAgent']

    reward = -1.0
    stopEpisode = False

    if values['currentNode'].goalNode:
        reward = 10.0
        stopEpisode = True

    return [reward, stopEpisode]


def trialBeginCallback(trial, rlAgent):
    """
    This is a callback function that is called in the beginning of each trial. Here, experimental behavior can be defined (ABA renewal and the like).

    trial:      the number of the finished trial
    rlAgent:    the employed reinforcement learning agent
    """
    if trial == rlAgent.trialNumber - 1:
        # end the experiment by setting the number of steps to a excessively large value, this stops the 'fit' routine
        rlAgent.agent.step = rlAgent.maxSteps + 1


def trialEndCallback(trial, rlAgent, logs):
    """
    This is a callback routine that is called when a single trial ends. Here, functionality for performance evaluation can be introduced.
    trial:      the number of the finished trial
    rlAgent:    the employed reinforcement learning agent
    logs:       output of the reinforcement learning subsystem
    """

    if visualOutput:
        # update the visual elements if required
        rlAgent.performanceMonitor.update(trial, logs)


def singleRun():
    """
    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output), or by a direct call (in this case, visual output can be used).

    combinations:           this is a combination of parameters used for a single experiment. Note: the combination values depend on the experimental design!
    """

    np.random.seed()

    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    mainWindow = None
    # if visual output is required, activate an output window
    if visualOutput:
        mainWindow = qg.GraphicsWindow(title="workingTitle_Framework")

    environment_filename = '/home/mknull/CoBeL-RL/envs/3DBall_single_agent'
    # a dictionary that contains all employed modules
    modules = dict()

    modules['interfaceOAI'] = unity2cobelRL(environment_filename, modules, visualOutput, rewardCallback)

    rlAgent = DQNAgentBaseline(modules['interfaceOAI'],memoryCapacity=5000,epsilon=0.3,
                               trialBeginFcn=trialBeginCallback,trialEndFcn=trialEndCallback)

    # set the experimental parameters
    rlAgent.trialNumber = 100

    perfMon = RLPerformanceMonitorBaseline(rlAgent, mainWindow, True)
    rlAgent.performanceMonitor = perfMon

    # eventually, allow the OAI class to access the robotic agent class
    modules['interfaceOAI'].rlAgent = rlAgent

    # and allow the topology class to access the rlAgent
    #modules['topologyGraph'].rlAgent = rlAgent

    # let the agent learn, with extremely large number of allowed maximum steps
    rlAgent.train(5)

    backend.clear_session()
    modules['interfaceOAI'].close()



if __name__ == "__main__":
    singleRun()




