# basic imports
import os
import numpy as np
import PyQt5 as qt
import pyqtgraph as qg
# tensorflow
import tensorflow as tf
# change directory
#os.chdir("D:/PhD/Code/CoBeL-RL-master/")
# CoBel-RL framework
from agents.dyna_dqn import DynaDQN
from interfaces.oai_gym_gridworlds import OAIGymInterface
from analysis.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline
from misc.gridworld_tools import makeOpenField, makeGridworld

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visualOutput = True


def trialEndCallback(trial, rlAgent, logs):
    '''
    This is a callback routine that is called when a single trial ends.
    Here, functionality for performance evaluation can be introduced.
    
    | **Args**
    | trial:                        The number of the finished trial.
    | rlAgent:                      The employed reinforcement learning agent.
    | logs:                         Output of the reinforcement learning subsystem.
    '''
    if visualOutput:
        # update the visual elements if required
        rlAgent.performanceMonitor.update(trial, logs)

        if hasattr(qt.QtGui, 'QApplication'):
            if qt.QtGui.QApplication.instance() is not None:
                qt.QtGui.QApplication.instance().processEvents()
        else:
            if qt.QtWidgets.QApplication.instance() is not None:
                qt.QtWidgets.QApplication.instance().processEvents()


def singleRun():
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    np.random.seed()
    
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    mainWindow = None
    # if visual output is required, activate an output window
    if visualOutput:
        mainWindow = qg.GraphicsWindow(title="workingTitle_Framework")
    
    # define environmental barriers
    invalidTransitions = [(3, 4), (4, 3), (8, 9), (9, 8), (13, 14), (14, 13), (18, 19), (19, 18)]
    
    # initialize world
    world = makeGridworld(5, 5, terminals=[4], rewards=np.array([[4, 10]]), goals=[4], invalidTransitions=invalidTransitions)
    world['startingStates'] = np.array([12])
    
    # a dictionary that contains all employed modules
    modules = dict()
    modules['rl_interface'] = OAIGymInterface(modules, world, visualOutput, mainWindow)
    
    # amount of trials
    numberOfTrials = 150
    # maximum steps per trial
    maxSteps = 50
    
    # initialize RL agent
    rlAgent = DynaDQN(interfaceOAI=modules['rl_interface'], epsilon=0.3, beta=5, learningRate=0.9,
                                   gamma=0.9, trialEndFcn=trialEndCallback)
    rlAgent.mask_actions = True
    rlAgent.policy = 'softmax'
    
    # initialize performance Monitor
    perfMon = RLPerformanceMonitorBaseline(rlAgent, numberOfTrials, mainWindow, visualOutput)
    rlAgent.performanceMonitor = perfMon
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rlAgent = rlAgent
    
    # let the agent learn
    rlAgent.train(numberOfTrials, maxSteps, replayBatchSize=32)
    
    # and also stop visualization
    if visualOutput:
        mainWindow.close()


if __name__ == "__main__":
    # run demo
    singleRun()
    # clear keras session (for performance)
    tf.keras.backend.clear_session()

