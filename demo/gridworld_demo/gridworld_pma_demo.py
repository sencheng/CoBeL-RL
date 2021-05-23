# basic imports
import os
import numpy as np
import PyQt5 as qt
import pyqtgraph as qg
# change directory
#os.chdir("D:/PhD/Code/CoBeL-RL-gridworld_and_dyna_q/")
# CoBel-RL framework
from agents.dyna_q_agent import PMAAgent
from interfaces.oai_gym_gridworlds import OAIGymInterface
from analysis.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline
from misc.gridworld_tools import makeOpenField, makeWindyGridworld

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
        if qt.QtGui.QApplication.instance() is not None:
            qt.QtGui.QApplication.instance().processEvents()

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
    
    # initialize world            
    world = makeOpenField(1, 10, 9, 10)
    world['startingStates'] = np.array([0])
    
    # a dictionary that contains all employed modules
    modules = dict()
    modules['rl_interface'] = OAIGymInterface(modules, world, visualOutput, mainWindow)
    
    # amount of trials
    numberOfTrials = 100
    # maximum steos per trial
    maxSteps = 30
    
    # initialize RL agent
    rlAgent = PMAAgent(interfaceOAI=modules['rl_interface'], epsilon=0.3, beta=5, learningRate=0.9,
                                   gamma=0.9, trialEndFcn=trialEndCallback)
    
    # initialize experiences
    for state in range(10):
        for action in range(4):
            rlAgent.M.states[state, action] = np.argmax(world['sas'][state, action])
    rlAgent.M.updateMask()
    
    # initialize performance Monitor
    perfMon = RLPerformanceMonitorBaseline(rlAgent, numberOfTrials, mainWindow, visualOutput)
    rlAgent.performanceMonitor = perfMon
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rlAgent = rlAgent
    
    # let the agent learn
    rlAgent.train(numberOfTrials, maxSteps, replayBatchSize=5)
    
    # and also stop visualization
    if visualOutput:
        mainWindow.close()
        
    for q in rlAgent.Q:
        print(q)


if __name__ == "__main__":
    singleRun()
