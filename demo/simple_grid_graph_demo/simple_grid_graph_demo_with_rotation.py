# basic imports
import sys
import shutil
import os
import multiprocessing as mp
import numpy as np
import time

import pyqtgraph as qg

from multiprocessing import Pool, Process
from pathlib import Path
from numpy import random
from tensorflow.keras import backend

try:
    sys.path.append(os.path.abspath(__file__ + "/../../../"))
except IndexError:
    pass

from frontends.frontends_blender import FrontendBlenderInterface
from spatial_representations.topology_graphs.manual_topology_graph_with_rotation import ManualTopologyGraphWithRotation
from agents.dqn_agents import DQNAgentBaseline
from observations.image_observations import ImageObservationBaseline
from interfaces.oai_gym_interface import OAIGymInterface
from analysis.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline


# shall the system provide visual output while performing the experiments? NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visualOutput=True



'''
This is a callback function that defines the reward provided to the robotic agent. Note: this function has to be adopted to the current experimental design.

values: a dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
'''
def rewardCallback(values):
    # the standard reward for each step taken is negative, making the agent seek short routes
    
    rlAgent=values['rlAgent']
        
    reward=-1.0
    stopEpisode=False
    
    
    
    if values['currentNode'].goalNode:
        reward=10.0
        stopEpisode=True
    
    
    return [reward,stopEpisode]




'''
This is a callback function that is called in the beginning of each trial. Here, experimental behavior can be defined (ABA renewal and the like).

trial:      the number of the finished trial
rlAgent:    the employed reinforcement learning agent
'''
def trialBeginCallback(trial,rlAgent):
        
    if trial==rlAgent.trialNumber-1:
        
        # end the experiment by setting the number of steps to a excessively large value, this stops the 'fit' routine
        rlAgent.agent.step=rlAgent.maxSteps+1
    
    


'''
This is a callback routine that is called when a single trial ends. Here, functionality for performance evaluation can be introduced.
trial:      the number of the finished trial
rlAgent:    the employed reinforcement learning agent
logs:       output of the reinforcement learning subsystem
'''

def trialEndCallback(trial,rlAgent,logs):

    if visualOutput:
        # update the visual elements if required
        rlAgent.interfaceOAI.modules['spatial_representation'].updateVisualElements()
        rlAgent.performanceMonitor.update(trial,logs)

'''
This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output), or by a direct call (in this case, visual output can be used).

combinations:           this is a combination of parameters used for a single experiment. Note: the combination values depend on the experimental design!
'''

def singleRun():
    
    np.random.seed()
    
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    mainWindow=None
    # if visual output is required, activate an output window
    if visualOutput:
        mainWindow = qg.GraphicsWindow( title="workingTitle_Framework" )
    
    
    
    # a dictionary that contains all employed modules
    modules=dict()
    
    
    
    modules['world']=FrontendBlenderInterface('simple_grid_graph_env/simple_grid_graph_maze.blend')
    modules['observation']=ImageObservationBaseline(modules['world'],mainWindow,visualOutput)
    modules['spatial_representation']=ManualTopologyGraphWithRotation(modules,{'startNodes':[0,4,8,12],'goalNodes':[15],'cliqueSize':4})
    modules['spatial_representation'].set_visual_debugging(visualOutput,mainWindow)
    modules['rl_interface']=OAIGymInterface(modules,visualOutput,rewardCallback)
    
    
    rlAgent=DQNAgentBaseline(modules['rl_interface'],5000,0.3,None,trialBeginCallback,trialEndCallback)
    
    
    # set the experimental parameters
    rlAgent.trialNumber=100
    
    perfMon=RLPerformanceMonitorBaseline(rlAgent,rlAgent.trialNumber,mainWindow,visualOutput)
    rlAgent.performanceMonitor=perfMon
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rlAgent=rlAgent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rlAgent=rlAgent
    
    # let the agent learn, with extremely large number of allowed maximum steps
    rlAgent.train(100000)
    
    backend.clear_session()
    modules['world'].stopBlender()



if __name__ == "__main__":
        
    singleRun()
    
    
    
    
