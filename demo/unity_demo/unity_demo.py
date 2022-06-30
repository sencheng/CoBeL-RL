# basic imports
import numpy as np
import os
import pyqtgraph as pg
# tensorflow
from tensorflow.keras import backend as K
# framework imports
from cobel.frontends.frontends_unity import FrontendUnityInterface
from cobel.spatial_representations.topology_graphs.four_connected_graph_rotation import Four_Connected_Graph_Rotation
from cobel.agents.dqn_agents import DQNAgentBaseline
from cobel.observations.image_observations import ImageObservationUnity
from cobel.interfaces.oai_gym_interface import OAIGymInterface
from cobel.analysis.rl_monitoring.rl_performance_monitors import UnityPerformanceMonitor

# set some python environment properties
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # reduces the amount of debug messages from tensorflow.
K.set_image_data_format(data_format='channels_last')
# shall the system provide visual output while performing the experiments? NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'!
visual_output = True


def reward_callback(values):
    '''
    This is a callback function that defines the reward provided to the robotic agent. Note: this function has to be adopted to the current experimental design.
    
    values: a dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
    '''
    # the standard reward for each step taken is negative, making the agent seek short routes
    reward = -1.0
    end_trial = False
    if values['currentNode'].goalNode:
        reward = 10.0
        end_trial = True

    return reward, end_trial


available_mazes = ['TMaze', 'TMaze_LV1', 'TMaze_LV2', 'DoubleTMaze', 'TunnelMaze_new']

def single_run():
    # the length of the edge in the topology graph
    step_size = 1.0
    # this is the main window for visual output
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = pg.GraphicsWindow(title='Unity Demo')
        # layout
        layout = pg.GraphicsLayout(border=(30, 30, 30))
        main_window.setCentralItem(layout)
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendUnityInterface('TMaze')
    modules['observation'] = ImageObservationUnity(modules['world'], main_window, visual_output, False, (30, 1, 3))
    modules['spatial_representation'] = Four_Connected_Graph_Rotation(modules, {'startNodes':[3], 'goalNodes':[0], 'start_ori': 90, 'cliqueSize':4}, step_size=step_size)
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)
    modules['rl_interface'] = OAIGymInterface(modules, visual_output, reward_callback)
    
    # amount of trials
    number_of_trials = 50
    # maximum steos per trial
    max_steps = 100
    
    # set the experimental parameters
    performanceMonitor = UnityPerformanceMonitor(main_window, visual_output)
    
    # initialize RL agent
    rl_agent = DQNAgentBaseline(modules['rl_interface'], 1000000, 0.3, None, custom_callbacks={'on_trial_end': [performanceMonitor.update]})

    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent

    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent

    # let the agent learn for 100 episode
    rl_agent.train(number_of_trials, max_steps)

    # clear keras session (for performance)
    K.clear_session()
    
    # stop Unity
    modules['world'].stopUnity()
    
    # and also stop visualization
    if visual_output:
        main_window.close()

if __name__ == '__main__':
    single_run()
