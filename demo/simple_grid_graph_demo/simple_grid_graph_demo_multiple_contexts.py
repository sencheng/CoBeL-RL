# basic imports
import os
import numpy as np
import pyqtgraph as qg
# tensorflow
from tensorflow.keras import backend as K
# framework imports
from cobel.frontends.frontends_blender import FrontendBlenderMultipleContexts
from cobel.spatial_representations.topology_graphs.manual_topology_graph_no_rotation import ManualTopologyGraphNoRotation
from cobel.agents.dqn_agents import DQNAgentBaseline
from cobel.observations.image_observations import ImageObservationBaseline
from cobel.interfaces.oai_gym_interface import OAIGymInterface
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True


def reward_callback(values):
    '''
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.
    
    | **Args**
    | values:                       A dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
    '''
    # the standard reward for each step taken is negative, making the agent seek short routes   
    reward = -1.0
    end_trial = False
    
    if values['currentNode'].goalNode:
        reward = 10.0
        end_trial = True
    
    return reward, end_trial
    
def trialEndCallback(logs):
    '''
    This is a callback routine that is called when a single trial ends. Here, functionality for performance evaluation can be introduced.
    
    | **Args**
    | trial:                        The number of the current trial.
    | rlAgent:                      The employed reinforcement learning agent.
    | rlAgent:                      The output of the reinforcement learning subsystem.
    '''
    # change context every 200 trials
    if logs['trial'] % 200 == 0:
        # determine current context
        current_context = int(logs['trial']/200)
        # determine local texture path
        path = os.path.abspath(__file__).split('cobel')[0] + '/cobel/environments/environments_blender/textures/symbols_v4/'
        # determine textures
        left = path + 'symbols_%02d.png' % (current_context * 4 + 20)
        front = path + 'symbols_%02d.png' % (current_context * 4 + 21)
        right = path + 'symbols_%02d.png' % (current_context * 4 + 22)
        back = path + 'symbols_%02d.png' % (current_context * 4 + 23)
        # update textures
        logs['rl_parent'].interface_OAI.modules['world'].set_wall_textures(left, front, right, back)
        # change the environment's lighting
        color = [0.3, 0.3, 0.3]
        color[current_context % 3] = 1.
        logs['rl_parent'].interface_OAI.modules['world'].setIllumination('Sun', color)


def single_run():
    '''
    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Demo: Multiple Contexts')
        
    # determine demo scene path
    demo_scene = os.path.abspath(__file__).split('cobel')[0] + '/cobel/environments/environments_blender/scene_multiple_contexts.blend'
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendBlenderMultipleContexts(demo_scene)
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, visual_output)
    modules['spatial_representation'] = ManualTopologyGraphNoRotation(modules, {'startNodes': [0], 'goalNodes': [15], 'cliqueSize': 4})
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)
    modules['rl_interface'] = OAIGymInterface(modules, visual_output, reward_callback)
    
    # amount of trials
    number_of_trials = 1000
    # maximum steos per trial
    max_steps = 30
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [-30, 10])
    
    # initialize RL agent
    rl_agent = DQNAgentBaseline(modules['rl_interface'], 1000000, 0.3, None, custom_callbacks={'on_trial_end': [reward_monitor.update, trialEndCallback]})
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    
    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.train(number_of_trials, max_steps)
    
    # clear keras session (for performance)
    K.clear_session()
    
    # stop simulation
    modules['world'].stopBlender()
    
    # and also stop visualization
    if visual_output:
        main_window.close()

if __name__ == '__main__':
    single_run()
