# basic imports
import os
import numpy as np
import pyqtgraph as qg
# tensorflow
from tensorflow.keras import backend as K
# framework imports
from cobel.frontends.frontends_blender import FrontendBlenderInterface
from cobel.spatial_representations.topology_graphs.manual_topology_graph_with_rotation import ManualTopologyGraphWithRotation
from cobel.agents.keras_rl.dqn import DQNAgentBaseline
from cobel.observations.image_observations import ImageObservationBaseline
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True


def reward_callback(values):
    '''
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.
    
    Parameters
    ----------
    values :                            A dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
    
    Returns
    ----------
    reward :                            The reward that will be provided.
    end_trial :                         Flag indicating whether the trial ended.
    '''
    # the standard reward for each step taken is negative, making the agent seek short routes   
    reward = -1.0
    end_trial = False
    
    if values['current_node'].goal_node:
        reward = 10.0
        end_trial = True
    
    return reward, end_trial

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
        main_window = qg.GraphicsWindow(title='Demo: DQN using rotation')
    
    # determine demo scene path
    demo_scene = os.path.abspath(__file__).split('cobel')[0] + '/cobel/environments/environments_blender/simple_grid_graph_maze.blend'
    
    # a dictionary that contains all employed modules
    modules=dict()
    modules['world'] = FrontendBlenderInterface(demo_scene)
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, visual_output)
    modules['spatial_representation'] = ManualTopologyGraphWithRotation(modules, {'start_nodes': [0], 'goal_nodes': [15], 'clique_size': 4})
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)
    modules['rl_interface'] = InterfaceBaseline(modules, visual_output, reward_callback)
    
    # amount of trials
    number_of_trials = 100
    # maximum steos per trial
    max_steps = 30
    
     # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [-30, 10])
    
    # initialize RL agent
    rl_agent = DQNAgentBaseline(modules['rl_interface'], 1000000, 0.3, None, custom_callbacks={'on_trial_end': [reward_monitor.update]})
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    
    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.train(number_of_trials, max_steps)
    
    # clear keras session (for performance)
    K.clear_session()
    
    # stop simulation
    modules['world'].stop_blender()
    
    # and also stop visualization
    if visual_output:
        main_window.close()


if __name__ == '__main__':
    single_run()