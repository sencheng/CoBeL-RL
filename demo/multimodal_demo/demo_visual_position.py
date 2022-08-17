# basic imports
import os
import numpy as np
import pyqtgraph as qg
import json
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
# framework imports

from cobel.frontends.frontends_blender import FrontendBlenderInterface
from cobel.observations.image_observations import ImageObservationBaseline
from cobel.observations.position_observations import PoseObservation
from cobel.observations.dictionary_observations import DictionaryObservations
from cobel.spatial_representations.topology_graphs.simple_topology_graph  import HexagonalGraph
from cobel.agents.multi_dict_dqn import DQNAgentMultiModal
from cobel.interfaces.oai_gym_interface import OAIGymInterface
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True
move_goal_at = 10
network = "../../networks/multimodal_position_visual.json"
json_file   = open(network, 'r')
loaded_model_json = json_file.read()
json_file.close()  
model = model_from_json(loaded_model_json)    
print(model.summary())

def reward_callback(values):
    '''
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.
    
    | **Args**
    | values: A dict of values that are transferred from the OAI module to the 
      reward function. This is flexible enough to accommodate for different 
      experimental setups.
    '''
    # the standard reward for each step taken is negative, making the agent seek short routes   
    reward = -1.0
    end_trial = False
    
    if values['currentNode'].goalNode:
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
        main_window = qg.GraphicsWindow(title='Demo: DQN')
        
    # determine demo scene path
    demo_scene = os.path.abspath(__file__).split('cobel')[0] + '/cobel/environments/environments_blender/simple_grid_graph_maze.blend'
    print(demo_scene)
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendBlenderInterface(demo_scene)
    position_observation = PoseObservation(None, main_window)
    image_observation = ImageObservationBaseline(modules['world'], main_window, visual_output,
                                                 imageDims=(72, 12))
    modules['observation'] = DictionaryObservations({'image_input':image_observation,
                                                     'position_input':position_observation})    
    modules['spatial_representation'] = HexagonalGraph(n_nodes_x=5, n_nodes_y=5,
                                                       n_neighbors=6, goal_nodes=[11],
                                                       visual_output=True, 
                                                       world_module=modules['world'],
                                                       use_world_limits=True, 
                                                       observation_module=modules['observation'], 
                                                       rotation=True)
    position_observation.add_topology_graph(modules['spatial_representation'])
    modules['spatial_representation'].set_visual_debugging(main_window)
    modules['rl_interface'] = OAIGymInterface(modules, visual_output, reward_callback)
    
    # amount of trials
    number_of_trials = 150
    # maximum steps per trial
    max_steps = 100
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, 
                                   [-max_steps, 10])
    
    # initialize RL agent
    rl_agent = DQNAgentMultiModal(modules['rl_interface'], 30000, 0.3, model=model, 
                                custom_callbacks={'on_trial_end': [reward_monitor.update]})
    
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
    # clear keras session (for performance)
    K.clear_session()