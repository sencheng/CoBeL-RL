# basic imports
import numpy as np
import pyqtgraph as qg
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import framework modules
from cobel.agents.dqn import SimpleDQN
from cobel.policy.greedy import EpsilonGreedy
from cobel.networks.network_torch import TorchNetwork
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.frontends.frontends_blender import FrontendBlenderMultipleContexts
from cobel.observations.image_observations import ImageObservationBaseline
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor
from cobel.spatial_representations.topology_graphs.manual_topology_graph_no_rotation import ManualTopologyGraphNoRotation


class Model(torch.nn.Module):
    
    def __init__(self, input_size: int | tuple, number_of_actions: int):
        super().__init__()
        input_size = input_size if type(input_size) is int else np.product(input_size)
        self.layer_dense_1 = nn.Linear(in_features=input_size, out_features=64)
        self.layer_dense_2 = nn.Linear(in_features=64, out_features=64)      
        self.layer_output = nn.Linear(in_features=64, out_features=number_of_actions)
        self.double()
        
    def forward(self, layer_input: torch.Tensor) -> torch.Tensor:
        x = torch.reshape(layer_input, (len(layer_input), -1))
        x = self.layer_dense_1(x) 
        x = F.tanh(x)
        x = self.layer_dense_2(x)
        x = F.tanh(x)
        x = self.layer_output(x)
        
        return x
    
def trial_end_callback(logs: dict):
    '''
    This is a callback routine that is called when a single trial ends. Here, functionality for performance evaluation can be introduced.
    
    Parameters
    ----------
    logs :                              The trial log.
    
    Returns
    ----------
    None
    '''
    # change context every 200 trials
    if logs['trial'] % 200 == 0:
        # determine current context
        current_context = int(logs['trial']/200)
        # determine local texture path
        path = '../../environments/environments_blender/textures/symbols_v4/'
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
        logs['rl_parent'].interface_OAI.modules['world'].set_illumination('Sun', color)

def single_run(visual_output: bool = True):
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
        main_window = qg.GraphicsLayoutWidget(title='Demo: Multiple Contexts')
        main_window.show()
        
    # determine demo scene path
    demo_scene = '../../environments/environments_blender/scene_multiple_contexts.blend'
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendBlenderMultipleContexts(demo_scene)
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, visual_output)
    modules['spatial_representation'] = ManualTopologyGraphNoRotation(modules, {'start_nodes': [0], 'goal_nodes': [15], 'clique_size': 4})
    for node in modules['spatial_representation'].nodes:
        node.node_reward_bias = -1
    modules['spatial_representation'].nodes[15].node_reward_bias = 10
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)
    modules['rl_interface'] = InterfaceBaseline(modules, visual_output)
    
    # amount of trials
    number_of_trials = 1000
    # maximum steos per trial
    max_steps = 30
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [-30, 10])
    
    # define policy
    policy = EpsilonGreedy(0.1)
    
    # initialize RL agent
    rl_agent = SimpleDQN(modules['rl_interface'], policy, policy, 0.8,
                         TorchNetwork(Model((90,), 4)), {'on_trial_end': [reward_monitor.update, trial_end_callback]})
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    
    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.train(number_of_trials, max_steps)
    
    # stop simulation
    modules['world'].stop_blender()
    
    # and also stop visualization
    if visual_output:
        main_window.close()

if __name__ == '__main__':
    single_run()
