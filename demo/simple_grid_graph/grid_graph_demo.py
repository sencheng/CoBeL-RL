# basic imports
import numpy as np
import pyqtgraph as qg
import PyQt5 as qt
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import framework modules
from cobel.agents.dqn import SimpleDQN
from cobel.policy.greedy import EpsilonGreedy
from cobel.networks.network_torch import TorchNetwork
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.frontends.frontends_blender import FrontendBlenderInterface
from cobel.observations.image_observations import ImageObservationBaseline
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor
from cobel.spatial_representations.topology_graphs.simple_topology_graph  import GridGraph


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
        main_window = qg.GraphicsLayoutWidget(title='Demo: DQN')
        main_window.show()
        
    # determine demo scene path
    demo_scene = '../../environments/environments_blender/simple_grid_graph_maze.blend'
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendBlenderInterface(demo_scene)
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, visual_output)
    modules['spatial_representation'] = GridGraph(n_nodes_x=5, n_nodes_y=5, start_nodes=[0], goal_nodes=[24],
                                                  visual_output=True, world_module=modules['world'],
                                                  use_world_limits=True, observation_module=modules['observation'])
    for node in modules['spatial_representation'].nodes:
        node.node_reward_bias = -1
    modules['spatial_representation'].nodes[24].node_reward_bias = 10
    modules['spatial_representation'].set_visual_debugging(main_window)
    modules['rl_interface'] = InterfaceBaseline(modules, visual_output)
    
    # amount of trials
    number_of_trials = 100
    # maximum steos per trial
    max_steps = 30
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [-30, 10])
    
    # prepare custom callbacks
    custom_callbacks = {'on_trial_end': [reward_monitor.update]}
    if visual_output:
        def update_visualization(logs: dict):
            qt.QtGui.QApplication.instance().processEvents()
        custom_callbacks['on_step_end'] = [update_visualization]
    
    # define policy
    policy = EpsilonGreedy(0.1)
    
    # initialize RL agent
    rl_agent = SimpleDQN(modules['rl_interface'], policy, policy, 0.8, TorchNetwork(Model((90,), 4)), custom_callbacks)
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    
    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.train(number_of_trials, max_steps, 32)
    
    # stop simulation
    modules['world'].stop_blender()
    
    # and also stop visualization
    if visual_output:
        main_window.close()

if __name__ == '__main__':
    single_run()
