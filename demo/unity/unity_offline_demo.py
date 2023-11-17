# basic imports
import numpy as np
import pyqtgraph as pg
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# framework imports
from cobel.agents.dqn import SimpleDQN
from cobel.policy.greedy import EpsilonGreedy
from cobel.networks.network_torch import TorchNetwork
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.frontends.frontends_unity import FrontendUnityOfflineInterface
from cobel.observations.image_observations import ImageObservationUnity
from cobel.analysis.rl_monitoring.rl_performance_monitors import UnityPerformanceMonitor
from cobel.spatial_representations.topology_graphs.four_connected_graph_rotation import FourConnectedGraphRotation


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

available_mazes = ['TMaze', 'TMaze_LV1', 'TMaze_LV2', 'DoubleTMaze', 'TunnelMaze_new']

def single_run(visual_output: bool = True):
    # the length of the edge in the topology graph
    step_size = 1.0
    # this is the main window for visual output
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = pg.GraphicsLayoutWidget(title='Unity Offline Demo')
        main_window.show()
        # layout
        layout = pg.GraphicsLayout(border=(30, 30, 30))
        main_window.setCentralItem(layout)
    
    # determine world info file path
    worldInfo = 'worldInfo/TMaze_Infos.pkl'
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendUnityOfflineInterface(worldInfo)
    modules['observation'] = ImageObservationUnity(modules['world'], main_window, visual_output, False, (30, 1, 3))
    modules['spatial_representation'] = FourConnectedGraphRotation(modules, {'start_nodes':[3], 'goal_nodes':[0], 'start_ori': 90, 'clique_size':4}, step_size=step_size)
    for node in modules['spatial_representation'].nodes:
        node.node_reward_bias = -1
    modules['spatial_representation'].nodes[0].node_reward_bias = 10
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)
    modules['rl_interface'] = InterfaceBaseline(modules, visual_output)
    
    # amount of trials
    number_of_trials = 50
    # maximum steos per trial
    max_steps = 100
    
    # set the experimental parameters
    performanceMonitor = UnityPerformanceMonitor(main_window, visual_output)
    
    # define policy
    policy = EpsilonGreedy(0.1)
    
    # initialize RL agent
    rl_agent = SimpleDQN(modules['rl_interface'], policy, policy, 0.8,
                         TorchNetwork(Model((90,), 4)), {'on_trial_end': [performanceMonitor.update]})

    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent

    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent

    # let the agent learn for 100 episode
    rl_agent.train(number_of_trials, max_steps)
    
    # stop Unity
    modules['world'].stop_unity()
    
    # and also stop visualization
    if visual_output:
        main_window.close()

if __name__ == '__main__':
    single_run()
