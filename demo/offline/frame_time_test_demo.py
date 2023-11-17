# basic imports
import time
import numpy as np
import pyqtgraph as qg
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# framework imports
from cobel.agents.dqn import SimpleDQN
from cobel.policy.greedy import EpsilonGreedy
from cobel.networks.network_torch import TorchNetwork
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.frontends.frontends_blender import ImageInterface
from cobel.observations.image_observations import ImageObservationBaseline
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor
from cobel.spatial_representations.topology_graphs.manual_topology_graph_no_rotation import ManualTopologyGraphNoRotation


class FrameTimeWorldWrapper(object):
    
    def __init__(self, world):
        '''
        A wrapper to test the frame time of a world class.
        
        Parameters
        ----------
        world :                             A world class instance.
        
        Returns
        ----------
        None
        '''
        self.world = world
        self.min_frame_time = 0.001
        self.abs_frame_time = []
        self.rel_frame_time = []
        
    def set_topology(self, topology_module):
        self.world.set_topology(topology_module)
 
    def get_manually_defined_topology_nodes(self):
        return self.world.get_manually_defined_topology_nodes()
        
    def get_manually_defined_topology_edges(self):
        return self.world.get_manually_defined_topology_edges()
         
    def step_simulation_without_physics(self, x, y, yaw):
        t0 = time.time()
        env_data = self.world.step_simulation_without_physics(x, y, yaw)
        t1 = time.time()
        self.env_data = self.world.env_data
        self.abs_frame_time.append(t1)
        self.rel_frame_time.append(max(t1-t0, self.min_frame_time))
        return env_data
        
    def actuate_robot(self, actuator_command):
        return self.world.actuate_robot(actuator_command)
    
    def get_limits(self):
        return self.world.get_limits()
     
    def get_wall_graph(self):
        return self.world.get_wall_graph()
        
    def get_mean_fps(self):
        return np.mean([1./ft for ft in self.rel_frame_time])
        
    def get_std_fps(self):
        return np.std([1./ft for ft in self.rel_frame_time])

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

def single_run(visual_output: bool = False):
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
        main_window = qg.GraphicsLayoutWidget(title='Demo: Offline')
        main_window.show()
    
    # determine world info file paths
    imageSet = 'worldInfo/images.npy'
    safeZoneDimensions = 'worldInfo/safeZoneDimensions.npy'
    safeZoneVertices = 'worldInfo/safeZoneVertices.npy'
    safeZoneSegments = 'worldInfo/safeZoneSegments.npy'
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrameTimeWorldWrapper(ImageInterface(imageSet, safeZoneDimensions, safeZoneVertices, safeZoneSegments))
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, visual_output)
    modules['spatial_representation'] = ManualTopologyGraphNoRotation(modules, {'start_nodes':[0], 'goal_nodes':[15], 'clique_size':4})
    for node in modules['spatial_representation'].nodes:
        node.node_reward_bias = -1
    modules['spatial_representation'].nodes[15].node_reward_bias = 10
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)
    modules['rl_interface'] = InterfaceBaseline(modules, visual_output)
    
    # amount of trials
    number_of_trials = 100
    # maximum steos per trial
    max_steps = 30
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [-max_steps, 10])
    
    # prepare step time callbacks
    time_start, time_end = [], []
    def step_start(logs: dict):
        time_start.append(time.time())
    def step_end(logs: dict):
        time_end.append(time.time())
        
    # define custom callbacks
    custom_callbacks={'on_trial_end': [reward_monitor.update],
                      'on_step_begin': [step_start],
                      'on_step_end': [step_end]}
    
    # define policy
    policy = EpsilonGreedy(0.1)
    
    # initialize RL agent
    rl_agent = SimpleDQN(modules['rl_interface'], policy, policy, 0.8, TorchNetwork(Model((90,), 4)), custom_callbacks)
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    
    print("measuring frame time...")
    
    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.train(number_of_trials, max_steps)
    
    # and also stop visualization
    if visual_output:
        main_window.close()
    
    # compute simulation step times
    step_times = 1/(np.array(time_end) - np.array(time_start))
    
    print("FPS in simulation step: {:0.2f} +/- {:0.2f}".format(np.mean(step_times), np.std(step_times, ddof=1)))
    print("FPS in render world: {:0.2f} +/- {:0.2f}".format(modules['world'].get_mean_fps(), modules['world'].get_std_fps()))

if __name__ == '__main__':
    single_run()
    
