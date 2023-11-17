# basic imports
import numpy as np
import pyqtgraph as qg
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# CoBel-RL framework
from cobel.agents.dqn import SimpleDQN
from cobel.policy.greedy import EpsilonGreedy
from cobel.networks.network_torch import TorchNetwork
from cobel.interfaces.simple_2d import InterfaceSimple2D
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor


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
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsLayoutWidget(title='Demo: DQN & Simple 2D Interface')
        main_window.show()
    
    # define reward locations
    rewards = np.array([[0.75, 0.75, 10]])
    # define discount factor for moving 0.1 units of distance (assuming the agent moves straight)
    gamma_base = 0.9
    # define step size
    step_size = 0.015
    # determine step gamma
    gamma = np.power(gamma_base, 1.0/(0.1/step_size))
    
    # a dictionary that contains all employed modules
    modules = dict()
    modules['rl_interface'] = InterfaceSimple2D(modules, 'step', rewards, visual_output, main_window)
    modules['rl_interface'].step_size = step_size
    
    # amount of trials
    number_of_trials = 300
    # maximum steos per trial
    max_steps = 250
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [0, 10])
    
    # define policy
    policy = EpsilonGreedy(0.1)
    
    # initialize RL agent
    model = TorchNetwork(Model((modules['rl_interface'].observation_space.shape,), 4))
    rl_agent = SimpleDQN(modules['rl_interface'], policy, policy, gamma, model, {'on_trial_end': [reward_monitor.update]})
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps)
    
    # stop visualization
    if visual_output:
        main_window.close()


if __name__ == '__main__':
    single_run()
