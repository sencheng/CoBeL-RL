# basic imports
import numpy as np
import pyqtgraph as qg
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# CoBel-RL framework
from cobel.agents.dyna_dqn import DynaDSR
from cobel.policy.greedy import EpsilonGreedy
from cobel.networks.network_torch import TorchNetwork
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor, EscapeLatencyMonitor
from cobel.misc.gridworld_tools import make_gridworld


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
        main_window = qg.GraphicsLayoutWidget(title='Demo: Dyna-DSR Agent')
        main_window.show()
    
    # define environmental barriers
    invalid_transitions = [(3, 4), (4, 3), (8, 9), (9, 8), (13, 14), (14, 13), (18, 19), (19, 18)]
    invalid_transitions = []
    
    # initialize world
    world = make_gridworld(5, 5, terminals=[4], rewards=np.array([[4, 10]]), goals=[4], invalid_transitions=invalid_transitions)
    world['starting_states'] = np.array([12])
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, visual_output, main_window)
    
    # amount of trials
    number_of_trials = 50
    # maximum steps per trial
    max_steps = 50
    
    # initialize monitors and prepare custom callbacks
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [0, 10])
    el_monitor = EscapeLatencyMonitor(number_of_trials, max_steps, main_window, visual_output)
    main_window.setGeometry(50, 50, 1600, 450)
    custom_callbacks = {'on_trial_end': [reward_monitor.update, el_monitor.update]}
    
    # build models
    model_SR = TorchNetwork(Model((25,), 25))
    model_reward = TorchNetwork(Model((25,), 1))
        
    # define policy
    policy = EpsilonGreedy(0.1)
    
    # initialize RL agent
    rl_agent = DynaDSR(interface_OAI=modules['rl_interface'], policy=policy, policy_test=policy, gamma=0.9,
                       model_SR=model_SR, model_reward=model_reward, custom_callbacks=custom_callbacks)
    #rl_agent.mask_actions = True
    #rl_agent.ignore_terminality = False
    #rl_agent.use_Deep_DR = True
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps, replay_batch_size=64)
    
    # and also stop visualization
    if visual_output:
        main_window.close()


if __name__ == "__main__":
    # run demo
    single_run()
