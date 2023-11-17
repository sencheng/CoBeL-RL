# basic imports
import numpy as np
import pyqtgraph as qg
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# CoBel-RL framework
from cobel.agents.ddpg import DDPGAgent
from cobel.policy.policy import AbstractPolicy
from cobel.networks.network_torch import FlexibleTorchNetwork
from cobel.interfaces.move_2d import InterfaceMove2D
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor

    
class Model(torch.nn.Module):
    
    def __init__(self, input_size: int | tuple, number_of_actions: int):
        super().__init__()
        input_size = input_size if type(input_size) is int else np.product(input_size)
        # actor network
        self.layer_actor_1 = nn.Linear(in_features=input_size, out_features=64)
        self.layer_actor_2 = nn.Linear(in_features=64, out_features=64)
        self.layer_actor_out = nn.Linear(in_features=64, out_features=number_of_actions)
        # critic network
        self.layer_critic_1 = nn.Linear(in_features=input_size, out_features=64)
        self.layer_critic_2 = nn.Linear(in_features=64, out_features=64)
        self.layer_critic_3 = nn.Linear(in_features=64 + number_of_actions, out_features=64)
        self.layer_critic_out = nn.Linear(in_features=64, out_features=1)
        self.double()
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # actor network
        x_actor = self.layer_actor_1(states)
        x_actor = F.relu(x_actor)
        x_actor = self.layer_actor_2(x_actor)
        x_actor = F.relu(x_actor)
        x_actor = self.layer_actor_out(x_actor)
        # prevent changes to the critic when updating the actor
        if actions.shape[1] == 0:
            # critic network
            with torch.inference_mode():
                x_critic = self.layer_critic_1(states)
                x_critic = F.relu(x_critic)
                x_critic = self.layer_critic_2(x_critic)
                x_critic = F.relu(x_critic)
                x_critic = torch.cat((x_actor, x_critic), 1)
                x_critic = self.layer_critic_3(x_critic) 
                x_critic = F.relu(x_critic)
                x_critic = self.layer_critic_out(x_critic)
                
            return x_actor, x_critic
        else:
            x_critic = self.layer_critic_1(states)
            x_critic = F.relu(x_critic)
            x_critic = self.layer_critic_2(x_critic)
            x_critic = F.relu(x_critic)
            x_critic = torch.cat((actions, x_critic), 1)
            x_critic = self.layer_critic_3(x_critic) 
            x_critic = F.relu(x_critic)
            x_critic = self.layer_critic_out(x_critic)
            
            return x_actor, x_critic
    
class ContinuousPolicy(AbstractPolicy):
    
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma
        
    def select_action(self, v: np.ndarray, mask: None | np.ndarray = None) -> np.ndarray:
        print(v)
        return  np.random.normal(v * 10 ** -1, self.sigma)
    
    def get_action_probs(self, v: np.ndarray, mask: None | np.ndarray = None) -> np.ndarray:
        return np.zeros(v.shape)
    
class DDPGLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        if targets.shape[1] == 0:
            return torch.mean(-inputs[1])
        else:
            return torch.mean((targets - inputs[1]) ** 2)
    

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
        main_window = qg.GraphicsLayoutWidget(title='Demo: DDPG & Move 2D Interface')
        main_window.show()
    
    # prevent overuse of threads for simple model
    torch.set_num_threads(1)
    
    # build models
    model = FlexibleTorchNetwork(Model(4, 2), loss=DDPGLoss())
    
    # a dictionary that contains all employed modules
    modules = dict()
    modules['rl_interface'] = InterfaceMove2D(modules, visual_output, main_window)
    modules['rl_interface'].static_goal = np.array([.5, .5])
    
    # amount of train and test trials
    trials_train, trials_test = 500, 10
    # amount of trials
    number_of_trials = trials_train + trials_test
    # maximum steos per trial
    max_steps = 100
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [-1.2 * max_steps, 1])
    
    # define action selection policies
    policy = ContinuousPolicy(0.01)
    policy_test = ContinuousPolicy(0.)
    
    # initialize RL agent
    rl_agent = DDPGAgent(modules['rl_interface'], policy, policy_test, 0.8, model, {'on_trial_end': [reward_monitor.update]})
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(trials_train, max_steps)
    
    # test the agent
    rl_agent.test(trials_test, max_steps)
    
    # stop visualization
    if visual_output:
        main_window.close()


if __name__ == '__main__':
    single_run()
