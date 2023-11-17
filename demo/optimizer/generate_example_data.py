# basic imports
import numpy as np
import pickle
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# framework imports
from cobel.agents.dqn import AssociativeDQN
from cobel.networks.network_torch import TorchNetwork
from cobel.interfaces.sequence import InterfaceSequence
from cobel.analysis.rl_monitoring.behavior import PredictionMonitor, PredictionErrorMonitor


def prepare_tasks():
    '''
    This function prepares a task dictionary. 
    
    Parameters
    ----------
    None\n
    
    Returns
    ----------
    tasks :                             The task dictionary.\n
    '''
    tasks = {task: {'sequence': [], 'observations': None} for task in ['task_1', 'task_2']}
    # acquisition
    for i in [0., 1., 1., 0., 1., 0., 0., 1.]:
        tasks['task_1']['sequence'].append([{'reward': i, 'observation': 'CSplus_A'}])
        tasks['task_1']['sequence'].append([{'reward': 0., 'observation': 'CSminus_A'}])
    for i in [1., 0., 1., 0., 0., 1., 0., 1.]:
        tasks['task_2']['sequence'].append([{'reward': i, 'observation': 'CSplus_A'}])
        tasks['task_2']['sequence'].append([{'reward': 0., 'observation': 'CSminus_A'}])
    # extinction
    for i in range(8):
        tasks['task_1']['sequence'].append([{'reward': 0., 'observation': 'CSplus_B'}])
        tasks['task_1']['sequence'].append([{'reward': 0., 'observation': 'CSminus_B'}])
        tasks['task_2']['sequence'].append([{'reward': 0., 'observation': 'CSplus_B'}])
        tasks['task_2']['sequence'].append([{'reward': 0., 'observation': 'CSminus_B'}])
    # recall
    for i in range(8):
        tasks['task_1']['sequence'].append([{'reward': 0., 'observation': 'CSplus_A'}])
        tasks['task_1']['sequence'].append([{'reward': 0., 'observation': 'CSminus_A'}])
        tasks['task_2']['sequence'].append([{'reward': 0., 'observation': 'CSplus_A'}])
        tasks['task_2']['sequence'].append([{'reward': 0., 'observation': 'CSminus_A'}])
    # define observations
    observations = {'CSplus_A': np.array([1., .2, .6, .4]),
                    'CSminus_A': np.array([.2, 1., .6, .4]),
                    'CSplus_B': np.array([1., .2, .4, .6]),
                    'CSminus_B': np.array([.2, 1., .4, .6])}
    tasks['task_1']['observations'] = observations
    tasks['task_2']['observations'] = observations
    
    return tasks

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

def simulation_run(task: dict, parameters: dict) -> dict:
    '''
    This function represents one simulation run. 
    
    Parameters
    ----------
    task :                              A dictionary containing the sequence of experiences and observations that the agent will be trained on.\n
    parameters :                        A dictionary containing the model parameters.\n
    
    Returns
    ----------
    simulation_data :                   A dictionary containing the simulation data.\n
    '''
    np.random.seed()
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceSequence(modules, task['sequence'], task['observations'])
    
    # network model
    model = TorchNetwork(Model(task['observations'][list(task['observations'].keys())[0]].shape, 1))
        
    # amount of trials and maximum steps per trial
    number_of_trials, max_steps = len(task['sequence']), 100
    
    # initialize performance Monitor
    prediction_monitor = PredictionMonitor()
    prediction_error_monitor = PredictionErrorMonitor()
    custom_callbacks = {'on_step_end': [prediction_monitor.update, prediction_error_monitor.update]}
        
    # initialize RL agent
    rl_agent = AssociativeDQN(modules['rl_interface'], model, batch_size=parameters['batch_size'],
                              training_repeats=parameters['training_repeats'], custom_callbacks=custom_callbacks)
    rl_agent.M.decay = parameters['decay']
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps)
    
    return {'predictions': prediction_monitor.prediction_trace, 'prediction_errors': prediction_error_monitor.prediction_error_trace}


if __name__ == '__main__':    
    # prepare tasks and store them
    tasks = prepare_tasks()
    pickle.dump(tasks, open('tasks.pkl', 'wb'))
    
    # define training parameters
    params = {'batch_size': 32, 'training_repeats': 6, 'decay': 0.9}
    number_of_runs = 25
    
    # run simulations and store the results
    data = {}
    for task in tasks:
        print('Running simulation for task: ', task)
        data[task] = []
        for run in range(number_of_runs):
            print('\tRun: ', str(run + 1))
            data[task].append(simulation_run(tasks[task], params))
    pickle.dump(data, open('data.pkl', 'wb'))
    