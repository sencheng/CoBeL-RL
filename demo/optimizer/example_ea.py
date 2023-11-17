# basic imports
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# framework imports
from cobel.agents.dqn import AssociativeDQN
from cobel.networks.network_torch import TorchNetwork
from cobel.interfaces.sequence import InterfaceSequence
from cobel.analysis.rl_monitoring.behavior import PredictionMonitor, PredictionErrorMonitor
from cobel.optimizers.evolution import EAOptimizer


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
    task :                              A dictionary containing the sequence of experiences and observations that the agent will be trained on.
    parameters :                        A dictionary containing the model parameters.
    
    Returns
    ----------
    simulation_data :                   A dictionary containing the simulation data.
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


def loss(simulation_data: dict, behavioral_data: dict) -> float:
    '''
    The loss function. 
    
    Parameters
    ----------
    simulation_data :                   A dictionary containing the simulation data produced by the model.
    behavioral_data :                   A dictionary containing the behavioral data.
    
    Returns
    ----------
    loss :                              The loss.
    '''
    # compute means for simulation an behavioral data
    simulation_mean, behavioral_mean = {}, {}
    for task in simulation_data:
        simulation_mean[task] = []
        behavioral_mean[task] = []
        # retrieve data
        for run in simulation_data[task]:
            simulation_mean[task].append(run['predictions'])
        for subject in behavioral_data[task]:
            behavioral_mean[task].append(subject['predictions'])
        # average data
        simulation_mean[task] = np.mean(np.array(simulation_mean[task]), axis=0)
        behavioral_mean[task] = np.mean(np.array(behavioral_mean[task]), axis=0)
    # compute the loss as the weighted MSE
    loss, number_of_subjects = 0., 0
    for task in simulation_mean:
        loss += np.sqrt(np.sum((simulation_mean[task] - behavioral_mean[task])**2)) * len(behavioral_data[task])
        number_of_subjects += len(behavioral_data[task])
    
    return loss/number_of_subjects


if __name__ == '__main__':
    # create directory for storing simulation data
    file_path = 'data_ea/'
    os.makedirs(file_path, exist_ok=True)
    
    # load task and behavioral data
    tasks = pickle.load(open('tasks.pkl', 'rb'))
    data = pickle.load(open('data.pkl', 'rb'))
    
    # define optimizer params
    parameters = {'batch_size':
                      {'init_range': {'low': 28, 'high': 36},
                       'param_range': {'a_min': 24, 'a_max': 40},
                       'param_type': int,
                       'mutator': (np.random.normal, {'scale': 10})},
                  'training_repeats':
                      {'init_range': {'low': 5, 'high': 7},
                       'param_range': {'a_min': 4, 'a_max': 8},
                       'param_type': int,
                       'mutator': (np.random.normal, {'scale': 3})},
                  'decay':
                      {'init_range': {'low': 0.85, 'high': 0.95},
                       'param_range': {'a_min': 0.8, 'a_max': 1.},
                       'param_type': float,
                       'mutator': (np.random.normal, {'scale': 0.1})}}
    population_size = 5
    runs = 1
    individuals = 10
    generations = 10
    
    # perform grid search
    optimizer = EAOptimizer(file_path, parameters, population_size)
    best_fit = optimizer.fit(simulation_run, tasks, data, loss, runs, individuals, generations)[0][-1][0]
    
    # print best fit
    print('Best fit: ', best_fit)
    print('Theoretical Best fit: batch size: 32, training repeats: 6, decay: 0.9')
    
    # run simulations with best fit params
    simulation_data_best = {task: [simulation_run(tasks[task], best_fit) for run in range(population_size)] for task in tasks}
    simulation_data = {task: [simulation_run(tasks[task], {'batch_size': 32, 'training_repeats': 6, 'decay': 0.9}) for run in range(population_size)] for task in tasks}
    
    # prepare data for plotting
    mean_behavior, mean_simulation_fit, mean_simulation = {}, {}, {}
    for task in data:
        mean_behavior[task] = []
        mean_simulation_fit[task] = []
        mean_simulation[task] = []
        for subject in data[task]:
            mean_behavior[task].append(subject['predictions'])
        for run in simulation_data_best[task]:
            mean_simulation_fit[task].append(run['predictions'])
        for run in simulation_data[task]:
            mean_simulation[task].append(run['predictions'])
        mean_behavior[task] = np.mean(np.array(mean_behavior[task]), axis=0)
        mean_simulation_fit[task] = np.mean(np.array(mean_simulation_fit[task]), axis=0)
        mean_simulation[task] = np.mean(np.array(mean_simulation[task]), axis=0)
        
    # retrieve CS indeces
    idx = {}
    for task in tasks:
        idx[task] = {'CSplus': [], 'CSminus': []}
        for t, trial in enumerate(tasks[task]['sequence']):
            if 'CSplus' in trial[0]['observation']:
                idx[task]['CSplus'].append(t)
            else:
                idx[task]['CSminus'].append(t)
        idx[task]['CSplus'] = np.array(idx[task]['CSplus'])
        idx[task]['CSminus'] = np.array(idx[task]['CSminus'])
        
    # plot behavioral data, best fit simulation data and theoretical best fit simulation data
    plt.figure(1, figsize=(18, 4.5))
    plt.subplots_adjust(wspace=0.2, hspace=0.8)
    for t, task in enumerate(tasks):
        plt.subplot(2, 3, t * 3 + 1)
        plt.title('Behavioral Data')
        plt.xlabel('Trial')
        plt.ylabel('Prediction')
        plt.plot(idx[task]['CSplus'] + 1, mean_behavior[task][idx[task]['CSplus']], color='r')
        plt.plot(idx[task]['CSminus'] + 1, mean_behavior[task][idx[task]['CSminus']], color='b')
        plt.axvline(16.5, linestyle='--', color='k')
        plt.axvline(32.5, linestyle='--', color='k')
        plt.xlim(1, 48)
        plt.ylim(np.amin(mean_behavior[task]), np.amax(mean_behavior[task]) * 1.1)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.subplot(2, 3, t * 3 + 2)
        plt.title('Best Fit\n(batch_size_%d_training_repeats_%d_decay_%f)' % tuple(best_fit.values()))
        plt.xlabel('Trial')
        plt.ylabel('Prediction')
        plt.plot(idx[task]['CSplus'] + 1, mean_simulation_fit[task][idx[task]['CSplus']], color='r')
        plt.plot(idx[task]['CSminus'] + 1, mean_simulation_fit[task][idx[task]['CSminus']], color='b')
        plt.axvline(16.5, linestyle='--', color='k')
        plt.axvline(32.5, linestyle='--', color='k')
        plt.xlim(1, 48)
        plt.ylim(np.amin(mean_simulation_fit[task]), np.amax(mean_simulation_fit[task]) * 1.1)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.subplot(2, 3, t * 3 + 3)
        plt.title('Theoretical Best Fit\n(batch_size_32_training_repeats_6_decay_0.9)')
        plt.xlabel('Trial')
        plt.ylabel('Prediction')
        plt.plot(idx[task]['CSplus'] + 1, mean_simulation[task][idx[task]['CSplus']], color='r')
        plt.plot(idx[task]['CSminus'] + 1, mean_simulation[task][idx[task]['CSminus']], color='b')
        plt.axvline(16.5, linestyle='--', color='k')
        plt.axvline(32.5, linestyle='--', color='k')
        plt.xlim(1, 48)
        plt.ylim(np.amin(mean_simulation[task]), np.amax(mean_simulation[task]) * 1.1)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    plt.savefig('example_ea.png', dpi=200, bbox_inches='tight')
