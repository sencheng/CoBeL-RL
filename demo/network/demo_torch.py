# basic imports
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# CoBel-RL framework
from cobel.networks.network_torch import TorchNetwork, FlexibleTorchNetwork


class Model(torch.nn.Module):
    
    def __init__(self, input_size: tuple = (25,), number_of_actions: int = 1):
        super().__init__()
        input_size = input_size[0] if len(input_size) == 1 else input_size
        self.layer_dense_1 = nn.Linear(in_features=input_size, out_features=64)
        self.layer_dense_2 = nn.Linear(in_features=64, out_features=64)      
        self.layer_output = nn.Linear(in_features=64, out_features=number_of_actions)
        self.double()
        
    def forward(self, layer_input):
        x = self.layer_dense_1(layer_input) 
        x = F.tanh(x)
        x = self.layer_dense_2(x)
        x = F.tanh(x)
        x = self.layer_output(x)
        x = F.sigmoid(x)
        
        return x

class MultiModel(torch.nn.Module):
    
    def __init__(self, input_size: tuple = (25,), number_of_actions: int = 1, dict_output: bool = True):
        super().__init__()
        input_size = input_size[0] if len(input_size) == 1 else input_size
        self.layer_dense_1 = nn.Linear(in_features=input_size, out_features=64)
        self.layer_dense_2 = nn.Linear(in_features=input_size, out_features=64)
        self.layer_dense_3 = nn.Linear(in_features=128, out_features=64)
        self.layer_output_1 = nn.Linear(in_features=64, out_features=number_of_actions)
        self.layer_output_2 = nn.Linear(in_features=64, out_features=number_of_actions)
        self.double()
        self.dict_output = dict_output
        
    def forward(self, layer_input_1, layer_input_2):
        # two separate input streams
        x_1 = self.layer_dense_1(layer_input_1) 
        x_1 = F.tanh(x_1)
        x_2 = self.layer_dense_2(layer_input_2)
        x_2 = F.tanh(x_2)
        # combine streams
        x = torch.cat((x_1, x_2), 1)
        x = self.layer_dense_3(x)
        x = F.tanh(x)
        # split into two separate output streams
        x_1 = self.layer_output_1(x)
        x_1 = F.sigmoid(x_1)
        x_2 = self.layer_output_2(x)
        x_2 = F.sigmoid(x_2)
        
        if self.dict_output:
            # if dicts should be used for the output; if targets are provided as lists the dict entries are accessed in order
            return {'layer_output_1': x_1, 'layer_output_2': x_2}
        else:
            return x_1, x_2

if __name__ == '__main__':
    # prepare observations
    observations = np.random.rand(5, 25)
    
    # use TorchNetwork class for simple (i.e., single input and single output) networks
    model = TorchNetwork(Model((25,), 1), activations={'layer_dense_1': F.tanh, 'layer_dense_2': F.tanh, 'layer_output': F.sigmoid})
    # use set_optimizer and and set_loss functions for setting the optimizer and loss function (accept optimizer/loss name or object as input)
    model.set_optimizer('adam')
    model.set_loss('mse')
    # use the set device function to move the model to a specific compute device, e.g., 'cpu', 'cuda', etc. (requires appropriate torch install)
    model.set_device('cpu')
    # predict_on_batch and train_on_batch functions for inference and training
    preds_1 = model.predict_on_batch(observations)
    for i in range(100):
        model.train_on_batch(observations, np.ones(5))
    # retrieve layer activity for a set of observations by specifing either layer name or layer index
    act_1 = model.get_layer_activity(observations, 'layer_output')
    act_2 = model.get_layer_activity(observations, -1)
    print(np.array_equal(act_1, act_2))
    
    # use FlexibleTorchNetwork class for complex networks
    activations = {'layer_dense_1': F.tanh, 'layer_dense_2': F.tanh, 'layer_dense_3': F.tanh, 'layer_output_1': F.sigmoid, 'layer_output_2': F.sigmoid}
    model_2 = FlexibleTorchNetwork(MultiModel((25,), 1), activations=activations)
    # in case of multiple outputs the loss function and its weighting can be set for each output 
    model_2.set_loss(['mse', 'mse'], loss_weights=[0.5, 0.5])
    # multiple inputs/outputs can be provided as list, dict or numpy.ndarray of object dtype
    preds_2 = model_2.predict_on_batch([observations, observations])
    for i in range(100):
        model_2.train_on_batch([observations, observations], [np.ones(5), np.zeros(5)])
    act_3 = model_2.get_layer_activity([observations, observations], 'layer_dense_2')
    # when using dicts the input keys must match the parameter names of the forward function
    act_4 = model_2.get_layer_activity({'layer_input_1': observations, 'layer_input_2': observations}, 'layer_dense_2')
    # when using dicts the forward function must return a dict with appropriate keys
    model_2.train_on_batch({'layer_input_1': observations, 'layer_input_2': observations}, {'layer_output_1': np.ones(5), 'layer_output_2': np.zeros(5)})
    model_2.set_loss({'layer_output_1': 'mse', 'layer_output_2': 'mse'}, loss_weights={'layer_output_1': 0.5, 'layer_output_2': 0.5})
    print(np.array_equal(act_3, act_4))
    
    # FlexibleTorchNetwork class also supports simple networks
    model_3 = FlexibleTorchNetwork(Model((25,), 1), activations={'layer_dense_1': F.tanh, 'layer_dense_2': F.sigmoid, 'layer_output': None})
    act_5 = model_3.get_layer_activity(observations, 'layer_output')
    act_6 = model_3.get_layer_activity([observations], 'layer_output')
    act_7 = model_3.get_layer_activity({'layer_input': observations}, 'layer_output')
    print(np.array_equal(act_5, act_6) and np.array_equal(act_6, act_7))
    