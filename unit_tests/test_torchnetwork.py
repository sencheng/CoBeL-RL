# basic imports
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# CoBeL-RL
from cobel.networks.network_torch import TorchNetwork


class Model(torch.nn.Module):
    
    def __init__(self, input_size: int | tuple, output_size: int):
        super().__init__()
        input_size = input_size if type(input_size) is int else np.product(input_size)
        self.layer_dense_1 = nn.Linear(in_features=input_size, out_features=64)
        self.layer_dense_2 = nn.Linear(in_features=64, out_features=64)      
        self.layer_output = nn.Linear(in_features=64, out_features=output_size)
        self.double()
        
    def forward(self, layer_input: torch.Tensor) -> torch.Tensor:
        x = torch.reshape(layer_input, (len(layer_input), -1))
        x = self.layer_dense_1(x) 
        x = F.tanh(x)
        x = self.layer_dense_2(x)
        x = F.tanh(x)
        x = self.layer_output(x)
        x = F.sigmoid(x)
        
        return x

if __name__ == '__main__':
    # prepare a simple data set
    data = np.random.rand(1000, 2)
    labels = ((data[:, 0] > 0.5) * (data[:, 1] > 0.5)).astype(float) + ((data[:, 0] < 0.5) * (data[:, 1] < 0.5).astype(float))
    # init network
    network = Model(2, 1)
    # prepare training hyper-parameters
    loss = torch.nn.HuberLoss()
    loss_params = {'delta': 1.1}
    optimizer = torch.optim.Adam(network.parameters())
    optimizer_params = {'lr': 0.005, 'betas': (0.91, 0.998)}
    activations = {'layer_dense_1': F.tanh, 'layer_dense_2': F.tanh, 'layer_output': F.sigmoid}
    
    # init model with default parameters
    model = TorchNetwork(network)
    # test default parameters
    assert type(model.criterion) is torch.nn.MSELoss, 'Default loss wasn\'t of type MSELoss.'
    assert type(model.optimizer) is torch.optim.Adam, 'Default optimizer wasn\'t of type Adam.'
    assert len(model.get_weights()) == 6, 'The number of weights doesn\'t match the expected value of 6.'
    assert type(model.activations) is dict, 'Default activations wasn\'t of type dict.'
    assert len(model.activations) == 0, 'Default activations wasn\'t empty.'
    
    # init model with parameters
    model = TorchNetwork(network, optimizer, loss, optimizer_params, loss_params, activations)
    # test model parameters
    assert type(model.criterion) is torch.nn.HuberLoss, 'Model loss wasn\'t of type HuberLoss.'
    assert model.criterion.delta == 1.1, 'Huber loss\' delta parameter doesn\'t match expected value of 1.1.'
    assert type(model.optimizer) is torch.optim.Adam, 'Model optimizer wasn\'t of type Adam.'
    state_dict = model.optimizer.state_dict()
    assert state_dict['param_groups'][0]['lr'] == 0.005 and state_dict['param_groups'][0]['betas'] == (0.91, 0.998), 'Model optimizer\'s parameters don\'t match the expected values (lr==0.005, betas==(0.91, 0.998)).'
    assert len(model.activations) == 3, 'Length of activations doesn\'t match the expected value of 3.'
    
    # test layer activity retrieval
    # with layer indices
    for i, size in enumerate([64, 64, 1]):
        layer_shape = model.get_layer_activity(data[:10], i).shape
        assert layer_shape == (size, 10), 'Shape of activity at layer %d didn\'t match the expected shape of (10, %d).' % (i, size)
    # with layer names
    for layer, size in {'layer_dense_1': 64, 'layer_dense_2': 64, 'layer_output': 1}.items():
        layer_shape = model.get_layer_activity(data[:10], layer).shape
        assert layer_shape == (size, 10), 'Shape of activity at layer %s didn\'t match the expected shape of (10, %d).' % (layer, size)
        
    # test training
    weights_before = model.get_weights()
    preds_before = model.predict_on_batch(data).flatten()
    error_before = np.mean((preds_before - labels) ** 2)
    for epoch in range(100):
        for batch in range(20):
            start, end = batch * 50, (batch + 1) * 50
            model.train_on_batch(data[start:end], labels[start:end])
    preds_after = model.predict_on_batch(data).flatten()
    error_after = np.mean((preds_after - labels) ** 2)
    assert error_after < error_before, 'Model didn\'t train properly.'
    # test weight setting
    model.set_weights(weights_before)
    for i, weight in enumerate(model.get_weights()):
        assert np.array_equal(weights_before[i], weight), 'Weights not set at layer %d.' % i
    
    # test model cloning
    model_cloned = model.clone_model()
    assert type(model.criterion) == type(model_cloned.criterion), 'Cloned model\'s loss function doesn\'t match.'
    assert model.criterion.delta == model_cloned.criterion.delta, 'Cloned model\'s loss function\'s delta parameter doesn\'t match.'
    assert type(model.optimizer) == type(model_cloned.optimizer), 'Cloned model\'s optimizer doesn\'t match.'
    param_group, param_group_cloned = model.optimizer.state_dict()['param_groups'][0], model_cloned.optimizer.state_dict()['param_groups'][0]
    assert param_group['lr'] == param_group_cloned['lr'] and param_group['betas'] == param_group_cloned['betas'], 'Cloned model\'s optimizer\'s parameter don\'t match.'
    assert type(model.activations) == type(model_cloned.activations), 'Cloned model\'s activations doesn\'t match.'
    for i, d in enumerate(model_cloned.activations):
        idx = d if type(d) is str else i
        assert model.activations[idx] is model_cloned.activations[idx], 'Cloned model\'s activation at layer %s doesn\'t match.' % idx
    assert model.model.parameters() != model_cloned.model.parameters(), 'Cloned model shares underlying parameters.'
    
    # test setters
    # loss function
    model.set_loss('mse')
    assert type(model.criterion) is torch.nn.MSELoss, 'Loss function wasn\'t set when using string identifier.'
    model.set_loss('huber', loss_params)
    assert model.criterion.delta == 1.1, 'Loss function\'s delta parameter wasn\'t set when using string identifier.'
    model.set_loss('mse')
    model.set_loss(loss, loss_params)
    assert type(model.criterion) is torch.nn.HuberLoss, 'Loss function wasn\'t set when using loss class.'
    assert model.criterion.delta == 1.1, 'Loss function\'s delta parameter wasn\'t set when using loss class.'
    # optimizer
    model.set_optimizer('adam', {'lr': 0.1})
    assert type(model.optimizer) is torch.optim.Adam, 'Optimizer wasn\'t set when using string identifier'
    assert model.optimizer.state_dict()['param_groups'][0]['lr'] == 0.1, 'Optimizer\'s lr parameter wasn\'t set when using string identifier.'
    model.set_optimizer(torch.optim.Adam(model.model.parameters()), {'lr': 0.01})
    assert type(model.optimizer) is torch.optim.Adam, 'Optimizer wasn\'t set when using string identifier'
    assert model.optimizer.state_dict()['param_groups'][0]['lr'] == 0.01, 'Optimizer\'s lr parameter wasn\'t set when using string identifier.'
    