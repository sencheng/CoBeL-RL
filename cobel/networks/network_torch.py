# basic imports
import numpy as np
import copy
# torch imports
import torch
import torch.optim as optim
# framework
from cobel.networks.network import AbstractNetwork


class TorchNetwork(AbstractNetwork):
    
    def __init__(self, model):
        '''
        This class provides an interface to Torch models. 
        
        Parameters
        ----------
        model :                             The network model.
        
        Returns
        ----------
        None
        '''
        super().__init__(model)
        # initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()
    
    def predict_on_batch(self, batch: np.ndarray) -> np.ndarray:
        '''
        This function computes network predictions for a batch of input samples.
        
        Parameters
        ----------
        batch :                             The batch of input samples.
        
        Returns
        ----------
        predictions :                       A batch of network predictions.
        '''
        with torch.inference_mode():
            return self.model(torch.tensor(batch)).detach().numpy()
    
    def train_on_batch(self, batch: np.ndarray, targets: np.ndarray):
        '''
        This function trains the network on a batch of input samples.
        
        Parameters
        ----------
        batch :                             The batch of input samples.
        targets :                           The batch of target values.
        
        Returns
        ----------
        None
        '''
        self.optimizer.zero_grad()
        predictions = self.model(torch.tensor(batch))
        loss = self.criterion(predictions, torch.tensor(targets))
        loss.backward()
        self.optimizer.step()
   
    def get_weights(self) -> list:
        '''
        This function returns the weights of the network.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        weights :                           A list of layer weights.
        '''
        # retrieve params from state_dict and save them as a list of numpy arrays
        weights = list(self.model.state_dict().values())
        for i in range(len(weights)):
            weights[i] = weights[i].numpy()
            
        return weights
    
    def set_weights(self, weights: list):
        '''
        This function sets the weights of the network.
        
        Parameters
        ----------
        weights :                           A list of layer weights.
        
        Returns
        ----------
        None
        '''
        # prepare a new state_dict
        new_state_dict = self.model.state_dict()
        for i, param in enumerate(new_state_dict):
            new_state_dict[param] = torch.tensor(weights[i])
        # load the state_dict
        self.model.load_state_dict(new_state_dict)
        
    def clone_model(self):
        '''
        This function returns a copy of the network.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        model :                             The network model's copy.
        '''
        return TorchNetwork(copy.deepcopy(self.model))
        
    def get_layer_activity(self, batch: np.ndarray, layer_index: int) -> np.ndarray:
        '''
        This function returns the activity of a specified layer for a batch of input samples.
        
        Parameters
        ----------
        batch :                             The batch of input samples.
        layer_index :                       The index of the layer from which activity should be retrieved.
        
        Returns
        ----------
        activity :                          The layer activities of the specified layer for the batch of input samples.
        '''
        return np.zeros((batch.shape[0], 1))
    