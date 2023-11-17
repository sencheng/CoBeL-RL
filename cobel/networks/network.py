# basic imports
import numpy as np


class AbstractNetwork():
    
    def __init__(self, model):
        '''
        This class implements an abstract network class. 
        
        Parameters
        ----------
        model :                             The network model.
        
        Returns
        ----------
        None
        '''
        self.model = model
    
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
        raise NotImplementedError('.predict_on_batch() function not implemented!')
    
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
        raise NotImplementedError('.train_on_batch() function not implemented!')
   
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
        raise NotImplementedError('.get_weights() function not implemented!')
    
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
        raise NotImplementedError('.set_weights() function not implemented!')
        
    def clone_model(self):
        '''
        This function returns a copy of the network model.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        model :                             The network model's copy.
        '''
        raise NotImplementedError('.clone_model() function not implemented!')
        
    def set_optimizer(self, optimizer: str, parameters: None | dict = None):
        '''
        This function sets the optimizer of the network model.
        
        Parameters
        ----------
        optimizer :                         The name of the optimizer.
        parameters :                        The parameters of the optimizer (e.g., learning rate).
        
        Returns
        ----------
        None
        '''
        raise NotImplementedError('.set_optimizer() function not implemented!')
        
    def set_loss(self, loss: str, parameters: None | dict = None):
        '''
        This function sets the loss of the network model.
        
        Parameters
        ----------
        loss :                              The name of the loss.
        parameters :                        The parameters of the loss.
        
        Returns
        ----------
        None
        '''
        raise NotImplementedError('.set_loss() function not implemented!')
        
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
        raise NotImplementedError('.get_layer_activity() function not implemented!')
        