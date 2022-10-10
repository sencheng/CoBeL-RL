# basic imports
import numpy as np
# tensorflow
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras.models import clone_model
from tensorflow.keras import backend as K
# framework
from cobel.networks.network import AbstractNetwork


class SequentialKerasNetwork(AbstractNetwork):
    
    def __init__(self, model):
        '''
        This class provides an interface to Sequential Keras models. 
        
        Parameters
        ----------
        model :                             The network model.
        
        Returns
        ----------
        None
        '''
        super().__init__(model)
    
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
        return self.model.predict_on_batch(batch)
    
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
        self.model.train_on_batch(batch, targets)
   
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
        return self.model.get_weights()
    
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
        self.model.set_weights(weights)
        
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
        # clone model
        model_clone = clone_model(self.model)
        # compile model
        model_clone.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        
        return SequentialKerasNetwork(model_clone)
    
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
        # construct "sub model"
        sub_model = K.function([self.model.layers[input_idx].input for input_idx in batch], [self.model.layers[layer_index].output])
        # compute activity maps
        activity_maps = sub_model([batch[input_idx] for input_idx in batch])[0].T
            
        return activity_maps
        