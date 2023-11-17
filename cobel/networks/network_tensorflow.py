# basic imports
import numpy as np
import copy
# tensorflow
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras.models import clone_model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, losses
# framework
from cobel.networks.network import AbstractNetwork


def get_optimizers() -> dict:
    '''
    This function returns a dictionary containing the available optimizers.
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    optimizers :                        The available optimizers.
    '''
    # return default optimizers if eager execution is enabled, else return the legacy optimizers
    if tf.executing_eagerly():
        return {**dict.fromkeys(['Adadelta', 'adadelta'], optimizers.Adadelta), **dict.fromkeys(['Adagrad', 'adagrad'], optimizers.Adagrad),
                **dict.fromkeys(['Adafactor', 'adafactor'], optimizers.Adafactor), **dict.fromkeys(['Adam', 'adam'], optimizers.Adam),
                **dict.fromkeys(['AdamW', 'Adamw', 'adamw'], optimizers.AdamW), **dict.fromkeys(['Adamax', 'AdaMax', 'adamax'], optimizers.Adamax),
                **dict.fromkeys(['Ftrl', 'ftrl'], optimizers.Ftrl), **dict.fromkeys(['NAdam', 'Nadam', 'nadam'], optimizers.Nadam),
                **dict.fromkeys(['RMSprop', 'rmsprop'], optimizers.RMSprop), **dict.fromkeys(['SGD', 'sgd'], optimizers.SGD)}
    else:
        return {**dict.fromkeys(['Adadelta', 'adadelta'], optimizers.legacy.Adadelta), **dict.fromkeys(['Adagrad', 'adagrad'], optimizers.legacy.Adagrad),
                **dict.fromkeys(['Adam', 'adam'], optimizers.legacy.Adam), **dict.fromkeys(['Adamax', 'AdaMax', 'adamax'], optimizers.legacy.Adamax),
                **dict.fromkeys(['Ftrl', 'ftrl'], optimizers.legacy.Ftrl), **dict.fromkeys(['NAdam', 'Nadam', 'nadam'], optimizers.legacy.Nadam),
                **dict.fromkeys(['RMSprop', 'rmsprop'], optimizers.legacy.RMSprop), **dict.fromkeys(['SGD', 'sgd'], optimizers.legacy.SGD)}
    

def get_losses() -> dict:
    '''
    This function returns a dictionary containing the available loss functions.
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    losses :                            The available loss functions.
    '''
    return {'binary_crossentropy': losses.BinaryCrossentropy, 'binary_focal_crossentropy': losses.BinaryFocalCrossentropy,
            'categorical_crossentropy': losses.CategoricalCrossentropy, 'sparse_categorical_crossentropy': losses.SparseCategoricalCrossentropy,
            'hinge': losses.Hinge, 'categorical_hinge': losses.CategoricalHinge, 'squared_hinge': losses.SquaredHinge,
            'poisson': losses.Poisson, 'cosine_similarity': losses.CosineSimilarity,
            'kl_divergence': losses.KLDivergence, 'log_cosh': losses.LogCosh,
            **dict.fromkeys(['mean_absolute_error', 'mae'], losses.MeanAbsoluteError),
            **dict.fromkeys(['mean_absolute_percentage_error', 'mape'], losses.MeanAbsolutePercentageError),
            **dict.fromkeys(['mean_squared_error', 'mse'], losses.MeanSquaredError),
            **dict.fromkeys(['mean_squared_logarithmic_error', 'msle'], losses.MeanSquaredLogarithmicError),
            **dict.fromkeys(['huber_loss', 'huber'], losses.Huber)}
    

def is_optimizer_instance(optimizer: str | optimizers.Optimizer | optimizers.legacy.Optimizer) -> bool:
    '''
    This function checks if a given input is an optimizer instance.
    
    Parameters
    ----------
    optimizer :                         The putative optimizer.
    
    Returns
    ----------
    is_instance :                       A flag indicating whether the given input is an optimizer instance.
    '''
    return isinstance(optimizer, optimizers.Optimizer) if tf.executing_eagerly() else isinstance(optimizer, optimizers.legacy.Optimizer)


class SequentialKerasNetwork(AbstractNetwork):
    
    def __init__(self, model, optimizer: None | str | optimizers.Optimizer = None, loss: None | str | losses.Loss = None,
                 optimizer_params: None | dict = None, loss_params: None | dict = None):
        '''
        This class provides an interface to Sequential Keras models. 
        
        Parameters
        ----------
        model :                             The network model.
        optimizer :                         The name of the optimizer.
        loss :                              The name of the loss.
        optimizer_params :                  The parameters of the optimizer (e.g., learning rate).
        loss_params :                       The parameters of the loss.
        
        Returns
        ----------
        None
        '''
        super().__init__(model)
        self.set_optimizer(optimizer, optimizer_params)
        self.set_loss(loss, loss_params)
    
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
        model_clone.compile(optimizer=type(self.model.optimizer).from_config(self.model.optimizer.get_config()), loss=copy.deepcopy(self.model.loss),
                            loss_weights=self.model.loss_weights if hasattr(self.model, 'loss_weights') else self.model.get_compile_config()['loss_weights'])
        
        return type(self)(model_clone, type(self.model.optimizer).from_config(self.model.optimizer.get_config()), loss=copy.deepcopy(self.model.loss))
    
    def set_optimizer(self, optimizer: None | str | optimizers.Optimizer, parameters: None | dict = None):
        '''
        This function sets the optimizer of the network model.
        
        Parameters
        ----------
        optimizer :                         The optimizer or name of the optimizer.
        parameters :                        The parameters of the optimizer (e.g., learning rate).
        
        Returns
        ----------
        None
        '''
        valid_optimizers = get_optimizers()
        # prepare optimizer
        if not is_optimizer_instance(optimizer):
            # use Adam by default
            if not optimizer in valid_optimizers:
                optimizer = 'Adam'
            optimizer = valid_optimizers[optimizer](**({} if parameters is None else parameters))
        elif parameters:
            optimizer = type(optimizer)(**({} if parameters is None else parameters))
        # recompile model
        self.model.compile(optimizer=optimizer, loss=self.model.loss,
                           loss_weights=self.model.loss_weights if hasattr(self.model, 'loss_weights') else self.model.get_compile_config()['loss_weights'])
        
    def set_loss(self, loss: None | str | losses.Loss, parameters: None | dict = None):
        '''
        This function sets the loss of the network model.
        
        Parameters
        ----------
        loss :                              The loss function or name of the loss.
        parameters :                        The parameters of the loss.
        
        Returns
        ----------
        None
        '''
        valid_losses = get_losses()
        # prepare loss
        if not isinstance(loss, losses.Loss):
            # use mean squared error by default
            if not loss in valid_losses:
                loss = 'mse'
            loss = valid_losses[loss](**({} if parameters is None else parameters))
        elif parameters:
            loss = type(loss)(**({} if parameters is None else parameters))
        # recompile model
        self.model.compile(optimizer=type(self.model.optimizer).from_config(self.model.optimizer.get_config()), loss=loss,
                           loss_weights=self.model.loss_weights if hasattr(self.model, 'loss_weights') else self.model.get_compile_config()['loss_weights'])
    
    def get_layer_activity(self, batch: np.ndarray, layer: int | str) -> np.ndarray:
        '''
        This function returns the activity of a specified layer for a batch of input samples.
        
        Parameters
        ----------
        batch :                             The batch of input samples.
        layer :                             The index or name of the layer from which activity should be retrieved.
        
        Returns
        ----------
        activity :                          The layer activities of the specified layer for the batch of input samples.
        '''
        # construct "sub model"
        def get_layer(model, key):
            return model.get_layer(index=key) if type(key) is int else model.get_layer(name=key)
        sub_model = K.function(self.model.layers[0].input, get_layer(self.model, layer).output)
        # compute activity maps
        activity_maps = sub_model(batch).T
            
        return activity_maps
    
    
class FunctionalKerasNetwork(SequentialKerasNetwork):
    
    def __init__(self, model, optimizer: None | str | optimizers.Optimizer = None, loss: None | str | list | dict | losses.Loss = None,
                 optimizer_params: None | dict = None, loss_params: None | dict = None, loss_weights: None | list | dict = None):
        '''
        This class provides an interface to Functional Keras models. 
        
        Parameters
        ----------
        model :                             The network model.
        optimizer :                         The name of the optimizer.
        loss :                              The name of the loss.
        optimizer_params :                  The parameters of the optimizer (e.g., learning rate).
        loss_params :                       The parameters of the loss.
        
        Returns
        ----------
        None
        '''
        super().__init__(model, optimizer, loss, optimizer_params, loss_params)
        if not loss_weights is None:
            self.set_loss(loss, loss_params, loss_weights)
    
    def predict_on_batch(self, batch: np.ndarray | list | dict) -> np.ndarray | list:
        '''
        This function computes network predictions for a batch of input samples.
        
        Parameters
        ----------
        batch :                             The batch of input samples.
        
        Returns
        ----------
        predictions :                       A batch of network predictions.
        '''
        return self.model.predict_on_batch(list(batch) if type(batch) is np.ndarray and batch.dtype == object else batch)
    
    def train_on_batch(self, batch: np.ndarray | list | dict, targets: np.ndarray | list | dict):
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
        self.model.train_on_batch(list(batch) if type(batch) is np.ndarray and batch.dtype == object else batch,
                                  list(targets) if type(targets) is np.ndarray and targets.dtype == object else targets)
        
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
        loss_weights = self.model.loss_weights if hasattr(self.model, 'loss_weights') else self.model.get_compile_config()['loss_weights']
        # compile model
        model_clone.compile(optimizer=type(self.model.optimizer).from_config(self.model.optimizer.get_config()),
                            loss=copy.deepcopy(self.model.loss), loss_weights=loss_weights)
        
        return type(self)(model_clone, model_clone.optimizer, loss=model_clone.loss)
    
    def set_loss(self, loss: str | list | dict | losses.Loss, parameters: None | list | dict = None, loss_weights: None | list | dict = None):
        '''
        This function sets the loss(es) of the network model.
        
        Parameters
        ----------
        loss :                              The loss function(s) or name(s) of the loss(es).
        parameters :                        The parameters of the loss(es).
        loss_weights :                      The weightings for different losses (optional).
        
        Returns
        ----------
        None
        '''
        valid_losses = get_losses()
        params = {} if parameters is None else copy.deepcopy(parameters)
        # use inherited method for simple loss
        if type(loss) is str or isinstance(loss, losses.Loss) or loss is None:
            super().set_loss(loss, parameters)
        else:
            # prepare loss functions
            loss_list = []
            loss, outputs = list(loss.values()) if type(loss) is dict else loss, list(loss) if type(loss) is dict else []
            params = list(params.values()) if type(params) is dict else params
            if len(params) == 0:
                params = [{} for i in range(len(loss))]
            for i, L in enumerate(loss):
                if isinstance(L, losses.Loss):
                    if params[i]:
                        loss_list.append(type(L)(**params[i]))
                    else:
                        loss_list.append(L)
                elif L in valid_losses:
                    loss_list.append(valid_losses[L](**params[i]))
                else: # use mean squared error by default
                    loss_list.append(valid_losses['mse']())
            # turn list into dict if outputs were specified
            loss_list = {outputs[i]: loss_list[i] for i in range(len(loss_list))} if len(outputs) != 0 else loss_list
            # recompile model
            self.model.compile(optimizer=type(self.model.optimizer).from_config(self.model.optimizer.get_config()), loss=loss_list, loss_weights=loss_weights)
    
    def get_layer_activity(self, batch: np.ndarray | list | dict, layer: int | str) -> np.ndarray:
        '''
        This function returns the activity of a specified layer for a batch of input samples.
        
        Parameters
        ----------
        batch :                             The batch of input samples.
        layer :                             The index or name of the layer from which activity should be retrieved.
        
        Returns
        ----------
        activity :                          The layer activities of the specified layer for the batch of input samples.
        '''
        # use inherited method for simple unimodal input
        if type(batch) is np.ndarray and not batch.dtype == object:
            return super().get_layer_activity(batch, layer)
        # construct "sub model"
        def get_layer(model, key):
            return model.get_layer(index=key) if type(key) is int else model.get_layer(name=key)
        sub_model = K.function([get_layer(self.model, key if type(key) in [int, str] else i).input for i, key in enumerate(batch)],
                               [get_layer(self.model, layer).output])
        # compute activity maps
        activity_maps = sub_model(batch)[0].T
            
        return activity_maps
        