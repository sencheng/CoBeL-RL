# basic imports
import numpy as np
import copy
# torch imports
import torch
import torch.optim as optimizers
# framework
from cobel.networks.network import AbstractNetwork


class TorchNetwork(AbstractNetwork):
    
    def __init__(self, model, optimizer: None | str | optimizers.Optimizer = None, loss: None | str | torch.nn.modules.loss._Loss = None,
                 optimizer_params: None | dict = None, loss_params: None | dict = None, activations: None | list | dict = None, device: str = 'cpu'):
        '''
        This class provides an interface to Torch models. 
        
        Parameters
        ----------
        model :                             The network model.
        optimizer :                         The name of the optimizer.
        loss :                              The name of the loss.
        optimizer_params :                  The parameters of the optimizer (e.g., learning rate).
        loss_params :                       The parameters of the loss.
        activations :                       A list of layer activation functions. Required for retrieving layer activity.
        device :                            The name of the device that the model will stored on (\'cpu\' by default).
        
        Returns
        ----------
        None
        '''
        super().__init__(model)
        # move model to specified device
        self.set_device(device)
        # initialize optimizer and loss function
        self.set_optimizer(optimizer, optimizer_params)
        self.set_loss(loss, loss_params)
        # list of layer activation functions
        self.activations = {} if activations is None else activations
    
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
            return self.model(torch.tensor(batch, device=self.device)).detach().cpu().numpy()
    
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
        predictions = self.model(torch.tensor(batch, device=self.device))
        loss = self.criterion(predictions, torch.tensor(targets.reshape((targets.shape[0], 1)) if len(targets.shape) == 1 else targets, device=self.device))
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
            weights[i] = weights[i].cpu().numpy()
            
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
            new_state_dict[param] = torch.tensor(weights[i], device=self.device)
        # load the state_dict
        self.model.load_state_dict(new_state_dict)
        
    def clone_model(self) -> AbstractNetwork:
        '''
        This function returns a copy of the network.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        model :                             The network model's copy.
        '''
        network = copy.deepcopy(self.model)
        optimizer = type(self.optimizer)(params=network.parameters())
        model_clone = TorchNetwork(network, optimizer, copy.deepcopy(self.criterion), activations=copy.deepcopy(self.activations))
        model_clone.criterion.load_state_dict(self.criterion.state_dict())
        model_clone.optimizer.load_state_dict(self.optimizer.state_dict())
        
        return model_clone
    
    def set_optimizer(self, optimizer: None | str | optimizers.Optimizer, parameters: None | dict = None):
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
        valid_optimizers = {**dict.fromkeys(['Adadelta', 'adadelta'], optimizers.Adadelta),
                            **dict.fromkeys(['Adagrad', 'adagrad'], optimizers.Adagrad),
                            **dict.fromkeys(['Adam', 'adam'], optimizers.Adam),
                            **dict.fromkeys(['AdamW', 'Adamw', 'adamw'], optimizers.AdamW),
                            **dict.fromkeys(['Adamax', 'AdaMax', 'adamax'], optimizers.Adamax),
                            **dict.fromkeys(['SparseAdam', 'Sparseadam', 'sparseadam'], optimizers.SparseAdam),
                            **dict.fromkeys(['ASGD', 'asgd'], optimizers.ASGD),
                            **dict.fromkeys(['LBFGS', 'lbfgs'], optimizers.LBFGS),
                            **dict.fromkeys(['NAdam', 'Nadam', 'nadam'], optimizers.NAdam),
                            **dict.fromkeys(['RAdam', 'Radam', 'radam'], optimizers.RAdam),
                            **dict.fromkeys(['RMSprop', 'rmsprop'], optimizers.RMSprop),
                            **dict.fromkeys(['Rprop', 'rprop'], optimizers.Rprop),
                            **dict.fromkeys(['SGD', 'sgd'], optimizers.SGD)}
        self.optimizer = optimizer
        if not isinstance(self.optimizer, optimizers.Optimizer):
            # use Adam by default
            if optimizer not in valid_optimizers:
                optimizer = 'Adam'
            self.optimizer = valid_optimizers[optimizer](**({'params': self.model.parameters()} if parameters is None else {'params': self.model.parameters()} | parameters))
        elif parameters:
            self.optimizer = type(self.optimizer)(**({'params': self.model.parameters()} if parameters is None else {'params': self.model.parameters()} | parameters))
        
    def set_loss(self, loss: None | str | torch.nn.modules.loss._Loss, parameters: None | dict = None):
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
        valid_losses = {'kl_divergence': torch.nn.KLDivLoss, 'cosine_similarity': torch.nn.CosineEmbeddingLoss,
                        'poisson': torch.nn.PoissonNLLLoss, 'gaussian': torch.nn.GaussianNLLLoss,
                        'binary_crossentropy': torch.nn.BCELoss, 'hinge': torch.nn.HingeEmbeddingLoss,
                        'margin_ranking': torch.nn.MarginRankingLoss, 'multi_label_margin_ranking': torch.nn.MultiLabelMarginLoss,
                        **dict.fromkeys(['mean_absolute_error', 'mae'], torch.nn.L1Loss),
                        **dict.fromkeys(['mean_squared_error', 'mse'], torch.nn.MSELoss),
                        **dict.fromkeys(['huber_loss', 'huber'], torch.nn.HuberLoss),
                        **dict.fromkeys(['categorical_crossentropy', 'crossentropy'], torch.nn.CrossEntropyLoss),
                        **dict.fromkeys(['connectionist_temporal_classification', 'CTC', 'ctc'], torch.nn.CTCLoss),
                        **dict.fromkeys(['negative_log_likelihood', 'NLL', 'nll'], torch.nn.NLLLoss)}
        self.criterion = loss
        if not isinstance(self.criterion, torch.nn.modules.loss._Loss):
            # use mean squared error by default
            if loss not in valid_losses:
                loss = 'mse'
            self.criterion = valid_losses[loss](**({} if parameters is None else parameters))
        elif parameters:
            self.criterion = type(self.criterion)(**parameters)
        
    def get_layer_activity(self, batch: np.ndarray | list | dict, layer: int | str) -> np.ndarray:
        '''
        This function returns the activity of a specified layer for a batch of input samples.
        
        Parameters
        ----------
        batch :                             The batch of input samples.
        layer_index :                       The index or name of the layer from which activity should be retrieved.
        
        Returns
        ----------
        activity :                          The layer activities of the specified layer for the batch of input samples.
        '''
        # define hook
        def get_activation(activation, layer):
            def hook(model, input, output):
                activation[layer] = output.detach()
            return hook
        # register forward hook and record activity
        activation = {}
        layer = list(self.model.named_children())[layer][0] if type(layer) is int else layer
        sub_module = self.model.get_submodule(layer)
        sub_module.register_forward_hook(get_activation(activation, layer))
        self.predict_on_batch(batch)
        act = activation[layer]
        # apply activation function if available
        if type(layer) is str and layer in self.activations or type(layer) is int and len(self.activations) >= layer:
            if not self.activations[layer] is None: act = self.activations[layer](act)
        
        return act.cpu().numpy().T
    
    def set_device(self, device: str = 'cpu'):
        '''
        This function moves the model to a specified device. 
        
        Parameters
        ----------
        device :                            The name of the device that the model will stored on (\'cpu\' by default).
        
        Returns
        ----------
        None
        '''
        self.device = torch.device(device)
        self.model.to(self.device)
        
        
class FlexibleTorchNetwork(TorchNetwork):
    
    def __init__(self, model, optimizer: str | optimizers.Optimizer = 'adam', loss: str | list | dict | torch.nn.modules.loss._Loss = 'mse',
                 optimizer_params:  dict = {}, loss_params: None | list | dict = None, loss_weights: None | list | dict = None,
                 activations: None | list | dict = None, device: str = 'cpu'):
        '''
        This class provides an interface to Torch models. 
        
        Parameters
        ----------
        model :                             The network model.
        optimizer :                         The name of the optimizer.
        loss :                              The name of the loss(es).
        optimizer_params :                  The parameters of the optimizer(s) (e.g., learning rate).
        loss_params :                       The parameters of the loss(es).
        loss_weights :                      The weighting for each loss.
        activations :                       A list of layer activation functions. Required for retrieving layer activity.
        device :                            The name of the device that the model will stored on (\'cpu\' by default).
        
        Returns
        ----------
        None
        '''
        super().__init__(model, optimizer, loss, optimizer_params, loss_params, activations, device)
        self.loss_weights = loss_weights
        if not self.loss_weights is None:
            assert len(loss_weights) == len(self.criteria)
        else:
            self.loss_weights = [1 for c in self.criteria] if type(self.criteria) is list else {c: 1 for c in self.criteria}
        
    def prepare_batch(self, batch: np.ndarray | list | dict) -> list | dict | torch.Tensor:
        '''
        This function prepares a batch of input samples.
        
        Parameters
        ----------
        batch :                             The batch of input samples.
        
        Returns
        ----------
        batch :                             The batch of input samples transformed to torch tensors.
        '''
        if type(batch) is np.ndarray and batch.dtype != object:
            batch = [torch.tensor(batch.reshape((batch.shape[0], 1)) if len(batch.shape) == 1 else batch, device=self.device)]
        else:
            for i, k in enumerate(batch):
                idx = k if type(k) is str else i
                batch[idx] = torch.tensor(batch[idx].reshape((batch[idx].shape[0], 1)) if len(batch[idx].shape) == 1 else batch[idx], device=self.device)
            
        return batch
    
    def predict_on_batch(self, batch: np.ndarray | list | dict) -> np.ndarray:
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
            batch = self.prepare_batch(batch)
            predictions = self.model(**batch) if type(batch) is dict else self.model(*batch)
            if type(predictions) is torch.Tensor:
                return predictions.detach().cpu().numpy()
            elif type(predictions) is tuple:
                return [prediction.detach().cpu().numpy() for prediction in predictions]
            else:
                return {prediction: predictions[prediction].detach().cpu().numpy() for prediction in predictions}
    
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
        self.optimizer.zero_grad()
        # prepare training batch and perform forward pass
        batch, targets = self.prepare_batch(batch), self.prepare_batch(targets)
        predictions = self.model(**batch) if type(batch) is dict else self.model(*batch)
        if type(predictions) is torch.Tensor: predictions = [predictions]
        elif type(predictions) is tuple: predictions = list(predictions)
        # perform backward pass for each loss function
        loss = []
        for i, criterion in enumerate(self.criteria):
            idx_c = idx_p = idx_t = criterion if type(criterion) is str else i
            if type(self.criteria) != type(predictions): idx_p = i if type(idx_c) is str else list(predictions)[idx_c]
            if type(self.criteria) != type(targets): idx_t = i if type(idx_c) is str else list(targets)[idx_c] 
            loss.append(self.criteria[idx_c](predictions[idx_p], targets[idx_t]) * self.loss_weights[idx_c])
        sum(loss).backward()
        self.optimizer.step()
        
    def clone_model(self) -> TorchNetwork:
        '''
        This function returns a copy of the network.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        model :                             The network model's copy.
        '''
        model_clone = FlexibleTorchNetwork(copy.deepcopy(self.model), type(self.optimizer), copy.deepcopy(self.criteria),
                                           loss_weights=copy.deepcopy(self.loss_weights), activations=copy.deepcopy(self.activations))
        for i, criterion in enumerate(self.criteria):
            idx = criterion if type(criterion) is str else i
            model_clone.criteria[idx].load_state_dict(self.criteria[idx].state_dict())
        model_clone.optimizer.load_state_dict(self.optimizer.state_dict())
        
        return model_clone
        
    def set_loss(self, loss: str | list | dict | torch.nn.modules.loss._Loss, parameters: None | list | dict = None, loss_weights: None | list | dict = None):
        '''
        This function sets the loss of the network model.
        
        Parameters
        ----------
        loss :                              The loss function(s) or name(s) of the loss(es).
        parameters :                        The parameters of the loss(es).
        loss_weights :                      The weightings for different losses (optional).
        
        Returns
        ----------
        None
        '''
        if type(loss) in [list, dict] and not parameters is None:
            assert type(loss) == type(parameters), 'Loss functions and their parameters should be provided in the same data type!'
        if not loss_weights is None:
            assert type(loss) == type(loss_weights), 'Loss functions and their weightings should be provided using in same data type!'
        valid_losses = {'kl_divergence': torch.nn.KLDivLoss, 'cosine_similarity': torch.nn.CosineEmbeddingLoss,
                        'poisson': torch.nn.PoissonNLLLoss, 'gaussian': torch.nn.GaussianNLLLoss,
                        'binary_crossentropy': torch.nn.BCELoss, 'hinge': torch.nn.HingeEmbeddingLoss,
                        'margin_ranking': torch.nn.MarginRankingLoss, 'multi_label_margin_ranking': torch.nn.MultiLabelMarginLoss,
                        **dict.fromkeys(['mean_absolute_error', 'mae'], torch.nn.L1Loss),
                        **dict.fromkeys(['mean_squared_error', 'mse'], torch.nn.MSELoss),
                        **dict.fromkeys(['huber_loss', 'huber'], torch.nn.HuberLoss),
                        **dict.fromkeys(['categorical_crossentropy', 'crossentropy'], torch.nn.CrossEntropyLoss),
                        **dict.fromkeys(['connectionist_temporal_classification', 'CTC', 'ctc'], torch.nn.CTCLoss),
                        **dict.fromkeys(['negative_log_likelihood', 'NLL', 'nll'], torch.nn.NLLLoss)}
        self.criteria = [loss] if type(loss) not in [list, dict] else loss
        self.loss_weights = loss_weights
        if self.loss_weights is None:
            self.loss_weights = [1 for c in self.criteria] if type(self.criteria) is list else {c: 1 for c in self.criteria}
        if parameters is None:
            parameters = [{} for c in self.criteria] if type(self.criteria) is list else {c: {} for c in self.criteria}
        for i, criterion in enumerate(self.criteria):
            idx = criterion if type(self.criteria) is dict else i
            if not isinstance(self.criteria[idx], torch.nn.modules.loss._Loss):
                # use mean squared error by default
                if self.criteria[idx] not in valid_losses:
                    self.criteria[idx], parameters[idx] = 'mse', {}
                self.criteria[idx] = valid_losses[self.criteria[idx]](**parameters[idx])            
            elif parameters[idx]:
                self.criteria[idx] = type(self.criteria[idx])(**parameters[idx])
    