# basic imports
import numpy as np
import copy
# torch imports
import torch
import torch.optim as optimizers
# framework
from .network import Network
# typing
from numpy.typing import NDArray
from typing import Self
from .network import Batch, ParamDict

Loss = str | torch.nn.modules.loss._Loss
FlexibleLoss = (
    list[Loss]
    | dict[str, Loss]
    | Loss
    | None
    | list[torch.nn.modules.loss._Loss]
    | dict[str, torch.nn.modules.loss._Loss]
)
FlexibleLossParameters = (
    None | list[ParamDict | None] | dict[str, ParamDict | None] | ParamDict
)


class TorchNetwork(Network):
    """
    This class provides an interface to Torch models.

    Parameters
    ----------
    model : torch.nn.Module
        The network model.
    optimizer : str, optimizers.Optimizer or None, optional
        The optimizer that should be used.
        Defaults to Adam if none was provided.
    loss : str, torch.nn.modules.loss._Loss or None, optional
        The loss that should be used.
        Defaults to MSE if none was provided.
    optimizer_params : dict or None, optional
        The parameters of the optimizer (e.g., learning rate).
    loss_params : dict or None, optional
        The parameters of the loss.
    activations : list, dict or None, optional
        A list/dict of layer activation functions.
        Required for retrieving layer activity.
        Defaults to an empty list/dict if none was provided.
    device : str, default='cpu'
        The name of the device that the model will be
        stored on ('cpu' by default)

    Attributes
    ----------
    model : torch.nn.Module
        The network model.
    device : torch.device
        The device that the model is stored on.
    optimizer : optimizers.Optimizer
        The optimizer that is being used.
    loss : torch.nn.modules.loss._Loss
        The loss that is being used.
    activations : list or dict
        A list/dict of layer activation functions.
        Required for retrieving layer activity.

    Examples
    --------

    Simpy define a PyTorch network which you want to use
    with classes like the DQN agent. ::

        >>> from collections import OrderedDict
        >>> from cobel.network import TorchNetwork
        >>> import torch.nn as nn
        >>> layers = [('dense', nn.Linear(64, 64)),
        ...           ('relu', nn.ReLU()),
        ...           ('out', nn.Linear(64, 4))]
        >>> network = nn.Sequential(OrderedDict(layers))
        >>> model = TorchNetwork(network)

    The network model can be easily moved to the GPU. ::

        >>> model.set_device('cuda')

    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: None | str | optimizers.Optimizer = None,
        loss: None | str | torch.nn.modules.loss._Loss = None,
        optimizer_params: None | dict = None,
        loss_params: None | dict = None,
        activations: None | list | dict = None,
        device: str = 'cpu',
    ) -> None:
        self.model = model
        # move model to specified device
        self.set_device(device)
        # initialize optimizer and loss function
        self.set_optimizer(optimizer, optimizer_params)
        self.set_loss(loss, loss_params)
        # list of layer activation functions
        self.activations = {} if activations is None else activations

    def predict_on_batch(self, batch: Batch) -> NDArray:
        """
        This function computes network predictions for a batch of input samples.

        Parameters
        ----------
        batch : Batch
            The batch of input samples.

        Returns
        -------
        predictions : NDArray
            A batch of network predictions.
        """
        with torch.inference_mode():
            return (
                self.model(torch.tensor(batch, device=self.device))
                .detach()
                .cpu()
                .numpy()
            )

    def train_on_batch(
        self, batch: Batch, targets: Batch, sample_weights: None | NDArray = None
    ) -> None:
        """
        This function trains the network on a batch of input samples.

        Parameters
        ----------
        batch : Batch
            The batch of input samples.
        targets : Batch
            The batch of target values.
        sample_weights : NDArray or None, optional
            An optional batch of sample weights.
        """
        assert type(targets) is np.ndarray, ''
        self.optimizer.zero_grad()
        predictions = self.model(torch.tensor(batch, device=self.device))
        loss = self.criterion(
            predictions,
            torch.tensor(
                targets.reshape((targets.shape[0], 1))
                if len(targets.shape) == 1
                else targets,
                device=self.device,
            ),
        )
        if sample_weights is not None:
            sample_weights = (
                sample_weights
                if len(sample_weights.shape) == 2
                else sample_weights.reshape(sample_weights.size, 1)
            )
            loss *= torch.tensor(sample_weights, device=self.device)
        loss.mean().backward()
        self.optimizer.step()

    def get_weights(self) -> list[NDArray]:
        """
        This function returns the weights of the network.

        Returns
        -------
        weights : list of NDArray
            A list of layer weights.
        """
        # retrieve params from state_dict and save them as a list of numpy arrays
        weights = list(self.model.state_dict().values())
        for i in range(len(weights)):
            weights[i] = weights[i].cpu().numpy()

        return weights

    def set_weights(self, weights: list[NDArray]) -> None:
        """
        This function sets the weights of the network.

        Parameters
        ----------
        weights : list of NDArray
            A list of layer weights.
        """
        # prepare a new state_dict
        new_state_dict = self.model.state_dict()
        for i, param in enumerate(new_state_dict):
            new_state_dict[param] = torch.tensor(weights[i], device=self.device)
        # load the state_dict
        self.model.load_state_dict(new_state_dict)

    def clone(self) -> Self:
        """
        This function returns a copy of the network.

        Returns
        -------
        model : Self
            The network model's copy.
        """
        network = copy.deepcopy(self.model)
        optimizer = type(self.optimizer)(params=network.parameters())  # type: ignore
        model_clone = type(self)(
            network,
            optimizer,
            copy.deepcopy(self.criterion),
            activations=copy.deepcopy(self.activations),
        )
        model_clone.criterion.load_state_dict(self.criterion.state_dict())
        model_clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return model_clone

    def set_optimizer(
        self,
        optimizer: None | str | optimizers.Optimizer,
        parameters: None | ParamDict = None,
    ) -> None:
        """
        This function sets the optimizer of the network model.

        Parameters
        ----------
        optimizer : str, optimizers.Optimizer or None
            The optimizer that should be used.
        parameters : ParamDict or None, optional
            The parameters of the optimizer (e.g., learning rate).
        """
        valid_optimizers = {
            **dict.fromkeys(['Adadelta', 'adadelta'], optimizers.Adadelta),
            **dict.fromkeys(['Adagrad', 'adagrad'], optimizers.Adagrad),
            **dict.fromkeys(['Adam', 'adam'], optimizers.Adam),
            **dict.fromkeys(['AdamW', 'Adamw', 'adamw'], optimizers.AdamW),
            **dict.fromkeys(['Adamax', 'AdaMax', 'adamax'], optimizers.Adamax),
            **dict.fromkeys(
                ['SparseAdam', 'Sparseadam', 'sparseadam'], optimizers.SparseAdam
            ),
            **dict.fromkeys(['ASGD', 'asgd'], optimizers.ASGD),
            **dict.fromkeys(['LBFGS', 'lbfgs'], optimizers.LBFGS),
            **dict.fromkeys(['NAdam', 'Nadam', 'nadam'], optimizers.NAdam),
            **dict.fromkeys(['RAdam', 'Radam', 'radam'], optimizers.RAdam),
            **dict.fromkeys(['RMSprop', 'rmsprop'], optimizers.RMSprop),
            **dict.fromkeys(['Rprop', 'rprop'], optimizers.Rprop),
            **dict.fromkeys(['SGD', 'sgd'], optimizers.SGD),
        }
        self.optimizer: optimizers.Optimizer
        if not isinstance(optimizer, optimizers.Optimizer):
            # use Adam by default
            if optimizer not in valid_optimizers:
                optimizer = 'Adam'
            self.optimizer = valid_optimizers[optimizer](
                **(
                    {'params': self.model.parameters()}
                    if parameters is None
                    else {'params': self.model.parameters()} | parameters
                )
            )
        else:
            self.optimizer = optimizer
            if type(parameters) is dict:
                self.optimizer = type(self.optimizer)(
                    **(
                        {'params': self.model.parameters()}
                        if parameters is None
                        else {'params': self.model.parameters()} | parameters  # type: ignore
                    )
                )

    def set_loss(
        self,
        loss: None | str | torch.nn.modules.loss._Loss,
        parameters: None | ParamDict = None,
    ) -> None:
        """
        This function sets the loss of the network model.

        Parameters
        ----------
        loss : str or torch.nn.modules.loss._Loss or None
            The loss that should be used.
        parameters : ParamDict or None, optional
            The parameters of the loss.
        """
        valid_losses = {
            'kl_divergence': torch.nn.KLDivLoss,
            'cosine_similarity': torch.nn.CosineEmbeddingLoss,
            'poisson': torch.nn.PoissonNLLLoss,
            'gaussian': torch.nn.GaussianNLLLoss,
            'binary_crossentropy': torch.nn.BCELoss,
            'hinge': torch.nn.HingeEmbeddingLoss,
            'margin_ranking': torch.nn.MarginRankingLoss,
            'multi_label_margin_ranking': torch.nn.MultiLabelMarginLoss,
            **dict.fromkeys(['mean_absolute_error', 'mae'], torch.nn.L1Loss),
            **dict.fromkeys(['mean_squared_error', 'mse'], torch.nn.MSELoss),
            **dict.fromkeys(['huber_loss', 'huber'], torch.nn.HuberLoss),
            **dict.fromkeys(
                ['categorical_crossentropy', 'crossentropy'], torch.nn.CrossEntropyLoss
            ),
            **dict.fromkeys(
                ['connectionist_temporal_classification', 'CTC', 'ctc'],
                torch.nn.CTCLoss,
            ),
            **dict.fromkeys(
                ['negative_log_likelihood', 'NLL', 'nll'], torch.nn.NLLLoss
            ),
        }
        self.criterion: torch.nn.modules.loss._Loss
        if not isinstance(loss, torch.nn.modules.loss._Loss):
            # use mean squared error by default
            if loss not in valid_losses:
                loss = 'mse'
            self.criterion = valid_losses[loss](
                **({} if parameters is None else parameters)
            )
        else:
            self.criterion = loss
            if parameters:
                self.criterion = type(self.criterion)(**parameters)
        # overwrite reduction so that we can weigh samples when updating the network
        self.criterion.reduction = 'none'

    def get_layer_activity(self, batch: Batch, layer: int | str) -> NDArray:
        """
        This function returns the activity of a specified layer
        for a batch of input samples.

        Parameters
        ----------
        batch : Batch
            The batch of input samples.
        layer_index : int or str
            The index or name of the layer from which
            activity should be retrieved.

        Returns
        -------
        activity : NDArray
            The layer activities of the specified layer
            for the batch of input samples.
        """

        # define hook
        def get_activation(activation, layer):
            def hook(model, input, output):
                activation[layer] = output.detach()

            return hook

        # register forward hook and record activity
        activation: dict[int | str, torch.Tensor] = {}
        layer_key = (
            layer
            if type(layer) is str
            else list(self.model.named_children())[int(layer)][0]
        )
        layer_index = layer if type(layer) is int else 10**10
        if type(layer) is str:
            for i, (name, _) in enumerate(self.model.named_children()):
                if name == layer:
                    layer_index = i
                    break
        sub_module = self.model.get_submodule(layer_key)
        sub_module.register_forward_hook(get_activation(activation, layer_key))
        self.predict_on_batch(batch)
        act: torch.Tensor = activation[layer_key]
        # apply activation function if available
        if type(self.activations) is list:
            assert len(self.activations) >= layer_index, 'Index not in activation list!'
            if self.activations[layer_index] is not None:
                act = self.activations[layer_index](act)
        else:
            assert type(self.activations) is dict, (
                'String layerkeys not compatible with activation lists!'
            )
            assert layer_key in self.activations, 'Key not in activation dict!'
            if self.activations[layer_key] is not None:
                act = self.activations[layer_key](act)

        return act.cpu().numpy().T

    def set_trainable(
        self,
        layers: list[int] | list[str],
        trainable: bool | list[bool] | dict[str, bool],
    ) -> None:
        """
        This function sets the trainability of specified network layers.

        Parameters
        ----------
        layers : list of int or list of str
            A list of layer indeces or layer names
            for which trainability will be set.
        trainable : bool, list of bool or dict of bool
            The trainability that will be set. Use a bool to
            set all layers with the same value, and use a list
            or dict to specify values for each layer.
        """
        weights: list[str] = []
        for param in self.model.state_dict():
            if '.weight' in param:
                weights.append(param.split('.')[0])
        requires_grad: bool
        if type(trainable) is bool:
            requires_grad = trainable
        if type(layers[0]) is int:
            for l, layer_idx in enumerate(layers):
                assert type(layer_idx) is int
                assert layer_idx < len(weights)
                if type(trainable) is list:
                    assert len(layers) == len(trainable)
                    requires_grad = trainable[l]
                self.model.get_parameter(
                    weights[layer_idx] + '.weight'
                ).requires_grad = requires_grad
                self.model.get_parameter(
                    weights[layer_idx] + '.bias'
                ).requires_grad = requires_grad
        else:
            for layer in layers:
                assert type(layer) is str
                assert layer in weights
                if type(trainable) is dict:
                    assert layer in trainable
                    requires_grad = trainable[layer]
                self.model.get_parameter(
                    layer + '.weight'
                ).requires_grad = requires_grad
                self.model.get_parameter(layer + '.bias').requires_grad = requires_grad

    def set_device(self, device: str = 'cpu') -> None:
        """
        This function moves the model to a specified device.

        Parameters
        ----------
        device : str
            The name of the device that the model
            will stored on ('cpu' by default).
        """
        self.device = torch.device(device)
        self.model.to(self.device)


class FlexibleTorchNetwork(Network):
    """
    This class provides an interface to Torch models.

    Parameters
    ----------
    model : torch.nn.Module
        The network model.
    optimizer : str, optimizers.Optimizer or None, optional
        The optimizer that should be used.
        Defaults to Adam if none was provided.
    loss : str, torch.nn.modules.loss._Loss, list, dict or None, optional
        The loss(es) that should be used.
        Defaults to MSE if none was provided.
    optimizer_params : dict or None, optional
        The parameters of the optimizer (e.g., learning rate).
    loss_params : ParamDict, list, dict or None, optional
        The parameters of the loss(es).
    loss_weights : list of float, dict of float or None, optional
        Optional weightings applied to different loss functions.
        Defaults to equal weightings when none were provided.
    activations : list, dict or None, optional
        A list/dict of layer activation functions.
        Required for retrieving layer activity.
        Defaults to an empty list/dict if none was provided.
    device : str, default='cpu'
        The name of the device that the model will be
        stored on ('cpu' by default)

    Attributes
    ----------
    model : torch.nn.Module
        The network model.
    device : torch.device
        The device that the model is stored on.
    optimizer : optimizers.Optimizer
        The optimizer that is being used.
    loss : torch.nn.modules.loss._Loss
        The loss that is being used.
    activations : list or dict
        A list/dict of layer activation functions.
        Required for retrieving layer activity.
    loss_weights : list of float or dict of float
        Weightings applied to different loss functions.
        Defaults to equal weightings when none were provided.

    Examples
    --------

    This class is compatible with networks that have multiple
    input and output streams. ::

        >>> from cobel.network import FlexibleTorchNetwork
        >>> import torch
        >>> import torch.nn as nn
        >>> Model(nn.Module):
        ...     def __init__(self):
        ...         super().__init__():
        ...         self.stream_1 = nn.Linear(64, 64)
        ...         self.stream_2 = nn.Linear(64, 64)
        ...         self.hidden = nn.Linear(128, 64)
        ...         self.output_1 = nn.Linear(64, 64)
        ...         self.output_2 = nn.Linear(64, 64)
        ...     def forward(self, input_1, input_2):
        ...         x_1 = self.stream(input_1)
        ...         x_2 = self.stream(input_2)
        ...         x = torch.cat((x_1, x_2), 1)
        ...         x = nn.functional.relu(x)
        ...         x = self.hidden(x)
        ...         return self.output_1(x), self.output_2(x)
        >>> model = FlexibleTorchNetwork(Model())

    This class also accepts simple models. ::

        >>> from collections import OrderedDict
        >>> from cobel.network import FlexibleTorchNetwork
        >>> import torch.nn as nn
        >>> layers = [('dense', nn.Linear(64, 64)),
        ...           ('relu', nn.ReLU()),
        ...           ('out', nn.Linear(64, 4))]
        >>> network = nn.Sequential(OrderedDict(layers))
        >>> model = FlexibleTorchNetwork(network)

    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: None | str | optimizers.Optimizer = None,
        loss: FlexibleLoss = None,
        optimizer_params: None | ParamDict = None,
        loss_params: FlexibleLossParameters = None,
        loss_weights: None | list[float] | dict[str, float] = None,
        activations: None | list | dict = None,
        device: str = 'cpu',
    ) -> None:
        self.model = model
        # move model to specified device
        self.set_device(device)
        # initialize optimizer and loss function
        self.set_optimizer(optimizer, optimizer_params)
        self.set_loss(loss, loss_params, loss_weights)
        # list of layer activation functions
        self.activations = {} if activations is None else activations

    def prepare_batch(
        self, batch: Batch
    ) -> dict[str, torch.Tensor] | list[torch.Tensor]:
        """
        This function prepares a batch of input samples.

        Parameters
        ----------
        batch : Batch
            The batch of input samples.

        Returns
        -------
        batch : list of torch.Tensor or dict of torch.Tensor
            The batch of input samples transformed into tensors.
        """
        if type(batch) is np.ndarray and batch.dtype != object:
            return [
                torch.tensor(
                    batch.reshape((batch.shape[0], 1))
                    if len(batch.shape) == 1
                    else batch,
                    device=self.device,
                )
            ]
        else:
            if type(batch) is dict:
                return {
                    s: torch.tensor(
                        v.reshape((v.shape[0], 1)) if len(v.shape) == 1 else v,
                        device=self.device,
                    )
                    for s, v in batch.items()
                }
            else:
                assert type(batch) is list
                return [
                    torch.tensor(
                        v.reshape((v.shape[0], 1)) if len(v.shape) == 1 else v,
                        device=self.device,
                    )
                    for v in batch
                ]

    def predict_on_batch(
        self, batch: Batch
    ) -> NDArray | list[NDArray] | dict[str, NDArray]:
        """
        This function computes network predictions for a batch of input samples.

        Parameters
        ----------
        batch : Batch
            The batch of input samples.

        Returns
        -------
        predictions : NDArray, list of NDArray or dict of NDArray
            A batch of network predictions.
        """
        with torch.inference_mode():
            batch_prepared = self.prepare_batch(batch)
            predictions = (
                self.model(**batch_prepared)
                if type(batch_prepared) is dict
                else self.model(*batch_prepared)
            )
            if type(predictions) is torch.Tensor:
                return predictions.detach().cpu().numpy()
            elif type(predictions) is tuple:
                return [prediction.detach().cpu().numpy() for prediction in predictions]
            else:
                return {
                    stream: prediction.detach().cpu().numpy()
                    for (stream, prediction) in predictions.values()
                }

    def train_on_batch(
        self, batch: Batch, targets: Batch, sample_weights: None | NDArray = None
    ) -> None:
        """
        This function trains the network on a batch of input samples.

        Parameters
        ----------
        batch : Batch
            The batch of input samples.
        targets : Batch
            The batch of target values.
        sample_weights : NDArray or None, optional
            An optional batch of sample weights.
        """
        self.optimizer.zero_grad()
        weights: None | torch.Tensor = None
        if sample_weights is not None:
            weights = (
                torch.tensor(sample_weights, device=self.device)
                if len(sample_weights.shape) == 2
                else torch.tensor(
                    sample_weights.reshape(sample_weights.size, 1), device=self.device
                )
            )
        batch_prepared = self.prepare_batch(batch)
        targets_prepared = self.prepare_batch(targets)
        predictions = (
            self.model(**batch_prepared)
            if type(batch_prepared) is dict
            else self.model(*batch_prepared)
        )
        pred_list: list[torch.Tensor]
        if type(predictions) is torch.Tensor:
            pred_list = [predictions]
        elif type(predictions) is tuple:
            pred_list = list(predictions)
        else:
            pred_list = predictions
        loss: list[torch.Tensor] = []
        if type(self.criteria) is list:
            assert type(self.loss_weights) is list
            target_list: list[torch.Tensor]
            if type(targets_prepared) is list:
                target_list = targets_prepared
            elif type(targets_prepared) is dict:
                target_list = list(targets_prepared.values())
            for i, criterion in enumerate(self.criteria):
                loss.append(
                    criterion(pred_list[i], target_list[i]) * self.loss_weights[i]
                )
                if weights is not None:
                    loss[-1] *= weights
        sum(loss).mean().backward()  # type: ignore
        self.optimizer.step()

    def get_weights(self) -> list[NDArray]:
        """
        This function returns the weights of the network.

        Returns
        -------
        weights : list of NDArray
            A list of layer weights.
        """
        # retrieve params from state_dict and save them as a list of numpy arrays
        weights = list(self.model.state_dict().values())
        for i in range(len(weights)):
            weights[i] = weights[i].cpu().numpy()

        return weights

    def set_weights(self, weights: list[NDArray]) -> None:
        """
        This function sets the weights of the network.

        Parameters
        ----------
        weights : list of NDArray
            A list of layer weights.
        """
        # prepare a new state_dict
        new_state_dict = self.model.state_dict()
        for i, param in enumerate(new_state_dict):
            new_state_dict[param] = torch.tensor(weights[i], device=self.device)
        # load the state_dict
        self.model.load_state_dict(new_state_dict)

    def clone(self) -> Self:
        """
        This function returns a copy of the network.

        Returns
        -------
        model : Self
            The network model's copy.
        """
        network = copy.deepcopy(self.model)
        optimizer = type(self.optimizer)(params=network.parameters())  # type: ignore
        model_clone = type(self)(
            network,
            optimizer,
            copy.deepcopy(self.criteria),
            loss_params=None,
            loss_weights=copy.deepcopy(self.loss_weights),
            activations=copy.deepcopy(self.activations),
        )
        if type(self.criteria) is dict:
            assert type(model_clone.criteria) is dict
            for criterion in self.criteria:
                model_clone.criteria[criterion].load_state_dict(
                    self.criteria[criterion].state_dict()
                )
        elif type(self.criteria) is list:
            assert type(model_clone.criteria) is list
            for i in range(len(self.criteria)):
                model_clone.criteria[i].load_state_dict(self.criteria[i].state_dict())
        model_clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return model_clone

    def set_optimizer(
        self,
        optimizer: None | str | optimizers.Optimizer,
        parameters: None | ParamDict = None,
    ) -> None:
        """
        This function sets the optimizer of the network model.

        Parameters
        ----------
        optimizer : str, optimizers.Optimizer or None
            The name of the optimizer.
        parameters : ParamDict or None, optional
            The parameters of the optimizer (e.g., learning rate).
        """
        valid_optimizers = {
            **dict.fromkeys(['Adadelta', 'adadelta'], optimizers.Adadelta),
            **dict.fromkeys(['Adagrad', 'adagrad'], optimizers.Adagrad),
            **dict.fromkeys(['Adam', 'adam'], optimizers.Adam),
            **dict.fromkeys(['AdamW', 'Adamw', 'adamw'], optimizers.AdamW),
            **dict.fromkeys(['Adamax', 'AdaMax', 'adamax'], optimizers.Adamax),
            **dict.fromkeys(
                ['SparseAdam', 'Sparseadam', 'sparseadam'], optimizers.SparseAdam
            ),
            **dict.fromkeys(['ASGD', 'asgd'], optimizers.ASGD),
            **dict.fromkeys(['LBFGS', 'lbfgs'], optimizers.LBFGS),
            **dict.fromkeys(['NAdam', 'Nadam', 'nadam'], optimizers.NAdam),
            **dict.fromkeys(['RAdam', 'Radam', 'radam'], optimizers.RAdam),
            **dict.fromkeys(['RMSprop', 'rmsprop'], optimizers.RMSprop),
            **dict.fromkeys(['Rprop', 'rprop'], optimizers.Rprop),
            **dict.fromkeys(['SGD', 'sgd'], optimizers.SGD),
        }
        self.optimizer: optimizers.Optimizer
        if not isinstance(optimizer, optimizers.Optimizer):
            # use Adam by default
            if optimizer not in valid_optimizers:
                optimizer = 'Adam'
            self.optimizer = valid_optimizers[optimizer](
                **(
                    {'params': self.model.parameters()}
                    if parameters is None
                    else {'params': self.model.parameters()} | parameters
                )
            )
        else:
            self.optimizer = optimizer
            if type(parameters) is dict:
                self.optimizer = type(self.optimizer)(
                    **(
                        {'params': self.model.parameters()}
                        if parameters is None
                        else {'params': self.model.parameters()} | parameters  # type: ignore
                    )
                )

    def set_loss(
        self,
        loss: FlexibleLoss,
        parameters: FlexibleLossParameters = None,
        loss_weights: None | list[float] | dict[str, float] = None,
    ) -> None:
        """
        This function sets the loss of the network model.

        Parameters
        ----------
        loss : str, torch.nn.modules.loss._Loss, list, dict or None, optional
            The loss(es) that should be used.
            Defaults to MSE if none was provided.
        parameters : ParamDict, list, dict or None, optional
            The parameters of the loss(es).
        loss_weights : list of float, dict of float or None, optional
            Optional weightings applied to different loss functions.
            Defaults to equal weightings when none were provided.
        """
        if type(loss) in [list, dict] and parameters is not None:
            assert type(loss) is type(parameters), (
                'Loss functions and their parameters should'
                ' be provided in the same data type!'
            )
        valid_losses = {
            'kl_divergence': torch.nn.KLDivLoss,
            'cosine_similarity': torch.nn.CosineEmbeddingLoss,
            'poisson': torch.nn.PoissonNLLLoss,
            'gaussian': torch.nn.GaussianNLLLoss,
            'binary_crossentropy': torch.nn.BCELoss,
            'hinge': torch.nn.HingeEmbeddingLoss,
            'margin_ranking': torch.nn.MarginRankingLoss,
            'multi_label_margin_ranking': torch.nn.MultiLabelMarginLoss,
            **dict.fromkeys(['mean_absolute_error', 'mae'], torch.nn.L1Loss),
            **dict.fromkeys(['mean_squared_error', 'mse'], torch.nn.MSELoss),
            **dict.fromkeys(['huber_loss', 'huber'], torch.nn.HuberLoss),
            **dict.fromkeys(
                ['categorical_crossentropy', 'crossentropy'], torch.nn.CrossEntropyLoss
            ),
            **dict.fromkeys(
                ['connectionist_temporal_classification', 'CTC', 'ctc'],
                torch.nn.CTCLoss,
            ),
            **dict.fromkeys(
                ['negative_log_likelihood', 'NLL', 'nll'], torch.nn.NLLLoss
            ),
        }
        # prepare loss and param lists
        criterion_list: list[Loss] = []
        if type(loss) is dict:
            criterion_list = list(loss.values())
        elif type(loss) is list:
            criterion_list = [l for l in loss]
        elif type(loss) is str or isinstance(loss, torch.nn.modules.loss._Loss):
            criterion_list = [loss]
        else:
            criterion_list = ['mse']
        param_list: list[ParamDict | None] = []
        if type(parameters) is dict[str, ParamDict | None]:
            param_list = list(parameters.values())
        elif type(parameters) is list:
            param_list = parameters
        elif type(parameters) is ParamDict:
            param_list = [parameters]
        else:
            param_list = [None] * len(criterion_list)
        # create list of loss items
        self.criteria: (
            list[torch.nn.modules.loss._Loss] | dict[str, torch.nn.modules.loss._Loss]
        ) = {} if type(loss) is dict else []
        for i, criterion in enumerate(criterion_list):
            _loss, _param = criterion, param_list[i]
            _criterion: torch.nn.modules.loss._Loss
            # create loss item if string identifier was provided
            if not isinstance(_loss, torch.nn.modules.loss._Loss):
                if _loss not in valid_losses:
                    _loss, _param = 'mse', {}
                _criterion = valid_losses[_loss]()
            else:
                _criterion = _loss
            # reinitialize when parameters where provided
            if _param:
                _criterion = type(_criterion)(**_param)
            # ensure no reduction is applied so that sample weighting
            # may be applied when training
            _criterion.reduction = 'none'
            # store loss
            if type(loss) is dict:
                assert type(self.criteria) is dict
                self.criteria[list(loss.keys())[i]] = _criterion
            else:
                assert type(self.criteria) is list
                self.criteria.append(_criterion)
        # create loss weightings (uniform by default)
        if loss_weights is not None:
            assert type(self.criteria) is type(loss_weights), (
                'Loss functions and their weightings should'
                ' be provided in the same data type!'
            )
        self.loss_weights: dict[str, float] | list[float] = (
            {i: 1.0 for i in self.criteria}
            if type(self.criteria) is dict
            else [1.0 for i in self.criteria]
        )
        if loss_weights is not None:
            self.loss_weights = loss_weights

    def get_layer_activity(self, batch: Batch, layer: int | str) -> NDArray:
        """
        This function returns the activity of a specified layer
        for a batch of input samples.

        Parameters
        ----------
        batch : Batch
            The batch of input samples.
        layer_index : int or str
            The index or name of the layer from which
            activity should be retrieved.

        Returns
        ----------
        activity : NDArray
            The layer activities of the specified layer
            for the batch of input samples.
        """

        # define hook
        def get_activation(activation, layer):
            def hook(model, input, output):
                activation[layer] = output.detach()

            return hook

        # register forward hook and record activity
        activation: dict[int | str, torch.Tensor] = {}
        layer_key = (
            layer
            if type(layer) is str
            else list(self.model.named_children())[int(layer)][0]
        )
        layer_index = layer if type(layer) is int else 10**10
        if type(layer) is str:
            for i, (name, _) in enumerate(self.model.named_children()):
                if name == layer:
                    layer_index = i
                    break
        sub_module = self.model.get_submodule(layer_key)
        sub_module.register_forward_hook(get_activation(activation, layer_key))
        self.predict_on_batch(batch)
        act: torch.Tensor = activation[layer_key]
        # apply activation function if available
        if type(self.activations) is list:
            assert len(self.activations) >= layer_index, 'Index not in activation list!'
            if self.activations[layer_index] is not None:
                act = self.activations[layer_index](act)
        else:
            assert type(self.activations) is dict, (
                'String layerkeys not compatible with activation lists!'
            )
            assert layer_key in self.activations, 'Key not in activation dict!'
            if self.activations[layer_key] is not None:
                act = self.activations[layer_key](act)

        return act.cpu().numpy().T

    def set_trainable(
        self,
        layers: list[int] | list[str],
        trainable: bool | list[bool] | dict[str, bool],
    ) -> None:
        """
        This function sets the trainability of specified network layers.

        Parameters
        ----------
        layers : list of int or list of str
            A list of layer indeces or layer names for which
            trainability will be set.
        trainable : bool, list of bool or dict of bool
            The trainability that will be set. Use a bool to set
            all layers with the same value, and use a list or dict
            to specify values for each layer.
        """
        weights: list[str] = []
        for param in self.model.state_dict():
            if '.weight' in param:
                weights.append(param.split('.')[0])
        requires_grad: bool
        if type(trainable) is bool:
            requires_grad = trainable
        if type(layers[0]) is int:
            for l, layer_idx in enumerate(layers):
                assert type(layer_idx) is int
                assert layer_idx < len(weights)
                if type(trainable) is list:
                    assert len(layers) == len(trainable)
                    requires_grad = trainable[l]
                self.model.get_parameter(
                    weights[layer_idx] + '.weight'
                ).requires_grad = requires_grad
                self.model.get_parameter(
                    weights[layer_idx] + '.bias'
                ).requires_grad = requires_grad
        else:
            for layer in layers:
                assert type(layer) is str
                assert layer in weights
                if type(trainable) is dict:
                    assert layer in trainable
                    requires_grad = trainable[layer]
                self.model.get_parameter(
                    layer + '.weight'
                ).requires_grad = requires_grad
                self.model.get_parameter(layer + '.bias').requires_grad = requires_grad

    def set_device(self, device: str = 'cpu') -> None:
        """
        This function moves the model to a specified device.

        Parameters
        ----------
        device : str
            The name of the device that the model will
            be stored on ('cpu' by default).
        """
        self.device = torch.device(device)
        self.model.to(self.device)
