# basic imports
import copy
import optax  # type: ignore
import numpy as np
from flax import nnx
# framework imports
from .network import Network
from .network import Batch, ParamDict
# typing
from typing import Self
from collections.abc import Callable
from numpy.typing import NDArray

flax_optimizers = {
    'adam': optax.adam,
    'adamw': optax.adamw,
    'adadelta': optax.adadelta,
    'adagrad': optax.adagrad,
    'adamax': optax.adamax,
    'lbfgs': optax.lbfgs,
    'nadam': optax.nadam,
    'radam': optax.radam,
    'rmsprop': optax.rmsprop,
    'rprop': optax.rprop,
    'sgd': optax.sgd,
}
flax_losses = {
    'kl_divergence': optax.kl_divergence,
    'cosine_similarity': optax.cosine_similarity,
    'hinge': optax.hinge_loss,
    'mse': optax.squared_error,
    'huber': optax.huber_loss,
    'ctc': optax.ctc_loss,
}


class FlaxNetwork(Network):
    """
    This class provides an interface to Flax models.

    Parameters
    ----------
    model : nnx.Module
        The network model.
    optimizer : str, nnx.Optimizer or None, optional
        The optimizer that should be used.
        Defaults to Adam if none was provided.
    loss : str, Callable or None, optional
        The loss that should be used.
        Defaults to MSE if none was provided.
    optimizer_params : dict or None, optional
        The parameters of the optimizer (e.g., learning rate).

    Attributes
    ----------
    model : nnx.Module
        The network model.
    optimizer : nnx.Optimizer
        The optimizer that is being used.
    loss : Callable
        The loss that is being used.

    Examples
    --------

    Simpy define a Flax network which you want to use
    with classes like the DQN agent. ::

        >>> flax import nnx
        >>> from cobel.network import FlaxNetwork
        >>> Model(nnx.Module):
        ...     def __init__(self):
        ...         rngs = nnx.Rngs(0)
        ...         self.hidden = Linear(6, 64, rngs=rngs)
        ...         self.out = Linear(64, 4, rngs=rngs)
        ...     def __call__(self, x):
        ...         self.x = self.hidden(x)
        ...         self.x = nnx.relu(x)
        ...         self.x = self.out(x)
        ...         return x
        >>> network = Model()
        >>> model = FlaxNetwork(network)

    In order to record layer activity you have to make
    it available via intermediates stored using `.sow()`
    in your network's `__call__` function.
    To make sure that only the values of the last model
    call are stored make sure that `reduce_fn` is defined
    accordingly. ::

        >>> def __call__(self, x):
        ...     x = self.hidden(x)
        ...     x = nnx.relu(x)
        ...     self.sow(nnx.Intermediate, 'act_hidden', x,
        ...              reduce_fn=(lambda xs, x: (x, )))
        ...     x = self.out(x)
        ...     return x

    The activity can then be retrieved with `get_layer_activity`
    by providing the intermediate's name. ::

        >>> model.get_layer_activity(np.random.rand(10, 6),
        ...                          'act_hidden')

    """

    def __init__(
        self,
        model: nnx.Module,
        optimizer: None | str | nnx.Optimizer = None,
        loss: None | str | Callable = None,
        optimizer_params: None | dict = None,
        loss_params: None | dict = None,
    ) -> None:
        self.model: nnx.Module = model
        self.optim: nnx.Optimizer
        if optimizer is None:
            self.optim = nnx.Optimizer(self.model, optax.adam(0.001), wrt=nnx.Param)
        elif type(optimizer) is str:
            opt_param = {'learning_rate': 0.001, 'wrt': nnx.Param}
            if optimizer_params is not None:
                opt_param = optimizer_params | {'wrt': nnx.Param}
            self.optim = flax_optimizers[optimizer.lower()](**opt_param)
        else:
            assert isinstance(optimizer, nnx.Optimizer)
            self.optim = optimizer
        self.loss: Callable
        if loss is None:
            self.loss = optax.squared_error
        elif type(loss) is str:
            self.loss = flax_losses[loss.lower()]
        else:
            self.loss = loss  # type: ignore

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
        assert callable(self.model)
        return np.array(self.model(batch))

    @nnx.jit(static_argnums=(0, 2))
    def _train_step(self, model, loss, optimizer, samples, targets):
        """
        Train step function which updates the network.
        """

        def loss_fn(model):
            return loss(model(samples), targets).mean()

        _, grad = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grad)

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
        self._train_step(self.model, self.loss, self.optim, batch, targets)

    def get_weights(self) -> list[NDArray]:
        """
        This function returns the weights of the network.

        Returns
        -------
        weights : list of NDArray
            A list of layer weights.
        """
        weights: list[NDArray] = []
        for child in self.model.iter_children():
            layer = child[1]
            assert hasattr(layer, 'kernel')
            assert hasattr(layer, 'bias')
            weights.append(np.array(layer.kernel.value))
            weights.append(np.array(layer.bias.value))

        return weights

    def set_weights(self, weights: list[NDArray]) -> None:
        """
        This function sets the weights of the network.

        Parameters
        ----------
        weights : list of NDArray
            A list of layer weights.
        """
        for i, child in enumerate(self.model.iter_children()):
            layer = child[1]
            assert hasattr(layer, 'kernel')
            assert hasattr(layer, 'bias')
            layer.kernel.value = weights[i * 2]
            layer.bias.value = weights[i * 2 + 1]

    def clone(self) -> Self:
        """
        This function returns a copy of the network.

        Returns
        -------
        model : Self
            The network model's copy.
        """
        network = copy.deepcopy(self.model)
        optim = copy.deepcopy(self.optim)
        model = type(self)(network, optim, self.loss)

        return model

    def set_optimizer(
        self, optimizer: str | nnx.Optimizer, parameters: None | ParamDict = None
    ) -> None:
        """
        This function sets the optimizer of the network model.

        Parameters
        ----------
        optimizer : str or nnx.Optimizer
            The optimizer that should be used.
        parameters : ParamDict or None, optional
            The parameters of the optimizer (e.g., learning rate).
        """
        optim: nnx.Optimizer
        if type(optimizer) is str:
            params: ParamDict
            if parameters is None:
                params = {'learning_rate': 0.001, 'wrt': nnx.Param}
            else:
                params = parameters
            optim = flax_optimizers[optimizer.lower()](**params)
        else:
            assert isinstance(optimizer, nnx.Optimizer)
            optim = optimizer
        self.optim = nnx.Optimizer(self.model, optim, wrt=nnx.Param)

    def set_loss(
        self, loss: str | Callable, parameters: None | ParamDict = None
    ) -> None:
        """
        This function sets the loss of the network model.

        Parameters
        ----------
        loss : str or Callable
            The loss that should be used.
        parameters : ParamDict or None, optional
            The parameters of the loss.
            Unused.
        """
        if type(loss) is str:
            self.loss = flax_losses[loss.lower()]
        else:
            self.loss = loss  # type: ignore

    def get_layer_activity(self, batch: Batch, layer: int | str) -> NDArray:
        """
        This function returns the activity of a specified layer
        for a batch of input samples.
        Currently only the recording of layer activity using intermediates
        stored with `.sow()` is supported.
        Define your network's `__call__` function accordingly.

        Parameters
        ----------
        batch : Batch
            The batch of input samples.
        layer_index : int or str
            The index or name of the layer from which
            activity should be retrieved.
            Will throw an assertion error when an index is given.

        Returns
        -------
        activity : NDArray
            The layer activities of the specified layer
            for the batch of input samples.
        """
        assert type(layer) is str, 'Index values are currently not supported!'
        assert callable(self.model), 'Model not callable!'
        self.model(batch)
        hasattr(self.model, layer)
        intermediate = getattr(self.model, layer)
        assert isinstance(intermediate, nnx.Intermediate)

        return np.array(intermediate.value[-1]).T

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
        print('Freezing of layers currently not supported!')
