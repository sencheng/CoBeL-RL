# imports
import numpy as np
from tensorflow.keras.models import Model, clone_model  # type: ignore
from tensorflow.keras.layers import Layer  # type: ignore
from tensorflow.keras import optimizers  # type: ignore
from tensorflow.keras import losses  # type: ignore
# CoBeL-RL
from .network import Network
# typing
from typing import Self
from numpy.typing import NDArray
from .network import Batch, ParamDict
# make optimizers and losses available via names
tf_optimizers = {
    'adadelta': optimizers.Adadelta,
    'adafactor': optimizers.Adafactor,
    'adagrad': optimizers.Adagrad,
    'adam': optimizers.Adam,
    'adamw': optimizers.AdamW,
    'ftrl': optimizers.Ftrl,
    'lion': optimizers.Lion,
    'lossscaleoptimizer': optimizers.LossScaleOptimizer,
    'nadam': optimizers.Nadam,
    'rmsprop': optimizers.RMSprop,
    'sgd': optimizers.SGD,
}
tf_losses = {
    'binarycrossentropy': losses.BinaryCrossentropy,
    'binaryfocalcrossentropy': losses.BinaryFocalCrossentropy,
    'ctc': losses.CTC,
    'categoricalcrossentropy': losses.CategoricalCrossentropy,
    'categoricalfocalcrossentropy': losses.CategoricalFocalCrossentropy,
    'categoricalhinge': losses.CategoricalHinge,
    'cosinesimilarity': losses.CosineSimilarity,
    'dice': losses.Dice,
    'hinge': losses.Hinge,
    'huber': losses.Huber,
    'kldivergence': losses.KLDivergence,
    'logcosh': losses.LogCosh,
    'mae': losses.MeanAbsoluteError,
    'mape': losses.MeanAbsolutePercentageError,
    'mse': losses.MeanSquaredError,
    'msle': losses.MeanSquaredLogarithmicError,
    'poisson': losses.Poisson,
    'sparsecategoricalcrossentropy': losses.SparseCategoricalCrossentropy,
    'squaredhinge': losses.SquaredHinge,
    'tversky': losses.Tversky,
}


class KerasNetwork(Network):
    """
    This class provides an interface to Tensorflow/Keras models.

    Parameters
    ----------
    model : Model
        The network model.
    device : str, default='/CPU:0'
        The name of the device that model will be stored on.
        Currently unused.

    Attributes
    ----------
    model : Model
        The network model.
    device : str, default='/CPU:0'
        The name of the device that model will be stored on.
        Currently unused.

    Examples
    --------

    Simply define a Sequential network which you want to use
    with classes like the DQN agent. ::

        >>> from tensorflow.keras.models import Sequential
        >>> from tensorflow.keras.layers import Input, Dense
        >>> from cobel.network import KerasNetwork
        >>> network = Sequential([Input((64)),
        ...                       Dense(64, activation='relu', name='hidden'),
        ...                       Dense(4, name='output')])
        >>> network.compile(optimizer='adam', loss='mse')
        >>> model = KerasNetwork(network)

    This class also accepts complex models created using the functional API. ::

        >>> from tensorflow.keras.models import Sequential
        >>> from tensorflow.keras.layers import Input, Dense, concatenate
        >>> from cobel.network import KerasNetwork
        >>> input_1, input_2 = Input((64, )), Input((64, ))
        >>> in_1 = Dense(64, activation='relu', 'in_1')
        >>> in_2 = Dense(64, activation='relu', 'in_1')
        >>> concat = concatenate([in_1, in_2], 'concat')
        >>> out_1 = Dense(4, 'out_1')
        >>> out_2 = Dense(4, 'out_1')
        >>> network = Model(inputs=[in_1, in_2], outputs=[out_1, out_2])
        >>> network.compile(optimizer='adam', loss='mse')
        >>> model = KerasNetwork(network)

    """

    def __init__(self, model: Model, device: str = '/CPU:0') -> None:
        self.model = model
        self.device = device

    def predict_on_batch(self, batch: Batch) -> Batch:
        """
        This function computes the network predictions for a batch of input samples.

        Parameters
        ----------
        batch : Batch
            The batch of input samples.

        Returns
        -------
        predictions : Batch
            A batch of network predictions.
        """
        preds = self.model(batch)
        if isinstance(preds, list):
            return [p.numpy() for p in preds]
        else:
            return preds.numpy()

    def train_on_batch(
        self, batch: Batch, targets: Batch, sample_weights: None | NDArray = None
    ) -> None:
        """
        This functions trains the network on a batch of input samples.

        Parameters
        ----------
        batch : Batch
            The batch of input samples.
        targets : Batch
            The batch of target values.
        sample_weights : NDArray or None, optional
            An optional batch of sample_weights
        """
        assert type(batch) is type(targets)
        self.model.train_on_batch(batch, targets, sample_weights)

    def get_weights(self) -> list[NDArray]:
        """
        This function returns the weights of the network.

        Returns
        -------
        weights : list of NDArray
            A list of layer weights.
        """
        return self.model.get_weights()

    def set_weights(self, weights: list[NDArray]) -> None:
        """
        This function sets the weights of the network.

        Parameters
        ----------
        weights : list of NDArray
            A list of layer weights.
        """
        self.model.set_weights(weights)

    def clone(self) -> Self:
        """
        This function returns a copy of the network.

        Returns
        -------
        model : Self
            The network model's copy.
        """
        return type(self)(clone_model(self.model))

    def set_optimizer(
        self, optimizer: str | optimizers.Optimizer, parameters: None | ParamDict = None
    ) -> None:
        """
        This function sets the optimizer of the network model.

        Parameters
        ----------
        optimizer : str or optimizers.Optimizer
            The optimizer that should be used.
        parameters : ParamDict or None, optional
            The parameters of the optimizer (e.g., learning rate).
        """
        config = self.model.get_compile_config()
        opt: optimizers.Optimizer
        if type(optimizer) is str:
            opt = tf_optimizers[optimizer.lower()]()
        else:
            opt = optimizer
        if parameters is not None:
            opt = type(opt)(**parameters)
        opt_config = {
            'module': 'keras.optimizers',
            'class_name': type(opt).__name__,
            'config': opt.get_config(),
            'registered_name': None,
        }
        config['optimizer'] = opt_config
        self.model.compile_from_config(config)

    def set_loss(
        self, loss: str | losses.Loss, parameters: None | ParamDict = None
    ) -> None:
        """
        This function sets the loss of the network model.

        Parameters
        ----------
        loss : str or losses.Loss
            The name of the loss.
        parameters : ParamDict or None, optional
            The paramters of the loss.
        """
        config = self.model.get_compile_config()
        new_loss: losses.Loss
        if type(loss) is str:
            new_loss = tf_losses[loss.lower()]()
        else:
            new_loss = loss
        if parameters is not None:
            new_loss = type(new_loss)(**parameters)
        loss_config = {
            'module': 'keras.losses',
            'class_name': type(new_loss).__name__,
            'config': new_loss.get_config(),
            'registered_name': None,
        }
        config['loss'] = loss_config
        self.model.compile_from_config(config)

    def get_layer_activity(self, batch: Batch, layer_index: int | str) -> NDArray:
        """
        This function return the activity of a specified layer
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
        # construct "sub model"
        layer: Layer
        if type(layer_index) is int:
            layer = self.model.get_layer(index=layer_index)
        else:
            layer = self.model.get_layer(name=layer_index)
        inputs = (
            self.model.layers[0].input
            if isinstance(batch, np.ndarray)
            else self.model.inputs
        )
        sub_model = Model(inputs=inputs, outputs=layer.output)
        # compute activity map
        activity = sub_model(batch).numpy().T

        return activity

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
            set all layer with the same value, and use a list
            or dict to specify values for each layer.
        """
        for idx in layers:
            layer: Layer
            if type(idx) is int:
                layer = self.model.get_layer(index=idx)
            else:
                layer = self.model.get_layer(name=idx)
            freeze: bool
            if type(trainable) is bool:
                freeze = trainable
            elif type(trainable) is list:
                assert type(idx) is int
                freeze = trainable[idx]
            else:
                assert type(trainable) is dict
                assert type(idx) is str
                freeze = trainable[idx]
            layer.trainable = freeze
