# basic imports
import abc

# typing
from typing import Any, Self
from numpy.typing import NDArray

Batch = NDArray | list[NDArray] | dict[str, NDArray]
ParamDict = dict[str, Any]


class Network(abc.ABC):
    """Implements an abstract network class."""

    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def predict_on_batch(self, batch: Batch) -> Batch:
        """
        Compute network predictions for a batch of input samples.

        Parameters
        ----------
        batch : cobel.network.network.Batch
            The batch of input samples.

        Returns
        -------
        predictions : cobel.network.network.Batch
            A batch of network predictions.
        """

    @abc.abstractmethod
    def train_on_batch(self, batch: Batch, targets: Batch) -> None:
        """
        Train the network on a batch of input samples.

        Parameters
        ----------
        batch : cobel.network.network.Batch
            The batch of input samples.
        targets : cobel.network.network.Batch
            The batch of target values.
        """

    @abc.abstractmethod
    def get_weights(self) -> list[NDArray]:
        """
        Return the weights of the network.

        Returns
        -------
        weights : list of numpy.ndarray
            A list of layer weights.
        """

    @abc.abstractmethod
    def set_weights(self, weights: list[NDArray]) -> None:
        """
        Set the weights of the network.

        Parameters
        ----------
        weights : list of numpy.ndarray
            A list of layer weights.
        """

    @abc.abstractmethod
    def clone(self) -> Self:
        """
        Return a copy of the network model.

        Returns
        -------
        model : cobel.network.network.Network
            The network model's copy.
        """

    @abc.abstractmethod
    def set_optimizer(
        self, optimizer: str, parameters: None | ParamDict = None
    ) -> None:
        """
        Set the optimizer of the network model.

        Parameters
        ----------
        optimizer : str
            The name of the optimizer.
        parameters : cobel.network.network.ParamDict or None, optional
            The parameters of the optimizer (e.g., learning rate).
        """

    @abc.abstractmethod
    def set_loss(self, loss: str, parameters: None | ParamDict = None) -> None:
        """
        Set the loss of the network model.

        Parameters
        ----------
        loss : str
            The name of the loss.
        parameters : cobel.network.network.ParamDict or None, optional
            The parameters of the loss.
        """

    @abc.abstractmethod
    def get_layer_activity(self, batch: Batch, layer_index: int | str) -> NDArray:
        """
        Return the activity of a specified
        layer for a batch of input samples.

        Parameters
        ----------
        batch : cobel.network.network.Batch
            The batch of input samples.
        layer_index : int or str
            The index of the layer from which activity
            should be retrieved.

        Returns
        -------
        activity : numpy.ndarray
            The layer activities of the specified layer for the batch of input samples.
        """

    @abc.abstractmethod
    def set_trainable(
        self,
        layers: list[int] | list[str],
        trainable: bool | list[bool] | dict[str, bool],
    ) -> None:
        """
        Set the trainability of specified network layers.

        Parameters
        ----------
        layers : list of int or list of str
            A list of layer indeces or layer names
            for which trainability will be set.
        trainable : bool, list of bool  or dict of bool
            The trainability that will be set. Use a bool to set
            all layers with the same value, and use a list or dict
            to specify values for each layer.
        """
