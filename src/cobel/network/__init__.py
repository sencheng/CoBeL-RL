"""
This module provides interface network classes for
commonly used deep learning frameworks.

Classes
-------
Network
    Abstract network class from which all network classes are derived
TorchNetwork
    A network interface class for simple feed-forward PyTorch networks
FlexibleTorchNetwork
    A network interface class for PyTorch networks with multiple input
    and output stream
KerasNetwork
    A network interface class for Tensorflow/Keras networks
FlaxNetwork
    A network interface class for Flax networks
    Experimental

"""

from importlib.util import find_spec
from .network import Network

if find_spec('torch') is not None:
    from .network_torch import TorchNetwork, FlexibleTorchNetwork
if find_spec('tensorflow') is not None:
    from .network_tensorflow import KerasNetwork
if find_spec('flax') is not None:
    from .network_flax import FlaxNetwork
