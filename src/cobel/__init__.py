"""
CoBeL-RL
========

CoBeL-RL is a closed-loop simulator of complex behavior
and learning based on reinforcement learning (RL) and
deep neural networks (DNNs). It provides a neuroscience-oriented
framework for efficiently setting up and running simulations.

Available subpackages
---------------------
agent
    Various implementations of RL agents
analysis
    Collection of analysis functions
interface
    Various types of RL environments
memory
    Various memory modules which can be used with RL agents
misc
    Miscellaneous functionality like environmental editors and template functions
monitor
    Various monitors for tracking variables of interest during simulations
network
    Abstract network classes which interface with common DL framworks
optimizer
    Offers functionality for fitting models to behavioral data
policy
    Various action selection policies
typing
    Provides easy access to the framework's custom types

Utilities
---------
gridworld-editor
    Start the gridworld editor GUI for creation and editing of gridworld environments.
__version__
    CoBeL-RL version string

"""

from importlib.metadata import distribution

__version__ = distribution('cobel').version
