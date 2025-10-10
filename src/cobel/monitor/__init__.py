"""
This module provides various monitor classes for the
tracking of relevant simulation variables.

Classes
-------
Monitor
    Abstract monitor class from which all monitor classes are derived
EscapeLatencyMonitor
    A monitor which tracks the escape latency of reinforcement learning agents
RewardMonitor
    A monitor which tracks the episodic reward
ResponseMonitor
    A monitor which tracks the responses of a reinforcement learning agent
    by applying a user-defined response coding
TrajectoryMonitor
    A monitor which tracks the spatial trajectory of reinforcement learning
    agents
QMonitor
    A monitor which tracks the learned Q-function
RepresentationMonitor
    A monitor which tracks the network representations for a given set
    of observations in a specified network layer

"""

from .monitor import Monitor
from .behavior import (
    EscapeLatencyMonitor,
    RewardMonitor,
    ResponseMonitor,
    TrajectoryMonitor,
    QMonitor,
)
from .network import RepresentationMonitor
