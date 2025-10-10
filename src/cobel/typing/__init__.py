"""
This module provides easy access to the framework's
custom types.
"""

# agent module types
from ..agent.agent import Logs, Callback, CallbackDict
# interface module types
from ..interface.interface import Observation, Action
from ..interface.topology import Node, NodeID, Pose
from ..interface.simulator.simulator import ImageInfo
from ..interface.sequence import Trial, TrialStep
# optimizer module types
from ..optimizer.optimizer import Fit, FitLoss, Simulation
