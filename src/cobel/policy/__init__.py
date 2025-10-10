"""
This module offers a variety of action selection policies.

Classes
-------
Policy
    Abstract policy class from which all other policies are derived
EpsilonGreedy
    An epsilon-greedy action selection policy
ExclusiveEpsilonGreedy
    An epsilon-greedy action selection policy which only
    considers non-optimal actions during exploration
Softmax
    A softmax action selection policy
Proportional
    An action selection policy which transforms a scalar value
    to binary actions proportionally
Threshold
    An action selection policy which transforms a scalar value
    to binary actions by applying a decision threshold
Sigmoid
    An action selection policy which transforms a scalar value
    to binary actions by applying a sigmoid function
RandomDiscrete
    Generates random discrete actions
RandomUniform
    Generates random continuous actions sampled uniformly from
    predefined intervals
RandomGaussian
    Generates random continuous actions sampled from predefined
    Gaussian distributions

"""

from .policy import Policy
from .greedy import EpsilonGreedy, ExclusiveEpsilonGreedy
from .softmax import Softmax
from .scalar import Proportional, Threshold, Sigmoid
from .random import RandomDiscrete, RandomUniform, RandomGaussian
