"""
This module offers a wide range of reinforcement learning agents.

Classes
-------
Agent
    Abstract agent class from which all other agents are derived
QAgent
    A simple (tabular) Q-learning agent
DynaQ
    A Dyna-Q agent which is used in combination with gridworld environments
DynaDQN
    A Dyna-Q-DQN hybrid agent which is used in combination with
    gridworld environments and maps observations to state indices
DynaDSR
    A Dyna-Q-DSR hybrid agent which is used in combination with
    gridworld environments and maps observations to state indices
DQN
    A deep Q-network agent
ADQN
    A variant of the deep Q-network agent which learns CS-US associations
AssociativeNetwork
    An agent based on the associative network model from Donoso et al. (2021)
RescorlaWagner
    A Rescorla-Wagner model based agent (return a single scalar
    actions which represents associativity)
BinaryRescorlaWagner
    A Rescorla-Wagner model based agent
    (transforms associativity to binary actions)
MFEC
    A model-free episodic control agent
SFMA
    A Dyna-Q agent which performs experience replay
    using SFMA (Diekmann and Cheng, 2023)
PMA
    A Dyna-Q agent which performs experience replay
    using PMA (Mattar and Daw, 2018)
SR
    An agent based on the successor representation which is
    used in combination with gridworld environments

"""

from .agent import Agent
from .q import QAgent
from .dyna_q import DynaQ, DynaDQN, DynaDSR
from .dqn import DQN
from .adqn import ADQN
from .anet import AssociativeNetwork
from .rw import RescorlaWagner, BinaryRescorlaWagner
from .mfec import MFEC
from .sfma import SFMA
from .pma import PMA
from .sr import SR
