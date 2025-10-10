"""
This module provides memory modules which are used by
the different reinforcement learning agents.

Classes
-------
DynaQMemory
    Memory module to be used with the Dyna-Q-type agents
DQNMemory
    Memory module to be used with the DQN-type agents
ADQNMemory
    Memory module to be used with the ADQN-type agents
SFMAMemory
    Memory module to be used with the SFMA agent
PMAMemory
    Memory module to be used with the PMA agent

"""

from .dyna_q import DynaQMemory
from .dqn import DQNMemory
from .adqn import ADQNMemory
from .sfma import SFMAMemory
from .pma import PMAMemory
