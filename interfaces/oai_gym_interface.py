# basic imports
import os
import time
import numpy as np
import gym
import subprocess
import signal
# OpenAI Gym
from gym import spaces
# ML-Agents
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
# Keras-RL
from rl.core import Processor


class OAIGymInterface(gym.Env):
    '''
    This is the Open AI gym interface class. The interface wraps the control path and ensures communication
    between the agent and the environment. The class descends from gym.Env, and is designed to be minimalistic (currently!).
    
    | **Args**
    | modules:                      The dict of all available modules in the system.
    | withGUI:                      If True, the module provides GUI control.
    | rewardCallback:               This callback function is invoked in the step routine in order to get the appropriate reward w.r.t. the experimental design.
    '''
    
    def __init__(self, modules, withGUI=True, rewardCallback=None):
        # store the modules
        self.modules = modules
        # store visual output variable
        self.withGUI = withGUI
        # memorize the reward callback function
        self.rewardCallback = rewardCallback
        self.world = self.modules['world']
        self.observations = self.modules['observation']
        # retrieve action space
        self.action_space = modules['spatial_representation'].get_action_space()
        # retrieve observation space
        self.observation_space = modules['observation'].getObservationSpace()
        # this observation variable is filled by the OBS modules 
        self.observation = None
        # required for the analysis of the agent's behavior
        self.forbiddenZoneHit = False
        self.finalNode = -1
        # a variable that allows the OAI class to access the robotic agent class
        self.rl_agent = None
        
    def updateObservation(self, observation):
        '''
        This function updates the observation provided by the environment.
        
        | **Args**
        | observation:                  The observation used to perform the update.
        '''
        self.observation = observation
    
    def step(self, action):
        '''
        The step function that propels the simulation.
        This function is called by the .fit function of the RL agent whenever a novel action has been computed.
        The action is used to decide on the next topology node to run to, and step then triggers the control path (including 'Blender')
        by making use of direct calls and signal/slot methods.
        
        | **Args**
        | action:                       The action to be executed.
        '''
        callbackValue = self.modules['spatial_representation'].generate_behavior_from_action(action)
        callbackValue['rlAgent'] = self.rl_agent
        callbackValue['modules'] = self.modules
        
        reward, stopEpisode = self.rewardCallback(callbackValue)
        self.observation = np.copy(self.modules['observation'].observation)
        
        return self.modules['observation'].observation, reward, stopEpisode, {}
         
    def reset(self):
        '''
        This function restarts the RL agent's learning cycle by initiating a new episode.
        '''
        self.modules['spatial_representation'].generate_behavior_from_action('reset')
        self.observation = np.copy(self.modules['observation'].observation)
        
        return self.modules['observation'].observation
