


import numpy as np
import gym
import PyQt5 as qt


from gym import spaces



### This is the Open AI gym interface class. The interface wraps the control path and ensures communication
### between the agent and the environment. The class descends from gym.Env, and is designed to be minimalistic (currently!).
class OAIGymInterface(gym.Env):
    
    
    # The constructor.
    # modules:          the dict of all available modules in the system
    # withGUI:          if True, the module provides GUI control
    # rewardCallback:   this callback function is invoked in the step routine in order to get the appropriate reward w.r.t. the experimental design
    
    def __init__(self, modules,withGUI=True,rewardCallback=None):
        
        # store the modules
        self.modules=modules
    
        # store visual output variable
        self.withGUI=withGUI
        
        # memorize the reward callback function
        self.rewardCallback=rewardCallback
        
        self.world=self.modules['world']
        self.observations=self.modules['observation']

        
        # second: action space
        self.action_space = gym.spaces.Discrete(modules['spatial_representation'].cliqueSize)
        
        
        # third: observation space
        self.observation_space=modules['observation'].getObservationSpace()
        
        # all OAI spaces have been initialized!
        
        # this observation variable is filled by the OBS modules 
        self.observation=None
        
        # required for the analysis of the agent's behavior
        self.forbiddenZoneHit=False
        self.finalNode=-1
        
        # a variable that allows the OAI class to access the robotic agent class
        self.rlAgent=None
        
        
    # This function (slot) updates the observation provided by the environment
    # 
    # observation:  the observation used to perform the update
    def updateObservation(self,observation):
        self.observation=observation
    
    
    
    
    
    
        
    
    # The step function that propels the simulation.
    # This function is called by the .fit function of the RL agent whenever a novel action has been computed.
    # The action is used to decide on the next topology node to run to, and step then triggers the control path (including 'Blender')
    # by making use of direct calls and signal/slot methods.
    # 
    # action:   the action to be executed
    
    def _step(self, action):
        
        
        callbackValue=self.modules['spatial_representation'].generate_behavior_from_action(action)
        callbackValue['rlAgent']=self.rlAgent
        callbackValue['modules']=self.modules
        
        reward,stopEpisode=self.rewardCallback(callbackValue)
        
        return self.modules['observation'].observation, reward, stopEpisode, {}
        
        
    # This function restarts the RL agent's learning cycle by initiating a new episode.
    # 
    def _reset(self):
        
        self.modules['spatial_representation'].generate_behavior_from_action('reset')
        
        
        # return the observation
        return self.modules['observation'].observation
        
