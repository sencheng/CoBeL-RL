# basic imports
import numpy as np
# framework imports
from cobel.interfaces.rl_interface import AbstractInterface


class InterfaceBaseline(AbstractInterface):
    
    def __init__(self, modules: dict, with_GUI=True, reward_callback=None):
        '''
        This is the Open AI gym interface class. The interface wraps the control path and ensures communication between the agent and the environment.
        The class descends from gym.Env, and is designed to be minimalistic (currently!).
        
        Parameters
        ----------
        modules :                           Contains framework modules.
        with_GUI :                          If true, observations and policy will be visualized.on.
        
        Returns
        ----------
        None
        '''
        super().__init__(modules, with_GUI)
        # memorize the reward callback function
        self.reward_callback = reward_callback
        self.world = self.modules['world']
        self.observations = self.modules['observation']
        # retrieve action space
        self.action_space = modules['spatial_representation'].get_action_space()
        # retrieve observation space
        self.observation_space = modules['observation'].get_observation_space()
        # this observation variable is filled by the OBS modules 
        self.observation = None
        # required for the analysis of the agent's behavior
        self.forbidden_zone_hit = False
        self.final_node = -1
        
    def update_observation(self, observation: np.ndarray):
        '''
        This function updates the observation provided by the environment.
        
        Parameters
        ----------
        observation :                       The observation used to perform the update.
        
        Returns
        ----------
        None
        '''
        self.observation = observation
    
    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        '''
        AI Gym's step function.
        Executes the agent's action and propels the simulation.
        
        Parameters
        ----------
        action :                            The action selected by the agent.
        
        Returns
        ----------
        observation :                       The observation of the new current state.
        reward :                            The reward received.
        end_trial :                         Flag indicating whether the trial ended.
        logs :                              The (empty) logs dictionary.
        '''
        callback_value = self.modules['spatial_representation'].generate_behavior_from_action(action)
        callback_value['rlAgent'] = self.rl_agent
        callback_value['modules'] = self.modules
        
        reward, end_trial = self.reward_callback(callback_value)
        self.observation = np.copy(self.modules['observation'].observation)
        
        return self.modules['observation'].observation, reward, end_trial, {}
         
    def reset(self) -> np.ndarray:
        '''
        AI Gym's reset function.
        Resets the environment and the agent's state.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        observation :                       The observation of the new current state.
        '''
        self.modules['spatial_representation'].generate_behavior_from_action('reset')
        self.observation = np.copy(self.modules['observation'].observation)
        
        return self.modules['observation'].observation
    
    def get_position(self) -> np.ndarray:
        '''
        This function returns the agent's position in the environment.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        position :                          Numpy array containing the agent's position.
        '''
        current_node = self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].current_node]
        
        return np.array([current_node.x, current_node.y])
