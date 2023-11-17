# basic imports
import numpy as np
import gymnasium as gym


class AbstractInterface(gym.Env):
    
    def __init__(self, modules: dict, with_GUI: bool = True):
        '''
        This is the abstract OpenAI gym interface class.
        
        Parameters
        ----------
        modules :                           Contains framework modules.
        with_GUI :                          If true, observations and policy will be visualized.on.
        
        Returns
        ----------
        None
        '''
        # store the modules
        self.modules = modules
        # store visual output variable
        self.with_GUI = with_GUI
        # a variable that allows the OAI class to access the robotic agent class
        self.rl_agent = None
    
    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        '''
        OpenAI Gym's step function.
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
        raise NotImplementedError('.step() function not implemented!')
         
    def reset(self) -> np.ndarray:
        '''
        OpenAI Gym's reset function.
        Resets the environment and the agent's state.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        observation :                       The observation of the new current state.
        '''
        raise NotImplementedError('.reset() function not implemented!')
        
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
        return np.array([])
