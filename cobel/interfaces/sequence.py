# basic imports
import numpy as np
import gymnasium as gym
# framework imports
from cobel.interfaces.rl_interface import AbstractInterface


class InterfaceSequence(AbstractInterface):
    
    def __init__(self, modules: dict, trials: list, observations: dict, number_of_actions: int = 1):
        '''
        The Open AI Gym interface.
        The agent is presented a predetermined sequences of experiences.
        
        Parameters
        ----------
        modules :                           Contains framework modules.\n
        trials :                            The predetermined sequence of experiences.\n
        observations :                      A dictionary containing the observations referenced by the trial sequence.\n
        number_of_actions :                 The agent's number of actions.\n
        
        Returns
        ----------
        None\n
        '''
        super().__init__(modules=modules, with_GUI=False)
        # the trial sequence
        self.trials = trials
        # the observations
        self.observations = observations
        # the current trial
        self.current_trial = 0
        # the current trial step
        self.current_step = 0
        # a variable that allows the OAI class to access the robotic agent class
        self.rl_agent = None
        # prepare observation and action spaces
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.observations[list(self.observations)[0]].shape[1:])
        self.action_space = gym.spaces.Discrete(number_of_actions)
    
    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        '''
        AI Gym's step function.
        Executes the agent's action and propels the simulation.
        
        Parameters
        ----------
        action :                            The action selected by the agent (will be ignored).\n
        
        Returns
        ----------
        observation :                       The observation of the new current state.\n
        reward :                            The reward received.\n
        end_trial :                         Flag indicating whether the trial ended.\n
        logs :                              The (empty) logs dictionary.\n
        '''
        # retrieve reward/reinforcement
        self.current_observation.fill(0)
        reward = self.trials[self.current_trial][self.current_step]['reward']
        # update trial step
        self.current_step += 1
        # determine this was the last trial step
        end_trial = len(self.trials[self.current_trial]) == self.current_step
        # retrieve observation only if a next step exists 
        if not end_trial:
            self.current_observation = np.copy(self.observations[self.trials[self.current_trial][self.current_step]['observation']])
        # increment current trial here to prevent redundant reset calls from doing so
        self.current_trial += end_trial
        
        return np.copy(self.current_observation), reward, end_trial, {}
         
    def reset(self) -> np.ndarray:
        '''
        AI Gym's reset function.
        Resets the environment and the agent's state.
        
        Parameters
        ----------
        None\n
        
        Returns
        ----------
        observation :                       The observation of the new current state.\n
        '''
        self.current_observation = np.copy(self.observations[self.trials[self.current_trial][0]['observation']])
        self.current_step = 0
        
        return np.copy(self.current_observation)
