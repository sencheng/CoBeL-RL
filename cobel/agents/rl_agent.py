# basic imports
import numpy as np
# framework imports
from cobel.interfaces.rl_interface import AbstractInterface


class AbstractRLAgent():
    
    def __init__(self, interface_OAI: AbstractInterface, custom_callbacks: None | dict = None):
        '''
        Abstract class of an RL agent.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        custom_callbacks :                  The custom callbacks defined by the user.
        
        Returns
        ----------
        None
        '''
        # store the Open AI Gym interface
        self.interface_OAI = interface_OAI
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.number_of_actions = self.interface_OAI.action_space.n
        # initialze callbacks class with customized callbacks
        self.engaged_callbacks = callbacks(self, custom_callbacks)

                
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 50):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the RL agent is trained.
        max_number_of_steps :               The maximum number of steps per trial.
        
        Returns
        ----------
        None
        '''
        raise NotImplementedError('.train() function not implemented!')
        
    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 50):
        '''
        This function is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the RL agent is tested.
        max_number_of_steps :               The maximum number of steps per trial.
        
        Returns
        ----------
        None
        '''
        raise NotImplementedError('.test() function not implemented!')
        
    def predict_on_batch(self, batch: np.ndarray) -> np.ndarray:
        '''
        This function retrieves Q-values for a batch of states/observations.
        
        Parameters
        ----------
        batch :                             The batch of states/observations for which Q-values should be retrieved.
        
        Returns
        ----------
        predictions :                       The batch of Q-value predictions.
        '''
        raise NotImplementedError('.predict_on_batch() function not implemented!')


class callbacks():
    
    def __init__(self, rl_parent: AbstractRLAgent, custom_callbacks: None | dict = None):
        '''
        Callback class. Used for visualization and scenario control.
        
        Parameters
        ----------
        rl_parent :                         Reference to the RL agent.
        custom_callbacks :                  The custom callbacks defined by the user.
        
        Returns
        ----------
        None
        '''
        # store the hosting class
        self.rl_parent = rl_parent
        # store the trial end callback function
        self.custom_callbacks = {} if custom_callbacks is None else custom_callbacks
    
    def on_trial_begin(self, logs: dict):
        '''
        The following function is called whenever a trial begins, and executes callbacks defined by the user.
        
        Parameters
        ----------
        logs :                              The trial log.
        
        Returns
        ----------
        None
        '''
        logs['rl_parent'] = self.rl_parent
        if 'on_trial_begin' in self.custom_callbacks:
            for custom_callback in self.custom_callbacks['on_trial_begin']:
                custom_callback(logs)
                
    def on_trial_end(self, logs: dict):
        '''
        The following function is called whenever a trial ends, and executes callbacks defined by the user.
        
        Parameters
        ----------
        logs :                              The trial log.
        
        Returns
        ----------
        None
        '''
        logs['rl_parent'] = self.rl_parent
        if 'on_trial_end' in self.custom_callbacks:
            for custom_callback in self.custom_callbacks['on_trial_end']:
                custom_callback(logs)
                
    def on_step_begin(self, logs: dict):
        '''
        The following function is called whenever a step begins, and executes callbacks defined by the user.
        
        Parameters
        ----------
        logs :                              The trial log.
        
        Returns
        ----------
        None
        '''
        logs['rl_parent'] = self.rl_parent
        if 'on_step_begin' in self.custom_callbacks:
            for custom_callback in self.custom_callbacks['on_step_begin']:
                custom_callback(logs)
                
    def on_step_end(self, logs: dict):
        '''
        The following function is called whenever a step ends, and executes callbacks defined by the user.
        
        Parameters
        ----------
        logs :                              The trial log.
        
        Returns
        ----------
        None
        '''
        logs['rl_parent'] = self.rl_parent
        if 'on_step_end' in self.custom_callbacks:
            for custom_callback in self.custom_callbacks['on_step_end']:
                custom_callback(logs)
