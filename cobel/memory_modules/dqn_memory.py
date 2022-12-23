# basic imports
import numpy as np


class SimpleMemory():
    
    def __init__(self, capacity=100000):
        '''
        This class implements a simple memory structure of the storing of experiences. 
        
        Parameters
        ----------
        capacity :                          The memory capacity.
        
        Returns
        ----------
        None
        '''
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []
        
    def store(self, experience: dict):
        '''
        This function stores an experience tuple. 
        
        Parameters
        ----------
        experience :                        The experience to be stored.
        
        Returns
        ----------
        None
        '''
        self.states.append(experience['state'])
        self.actions.append(experience['action'])
        self.rewards.append(experience['reward'])
        self.next_states.append(experience['next_state'])
        self.terminals.append(experience['terminal'])
        # remove the oldest experience when the memory is over capacity
        if len(self.states) > self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.terminals.pop(0)
            
    def retrieve_batch(self, batch_size=32) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        '''
        This function retrieves a random batch of experiences. 
        
        Parameters
        ----------
        batch_size :                        The size of the batch.
        
        Returns
        ----------
        states :                            The batch of current states.
        actions :                           The batch of actions.
        rewards :                           The batch of rewards.
        next_states :                       The batch of next states.
        terminals :                         The batch of terminals.
        '''
        idx = np.random.randint(len(self.states), size=batch_size)
        
        return np.array(self.states)[idx], np.array(self.actions)[idx], np.array(self.rewards)[idx], np.array(self.next_states)[idx], np.array(self.terminals)[idx]
