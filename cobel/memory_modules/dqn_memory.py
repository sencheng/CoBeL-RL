# basic imports
import numpy as np


class SimpleMemory():
    
    def __init__(self, capacity: int = 100000):
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
            
    def retrieve_batch(self, batch_size: int = 32) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
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


class PrioritizedMemory():
    
    def __init__(self, decay: float = 1, prioritize_RPE: bool = True):
        '''
        Simple memory module for storing data samples. To be used with the Associative DQN agent.
        
        Parameters
        ----------
        decay :                             The decay parameter used to np.sum(self.priorities) sample weightings.
        prioritize_RPE :                    If true, batches are drawn using the prioritized replay method.
        
        Returns
        ----------
        None
        '''
        # initialize memory structures
        self.observations = []
        self.reinforcements = []
        self.errors = []
        self.priorities = np.array([])
        # learning parameters
        self.decay = np.clip(decay, a_min=0, a_max=1)
        self.prioritize_RPE = prioritize_RPE
        
    def store(self, experience: dict):
        '''
        This function stores a given experience.
        
        Parameters
        ----------
        experience :                        The experience to be stored.
        
        Returns
        ----------
        None
        '''
        # store experience
        self.observations.append(experience['state'])
        self.reinforcements.append(experience['reward'])
        self.errors.append(np.abs(experience['action'] - experience['reward']))
        # decay priorities
        self.priorities *= self.decay
        # update priorities
        if self.prioritize_RPE:
            self.priorities = np.append(self.priorities, np.array([self.errors[-1]]))
        else:
            self.priorities = np.append(self.priorities, np.array([1]))
        
    def sample_batch(self, batch_size: int) -> (np.ndarray, np.ndarray):
        '''
        This function samples a random batch of experiences.
        
        Parameters
        ----------
        batch_size :                        The size of the batch.
        
        Returns
        ----------
        observations :                      The observation batch.
        reinforcements :                    The reinforcement batch.
        '''
        # draw mini batch using prioritized replay method
        probs = np.ones(len(self.observations))/len(self.observations)
        prob_sum = np.sum(self.priorities)
        if prob_sum != 0:
            probs = self.priorities/prob_sum
        idx = np.random.choice(len(self.observations), p=probs, size=batch_size, replace=True)
            
        return np.array(self.observations)[idx], np.array(self.reinforcements)[idx]
