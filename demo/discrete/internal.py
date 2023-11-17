# basic imports
import numpy as np
# framework imports
from cobel.interfaces.rl_interface import AbstractInterface
from cobel.networks.network import AbstractNetwork


class InternalModel(AbstractInterface):
    
    def __init__(self, number_of_actions: int, initial_observations: np.ndarray, model_transition: AbstractNetwork,
                 model_reward: AbstractNetwork, model_terminal: AbstractNetwork, capacity: int = 10000, batch_size: int = 32, epochs: int = 1):
        '''
        Interface for the environmental model of the agent.
        Keeps track of the most recent experiences and learns a model of the environment.
        
        Parameters
        ----------
        rl_parent :                         Reference to the RL agent.
        window :                            Window length of the running batch.
        
        Returns
        ----------
        None
        '''
        self.number_of_actions = number_of_actions
        self.model_transition = model_transition
        self.model_reward = model_reward
        self.model_terminal = model_terminal
        # possible initial observations
        self.initial_observations = initial_observations
        # number of recent experiences stored
        self.capacity = capacity
        self.batch_size = batch_size
        # memory structures
        self.states, self.actions, self.rewards, self.next_states, self.terminals = [], [], [], [], []
        # current simulated observation
        self.observation = np.array([self.initial_observations[np.random.randint(self.initial_observations.shape[0])]])
        # the threshold at which a state is considered terminal
        self.terminal_threshold = .5
        # the number of epochs that the environmental model is trained for on each step
        self.epochs = epochs
        
    def update(self, experience: dict):
        '''
        The following function stores the experience contained in the logs dictionary
        and trains the environmental model on the most recent experiences.
        
        Parameters
        ----------
        logs :                              A dictionary containing logs of the simulation.
        
        Returns
        ----------
        None
        '''
        # leep track of running batch
        self.states.append(experience['state'])
        action = np.zeros(self.number_of_actions)
        action[experience['action']] = 1.
        self.actions.append(action)
        self.rewards.append(experience['reward'])
        self.next_states.append(experience['next_state'])
        self.terminals.append(float(not experience['terminal']))
        # ensure capacity limit
        if len(self.states) > self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.terminals.pop(0)
        # update environmental models
        for epoch in range(self.epochs):
            idx = np.random.choice(np.arange(len(self.states)), self.batch_size)
            self.model_transition.train_on_batch([np.array(self.states)[idx], np.array(self.actions)[idx]], np.array(self.next_states)[idx])
            self.model_reward.train_on_batch(np.array(self.next_states)[idx], np.array(self.rewards)[idx])
            self.model_terminal.train_on_batch(np.array(self.next_states)[idx], np.array(self.terminals)[idx])
            
    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        '''
        The interal model's step function.
        Experiences are simulated using the environmental model.
        
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
        # one-hot encoding of the selected action
        action_enc = np.zeros((1, self.number_of_actions))
        action_enc[0, action] = 1.
        # simulate experience
        self.observation = self.model_transition.predict_on_batch([self.observation, action_enc])
        reward = self.model_reward.predict_on_batch(self.observation)[0][0]
        terminal = self.model_terminal.predict_on_batch(self.observation)[0][0]
        
        return self.observation[0], reward, terminal > self.terminal_threshold, {}
    
    def reset(self):
        '''
        The internal model's reset function which resets the intermodal to one of the possible initial observations.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        observation :                       The observation of the new current state.
        '''
        self.observation = np.array([self.initial_observations[np.random.randint(self.initial_observations.shape[0])]])
        
        return self.observation[0]
    