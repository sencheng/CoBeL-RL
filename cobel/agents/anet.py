# basic imports
import numpy as np
# CoBeL-RL imports
from cobel.agents.rl_agent import AbstractRLAgent
from cobel.interfaces.rl_interface import AbstractInterface


class AssociativeNetworkAgent(AbstractRLAgent):
    
    def __init__(self, interface_OAI: AbstractInterface, custom_callbacks: None | dict = None):
        '''
        This class implements the associative net model from Donoso et al. (2021):
        https://doi.org/10.1007/s10071-021-01521-4
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        custom_callbacks :                  The custom callbacks defined by the user.
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, custom_callbacks=custom_callbacks)
        self.input_size = np.product(self.interface_OAI.observation_space.shape)
        # initialize weights
        self.weights = {'excitatory': np.zeros((self.input_size, self.number_of_actions - 1)),
                        'inhibitory': np.zeros((self.input_size, self.number_of_actions - 1))}
        # define saturation values for all weights
        self.saturation = {'excitatory': np.zeros((self.input_size, self.number_of_actions - 1)),
                           'inhibitory': np.zeros((self.input_size, self.number_of_actions - 1))}
        # define learning rates for all weights
        self.learning_rates = {'excitatory': np.zeros((self.input_size, self.number_of_actions - 1)),
                               'inhibitory': np.zeros((self.input_size, self.number_of_actions - 1))}
        # if true, saturation is ignored when updating the weights
        self.linear_update = False
        # noise strength
        self.noise_amplitude = 1
        # scales the weight update? wasn't mentioned in the paper
        self.alpha = 1
        self.d_alpha = 0.
        # the session trial
        self.current_trial = 0
        self.stop = False
        
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 100):
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
        for trial in range(number_of_trials):
            # prepare trial log
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                # determine next action
                action = self.select_action(state)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # update Q-function with experience when a response was given
                if reward != 0:
                    self.update_Q(experience)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                step_end_log = logs.copy()
                step_end_log.update(experience)
                self.engaged_callbacks.on_step_end(step_end_log)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            if self.stop:
                break
            
    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 100):
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
        for trial in range(number_of_trials):
            # prepare trial log
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                # determine next action
                action = self.select_action(state)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                step_end_log = logs.copy()
                step_end_log.update(experience)
                self.engaged_callbacks.on_step_end(step_end_log)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
        
    def rescale_weights(self, factor: dict):
        '''
        This function is rescales the excitatory and inhibitory weights of the network.
        
        Parameters
        ----------
        factor :                            A dictionary containing the rescaling factors for excitatory and inhibitory weights.
        
        Returns
        ----------
        None
        '''
        for weight in self.weights:
            self.weights[weight] *= factor[weight]
            
    def retrieve_Q(self, observation: np.ndarray) -> np.ndarray:
        '''
        This function predicts Q-values for a given observation.
        
        Parameters
        ----------
        observation :                       The observation for which Q-values should be retrieved.
        
        Returns
        ----------
        q_values :                          The predicted Q-values.
        '''
        noise = self.noise_amplitude * np.random.rand(self.number_of_actions - 1)
        
        return np.matmul(observation, self.weights['excitatory']) - np.matmul(observation, self.weights['inhibitory']) + noise
            
    def predict_on_batch(self, batch: np.ndarray) -> np.ndarray:
        '''
        This function predicts Q-values for a batch of observations.
        
        Parameters
        ----------
        batch :                             The batch of observations for which Q-values should be predicted.
        
        Returns
        ----------
        predictions :                       The batch of Q-value predictions.
        '''
        predictions = []
        for observation in batch:
            predictions.append(self.retrieve_Q(observation))
            
        return np.array(predictions)
            
    def select_action(self, observation: np.ndarray) -> int:
        '''
        This function selects an action according to the Q-values of the current observation.
        
        Parameters
        ----------
        observation :                       The current observation.
        
        Returns
        ----------
        action :                            The selected action.
        '''

        action = 2
        q = self.retrieve_Q(observation)
        if np.amax(q) > 0:
            action = np.argmax(q)
            
        return action
    
    def update_Q(self, experience: dict) -> np.ndarray:
        '''
        This function updates the Q-function with a given experience.
        
        Parameters
        ----------
        experience :                        The experience with which the Q-function will be updated.
        
        Returns
        ----------
        None
        '''
        # compute weight update mask
        action_vector = (np.arange(2).astype(int) == experience['action']).astype(int)
        update_mask = np.outer((experience['state'] != 0).astype(int), action_vector)
        # strengthen excitatory weights for correct response, inhibitory weights for incorrect response
        weight = 'excitatory' if experience['reward'] > 0 else 'inhibitory'
        delta = self.alpha * (self.saturation[weight] - self.weights[weight])
        if self.linear_update:
            delta.fill(1.)
        self.weights[weight] += self.learning_rates[weight] * delta * update_mask                
        
        
class MultiContextANetAgent(AssociativeNetworkAgent):
    
    def __init__(self, interface_OAI: AbstractInterface, number_of_stimuli: int, number_of_contexts: int, custom_callbacks: None | dict = None):
        '''
        This class implements an extended version of the associative net model from Donoso et al. (2021):
        https://doi.org/10.1007/s10071-021-01521-4
        The extended model expects input in the form of [left stimulus, right stimulus, contextual cues] (one-hot encoded).
        Weights are shared between left and right stimulus.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        number_of_stimuli :                 The number of stimuli that will be presented to the agent.
        number_of_contexts :                The number of contexts that will be presented to the agent.
        custom_callbacks :                  The custom callbacks defined by the user.
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, custom_callbacks=custom_callbacks)
        self.input_size = np.product(self.interface_OAI.observation_space.shape)
        self.number_of_stimuli = number_of_stimuli
        self.number_of_contexts = number_of_contexts
        assert self.input_size == 2 * self.number_of_stimuli + self.number_of_contexts
        # define saturation values for all weights
        self.saturation = {'excitatory': 0, 'inhibitory': 0}
        # define learning rates for all weights
        self.learning_rates = {'excitatory': 0, 'inhibitory': 0}
    
    def update_Q(self, experience: dict) -> np.ndarray:
        '''
        This function updates the Q-function with a given experience.
        
        Parameters
        ----------
        experience :                        The experience with which the Q-function will be updated.
        
        Returns
        ----------
        None
        '''
        # to achieve weight sharing we duplicate the relevant stimulus for left and right
        input_vector = np.copy(experience['state'])
        start = self.number_of_stimuli * experience['action']
        end = start + self.number_of_stimuli
        input_vector[:(self.number_of_stimuli * 2)] = np.tile(input_vector[start:end], 2)
        # compute weight update mask
        action_vector = (np.arange(2).astype(int) == experience['action']).astype(int)
        update_mask = np.outer((input_vector != 0).astype(int), action_vector)
        # strengthen excitatory weights for correct response, inhibitory weights for incorrect response
        weight = 'excitatory' if experience['reward'] > 0 else 'inhibitory'
        delta = self.alpha * (self.saturation[weight] - self.weights[weight]) * input_vector.reshape(self.input_size, 1)
        if self.linear_update:
            delta.fill(1.)
        self.weights[weight] += self.learning_rates[weight] * delta * update_mask
        