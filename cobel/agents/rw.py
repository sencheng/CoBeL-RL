# basic imports
import numpy as np
# CoBeL-RL
from cobel.agents.rl_agent import AbstractRLAgent
from cobel.policy.policy import AbstractPolicy
from cobel.interfaces.rl_interface import AbstractInterface


class RescorlaWagnerAgent(AbstractRLAgent):
    
    def __init__(self, interface_OAI: AbstractInterface, learning_rates: float | np.ndarray = 0.1, custom_callbacks: None | dict = None):
        '''
        This class implements an (associative) agent based on the Rescorla-Wagner model.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.\n
        learning_rates :                    The learning rate(s) for the different input cues.\n
        custom_callbacks :                  The custom callbacks defined by the user.\n
        
        Returns
        ----------
        None\n
        '''
        super().__init__(interface_OAI, custom_callbacks)
        assert len(self.interface_OAI.observation_space.shape) == 1, 'Incompatible observation space!'
        self.W = np.zeros(self.interface_OAI.observation_space.shape)
        self.learning_rates = np.ones(self.interface_OAI.observation_space.shape) * learning_rates
        self.current_trial = 0
        
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 100):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials that the agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
        Returns
        ----------
        None\n
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                # determine next action
                action = self.predict_on_batch(np.array([state]))[0]
                # callback
                self.engaged_callbacks.on_step_begin({'observation': state, 'action': action})
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # update weights
                self.W -= self.learning_rates * (action - reward) * state
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                # callback
                self.engaged_callbacks.on_step_end({'observation': state, 'action': action, 'reward': reward})
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            
    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 100):
        '''
        This function is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials that the agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
        Returns
        ----------
        None\n
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                # determine next action
                action = self.predict_on_batch(np.array([state]))[0]
                # callback
                self.engaged_callbacks.on_step_begin({'observation': state, 'action': action})
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                # callback
                self.engaged_callbacks.on_step_end({'observation': state, 'action': action, 'reward': reward})
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            
    def predict_on_batch(self, batch: np.ndarray) -> np.ndarray:
        '''
        This function retrieves values for a batch of observations.
        
        Parameters
        ----------
        batch :                             The batch of observations for which values should be retrieved.\n
        
        Returns
        ----------
        predictions :                       The batch of value predictions.\n
        '''
        return self.W @ batch.T
            

class BinaryRWAgent(RescorlaWagnerAgent):
    
    def __init__(self, interface_OAI: AbstractInterface, policy: AbstractPolicy, policy_test: None | AbstractPolicy = None,
                 learning_rates: float | np.ndarray = 0.1, custom_callbacks: None | dict = None):
        '''
        This class implements an (binary choice) agent based on the Rescorla-Wagner model.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.\n
        policy :                            The agent's action selection policy.\n
        policy_test :                       The agent's action selection policy during testing. If unspecified the agent uses the train policy.\n
        learning_rates :                    The learning rate(s) for the different input cues.\n
        custom_callbacks :                  The custom callbacks defined by the user.\n
        
        Returns
        ----------
        None\n
        '''
        super().__init__(interface_OAI, learning_rates, custom_callbacks)
        assert self.number_of_actions == 2, 'Incompatible action space!'
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 100):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials that the agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
        Returns
        ----------
        None\n
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                # determine next action
                v = self.predict_on_batch(np.array([state]))[0]
                action = self.policy.select_action(v)
                # callback
                self.engaged_callbacks.on_step_begin({'observation': state, 'action': action})
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # update weights
                target = 1. if (callback_value['action'] == 0 and reward > 0) or (callback_value['action'] == 1 and reward < 0) else 0.
                self.W -= self.learning_rates * (v - target) * state
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                # callback
                self.engaged_callbacks.on_step_end({'observation': state, 'action': action, 'reward': reward})
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            
    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 100):
        '''
        This function is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials that the agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
        Returns
        ----------
        None\n
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                # determine next action
                v = self.predict_on_batch(np.array([state]))[0]
                action = self.policy_test.select_action(v)
                # callback
                self.engaged_callbacks.on_step_begin({'observation': state, 'action': action})
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                # callback
                self.engaged_callbacks.on_step_end({'observation': state, 'action': action, 'reward': reward})
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
