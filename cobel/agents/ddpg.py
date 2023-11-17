# basic imports
import numpy as np
# framework imports
from cobel.agents.rl_agent import AbstractRLAgent
from cobel.policy.policy import AbstractPolicy
from cobel.interfaces.rl_interface import AbstractInterface
from cobel.networks.network import AbstractNetwork
from cobel.memory_modules.dqn_memory import SimpleMemory


class DDPGAgent(AbstractRLAgent):

    def __init__(self, interface_OAI: AbstractInterface, policy: AbstractPolicy, policy_test: None | AbstractPolicy = None,
                 gamma: float = 0.99, model: None | AbstractNetwork = None, custom_callbacks: None | dict = None):
        '''
        This class implements a simple deep Q-network. 
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.\n
        policy :                            The agent's action selection policy.\n
        policy_test :                       The agent's action selection policy during testing. If unspecified the agent uses the train policy.\n
        gamma :                             The discount factor used for computing the target values.\n
        model :                             The network model to be used by the agent.\n
        custom_callbacks :                  The custom callbacks defined by the user.\n
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, custom_callbacks=custom_callbacks)
        # build target and online models
        self.model_target = model
        self.model_online = self.model_target.clone_model()
        # memory module
        self.M = SimpleMemory(10 ** 6)
        # the rate at which the target model is updated (for values < 1 the target model is blended with the online model)
        self.target_model_update = 10 ** -2
        # count the steps since the last update of the target model
        self.steps_since_last_update = 0
        # the current trial across training and test sessions
        self.current_trial = 0
        # action selection policies
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        # the discount factor
        self.gamma = gamma
        self.stop = False
        
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 50, replay_batch_size: int = 100, no_replay: bool = False):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials that the agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        replay_batch_size :                 The number of random experiences that will be replayed.\n
        no_replay :                         If true, experiences are not replayed.\n
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                self.steps_since_last_update += 1
                # determine next action
                action = self.policy.select_action(self.retrieve_Q(state))
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # store experience
                self.M.store(experience)
                # update current state
                state = next_state
                # perform experience replay
                if not no_replay:
                    self.replay(replay_batch_size)
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            if self.stop:
                break
            
    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 50):
        '''
        This function is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials that the agent is tested.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                # determine next action
                action = self.policy.select_action(self.retrieve_Q(state))
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            if self.stop:
                break
        
    def replay(self, replay_batch_size: int = 200):
        '''
        This function replay experiences to update the Q-function.
        
        Parameters
        ----------
        replay_batch_size :                 The number of random experiences that will be replayed.\n
        
        Returns
        ----------
        None
        '''
        # sample random batch of experiences
        states, actions, rewards, next_states, terminals = self.M.retrieve_batch(replay_batch_size)
        bootstrap_actions, bootstrap_q = self.model_target.predict_on_batch([next_states, np.zeros((replay_batch_size, 0))])
        targets_q = rewards.reshape((rewards.shape[0], 1)) + self.gamma * bootstrap_q * terminals.reshape((rewards.shape[0], 1))
        # update online models
        self.model_online.train_on_batch([states, actions], targets_q)
        self.model_online.train_on_batch([states, actions], np.zeros((replay_batch_size, 0)))        
        # update target models
        if self.target_model_update < 1.:
            weights_target = np.array(self.model_target.get_weights(), dtype=object)
            weights_online = np.array(self.model_online.get_weights(), dtype=object)
            weights_target += self.target_model_update * (weights_online - weights_target)
            self.model_target.set_weights(weights_target)
        elif self.steps_since_last_update >= self.target_model_update:
            self.model_target.set_weights(self.model_online.get_weights())
            self.steps_since_last_update = 0
    
    def retrieve_Q(self, state: np.ndarray) -> np.ndarray:
        '''
        This function retrieves Q-values for a given state.
        
        Parameters
        ----------
        state :                             The state for which Q-values should be retrieved.\n
        
        Returns
        ----------
        q_values :                          The state's Q-values.\n
        '''
        return self.model_online.predict_on_batch([np.array([state]), np.zeros((1, self.number_of_actions))])[0][0]
    
    def predict_on_batch(self, batch: np.ndarray) -> np.ndarray:
        '''
        This function retrieves Q-values for a batch of states.
        
        Parameters
        ----------
        batch :                             The batch of states for which Q-values should be retrieved.\n
        
        Returns
        ----------
        predictions :                       The batch of Q-value predictions.\n
        '''
        return self.model_online.predict_on_batch([batch, np.zeros((batch.shape[0], self.number_of_actions))])[0]
    