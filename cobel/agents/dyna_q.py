# basic imports
import numpy as np
# memory module
from cobel.agents.rl_agent import AbstractRLAgent, callbacks
from cobel.policy.policy import AbstractPolicy
from cobel.interfaces.rl_interface import AbstractInterface
from cobel.memory_modules.dyna_q_memory import DynaQMemory, PMAMemory


class AbstractDynaQAgent(AbstractRLAgent):
            
    def __init__(self, interface_OAI: AbstractInterface, policy: AbstractPolicy, policy_test: None | AbstractPolicy = None,
                 learning_rate: float = 0.9, gamma: float = 0.99, custom_callbacks: None | dict = None):
        '''
        Implementation of an abstract Dyna-Q agent class.
        The Q-function is represented as a static table.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.\n
        policy :                            The agent's action selection policy.\n
        policy_test :                       The agent's action selection policy during testing. If unspecified the agent uses the train policy.\n
        learning_rate :                     The learning rate with which the Q-function is updated.\n
        gamma :                             The discount factor used for computing the target values.\n
        custom_callbacks :                  The custom callbacks defined by the user.\n
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, custom_callbacks=custom_callbacks)
        # the number of discrete states, retrieved from the Open AI Gym interface
        self.number_of_states = self.interface_OAI.world['states']
        # Q-learning parameters
        self.gamma = gamma
        self.learning_rate = learning_rate
        # action selection policy
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        # keeps track of current trial
        self.current_trial = 0 # trial count across all sessions (i.e. calls to the train/simulate method)
        # mask invalid actions?
        self.mask_actions = False
        self.compute_action_mask()
        self.stop = False
        
    def replay(self, replay_batch_size: int = 200):
        '''
        This function replays experiences to update the Q-function.
        
        Parameters
        ----------
        replay_batch_size :                 The number of random experiences that will be replayed.\n
        
        Returns
        ----------
        None
        '''
        # sample random batch of experiences
        replay_batch = self.M.retrieve_batch(replay_batch_size)
        # update the Q-function with each experience
        for experience in replay_batch:
            self.update_Q(experience)
        
    def compute_action_mask(self):
        '''
        This function computes the action mask which prevents the selection of invalid actions.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # retrieve number of states and actions
        s, a = self.interface_OAI.world['states'], self.number_of_actions
        # determine follow-up states
        self.action_mask = self.interface_OAI.world['sas'].reshape((s * a, s), order='F')
        self.action_mask = np.argmax(self.action_mask, axis=1)
        # make action mask
        self.action_mask = (self.action_mask !=  np.tile(np.arange(s), a)).reshape((s, a), order='F')
        
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 50, replay_batch_size: int = 100, no_replay: bool = False):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the Dyna-Q agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        replay_batch_size :                 The number of random experiences that will be replayed.\n
        no_replay :                         If true, experiences are not replayed.\n
        
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
        number_of_trials :                  The number of trials the Dyna-Q agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
        Returns
        ----------
        None
        '''
        raise NotImplementedError('.test() function not implemented!')
    
    def update_Q(self, experience: dict):
        '''
        This function updates the Q-function with a given experience.
        
        Parameters
        ----------
        experience :                        The experience with which the Q-function will be updated.\n
        
        Returns
        ----------
        None
        '''
        raise NotImplementedError('.update_Q() function not implemented!')
    
    def retrieve_Q(self, state: int) -> np.ndarray:
        '''
        This function retrieves Q-values for a given state.
        
        Parameters
        ----------
        state :                             The state for which Q-values should be retrieved.\n
        
        Returns
        ----------
        q_values :                          The state's Q-values.\n
        '''
        raise NotImplementedError('.retrieve_Q() function not implemented!')
    
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
        raise NotImplementedError('.predict_on_batch() function not implemented!')
        

class DynaQAgent(AbstractDynaQAgent):
            
    def __init__(self, interface_OAI: AbstractInterface, policy: AbstractPolicy, policy_test: None | AbstractPolicy = None,
                 learning_rate: float = 0.9, gamma: float = 0.99, custom_callbacks: None | dict = None):
        '''
        Implementation of a Dyna-Q agent.
        The Q-function is represented as a static table.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.\n
        policy :                            The agent's action selection policy.\n
        policy_test :                       The agent's action selection policy during testing. If unspecified the agent uses the train policy.\n
        learning_rate :                     The learning rate with which the Q-function is updated.\n
        gamma :                             The discount factor used for computing the target values.\n
        custom_callbacks :                  The custom callbacks defined by the user.\n
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, policy=policy, policy_test=policy_test, learning_rate=learning_rate, gamma=gamma, custom_callbacks=custom_callbacks)
        # Q-table
        self.Q = np.zeros((self.number_of_states, self.number_of_actions))
        # memory module
        self.M = DynaQMemory(self.number_of_states, self.number_of_actions)
        # perform replay at the end of an episode instead of each step
        self.episodic_replay = False
        
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 50, replay_batch_size: int = 100, no_replay: bool = False):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the Dyna-Q agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        replay_batch_size :                 The number of random experiences that will be replayed.\n
        no_replay :                         If true, experiences are not replayed.\n
        
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
                action = self.policy.select_action(self.retrieve_Q(state), self.action_mask[state] if self.mask_actions else None)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # update Q-function with experience
                self.update_Q(experience)
                # store experience
                self.M.store(experience)
                # update current state
                state = next_state
                # perform experience replay
                if not no_replay and not self.episodic_replay:
                    self.replay(replay_batch_size)
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # perform experience replay
            if not no_replay and self.episodic_replay:
                self.replay(replay_batch_size)
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            if self.stop:
                break
            
    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 50):
        '''
        This function is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the Dyna-Q agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
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
                action = self.policy_test.select_action(self.retrieve_Q(state), self.action_mask[state] if self.mask_actions else None)
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
    
    def update_Q(self, experience: dict):
        '''
        This function updates the Q-function with a given experience.
        
        Parameters
        ----------
        experience :                        The experience with which the Q-function will be updated.\n
        
        Returns
        ----------
        None
        '''
        # compute TD-error
        td = experience['reward']
        td += self.gamma * experience['terminal'] * np.amax(self.retrieve_Q(experience['next_state']))
        td -= self.retrieve_Q(experience['state'])[experience['action']]
        # update Q-function with TD-error
        self.Q[experience['state']][experience['action']] += self.learning_rate * td
    
    def retrieve_Q(self, state: int) -> np.ndarray:
        '''
        This function retrieves Q-values for a given state.
        
        Parameters
        ----------
        state :                             The state for which Q-values should be retrieved.\n
        
        Returns
        ----------
        q_values :                          The state's Q-values.\n
        '''
        return self.Q[state]
    
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
        return self.Q[batch]
    
    
class PMAAgent(AbstractDynaQAgent):
    
    class callbacksPMA(callbacks):
        
        def __init__(self, rl_parent: AbstractDynaQAgent, custom_callbacks: None | dict = None):
            '''
            Callback class of the PMA agent. Used for visualization and scenario control.
            
            Parameters
            ----------
            rl_parent :                         Reference to the RL agent.\n
            custom_callbacks :                  The custom callbacks defined by the user.\n
            
            Returns
            ----------
            None
            '''
            # store the hosting class
            self.rl_parent = rl_parent
            # store the trial end callback function
            self.custom_callbacks = {} if custom_callbacks is None else custom_callbacks
                    
        def on_replay_end(self, logs: dict):
            '''
            This function is called on the end of replay.
            
            Parameters
            ----------
            logs :                              The trial log.\n
            
            Returns
            ----------
            None
            '''
            logs['rl_parent'] = self.rl_parent
            if 'on_replay_end' in self.custom_callbacks:
                for custom_callback in self.custom_callbacks['on_replay_end']:
                    custom_callback(logs)
    
    def __init__(self, interface_OAI: AbstractInterface, policy: AbstractPolicy, policy_test: None | AbstractPolicy = None,
                 learning_rate: float = 0.9, gamma: float = 0.99, gamma_SR: float = 0.99, custom_callbacks: None | dict = None):
        '''
        Implementation of a Dyna-Q agent using the Prioritized Memory Access (PMA) method described by Mattar & Daw (2018):
        https://doi.org/10.1038/s41593-018-0232-z
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.\n
        policy :                            The agent's action selection policy.\n
        policy_test :                       The agent's action selection policy during testing. If unspecified the agent uses the train policy.\n
        learning_rate :                     The learning rate with which the Q-function is updated.\n
        gamma :                             The discount factor used for computing the target values.\n
        gamma_SR :                          The discount factor used for computing the successor representation.\n
        custom_callbacks :                  The custom callbacks defined by the user.\n
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, policy=policy, policy_test=policy_test, learning_rate=learning_rate, gamma=gamma, custom_callbacks=custom_callbacks)
        # Q-table
        self.Q = np.zeros((self.number_of_states, self.number_of_actions))
        # memory module
        self.M = PMAMemory(self.interface_OAI, self, self.number_of_states, self.number_of_actions, gamma_SR)
        # initialze callbacks class with customized callbacks
        self.engaged_callbacks = self.callbacksPMA(self, custom_callbacks)
        
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 50, replay_batch_size: int = 100, no_replay: bool = False):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the Dyna-Q agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        replay_batch_size :                 The number of random experiences that will be replayed.\n
        no_replay :                         If true, experiences are not replayed.\n
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # reset environment
            state = self.interface_OAI.reset()
            # log cumulative reward
            logs = {'trial_reward': 0, 'steps': 0, 'step': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # perform experience replay
            if not no_replay:
                logs['replay'] = self.M.replay(replay_batch_size, state)
                self.engaged_callbacks.on_replay_end(logs)
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                # determine next action
                action = self.policy.select_action(self.retrieve_Q(state), self.action_mask[state] if self.mask_actions else None)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # update Q-function with experience
                self.update_Q([experience])
                # store experience
                self.M.store(experience)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'], logs['step'] = step, step
            # perform experience replay
            if not no_replay:
                self.M.update_SR()
                logs['replay'] = self.M.replay(replay_batch_size, next_state)
                self.engaged_callbacks.on_replay_end(logs)
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            
    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 50):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the Dyna-Q agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # reset environment
            state = self.interface_OAI.reset()
            # log cumulative reward
            logs = {'trial_reward': 0, 'steps': 0, 'step': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            for step in range(max_number_of_steps):
                # determine next action
                action = self.policy_test.select_action(self.retrieve_Q(state), self.action_mask[state] if self.mask_actions else None)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'], logs['step'] = step, step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
        
    def update_Q(self, update: list):
        '''
        This function updates the Q-function.
        
        Parameters
        ----------
        update :                            A list containing experiences for an n-step update.\n
        
        Returns
        ----------
        None
        '''
        # expected future value
        future_value = np.amax(self.Q[update[-1]['next_state']]) * update[-1]['terminal']
        for s, step in enumerate(update):
            # sum rewards over remaining trajectory
            R = 0.
            intermediate_terminal = 1
            for following_steps in range(len(update) - s):
                # check for intermediate terminal transitions
                if update[s + following_steps]['terminal'] == 0 and s != len(update) - 1:
                    intermediate_terminal = 0
                    break
                R += update[s + following_steps]['reward'] * (self.gamma ** following_steps)
            # abort in case a terminal transition occurs within the n-step sequence
            if intermediate_terminal == 0:
                break
            # compute TD-error
            td = R + future_value * (self.gamma ** (following_steps + 1))
            td -= self.retrieve_Q(step['state'])[step['action']]
            # update Q-function with TD-error
            self.Q[step['state']][step['action']] += self.learning_rate * td
        
    def retrieve_Q(self, state: int) -> np.ndarray:
        '''
        This function retrieves Q-values for a given state.
        
        Parameters
        ----------
        state :                             The state for which Q-values should be retrieved.\n
        
        Returns
        ----------
        q_values :                          The state's Q-values.\n
        '''
        return self.Q[state]
    
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
        return self.Q[batch]
