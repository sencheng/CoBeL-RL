# basic imports
import numpy as np
# framework imports
from cobel.agents.rl_agent import callbacks
from cobel.agents.dyna_q import AbstractDynaQAgent
from cobel.policy.policy import AbstractPolicy
from cobel.interfaces.rl_interface import AbstractInterface
from cobel.memory_modules.sfma_memory import SFMAMemory
from cobel.networks.network import AbstractNetwork

    
class SFMAAgent(AbstractDynaQAgent):
    
    class callbacksSFMA(callbacks):
        
        def __init__(self, rl_parent: AbstractDynaQAgent, custom_callbacks: None | dict = None):
            '''
            Callback class. Used for visualization and scenario control.
            
            Parameters
            ----------
            rl_parent :                         Reference to the RL agent.\n
            custom_callbacks :                  The custom callbacks defined by the user.\n
            
            Returns
            ----------
            None
            '''
            super().__init__(rl_parent, custom_callbacks)
                    
        def on_replay_begin(self, logs: dict):
            '''
            The following function is called whenever experiences are replayed.
            
            Parameters
            ----------
            logs :                              The trial log.\n
            
            Returns
            ----------
            None
            '''
            logs['rl_parent'] = self.rl_parent
            if 'on_replay_begin' in self.custom_callbacks:
                for custom_callback in self.custom_callbacks['on_replay_begin']:
                    custom_callback(logs)
        
        def on_replay_end(self, logs: dict):
            '''
            The following function is called after experiences were replayed.
            
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
        Implementation of a Dyna-Q agent using the Spatial Structure and Frequency-weighted Memory Access (SFMA) memory module.
        Q-function is represented as a static table.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.\n
        policy :                            The agent's action selection policy.\n
        policy_test :                       The agent's action selection policy during testing. If unspecified the agent uses the train policy.\n
        learning_rate :                     The learning rate with which the Q-function is updated.\n
        gamma :                             The discount factor used to compute the TD-error.\n
        gamma_SR :                          The discount factor used by the SMA memory module.\n
        custom_callbacks :                  The custom callbacks defined by the user.\n
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, policy=policy, policy_test=policy_test, learning_rate=learning_rate, gamma=gamma)
        # Q-table
        self.Q = np.zeros((self.number_of_states, self.number_of_actions))
        # memory module
        self.M = SFMAMemory(self.interface_OAI, self.number_of_actions, gamma_SR)
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engaged_callbacks = self.callbacksSFMA(self, custom_callbacks)
        # training
        self.replays_per_trial = 1 # number of replay batches
        self.random_replay = False # if true, random replay batches are sampled
        self.dynamic_mode = False # if true, the replay mode is determined by the cumulative td-error
        self.offline = False # if true, the agent learns only with experience replay
        self.start_replay = False # if true, a replay trace is generated at the start of each trial
        self.td = 0. # stores the temporal difference errors accounted during each trial
        
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 50, replay_batch_size: int = 100, no_replay: bool = False):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the SFMA agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        replay_batch_size :                 The number of experiences that will be replayed.\n
        no_replay :                         If true, experiences are not replayed.\n
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # prepare trial log
            trial_log = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial, 'steps': 0, 'replay_mode': self.M.mode}
            # callback
            self.engaged_callbacks.on_trial_begin(trial_log)
            # reset environment
            state = self.interface_OAI.reset()
            if self.start_replay:
                self.engaged_callbacks.on_replay_begin(trial_log)
                trial_log['replay'] = self.M.replay(replay_batch_size, state)
                self.engaged_callbacks.on_replay_end(trial_log)
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(trial_log)
                # determine next action
                action = self.policy.select_action(self.retrieve_Q(state), self.action_mask[state] if self.mask_actions else None)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # update Q-function with experience
                self.update_Q(experience, self.offline)
                # store experience
                self.M.store(experience)
                # log reward
                trial_log['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(trial_log)
                # update current state
                state = next_state
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            # log steps
            trial_log['steps'] = step
            # perform experience replay
            if not no_replay:
                # determine replay mode if modes are chosen dynamically
                if self.dynamic_mode:
                    p_mode = 1 / (1 + np.exp(-(self.td * 5 - 2)))
                    trial_log['replay_mode'] = ['reverse', 'default'][np.random.choice(np.arange(2), p=np.array([p_mode, 1 - p_mode]))]
                    self.M.mode = trial_log['replay_mode']
                    self.td = 0.
                # replay
                for i in range(self.replays_per_trial):
                    self.engaged_callbacks.on_replay_begin(trial_log)
                    trial_log['replay'] = self.replay(replay_batch_size, next_state)
                    self.engaged_callbacks.on_replay_end(trial_log)
                self.M.T.fill(0)
            # callback
            self.engaged_callbacks.on_trial_end(trial_log)
            if self.stop:
                break
            
    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 50):
        '''
        This function is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the SFMA agent is tested.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
           # prepare trial log
            trial_log = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial, 'steps': 0, 'replay_mode': self.M.mode}
            # callback
            self.engaged_callbacks.on_trial_begin(trial_log)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(trial_log)
                # determine next action
                action = self.policy_test.select_action(self.retrieve_Q(state), self.action_mask[state] if self.mask_actions else None)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # log reward
                trial_log['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(trial_log)
                # update current state
                state = next_state
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            # log steps
            trial_log['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(trial_log)
            if self.stop:
                break
            
    def replay(self, replay_batch_size: int = 200, state: None | int = None) -> list:
        '''
        This function replays experiences to update the Q-function.
        
        Parameters
        ----------
        replay_batch_size :                 The number of experiences that will be replayed.\n
        state :                             The state at which replay should be initiated.\n
        
        Returns
        ----------
        replay_batch :                      The batch of replayed experiences.\n
        '''
        # sample batch of experiences
        replay_batch = []
        if self.random_replay:
            mask = np.ones((self.number_of_states * self.number_of_actions))
            if self.mask_actions:
                mask = np.copy(self.action_mask).flatten(order='F')
            replay_batch = self.M.retrieve_random_batch(replay_batch_size, mask)
        else:
            replay_batch = self.M.replay(replay_batch_size, state)
        # update the Q-function with each experience
        for experience in replay_batch:
            self.update_Q(experience)
            
        return replay_batch
    
    def update_Q(self, experience: dict, no_update: bool = False) -> float:
        '''
        This function updates the Q-function with a given experience.
        
        Parameters
        ----------
        experience :                        The experience with which the Q-function will be updated.\n
        no_update :                         If true, the Q-function is not updated.\n
        
        Returns
        ----------
        td :                                The update's TD-error.\n
        '''
        # make mask
        mask = np.arange(self.number_of_actions)
        if self.mask_actions:
            mask = self.action_mask[experience['next_state']]
        # compute TD-error
        td = experience['reward']
        td += self.gamma * experience['terminal'] * np.amax(self.retrieve_Q(experience['next_state'])[mask])
        td -= self.retrieve_Q(experience['state'])[experience['action']]
        # update Q-function with TD-error
        if not no_update:
            self.Q[experience['state']][experience['action']] += self.learning_rate * td
        # store temporal difference error
        experience['error'] = td
        #if self.dynamic_mode:
        self.td += np.abs(td)
        
        return td
    
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


class DeepSFMAAgent(AbstractDynaQAgent):               
            
    def __init__(self, interface_OAI: AbstractInterface, policy: AbstractPolicy, policy_test: None | AbstractPolicy = None,
                 gamma: float = 0.99, gamma_SR: float = 0.99, observations: None | np.ndarray = None,
                 model: None | AbstractNetwork = None, custom_callbacks: None | dict = None):
        '''
        Implementation of a Dyna-Q agent using the Spatial Structure and Frequency-weighted Memory Access (SFMA) memory module.
        Q-function is represented as a deep Q-network (DQN).
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.\n
        policy :                            The agent's action selection policy.\n
        policy_test :                       The agent's action selection policy during testing. If unspecified the agent uses the train policy.\n
        gamma :                             The discount factor used for computing the target values.\n
        gamma_SR :                          The discount factor used by the SMA memory module.\n
        observations :                      The set of observations that will be mapped to environmental states. If undefined, a one-hot encoding for will be generated.\n
        model :                             The network model to be used by the agent.\n
        custom_callbacks :                  The custom callbacks defined by the user.\n
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, policy=policy, policy_test=policy_test, learning_rate=0.9, gamma=gamma, custom_callbacks=custom_callbacks)
        # observations
        self.observations = observations
        if self.observations is None:
            self.observations = np.eye(self.number_of_states)
        # build target and online models
        self.model_target = model
        self.model_online = self.model_target.clone_model()
        # memory module
        self.M = SFMAMemory(self.interface_OAI, self.number_of_actions, gamma_SR)
        # training
        self.replays_per_trial = 1 # number of replay batches
        self.random_replay = False # if true, random replay batches are sampled
        self.dynamic_mode = False # if true, the replay mode is determined by the cumulative td-error
        self.td = 0. # stores the temporal difference errors accounted during each trial
        self.target_model_update = 10 ** -2 # target model blending factor
        self.updates_per_replay = 1 # number of BP updates per replay batch
        self.local_targets = True # if true, the model will be updated with locally computed target values
        self.randomize_subsequent_replays = False # if true, only the first replay after each trial uses SFMA
        
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 50, replay_batch_size: int = 100, no_replay: bool = False):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the Deep SFMA agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        replay_batch_size :                 The number of experiences that will be replayed.\n
        no_replay :                         If true, experiences are not replayed.\n
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # prepare trial log
            trial_log = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial, 'steps': 0, 'replay_mode': self.M.mode}
            # callback
            self.engaged_callbacks.on_trial_begin(trial_log)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(trial_log)
                # determine next action
                action = self.policy.select_action(self.retrieve_Q(state), self.action_mask[state] if self.mask_actions else None)
                # perform action
                next_state, reward, stop_episode, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - stop_episode)}
                # store experience
                self.M.store(experience)
                # log behavior and reward
                trial_log['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(trial_log)
                # update current state
                state = next_state
                # stop trial when the terminal state is reached
                if stop_episode:
                    break
            self.current_trial += 1
            # log steps
            trial_log['steps'] = step
            # perform experience replay
            if not no_replay:
                # determine replay mode if modes are chosen dynamically
                if self.dynamic_mode:
                    p_mode = 1 / (1 + np.exp(-(self.td * 5 - 2)))
                    self.M.mode = ['reverse', 'default'][np.random.choice(np.arange(2), p=np.array([p_mode, 1 - p_mode]))]
                    self.td = 0.
                # replay
                replays = []
                for i in range(self.replays_per_trial):
                    if i > 0 and self.randomize_subsequent_replays:
                        replays.append(self.replay(replay_batch_size, next_state, True))
                    else:
                        replays.append(self.replay(replay_batch_size, next_state, self.random_replay))
                # update model with generated replay
                if self.local_targets:
                    self.update_local(replays)
                else:
                    self.update_step_wise(replays)
                self.M.T.fill(0)
            # callback
            self.engaged_callbacks.on_trial_end(trial_log)
            if self.stop:
                break
            
    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 50):
        '''
        This function is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the Deep SFMA agent is tested.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # prepare trial log
            trial_log = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial, 'steps': 0, 'replay_mode': self.M.mode}
            # callback
            self.engaged_callbacks.on_trial_begin(trial_log)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(trial_log)
                # determine next action
                action = self.policy_test.select_action(self.retrieve_Q(state), self.action_mask[state] if self.mask_actions else None)
                # perform action
                next_state, reward, stop_episode, callback_value = self.interface_OAI.step(action)
                # log behavior and reward
                trial_log['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(trial_log)
                # update current state
                state = next_state
                # stop trial when the terminal state is reached
                if stop_episode:
                    break
            self.current_trial += 1
            # log steps
            trial_log['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(trial_log)
            if self.stop:
                break
            
    def replay(self, replay_batch_size: int = 200, state: None | int = None, random_replay: bool = False) -> list:
        '''
        This function replays experiences to update the Q-function.
        
        Parameters
        ----------
        replay_batch_size :                 The number of experiences that will be replayed.\n
        state :                             The  state at which replay should be initiated.\n
        random_replay :                     If true, than a batch of random experiences is replayed.\n
        
        Returns
        ----------
        replay_batch :                      The batch of replayed experiences.\n
        '''
        # sample batch of experiences
        replay_batch = []
        if random_replay:
            mask = np.ones((self.number_of_states * self.number_of_actions))
            if self.mask_actions:
                mask = np.copy(self.action_mask).flatten(order='F')
            replay_batch = self.M.retrieve_random_batch(replay_batch_size, mask)
        else:
            replay_batch = self.M.replay(replay_batch_size, state)
            
        return replay_batch
    
    def update_local(self, replays: list):
        '''
        This function locally computes new target values to update the model.
        
        Parameters
        ----------
        replays :                           The replays with which the model will be updated.\n
        
        Returns
        ----------
        None
        '''
        for replay in replays:
            # compute local Q-function
            Q_local = self.model_target.predict_on_batch(self.observations)
            # update local Q-function
            batch_states = []
            for experience in replay:
                batch_states.append(experience['state'])
                # prepare action mask
                mask = np.ones(self.number_of_actions).astype(bool)
                if self.mask_actions:
                    mask = self.action_mask[experience['next_state']]
                # compute target
                target = experience['reward']
                target += self.gamma * experience['terminal'] * np.amax(Q_local[experience['next_state']][mask])
                # update local Q-function
                Q_local[experience['state']][experience['action']] = target
            # update model
            self.update_model(self.observations[batch_states], Q_local[batch_states], self.updates_per_replay)
            
    def update_step_wise(self, replays: list):
        '''
        This function updates the model step-wise on all sequences.
        
        Parameters
        ----------
        replays :                           The replays with which the model will be updated.\n
        
        Returns
        ----------
        None
        '''
        # update step-wise
        for step in range(len(replays[0])):
            states, targets = [], []
            Q_local = self.model_target.predict_on_batch(self.observations)
            for replay in replays:
                # recover variables from experience
                states.append(replay[step]['state'])
                reward, action = replay[step]['reward'], replay[step]['action']
                terminal, next_state = replay[step]['terminal'], replay[step]['next_state']
                # prepare action mask
                mask = np.ones(self.number_of_actions).astype(bool)
                if self.mask_actions:
                    mask = self.action_mask[next_state]
                # compute target
                target = Q_local[replay[step]['state']]
                target[action] = reward + self.gamma * terminal * np.amax(Q_local[next_state][mask])
                targets.append(target)
            # update model
            self.update_model(self.observations[states], np.array(targets), self.updates_per_replay)
            
    def update_model(self, observations: np.ndarray, targets: np.ndarray, number_of_updates: int = 1):
        '''
        This function updates the model on a batch of experiences.
        
        Parameters
        ----------
        observations :                      The observations.\n
        targets :                           The targets.\n
        number_of_updates :                 The number of backpropagation updates that should be performed on this batch.\n
        
        Returns
        ----------
        None
        '''
        # update online model
        for update in range(number_of_updates):
            self.model_online.train_on_batch(observations, targets)
        # update target model by blending it with the online model
        weights_target = np.array(self.model_target.get_weights(), dtype=object)
        weights_online = np.array(self.model_online.get_weights(), dtype=object)
        weights_target += self.target_model_update * (weights_online - weights_target)
        self.model_target.set_weights(weights_target)
    
    def update_Q(self, experience: dict):
        '''
        This function is a dummy function and does nothing (implementation required by parent class).
        
        Parameters
        ----------
        experience :                        The experience with which the Q-function will be updated.\n
        
        Returns
        ----------
        None
        '''
        pass
    
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
        # retrieve Q-values, if entry exists
        return self.model_online.predict_on_batch(np.array([self.observations[state]]))[0]
    
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
        return self.model_online.predict_on_batch(self.observations[batch])
