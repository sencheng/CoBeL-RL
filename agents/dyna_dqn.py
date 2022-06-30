# basic imports
import numpy as np
# keras imports
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
# framework imports
from cobel.agents.dyna_q_agent import AbstractDynaQAgent
from cobel.memory_modules.dyna_q_memory import DynaQMemory


class DynaDQN(AbstractDynaQAgent):
    '''
    Implementation of a DQN agent using the Dyna-Q model.
    This agent uses the Dyna-Q agent's memory module and then maps gridworld states to predefined observations.
    
    | **Args**
    | interface_OAI:                The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learning_rate:                The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | observations:                 The set of observations that will be mapped to the gridworld states.
    | model:                        The DNN model to be used by the agent.
    | custom_callbacks:             The custom callbacks defined by the user.
    '''

    def __init__(self, interface_OAI, epsilon=0.3, beta=5, learning_rate=0.9, gamma=0.99, observations=None, model=None, custom_callbacks={}):
        super().__init__(interface_OAI, epsilon=epsilon, beta=beta, learning_rate=learning_rate, gamma=gamma, custom_callbacks=custom_callbacks)
        # prepare observations
        if observations is None or observations.shape[0] != self.number_of_states:
            # one-hot encoding of states
            self.observations = np.eye(self.number_of_states)
        else:
            self.observations = observations
        # build target and online models
        self.build_model(model)
        self.current_predictions = self.model_online.predict_on_batch(self.observations)
        # memory module
        self.M = DynaQMemory(self.number_of_states, self.number_of_actions)
        # perform replay at the end of an episode instead of each step
        self.episodic_replay = False
        # the rate at which the target model is updated (for values < 1 the target model is blended with the online model)
        self.target_model_update = 10**-2
        # count the steps since the last update of the target model
        self.steps_since_last_update = 0
        
    def build_model(self, model=None):
        '''
        This function builds the DQN's target and online models.
        
        | **Args**
        | model:                        The DNN model to be used by the agent. If None, a small dense DNN is created by default.
        '''
        # build target model
        if model is None:
            self.model_target = Sequential()
            self.model_target.add(Dense(units=64, input_shape=(self.observations.shape[1],), activation='tanh'))
            self.model_target.add(Dense(units=64, activation='tanh'))
            self.model_target.add(Dense(units=self.number_of_actions, activation='linear'))
        else:
            self.model_target = model
        self.model_target.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        # build online model by cloning the target model
        self.model_online = clone_model(self.model_target)
        self.model_online.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        
    def train(self, number_of_trials=100, max_number_of_steps=50, replay_batch_size=100, no_replay=False):
        '''
        This function is called to train the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the Dyna-Q agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
        | replay_batch_size:            The number of random that will be replayed.
        | no_replay:                    If true, experiences are not replayed.
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.steps_since_last_update += 1
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # store experience
                self.M.store(experience)
                # update current state
                state = next_state
                # perform experience replay
                if not no_replay and not self.episodic_replay:
                    self.replay(replay_batch_size)
                # update cumulative reward
                logs['trial_reward'] += reward
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # to save performance store state predictions after each trial only
            self.current_predictions = self.model_online.predict_on_batch(self.observations)
            # perform experience replay
            if not no_replay and self.episodic_replay:
                self.replay(replay_batch_size)
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            
    def test(self, number_of_trials=100, max_number_of_steps=50):
        '''
        This function is called to train the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the Dyna-Q agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
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
                action = self.select_action(state, self.epsilon, self.beta)
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
            logs['steps'] = step
            # to save performance store state predictions after each trial only
            self.current_predictions = self.model_online.predict_on_batch(self.observations)
            # callback
            self.engaged_callbacks.on_trial_end(logs)
        
    def replay(self, replay_batch_size=200):
        '''
        This function replays experiences to update the Q-function.
        
        | **Args**
        | replayBatchSize:              The number of random that will be replayed.
        '''
        # sample random batch of experiences
        replay_batch = self.M.retrieve_batch(replay_batch_size)
        # compute update targets
        states, next_states = np.zeros((replay_batch_size, self.observations.shape[1])), np.zeros((replay_batch_size, self.observations.shape[1]))
        rewards, terminals, actions = np.zeros(replay_batch_size),  np.zeros(replay_batch_size),  np.zeros(replay_batch_size)
        for e, experience in enumerate(replay_batch):
            states[e] = self.observations[experience['state']]
            next_states[e] = self.observations[experience['next_state']]
            rewards[e] = experience['reward']
            actions[e] = experience['action']
            terminals[e] = experience['terminal']
        future_values = np.amax(self.model_target.predict_on_batch(next_states), axis=1) * terminals
        targets = self.model_target.predict_on_batch(states)
        for a, action in enumerate(actions):
            targets[a, int(action)] = rewards[a] + self.gamma * future_values[a]
        # update online model
        self.model_online.train_on_batch(np.array(states), np.array(targets))
        # update target model
        if self.target_model_update < 1.:
            weights_target = np.array(self.model_target.get_weights(), dtype=object)
            weights_online = np.array(self.model_online.get_weights(), dtype=object)
            weights_target += self.target_model_update * (weights_online - weights_target)
            self.model_target.set_weights(weights_target)
        elif self.steps_since_last_update >= self.target_model_update:
            self.model_target.set_weights(self.model_online.get_weights())
            self.steps_since_last_update = 0
        
    
    def update_Q(self, experience):
        '''
        This function is a dummy function and does nothing (implementation required by parent class).
        
        | **Args**
        | experience:                   The experience with which the Q-function will be updated.
        '''
        pass
    
    def retrieve_Q(self, state):
        '''
        This function retrieves Q-values for a given state.
        
        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        return self.model_online.predict_on_batch(np.array([self.observations[state]]))[0]
    
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''
        return self.current_predictions[batch]
    
    
class DynaDSR(AbstractDynaQAgent):
    '''
    Implementation of a Deep Successor Representation agent using the Dyna-Q model.
    This agent uses the Dyna-Q agent's memory module and then maps gridworld states to predefined observations.
    
    | **Args**
    | interface_OAI:                The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learning_Rate:                The learning rate with which the SR is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | observations:                 The set of observations that will be mapped to the gridworld states.
    | model_SR:                     The DNN model to be used by the agent for the successor representation.
    | model_reward:                 The DNN model to be used by the agent for the reward model.
    | custom_callbacks:             The custom callbacks defined by the user.
    '''
           
    def __init__(self, interface_OAI, epsilon=0.3, beta=5, learning_rate=0.9, gamma=0.99, observations=None, model_SR=None, model_reward=None, custom_callbacks={}):
        super().__init__(interface_OAI, epsilon=epsilon, beta=beta, learning_rate=learning_rate, gamma=gamma, custom_callbacks=custom_callbacks)
        # prepare observations
        if observations is None or observations.shape[0] != self.number_of_states:
            # one-hot encoding of states
            self.observations = np.eye(self.number_of_states)
        else:
            self.observations = observations
        # build target and online models
        self.build_models(model_SR, model_reward)
        # used for visualization (currently all-zeros array)        
        self.current_predictions = np.zeros((self.number_of_states, self.number_of_actions))
        # memory module
        self.M = DynaQMemory(self.number_of_states, self.number_of_actions)
        # perform replay at the end of an episode instead of each step
        self.episodic_replay = False
        # the rate at which the target model is updated (for values < 1 the target model is blended with the online model)
        self.target_model_update = 10**-2
        # count the steps since the last update of the target model
        self.steps_since_last_update = 0   
        # compute DR instead of SR
        self.use_Deep_DR = False
        # computes SR based on the follow-up state (i.e. each action stream represents the SR of the follow-up state)
        self.use_follow_up_state = False
        
    def build_models(self, model_SR=None, model_reward=None):
        '''
        This function builds the Dyna-DSR's target and online models.
        
        | **Args**
        | model_SR:                     The DNN model to be used for the SR. If None, a small dense DNN is created by default.
        | model_reward:                 The DNN model to be used for the reward function. If None, a small dense DNN is created by default.
        '''
        # build target and online models for all actions
        self.models_target, self.models_online = {}, {}
        for action in range(self.number_of_actions):
            # build target model
            if model_SR is None:
                self.models_target[action] = Sequential()
                self.models_target[action].add(Dense(units=64, input_shape=(self.observations.shape[1],), activation='tanh'))
                self.models_target[action].add(Dense(units=64, activation='tanh'))
                self.models_target[action].add(Dense(units=self.number_of_states, activation='linear'))
            else:
                self.model_arget[action] = clone_model(model_SR)
            self.models_target[action].compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            # build online model
            self.models_online[action] = clone_model(self.models_target[action])
            self.models_online[action].compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        # build reward model
        if model_reward is None:
            self.model_reward = Sequential()
            self.model_reward.add(Dense(units=64, input_shape=(self.observations.shape[1],), activation='tanh'))
            self.model_reward.add(Dense(units=64, activation='tanh'))
            self.model_reward.add(Dense(units=1, activation='linear'))
        else:
            self.model_reward = model_reward
        self.model_reward.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        
    def train(self, number_of_trials=100, max_number_of_steps=50, replay_batch_size=100, no_replay=False):
        '''
        This function is called to train the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the Dyna-Q agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
        | replay_batch_size:            The number of random that will be replayed.
        | no_replay:                    If true, experiences are not replayed.
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.steps_since_last_update += 1
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # store experience
                self.M.store(experience)
                # update current state
                state = next_state
                # perform experience replay
                if not no_replay and not self.episodic_replay:
                    self.replay(replay_batch_size)
                # update cumulative reward
                logs['trial_reward'] += reward
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # to save performance store state predictions after each trial only
            #self.current_predictions = self.model_online.predict_on_batch(self.observations)
            # perform experience replay
            if not no_replay and self.episodic_replay:
                self.replay(replay_batch_size)
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            
    def test(self, number_of_trials=100, max_number_of_steps=50):
        '''
        This function is called to train the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the Dyna-Q agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.steps_since_last_update += 1
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
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
            logs['steps'] = step
            # to save performance store state predictions after each trial only
            #self.current_predictions = self.model_online.predict_on_batch(self.observations)
            # callback
            self.engaged_callbacks.on_trial_end(logs)
        
    def replay(self, replay_batch_size=200):
        '''
        This function replays experiences to update the DSR and reward function.
        
        | **Args**
        | replay_batch_size:            The number of random that will be replayed.
        '''
        # sample random batch of experiences
        replay_batch = self.M.retrieve_batch(replay_batch_size)
        # recover variables from experiences
        states, next_states = np.zeros((replay_batch_size, self.observations.shape[1])), np.zeros((replay_batch_size, self.observations.shape[1]))
        rewards, terminals, actions = np.zeros(replay_batch_size),  np.zeros(replay_batch_size),  np.zeros(replay_batch_size)
        for e, experience in enumerate(replay_batch):
            states[e] = self.observations[experience['state']]
            next_states[e] = self.observations[experience['next_state']]
            rewards[e] = experience['reward']
            actions[e] = experience['action']
            terminals[e] = experience['terminal']
        # compute the follow-up states' SR streams and values    
        future_SR, future_values = {}, {}
        for action in range(self.number_of_actions):
            future_SR[action] = self.models_target[action].predict_on_batch(next_states)
            future_values[action] = self.model_reward.predict_on_batch(future_SR[action])
        # compute targets
        inputs, targets = {}, {}
        for action in range(self.number_of_actions):
            # filter out experiences irrelevant for this action stream
            idx = (actions == action)
            inputs[action] = states[idx]
            if self.use_follow_up_state:
                targets[action] = next_states[idx]
            else:
                targets[action] = states[idx]
            # compute indices of relevant experiences
            idx = np.arange(len(replay_batch))[idx]
            for i, index in enumerate(idx):
                # Deep SR
                if not self.use_Deep_DR:
                    best = np.argmax(np.array([future_values[action][index] for action in future_values]))
                    targets[action][i] += self.gamma * future_SR[best][index]
                # Deep DR
                else:
                    targets[action][i] += self.gamma * np.mean(np.array([future_SR[stream][index] for stream in future_SR]), axis=0)
        # update online models
        for action in range(self.number_of_actions):
            if inputs[action].shape[0] > 0:
                self.models_online[action].train_on_batch(inputs[action], targets[action])
        # update reward model
        self.model_reward.train_on_batch(np.array(next_states), np.array(rewards))
        # update target models
        if self.target_model_update < 1.:
            for action in range(self.number_of_actions):
                weights_target = np.array(self.models_target[action].get_weights(), dtype=object)
                weights_online = np.array(self.models_online[action].get_weights(), dtype=object)
                weights_target += self.target_model_update * (weights_online - weights_target)
                self.models_target[action].set_weights(weights_target)
        elif self.steps_since_last_update >= self.target_model_update:
            for action in range(self.number_of_actions):
                # copy weights from online model
                self.models_target[action].set_weights(self.model_online[action].get_weights())
            # reset update timer
            self.steps_since_last_update = 0
    
    def update_Q(self, experience):
        '''
        This function is a dummy function and does nothing (implementation required by parent class).
        
        | **Args**
        | experience:                   The experience with which the Q-function will be updated.
        '''
        pass
    
    def retrieve_Q(self, state):
        '''
        This function retrieves Q-values for a given state.
        
        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        Q = []
        for action in range(self.number_of_actions):
            SR = self.models_online[action].predict_on_batch(np.array([self.observations[state]]))[0]
            Q += [self.model_reward.predict_on_batch(np.array([SR]))[0]]
        Q = np.array(Q)[:, 0]
        self.current_predictions[state] = Q
        
        return Q
    
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''
        Q = []
        for action in range(self.number_of_actions):
            SR = self.models_online[action].predict_on_batch(self.observations[batch])
            Q += [self.model_reward.predict_on_batch(SR)[:, 0]]
        
        return np.array(Q).T
