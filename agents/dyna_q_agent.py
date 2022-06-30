# basic imports
import numpy as np
# memory module
from cobel.agents.rl_agent import AbstractRLAgent, callbacks
from cobel.memory_modules.dyna_q_memory import DynaQMemory, PMAMemory


class AbstractDynaQAgent(AbstractRLAgent):
    '''
    Implementation of a Dyna-Q agent.
    Q-function is represented as a static table.
    
    | **Args**
    | interface_OAI:                The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learning_rate:                The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | custom_callbacks:             The custom callbacks defined by the user.
    '''                
            
    def __init__(self, interface_OAI, epsilon=0.3, beta=5, learning_rate=0.9, gamma=0.99, custom_callbacks={}):
        super().__init__(interface_OAI, custom_callbacks=custom_callbacks)
        # the number of discrete states, retrieved from the Open AI Gym interface
        self.number_of_states = self.interface_OAI.world['states']
        # Q-learning parameters
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.policy = 'greedy'
        # keeps track of current trial
        self.current_trial = 0 # trial count across all sessions (i.e. calls to the train/simulate method)
        # mask invalid actions?
        self.mask_actions = False
        self.compute_action_mask()
        
    def replay(self, replay_batch_size=200):
        '''
        This function replays experiences to update the Q-function.
        
        | **Args**
        | replayBatchSize:              The number of random that will be replayed.
        '''
        # sample random batch of experiences
        replay_batch = self.M.retrieve_batch(replay_batch_size)
        # update the Q-function with each experience
        for experience in replay_batch:
            self.update_Q(experience)
        
    def select_action(self, state, epsilon=0.3, beta=5):
        '''
        This function selects an action according to the Q-values of the current state.
        
        | **Args**
        | state:                        The current state.
        | epsilon:                      The epsilon parameter used under greedy action selection.
        | beta:                         The temperature parameter used when applying the softmax function to the Q-values.
        '''
        # revert to 'greedy' in case that the method name is not valid
        if not self.policy in ['greedy', 'softmax']:
            self.policy = 'greedy'
        # retrieve Q-values
        qVals = self.retrieve_Q(state)
        actions = np.arange(qVals.shape[0])
        # remove masked actions
        if self.mask_actions:
            qVals = qVals[self.action_mask[state]]
            actions = actions[self.action_mask[state]]
        # select action with highest value
        if self.policy == 'greedy':
            # act greedily and break ties
            action = np.argmax(qVals)
            ties = np.arange(qVals.shape[0])[(qVals == qVals[action])]
            action = ties[np.random.randint(ties.shape[0])]
            # in case that Q-values are equal select a random action
            if np.random.rand() < epsilon:
                action = np.random.randint(qVals.shape[0])
            return actions[action]
        # select action probabilistically
        elif self.policy == 'softmax':
            qVals -= np.amax(qVals)
            probs = np.exp(beta * qVals)/np.sum(np.exp(beta * qVals))
            action = np.random.choice(qVals.shape[0], p=probs)
            return actions[action]
        
    def compute_action_mask(self):
        '''
        This function computes the action mask which prevents the selection of invalid actions.
        '''
        # retrieve number of states and actions
        s, a = self.interface_OAI.world['states'], self.number_of_actions
        # determine follow-up states
        self.action_mask = self.interface_OAI.world['sas'].reshape((s * a, s), order='F')
        self.action_mask = np.argmax(self.action_mask, axis=1)
        # make action mask
        self.action_mask = (self.action_mask !=  np.tile(np.arange(s), a)).reshape((s, a), order='F')
        
    def train(self, number_of_trials=100, max_number_of_steps=50, replay_batch_size=100, no_replay=False):
        '''
        This function is called to train the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the Dyna-Q agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
        | replay_batch_size:            The number of random that will be replayed.
        | no_replay:                    If true, experiences are not replayed.
        '''
        raise NotImplementedError('.train() function not implemented!')
    
    def update_Q(self, experience):
        '''
        This function updates the Q-function with a given experience.
        
        | **Args**
        | experience:                   The experience with which the Q-function will be updated.
        '''
        raise NotImplementedError('.update_Q() function not implemented!')
    
    def retrieve_Q(self, state):
        '''
        This function retrieves Q-values for a given state.
        
        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        raise NotImplementedError('.retrieve_Q() function not implemented!')
    
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''
        raise NotImplementedError('.predict_on_batch() function not implemented!')
        

class DynaQAgent(AbstractDynaQAgent):
    '''
    Implementation of a Dyna-Q agent.
    Q-function is represented as a static table.
    
    | **Args**
    | interface_OAI:                The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learning_rate:                The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | custom_callbacks:             The custom callbacks defined by the user.
    '''                
            
    def __init__(self, interface_OAI, epsilon=0.3, beta=5, learning_rate=0.9, gamma=0.99, custom_callbacks={}):
        super().__init__(interface_OAI, epsilon=epsilon, beta=beta, learning_rate=learning_rate, gamma=gamma, custom_callbacks=custom_callbacks)
        # Q-table
        self.Q = np.zeros((self.number_of_states, self.number_of_actions))
        # memory module
        self.M = DynaQMemory(self.number_of_states, self.number_of_actions)
        # perform replay at the end of an episode instead of each step
        self.episodic_replay = False
        
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
            # prepare trial log
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
            
    def test(self, number_of_trials=100, max_number_of_steps=50):
        '''
        This function is called to test the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the Dyna-Q agent is tested.
        | max_number_of_steps:          The maximum number of steps per trial.
        '''
        for trial in range(number_of_trials):
            # prepare trial log
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
            # callback
            self.engaged_callbacks.on_trial_end(logs)
    
    def update_Q(self, experience):
        '''
        This function updates the Q-function with a given experience.
        
        | **Args**
        | experience:                   The experience with which the Q-function will be updated.
        '''
        # compute TD-error
        td = experience['reward']
        td += self.gamma * experience['terminal'] * np.amax(self.retrieve_Q(experience['next_state']))
        td -= self.retrieve_Q(experience['state'])[experience['action']]
        # update Q-function with TD-error
        self.Q[experience['state']][experience['action']] += self.learning_rate * td
    
    def retrieve_Q(self, state):
        '''
        This function retrieves Q-values for a given state.
        
        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        # retrieve Q-values, if entry exists
        return self.Q[state]
    
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''
        predictions = []
        for state in batch:
            predictions += [self.retrieve_Q(state)]
            
        return np.array(predictions)
    
    
class PMAAgent(AbstractDynaQAgent):
    '''
    Implementation of a Dyna-Q agent using the Prioritized Memory Access (PMA) method described by Mattar & Daw (2018).
    
    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learningRate:                 The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | trialEndFcn:                  The callback function called at the end of each trial, defined for more flexibility in scenario control.
    '''                
    
    class callbacksPMA(callbacks):
        '''
        Callback class. Used for visualization and scenario control.
        
        | **Args**
        | rl_parent:                    Reference to the RL agent.
        | custom_callbacks:             The custom callbacks defined by the user.
        '''
        
        def __init__(self, rl_parent, custom_callbacks={}):
            # store the hosting class
            self.rl_parent = rl_parent
            # store the trial end callback function
            self.custom_callbacks = custom_callbacks
                    
        def on_replay_end(self, logs):
            '''
            The following function is called whenever an episode ends, and executes callbacks defined by the user.
            
            | **Args**
            | logs:                         The trial log.
            '''
            logs['rl_parent'] = self.rl_parent
            if 'on_trial_end' in self.custom_callbacks:
                for custom_callback in self.custom_callbacks['on_trial_end']:
                    custom_callback(logs)
    
    def __init__(self, interface_OAI, epsilon=0.3, beta=5, learning_rate=0.9, gamma=0.99, gamma_SR=0.99, custom_callbacks={}):
        super().__init__(interface_OAI, epsilon=epsilon, beta=beta, learning_rate=learning_rate, gamma=gamma, custom_callbacks=custom_callbacks)
        # Q-table
        self.Q = np.zeros((self.number_of_states, self.number_of_actions))
        # memory module
        self.M = PMAMemory(self.interface_OAI, self, self.number_of_states, self.number_of_actions, gamma_SR)
        # initialze callbacks class with customized callbacks
        self.engaged_callbacks = self.callbacksPMA(self, custom_callbacks)
        
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
            # reset environment
            state = self.interface_OAI.reset()
            # log cumulative reward
            logs = {'trial_reward': 0, 'steps': 0, 'step': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # perform experience replay
            if not no_replay:
                logs['replay_batch'] = self.M.replay(replay_batch_size, state)
                self.engaged_callbacks.on_replay_end(logs)
            for step in range(max_number_of_steps):
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
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
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'], logs['step'] = step, step
            # perform experience replay
            if not no_replay:
                self.M.update_SR()
                logs['replay_batch'] = self.M.replay(replay_batch_size, next_state)
                self.engaged_callbacks.on_replay_end(logs)
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            
    def test(self, number_of_trials=100, max_number_of_steps=50):
        '''
        This function is called to test the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the Dyna-Q agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
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
            logs['steps'], logs['step'] = step, step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
        
    def update_Q(self, update):
        '''
        This function updates the Q-function with a given experience.
        
        | **Args**
        | experience:                   The experience with which the Q-function will be updated.
        '''
        # expected future value
        future_value = np.amax(self.Q[update[-1]['next_state']]) * update[-1]['terminal']
        for s, step in enumerate(update):
            # sum rewards over remaining trajectory
            R = 0.
            for following_steps in range(len(update) - s):
                R += update[s + following_steps]['reward'] * (self.gamma ** following_steps)
            # compute TD-error
            td = R + future_value * (self.gamma ** (following_steps + 1))
            td -= self.retrieve_Q(step['state'])[step['action']]
            # update Q-function with TD-error
            self.Q[step['state']][step['action']] += self.learning_rate * td
        
    def retrieve_Q(self, state):
        '''
        This function retrieves Q-values for a given state.
        
        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        # retrieve Q-values, if entry exists
        return self.Q[state]
    
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''
        predictions = []
        for state in batch:
            predictions += [self.retrieve_Q(state)]
            
        return np.array(predictions)
