# basic imports
import numpy as np
# memory module
from memory_modules.dyna_q_memory import DynaQMemory, PMAMemory


class AbstractDynaQAgent():
    '''
    Implementation of a Dyna-Q agent.
    Q-function is represented as a static table.
    
    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learningRate:                 The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    '''                
            
    def __init__(self, interfaceOAI, epsilon=0.3, beta=5, learningRate=0.9, gamma=0.99):
        # store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI        
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.numberOfActions = self.interfaceOAI.action_space.n
        # Q-learning parameters
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.learningRate = learningRate
        self.policy = 'greedy'
        # mask invalid actions?
        self.mask_actions = False
        self.compute_action_mask()
        
    def replay(self, replayBatchSize=200):
        '''
        This function replays experiences to update the Q-function.
        
        | **Args**
        | replayBatchSize:              The number of random that will be replayed.
        '''
        # sample random batch of experiences
        replayBatch = self.M.retrieve_batch(replayBatchSize)
        # update the Q-function with each experience
        for experience in replayBatch:
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
            probs = np.exp(beta * qVals)/np.sum(np.exp(beta * qVals))
            action = np.random.choice(qVals.shape[0], p=probs)
            return actions[action]
        
    def compute_action_mask(self):
        '''
        This function computes the action mask which prevents the selection of invalid actions.
        '''
        # retrieve number of states and actions
        s, a = self.interfaceOAI.world['states'], self.numberOfActions
        # determine follow-up states
        self.action_mask = self.interfaceOAI.world['sas'].reshape((s * a, s), order='F')
        self.action_mask = np.argmax(self.action_mask, axis=1)
        # make action mask
        self.action_mask = (self.action_mask !=  np.tile(np.arange(s), a)).reshape((s, a), order='F')
        
    def train(self, numberOfTrials=100, maxNumberOfSteps=50, replayBatchSize=100, noReplay=False):
        '''
        This function is called to train the agent.
        
        | **Args**
        | numberOfTrials:               The number of trials the Dyna-Q agent is trained.
        | maxNumberOfSteps:             The maximum number of steps per trial.
        | replayBatchSize:              The number of random that will be replayed.
        | noReplay:                     If true, experiences are not replayed.
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
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learningRate:                 The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | trialEndFcn:                  The callback function called at the end of each trial, defined for more flexibility in scenario control.
    '''
    
    class callbacks():
        '''
        Callback class. Used for visualization and scenario control.
        
        | **Args**
        | rlParent:                     Reference to the Dyna-Q agent.
        | trialEndFcn:                  Maximum number of experiences that will be stored by the memory module.
        '''

        def __init__(self, rlParent, trialEndFcn=None):
            super(DynaQAgent.callbacks, self).__init__()
            # store the hosting class
            self.rlParent = rlParent
            # store the trial end callback function
            self.trialEndFcn = trialEndFcn
        
        def on_episode_end(self, epoch, logs):
            '''
            The following function is called whenever an episode ends, and updates the reward accumulator,
            simultaneously updating the visualization of the reward function.
            
            | **Args**
            | rlParent:                     Reference to the Dyna-Q agent.
            | trialEndFcn:                  Maximum number of experiences that will be stored by the memory module.
            '''
            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch, self.rlParent, logs)
                
            
    def __init__(self, interfaceOAI, epsilon=0.3, beta=5, learningRate=0.9, gamma=0.99, trialEndFcn=None):
        super().__init__(interfaceOAI, epsilon=epsilon, beta=beta, learningRate=learningRate, gamma=gamma)
        # Q-table
        self.Q = np.zeros((self.interfaceOAI.world['states'], self.numberOfActions))
        # memory module
        self.M = DynaQMemory(self.interfaceOAI.world['states'], self.numberOfActions)
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks = self.callbacks(self,trialEndFcn)
        # perform replay at the end of an episode instead of each step
        self.episodic_replay = False
        
    def train(self, numberOfTrials=100, maxNumberOfSteps=50, replayBatchSize=100, noReplay=False):
        '''
        This function is called to train the agent.
        
        | **Args**
        | numberOfTrials:               The number of trials the Dyna-Q agent is trained.
        | maxNumberOfSteps:             The maximum number of steps per trial.
        | replayBatchSize:              The number of random that will be replayed.
        | noReplay:                     If true, experiences are not replayed.
        '''
        for trial in range(numberOfTrials):
            # reset environment
            state = self.interfaceOAI.reset()
            # log cumulative reward
            logs = {'episode_reward': 0}
            for step in range(maxNumberOfSteps):
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
                # perform action
                next_state, reward, stopEpisode, callbackValue = self.interfaceOAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - stopEpisode)}
                # update Q-function with experience
                self.update_Q(experience)
                # store experience
                self.M.store(experience)
                # update current state
                state = next_state
                # perform experience replay
                if not noReplay and not self.episodic_replay:
                    self.replay(replayBatchSize)
                # update cumulative reward
                logs['episode_reward'] += reward
                # stop trial when the terminal state is reached
                if stopEpisode:
                    break
            # perform experience replay
            if not noReplay and self.episodic_replay:
                self.replay(replayBatchSize)
            # callback
            self.engagedCallbacks.on_episode_end(trial, logs)
    
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
        self.Q[experience['state']][experience['action']] += self.learningRate * td
    
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
    Implementation of a Dyna-Q agent using the Prioritized Memory Access (PMA) method described by Mattar & Daw (2019).
    
    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learningRate:                 The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | trialEndFcn:                  The callback function called at the end of each trial, defined for more flexibility in scenario control.
    '''
    
    class callbacks():
        '''
        Callback class. Used for visualization and scenario control.
        
        | **Args**
        | rlParent:                     Reference to the Dyna-Q agent.
        | trialEndFcn:                  Maximum number of experiences that will be stored by the memory module.
        '''

        def __init__(self, rlParent, trialEndFcn=None):
            super(PMAAgent.callbacks, self).__init__()
            # store the hosting class
            self.rlParent = rlParent
            # store the trial end callback function
            self.trialEndFcn = trialEndFcn
        
        def on_episode_end(self, epoch, logs):
            '''
            The following function is called whenever an episode ends, and updates the reward accumulator,
            simultaneously updating the visualization of the reward function.
            
            | **Args**
            | rlParent:                     Reference to the Dyna-Q agent.
            | trialEndFcn:                  Maximum number of experiences that will be stored by the memory module.
            '''
            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch, self.rlParent, logs)
                
            
    def __init__(self, interfaceOAI, epsilon=0.3, beta=5, learningRate=0.9, gamma=0.99, trialEndFcn=None, gammaSR=0.99):
        super().__init__(interfaceOAI, epsilon=epsilon, beta=beta, learningRate=learningRate, gamma=gamma)
        # Q-table
        self.Q = np.zeros((self.interfaceOAI.world['states'], self.numberOfActions))
        # memory module
        self.M = PMAMemory(self.interfaceOAI, self, self.interfaceOAI.world['states'], self.numberOfActions, gammaSR)
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks = self.callbacks(self,trialEndFcn)
        # replay traces
        self.replay_traces = {'start': [], 'end': []}
        # logging
        self.rewards = []
        
    def train(self, numberOfTrials=100, maxNumberOfSteps=50, replayBatchSize=100, noReplay=False):
        '''
        This function is called to train the agent.
        
        | **Args**
        | numberOfTrials:               The number of trials the Dyna-Q agent is trained.
        | maxNumberOfSteps:             The maximum number of steps per trial.
        | replayBatchSize:              The number of random that will be replayed.
        | noReplay:                     If true, experiences are not replayed.
        '''
        for trial in range(numberOfTrials):
            # reset environment
            state = self.interfaceOAI.reset()
            # log cumulative reward
            logs = {'episode_reward': 0}
            # perform experience replay
            if not noReplay:
                self.replay_traces['start'] += [self.M.replay(replayBatchSize, state)]
            for step in range(maxNumberOfSteps):
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
                # perform action
                next_state, reward, stopEpisode, callbackValue = self.interfaceOAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - stopEpisode)}
                # update Q-function with experience
                self.update_Q([experience])
                # store experience
                self.M.store(experience)
                # update current state
                state = next_state
                # update cumulative reward
                logs['episode_reward'] += reward
                # stop trial when the terminal state is reached
                if stopEpisode:
                    break
            # perform experience replay
            if not noReplay:
                self.M.updateSR()
                self.replay_traces['end'] += [self.M.replay(replayBatchSize, next_state)]
            self.rewards += [logs['episode_reward']]
            # callback
            self.engagedCallbacks.on_episode_end(trial, logs)
        
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
            self.Q[step['state']][step['action']] += self.learningRate * td
        
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
