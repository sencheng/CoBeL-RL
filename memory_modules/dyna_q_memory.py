# basic imports
import numpy as np
from scipy import linalg


class DynaQMemory():
    '''
    Memory module to be used with the Dyna-Q agent.
    Experiences are stored as a static table.
    
    | **Args**
    | numberOfStates:               The number of environmental states.
    | numberOfActions:              The number of the agent's actions.
    | learningRate:                 The learning rate with which experiences are updated.
    '''
    
    def __init__(self, numberOfStates, numberOfActions, learningRate=0.9):
        # initialize variables
        self.learningRate = learningRate
        self.numberOfStates = numberOfStates
        self.numberOfActions = numberOfActions
        self.rewards = np.zeros((numberOfStates, numberOfActions))
        self.states = np.tile(np.arange(self.numberOfStates).reshape(self.numberOfStates, 1), self.numberOfActions).astype(int)
        self.terminals = np.zeros((numberOfStates, numberOfActions)).astype(int)
        
    def store(self, experience):
        '''
        This function stores a given experience.
        
        | **Args**
        | experience:                   The experience to be stored.
        '''
        # update experience
        self.rewards[experience['state']][experience['action']] += self.learningRate * (experience['reward'] - self.rewards[experience['state']][experience['action']])
        self.states[experience['state']][experience['action']] = experience['next_state']
        self.terminals[experience['state']][experience['action']] = experience['terminal']
            
    def retrieve(self, state, action):
        '''
        This function retrieves a specific experience.
        
        | **Args**
        | state:                        The environmental state.
        | action:                       The action selected.
        '''
        return {'state': state, 'action': action, 'reward': self.rewards[state][action],
                'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}
        
    def retrieve_batch(self, numberOfExperiences=1):
        '''
        This function retrieves a number of random experiences.
        
        | **Args**
        | numberOfExperiences:          The number of random experiences to be drawn.
        '''
        # draw random experiences
        idx = np.random.randint(0, self.numberOfStates * self.numberOfActions, numberOfExperiences)
        # determine indeces
        idx = np.array(np.unravel_index(idx, (self.numberOfStates, self.numberOfActions)))
        # build experience batch
        experiences = []
        for exp in range(numberOfExperiences):
            state, action = idx[0, exp], idx[1, exp]
            experiences += [{'state': state, 'action': action, 'reward': self.rewards[state][action],
                             'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}]
            
        return experiences
    
    
class PMAMemory():
    '''
    Memory module to be used with the Dyna-Q agent.
    Experiences are stored as a static table.
    
    | **Args**
    | numberOfStates:               The number of environmental states.
    | numberOfActions:              The number of the agent's actions.
    | learningRate:                 The learning rate with which experiences are updated.
    '''
    
    def __init__(self, interfaceOAI, rlAgent, numberOfStates, numberOfActions, learningRate=0.9, gamma=0.9):
        # initialize variables
        self.learningRate = learningRate
        self.learningRateT = 0.9
        self.gamma = gamma
        self.numberOfStates = numberOfStates
        self.numberOfActions = numberOfActions
        self.minGain = 10 ** -6
        self.minGainMode = 'original'
        self.equal_need = False
        self.equal_gain = False
        self.ignore_barriers = True
        self.allow_loops = False
        # initialize memory structures
        self.rewards = np.zeros((numberOfStates, numberOfActions))
        self.states = np.zeros((numberOfStates, numberOfActions)).astype(int)
        self.terminals = np.zeros((numberOfStates, numberOfActions)).astype(int)
        # store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI
        # store reference to agent
        self.rlParent = rlAgent
        # compute state-state transition matrix
        self.T = np.sum(self.interfaceOAI.world['sas'], axis=1)/self.interfaceOAI.action_space.n
        # compute successor representation
        self.SR = np.linalg.inv(np.eye(self.T.shape[0]) - self.gamma * self.T)
        # determine transitions that should be ignored (i.e. those that lead into the same state)
        self.update_mask = (self.states.flatten(order='F') != np.tile(np.arange(self.numberOfStates), self.numberOfActions))

        
    def store(self, experience):
        '''
        This function stores a given experience.
        
        | **Args**
        | experience:                   The experience to be stored.
        '''
        # update experience
        self.rewards[experience['state']][experience['action']] += self.learningRate * (experience['reward'] - self.rewards[experience['state']][experience['action']])
        self.states[experience['state']][experience['action']] = experience['next_state']
        self.terminals[experience['state']][experience['action']] = experience['terminal']
        # update T
        self.T[experience['state']] += self.learningRateT * ((np.arange(self.numberOfStates) == experience['next_state']) - self.T[experience['state']])
        
    def replay(self, replayLength, current_state, force_first=None):
        '''
        This function replays experiences.
        
        | **Args**
        | replayLength:                 The number of experiences that will be replayed.
        | current_state:                State at which replay should start.
        '''
        performed_updates = []
        last_seq = 0
        for update in range(replayLength):
            # make list of 1-step backups
            updates = []
            for i in range(self.numberOfStates * self.numberOfActions):
                s, a = i % self.numberOfStates, int(i/self.numberOfStates)
                updates += [[{'state': s, 'action': a, 'reward': self.rewards[s, a], 'next_state': self.states[s, a], 'terminal': self.terminals[s, a]}]]
            # extend current update sequence
            extend = -1
            if len(performed_updates) > 0:
                # determine extending state
                extend = performed_updates[-1]['next_state']
                # check for loop
                loop = False
                for step in performed_updates[last_seq:]:
                    if extend == step['state']:
                        loop = True
                # extend update
                if not loop or self.allow_loops:
                    # determine extending action which yields max future value
                    extending_action = self.actionProbs(self.rlParent.Q[extend])
                    extending_action = np.random.choice(np.arange(self.numberOfActions), p=extending_action)
                    # determine extending step
                    extend += extending_action * self.numberOfStates
                    #updates[extend] = performed_updates[last_seq:] + updates[extend]
                    updates[extend] = performed_updates[-1:] + updates[extend]
            # compute gain and need
            gain = self.computeGain(updates)
            if self.equal_gain:
                gain.fill(1)
            need = self.computeNeed(current_state)
            if self.equal_need:
                need.fill(1)
            # determine backup with highest utility
            utility = gain * need
            if self.ignore_barriers:
                utility *= self.update_mask
            # determine backup with highest utility
            ties = (utility == np.amax(utility))
            utility_max = np.random.choice(np.arange(self.numberOfStates * self.numberOfActions), p=ties/np.sum(ties))
            # force initial update
            if len(performed_updates) == 0 and force_first is not None:
                utility_max = force_first +  np.random.randint(self.numberOfActions) * self.numberOfStates
            # perform update
            self.rlParent.update_Q(updates[utility_max])
            # add update to list
            performed_updates += [updates[utility_max][-1]]
            if extend != utility_max:
                last_seq = update
            
        return performed_updates
            
    def computeGain(self, updates):
        '''
        This function computes the gain for each possible n-step backup in updates.
        
        | **Args**
        | updates:                      A list of n-step updates.
        '''
        gains = []
        for update in updates:
            gain = 0.
            # expected future value
            future_value = np.amax(self.rlParent.Q[update[-1]['next_state']]) * update[-1]['terminal']
            # gain is accumulated over the whole trajectory
            for s, step in enumerate(update):
                # policy before update
                policy_before = self.actionProbs(self.rlParent.Q[step['state']])
                # sum rewards over subsequent n-steps
                R = 0.
                for following_steps in range(len(update) - s):
                    R += update[s + following_steps]['reward'] * (self.gamma ** following_steps)
                # compute new Q-value
                q_target = np.copy(self.rlParent.Q[step['state']])
                q_target[step['action']] = R + future_value * (self.rlParent.gamma ** (following_steps + 1))
                q_new = self.rlParent.Q[step['state']] + self.rlParent.learningRate * (q_target - self.rlParent.Q[step['state']])
                # policy after update
                policy_after = self.actionProbs(q_new)
                # compute gain
                step_gain = np.sum(q_new * policy_after) - np.sum(q_new * policy_before)
                if self.minGainMode == 'original':
                    step_gain = max(step_gain, self.minGain)
                # add gain
                gain += step_gain
            # store gain for current update
            gains += [max(gain, self.minGain)]
        
        return np.array(gains)
    
    def computeNeed(self, currentState=None):
        '''
        This function computes the need for each possible n-step backup in updates.
        
        | **Args**
        | currentState:                 The state that the agent currently is in. 
        '''
        # use standing distribution of the MDP for 'offline' replay
        if currentState is None:
            # compute left eigenvectors
            eig, vec = linalg.eig(self.T, left=True, right=False)
            best = np.argmin(np.abs(eig - 1))
            
            return np.tile(np.abs(vec[:,best].T), self.numberOfActions)
        # use SR given the current state for 'awake' replay
        else:
            return np.tile(self.SR[currentState], self.numberOfActions)
    
    def updateSR(self):
        '''
        This function updates the SR given the current state-state transition matrix T. 
        '''
        self.SR = np.linalg.inv(np.eye(self.T.shape[0]) - self.gamma * self.T)
        
    def updateMask(self):
        '''
        This function updates the update mask. 
        '''
        self.update_mask = (self.states.flatten(order='F') != np.tile(np.arange(self.numberOfStates), self.numberOfActions))
    
    def actionProbs(self, q):
        '''
        This function computes the action probabilities given a set of Q-values.
        
        | **Args**
        | q:                            The set of Q-values.
        '''
        # assume greedy policy per default
        ties = (q == np.amax(q))
        #p = ties/np.sum(ties)
        p = np.ones(self.numberOfActions) * (self.rlParent.epsilon/self.numberOfActions)
        p[ties] += (1. - self.rlParent.epsilon)/np.sum(ties)
        #p = np.arange(self.numberOfActions) == np.argmax(q)
        # softmax when 'on-policy'
        if self.rlParent.policy == 'softmax':
            # catch all zero case
            if np.all(q == q[0]):
                q.fill(1)
            p = np.exp(q * self.rlParent.beta)
            p /= np.sum(p)
            
        return p
