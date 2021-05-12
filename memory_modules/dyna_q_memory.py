# basic imports
import numpy as np


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
        self.states = np.tile(np.eye(numberOfStates) * np.arange(numberOfStates), numberOfActions).astype(int)
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