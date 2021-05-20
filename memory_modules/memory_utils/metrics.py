# basic imports
import numpy as np


class AbstractMetric():
    '''
    Abstract metrics class.
    
    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    '''
    
    def __init__(self, interfaceOAI):
        # store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI        
    
    def updateTransitions(self):
        '''
        This function updates the metrics when changes in the environment occur.
        '''
        raise NotImplementedError('.updateTransitions() function not implemented!')
        
class Euclidean(AbstractMetric):
    '''
    Euclidean metrics class.
    
    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    '''
    
    def __init__(self, interfaceOAI):
        super().__init__(interfaceOAI)
        # prepare similarity matrix
        self.D = np.zeros((self.interfaceOAI.world['coordinates'].shape[0], self.interfaceOAI.world['coordinates'].shape[0]))
        # compute euclidean distances between all state pairs (exp(-distance) serves as the similarity measure)
        for s1 in range(self.interfaceOAI.world['coordinates'].shape[0]):
            for s2 in range(self.interfaceOAI.world['coordinates'].shape[0]):
                distance = np.sqrt(np.sum((self.interfaceOAI.world['coordinates'][s1] - self.interfaceOAI.world['coordinates'][s2])**2))
                self.D[s1, s2] = np.exp(-distance)
                self.D[s2, s1] = np.exp(-distance)
                
    def updateTransitions(self):
        '''
        This function updates the metric when changes in the environment occur.
        '''
        pass
    
class SR(AbstractMetric):
    '''
    Successor Representation (SR) metrics class.
    
    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    '''
    
    def __init__(self, interfaceOAI, gamma):
        super().__init__(interfaceOAI)
        self.gamma = gamma
        # prepare similarity matrix
        self.D = np.sum(self.interfaceOAI.world['sas'], axis=1)/self.interfaceOAI.action_space.n
        self.D = np.linalg.inv(np.eye(self.D.shape[0]) - self.gamma * self.D)
                
    def updateTransitions(self):
        '''
        This function updates the metric when changes in the environment occur.
        '''
        self.D = np.sum(self.interfaceOAI.world['sas'], axis=1)/self.interfaceOAI.action_space.n
        self.D = np.linalg.inv(np.eye(self.D.shape[0]) - self.gamma * self.D)
        
class DR(AbstractMetric):
    '''
    Default Representation (DR) metrics class.
    
    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    '''
    
    def __init__(self, interfaceOAI, gamma):
        super().__init__(interfaceOAI)
        self.gamma = gamma
        # compute DR
        self.buildDefaultTransitionMatrix()
        self.D0 = np.linalg.inv(np.eye(self.T_default.shape[0]) - self.gamma * self.T_default)
        # compute new transition matrix
        self.T_new = np.sum(self.interfaceOAI.world['sas'], axis=1)/self.interfaceOAI.action_space.n
        # prepare update matrix B
        self.B = np.zeros((self.interfaceOAI.world['states'], self.interfaceOAI.world['states']))
        if len(self.interfaceOAI.world['invalidTransitions']) > 0:
            # determine affected states
            self.states = np.unique(np.array(self.interfaceOAI.world['invalidTransitions'])[:,0])
            # compute delta
            L = np.eye(self.T_new.shape[0]) - self.gamma * self.T_new
            L0 = np.eye(self.T_default.shape[0]) - self.gamma * self.T_default
            delta = L[self.states] - L0[self.states]
            # compute update matrix B
            alpha = np.linalg.inv(np.eye(self.states.shape[0]) + np.matmul(delta, self.D0[:, self.states]))
            self.B = np.matmul(np.matmul(self.D0[:, self.states], alpha), np.matmul(delta, self.D0))
        # update DR with B
        self.D = self.D0 - self.B
                
    def updateTransitions(self):
        '''
        This function updates the metric when changes in the environment occur.
        '''
        # compute new transition matrix
        self.T_new = np.sum(self.interfaceOAI.world['sas'], axis=1)/self.interfaceOAI.action_space.n
        # prepare update matrix B
        self.B = np.zeros((self.interfaceOAI.world['states'], self.interfaceOAI.world['states']))
        if len(self.interfaceOAI.world['invalidTransitions']) > 0:
            # determine affected states
            self.states = np.unique(np.array(self.interfaceOAI.world['invalidTransitions'])[:,0])
            # compute delta
            L = np.eye(self.T_new.shape[0]) - self.gamma * self.T_new
            L0 = np.eye(self.T_default.shape[0]) - self.gamma * self.T_default
            delta = L[self.states] - L0[self.states]
            # compute update matrix B
            alpha = np.linalg.inv(np.eye(self.states.shape[0]) + np.matmul(delta, self.D0[:, self.states]))
            self.B = np.matmul(np.matmul(self.D0[:, self.states], alpha), np.matmul(delta, self.D0))
        # update DR with B
        self.D = self.D0 - self.B
        
    def buildDefaultTransitionMatrix(self):
        '''
        This function builds the default transition graph in an open field environment under a uniform policy.
        '''
        self.T_default = np.zeros((self.interfaceOAI.world['states'], self.interfaceOAI.world['states']))
        for state in range(self.interfaceOAI.world['states']):
            for action in range(4):
                h = int(state/self.interfaceOAI.world['width'])
                w = state - h * self.interfaceOAI.world['width']
                # left
                if action == 0:
                    w = max(0, w-1)
                # up
                elif action == 1:
                    h = max(0, h-1)
                # right
                elif  action == 2:
                    w = min(self.interfaceOAI.world['width'] - 1, w+1)
                # down
                else:
                    h = min(self.interfaceOAI.world['height'] - 1, h+1)
                # determine next state
                nextState = int(h * self.interfaceOAI.world['width'] + w)
                self.T_default[state][nextState] += 0.25