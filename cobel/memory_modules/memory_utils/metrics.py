# basic imports
import numpy as np


class AbstractMetric():
    
    def __init__(self, interface_OAI):
        '''
        Abstract metrics class.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        
        Returns
        ----------
        None
        '''
        # store the Open AI Gym interface
        self.interface_OAI = interface_OAI        
    
    def update_transitions(self):
        '''
        This function updates the metrics when changes in the environment occur.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        raise NotImplementedError('.update_transitions() function not implemented!')
        

class Euclidean(AbstractMetric):
    
    def __init__(self, interface_OAI):
        '''
        Euclidean metrics class.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI)
        # prepare similarity matrix
        self.D = np.zeros((self.interface_OAI.world['coordinates'].shape[0], self.interface_OAI.world['coordinates'].shape[0]))
        # compute euclidean distances between all state pairs (exp(-distance) serves as the similarity measure)
        for s1 in range(self.interface_OAI.world['coordinates'].shape[0]):
            for s2 in range(self.interface_OAI.world['coordinates'].shape[0]):
                distance = np.sqrt(np.sum((self.interface_OAI.world['coordinates'][s1] - self.interface_OAI.world['coordinates'][s2])**2))
                self.D[s1, s2] = np.exp(-distance)
                self.D[s2, s1] = np.exp(-distance)
                
    def update_transitions(self):
        '''
        This function updates the metrics when changes in the environment occur.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        pass
    

class SR(AbstractMetric):
    
    def __init__(self, interface_OAI, gamma: float):
        '''
        Successor Representation (SR) metrics class.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        gamma :                             The discount factor used to compute the SR.
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI)
        self.gamma = gamma
        # prepare similarity matrix
        self.D = np.sum(self.interface_OAI.world['sas'], axis=1)/self.interface_OAI.action_space.n
        self.D = np.linalg.inv(np.eye(self.D.shape[0]) - self.gamma * self.D)
                
    def update_transitions(self):
        '''
        This function updates the metric when changes in the environment occur.
        '''
        self.D = np.sum(self.interface_OAI.world['sas'], axis=1)/self.interface_OAI.action_space.n
        self.D = np.linalg.inv(np.eye(self.D.shape[0]) - self.gamma * self.D)
        

class DR(AbstractMetric):
    
    def __init__(self, interface_OAI, gamma: float, T_default=None):
        '''
        Default Representation (DR) metrics class.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        gamma :                             The discount factor used to compute the DR.
        T_default :                         The default state-state transition matrix. If none, then one is created assuming an open field gridworld.
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI)
        self.gamma = gamma
        # compute DR
        self.T_default = T_default
        if self.T_default is None:
            self.build_default_transition_matrix()
        self.D0 = np.linalg.inv(np.eye(self.T_default.shape[0]) - self.gamma * self.T_default)
        # compute new transition matrix
        self.T_new = np.sum(self.interface_OAI.world['sas'], axis=1)/self.interface_OAI.action_space.n
        # prepare update matrix B
        self.B = np.zeros((self.interface_OAI.world['states'], self.interface_OAI.world['states']))
        if len(self.interface_OAI.world['invalid_transitions']) > 0:
            # determine affected states
            self.states = np.unique(np.array(self.interface_OAI.world['invalid_transitions'])[:,0])
            # compute delta
            L = np.eye(self.T_new.shape[0]) - self.gamma * self.T_new
            L0 = np.eye(self.T_default.shape[0]) - self.gamma * self.T_default
            delta = L[self.states] - L0[self.states]
            # compute update matrix B
            alpha = np.linalg.inv(np.eye(self.states.shape[0]) + np.matmul(delta, self.D0[:, self.states]))
            self.B = np.matmul(np.matmul(self.D0[:, self.states], alpha), np.matmul(delta, self.D0))
        # update DR with B
        self.D = self.D0 - self.B
                
    def update_transitions(self):
        '''
        This function updates the metric when changes in the environment occur.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # compute new transition matrix
        self.T_new = np.sum(self.interface_OAI.world['sas'], axis=1)/self.interface_OAI.action_space.n
        # prepare update matrix B
        self.B = np.zeros((self.interface_OAI.world['states'], self.interface_OAI.world['states']))
        if len(self.interface_OAI.world['invalid_transitions']) > 0:
            # determine affected states
            self.states = np.unique(np.array(self.interface_OAI.world['invalid_transitions'])[:,0])
            # compute delta
            L = np.eye(self.T_new.shape[0]) - self.gamma * self.T_new
            L0 = np.eye(self.T_default.shape[0]) - self.gamma * self.T_default
            delta = L[self.states] - L0[self.states]
            # compute update matrix B
            alpha = np.linalg.inv(np.eye(self.states.shape[0]) + np.matmul(delta, self.D0[:, self.states]))
            self.B = np.matmul(np.matmul(self.D0[:, self.states], alpha), np.matmul(delta, self.D0))
        # update DR with B
        self.D = self.D0 - self.B
        
    def build_default_transition_matrix(self):
        '''
        This function builds the default transition graph in an open field environment under a uniform policy.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        self.T_default = np.zeros((self.interface_OAI.world['states'], self.interface_OAI.world['states']))
        for state in range(self.interface_OAI.world['states']):
            for action in range(4):
                h = int(state/self.interface_OAI.world['width'])
                w = state - h * self.interface_OAI.world['width']
                # left
                if action == 0:
                    w = max(0, w-1)
                # up
                elif action == 1:
                    h = max(0, h-1)
                # right
                elif  action == 2:
                    w = min(self.interface_OAI.world['width'] - 1, w+1)
                # down
                else:
                    h = min(self.interface_OAI.world['height'] - 1, h+1)
                # determine next state
                nextState = int(h * self.interface_OAI.world['width'] + w)
                self.T_default[state][nextState] += 0.25
