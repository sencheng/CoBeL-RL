# basic imports
import numpy as np
from scipy import linalg


class DynaQMemory():
    
    def __init__(self, number_of_states: int, number_of_actions: int, learning_rate=0.9):
        '''
        Memory module to be used with the Dyna-Q agent.
        Experiences are stored as a static table.
        
        Parameters
        ----------
        number_of_states :                  The number of environmental states.
        number_of_actions :                 The number of the agent's actions.
        learning_rate :                     The learning rate with which experiences are updated.
        
        Returns
        ----------
        None
        '''
        # initialize variables
        self.learning_rate = learning_rate
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.rewards = np.zeros((number_of_states, number_of_actions))
        self.states = np.tile(np.arange(self.number_of_states).reshape(self.number_of_states, 1), self.number_of_actions).astype(int)
        self.terminals = np.zeros((number_of_states, number_of_actions)).astype(int)
        
    def store(self, experience: dict):
        '''
        This function stores a given experience.
        
        Parameters
        ----------
        experience :                        The experience to be stored.
        
        Returns
        ----------
        None
        '''
        # update experience
        self.rewards[experience['state']][experience['action']] += self.learning_rate * (experience['reward'] - self.rewards[experience['state']][experience['action']])
        self.states[experience['state']][experience['action']] = experience['next_state']
        self.terminals[experience['state']][experience['action']] = experience['terminal']
            
    def retrieve(self, state: int, action: int) -> dict:
        '''
        This function retrieves a the experience for a specified state-action pair.
        
        Parameters
        ----------
        state :                             The state.
        action :                            The action.
        
        Returns
        ----------
        experience :                        The experience that is retrieved.
        '''
        return {'state': state, 'action': action, 'reward': self.rewards[state][action],
                'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}
        
    def retrieve_batch(self, number_of_experiences=1) -> list:
        '''
        This function retrieves a number of random experiences.
        
        Parameters
        ----------
        number_of_experiences :             The number of random experiences that will be retrieved.
        
        Returns
        ----------
        batch :                             The batch of experiences that is retrieved.
        '''
        # draw random experiences
        idx = np.random.randint(0, self.number_of_states * self.number_of_actions, number_of_experiences)
        # determine indeces
        idx = np.array(np.unravel_index(idx, (self.number_of_states, self.number_of_actions)))
        # build experience batch
        experiences = []
        for exp in range(number_of_experiences):
            state, action = idx[0, exp], idx[1, exp]
            experiences += [{'state': state, 'action': action, 'reward': self.rewards[state][action],
                             'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}]
            
        return experiences
    
    
class PMAMemory():
    
    def __init__(self, interface_OAI, rl_parent, number_of_states: int, number_of_actions: int, learning_rate=0.9, gamma=0.9):
        '''
        Memory module to be used with the PMA agent.
        Experiences are stored as a static table.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        rl_parent :                         The MA agent using this memory module.
        number_of_states :                  The number of environmental states.
        number_of_actions :                 The number of the agent's actions.
        learning_rate :                     The learning rate with which experiences are updated.
        gamma :                             The discount factor that is used to compute the successor representation.
        
        Returns
        ----------
        None
        '''
        # initialize variables
        self.learning_rate = learning_rate
        self.learning_rate_T = 0.9
        self.gamma = gamma
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.min_gain = 10 ** -6
        self.min_gain_mode = 'original'
        self.equal_need = False
        self.equal_gain = False
        self.ignore_barriers = True
        self.allow_loops = False
        # initialize memory structures
        self.rewards = np.zeros((self.number_of_states, self.number_of_actions))
        self.states = np.zeros((self.number_of_states, self.number_of_actions)).astype(int)
        self.terminals = np.zeros((self.number_of_states, self.number_of_actions)).astype(int)
        # store the Open AI Gym interface
        self.interface_OAI = interface_OAI
        # store reference to agent
        self.rl_parent = rl_parent
        # compute state-state transition matrix
        self.T = np.sum(self.interface_OAI.world['sas'], axis=1)/self.interface_OAI.action_space.n
        # compute successor representation
        self.SR = np.linalg.inv(np.eye(self.T.shape[0]) - self.gamma * self.T)
        # determine transitions that should be ignored (i.e. those that lead into the same state)
        self.update_mask = (self.states.flatten(order='F') != np.tile(np.arange(self.number_of_states), self.number_of_actions))
        
    def store(self, experience: dict):
        '''
        This function stores a given experience.
        
        Parameters
        ----------
        experience :                        The experience to be stored.
        
        Returns
        ----------
        None
        '''
        # update experience
        self.rewards[experience['state']][experience['action']] += self.learning_rate * (experience['reward'] - self.rewards[experience['state']][experience['action']])
        self.states[experience['state']][experience['action']] = experience['next_state']
        self.terminals[experience['state']][experience['action']] = experience['terminal']
        # update T
        self.T[experience['state']] += self.learning_rate_T * ((np.arange(self.number_of_states) == experience['next_state']) - self.T[experience['state']])
        
    def replay(self, replay_length, current_state, force_first=None) -> list:
        '''
        This function replays experiences.
        
        Parameters
        ----------
        replay_length :                     The number of experiences that will be replayed.
        current_state :                     State at which replay should start.
        force_first :                       If a state is specified, replay is forced to start here.
        
        Returns
        ----------
        updates :                           The updates that were executed.
        '''
        performed_updates = []
        last_seq = 0
        for update in range(replay_length):
            # make list of 1-step backups
            updates = []
            for i in range(self.number_of_states * self.number_of_actions):
                s, a = i % self.number_of_states, int(i/self.number_of_states)
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
                    extending_action = self.action_probs(self.rl_parent.Q[extend])
                    extending_action = np.random.choice(np.arange(self.number_of_actions), p=extending_action)
                    # determine extending step
                    extend += extending_action * self.number_of_states
                    updates[extend] = performed_updates[last_seq:] + updates[extend]
            # compute gain and need
            gain = self.compute_gain_batch() # ! check for consistency
            if extend != -1:
                gain[extend] = self.compute_gain([updates[extend]])
            #gain = self.compute_gain(updates) # (old) iterative version
            if self.equal_gain:
                gain.fill(1)
            need = self.compute_need(current_state)
            if self.equal_need:
                need.fill(1)
            # determine backup with highest utility
            utility = gain * need
            if self.ignore_barriers:
                utility *= self.update_mask
            # determine backup with highest utility
            ties = (utility == np.amax(utility))
            utility_max = np.random.choice(np.arange(self.number_of_states * self.number_of_actions), p=ties/np.sum(ties))
            # force initial update
            if len(performed_updates) == 0 and force_first is not None:
                utility_max = force_first +  np.random.randint(self.number_of_actions) * self.number_of_states
            # perform update
            self.rl_parent.update_Q(updates[utility_max])
            # add update to list
            performed_updates += [updates[utility_max][-1]]
            if extend != utility_max:
                last_seq = update
            
        return performed_updates
            
    def compute_gain(self, updates: list) -> np.ndarray:
        '''
        This function computes the gain for each possible n-step backup in updates.
        
        Parameters
        ----------
        updates :                           A list of n-step updates.
        
        Returns
        ----------
        gains :                             The gain values for the n-step updates.
        '''
        gains = []
        for update in updates:
            gain = 0.
            # expected future value
            future_value = np.amax(self.rl_parent.Q[update[-1]['next_state']]) * update[-1]['terminal']
            # gain is accumulated over the whole trajectory
            for s, step in enumerate(update):
                # policy before update
                policy_before = self.action_probs(self.rl_parent.Q[step['state']])
                # sum rewards over subsequent n-steps
                R = 0.
                for following_steps in range(len(update) - s):
                    R += update[s + following_steps]['reward'] * (self.gamma ** following_steps)
                # compute new Q-value
                q_target = np.copy(self.rl_parent.Q[step['state']])
                q_target[step['action']] = R + future_value * (self.rl_parent.gamma ** (following_steps + 1))
                q_new = self.rl_parent.Q[step['state']] + self.rl_parent.learning_rate * (q_target - self.rl_parent.Q[step['state']])
                # policy after update
                policy_after = self.action_probs(q_new)
                # compute gain
                step_gain = np.sum(q_new * policy_after) - np.sum(q_new * policy_before)
                if self.min_gain_mode == 'original':
                    step_gain = max(step_gain, self.min_gain)
                # add gain
                gain += step_gain
            # store gain for current update
            gains += [max(gain, self.min_gain)]
        
        return np.array(gains)
    
    def compute_gain_batch(self) -> np.ndarray:
        '''
        This function computes the gain for each possible 1-step backup.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        gains :                             The gain values for the 1-step updates.
        '''
        # prepare updates
        updates = np.tile(self.rl_parent.Q, (self.rl_parent.Q.shape[1], 1))
        # bootstrap target values
        next_states = self.rl_parent.M.states.flatten(order='F')
        targets = self.rl_parent.Q[next_states]
        # prepare target mask
        target_mask = np.zeros(updates.shape)
        for action in range(self.rl_parent.Q.shape[1]):
            target_mask[(self.rl_parent.Q.shape[0] * action):(self.rl_parent.Q.shape[0] * (action + 1)), action] = 1.
        # compute updated Q-values
        Q_new = np.copy(updates)
        Q_new += self.rl_parent.learning_rate * target_mask * (np.tile(self.rewards, (self.rl_parent.Q.shape[1], 1)) + self.rl_parent.gamma * np.amax(targets, axis=1).reshape(targets.shape[0], 1) * self.terminals.flatten(order='F').reshape(targets.shape[0], 1) - Q_new)
        # compute policies pre and post update
        policy_old = self.action_probs_batch(updates, self.rl_parent.beta, None)
        policy_new = self.action_probs_batch(Q_new, self.rl_parent.beta, None)
        # comute gain for all updates
        gain = np.sum(policy_new * Q_new, axis=1) - np.sum(policy_old * Q_new, axis=1)
        
        return np.clip(gain, a_min=self.min_gain, a_max=None)
    
    def compute_need(self, current_state=None) -> np.ndarray:
        '''
        This function computes the need for each possible n-step backup in updates.
        
        Parameters
        ----------
        current_state :                     The state that the agent currently is in.
        
        Returns
        ----------
        needs :                             The gain values for the n-step updates.
        '''
        # use standing distribution of the MDP for 'offline' replay
        if current_state is None:
            # compute left eigenvectors
            eig, vec = linalg.eig(self.T, left=True, right=False)
            best = np.argmin(np.abs(eig - 1))
            
            return np.tile(np.abs(vec[:, best].T), self.number_of_actions)
        # use SR given the current state for 'awake' replay
        else:
            return np.tile(self.SR[current_state], self.number_of_actions)
    
    def update_SR(self):
        '''
        This function updates the SR given the current state-state transition matrix T.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        self.SR = np.linalg.inv(np.eye(self.T.shape[0]) - self.gamma * self.T)
        
    def compute_update_mask(self):
        '''
        This function updates the update mask.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        self.update_mask = (self.states.flatten(order='F') != np.tile(np.arange(self.number_of_states), self.number_of_actions))
    
    def action_probs(self, q: np.ndarray) -> np.ndarray:
        '''
        This function computes the action selection probabilities given a set of Q-values.
        
        Parameters
        ----------
        q :                                 The Q-values for which action selection probabilities will be computed.
        
        Returns
        ----------
        p :                                 The action selection probabilities.
        '''
        # assume greedy policy per default
        ties = (q == np.amax(q))
        p = np.ones(self.number_of_actions) * (self.rl_parent.epsilon/self.number_of_actions)
        p[ties] += (1. - self.rl_parent.epsilon)/np.sum(ties)
        # softmax when 'on-policy'
        if self.rl_parent.policy == 'softmax':
            # catch all zero case
            if np.all(q == q[0]):
                q.fill(1)
            p = np.exp(q * self.rl_parent.beta)
            p /= np.sum(p)
            
        return p
    
    def action_probs_batch(self, Q: np.ndarray, beta=0., action_mask=None) -> np.ndarray:
        '''
        This function computes the action selection probabilities for a table of Q-values.
        
        Parameters
        ----------
        Q :                                 The Q-values.
        beta :                              The inverse temperature parameter used when computing the action selection probabilities under a softmax policy.
        action_mask :                       The action mask (optional).
        
        Returns
        ----------
        probs :                             The action selection probabilities.
        '''
        # assume random policy by default
        P = np.ones(Q.shape)
        if action_mask is None:
            action_mask = np.ones(P.shape)
        # compute action selection probabilities
        if self.rl_parent.policy == 'greedy':
            ties = (Q == np.amax(Q, axis=1).reshape(Q.shape[0], 1))
            P.fill(self.rl_parent.epsilon/self.number_of_actions)
            P += (ties * (1. - self.rl_parent.epsilon))/np.sum(ties, axis=1).reshape(ties.shape[0], 1)
        elif self.rl_parent.policy == 'softmax':
            Q_mask = np.exp(Q * beta)
            P = Q_mask * action_mask
    
        return P/np.sum(P, axis=1).reshape(P.shape[0], 1) 
