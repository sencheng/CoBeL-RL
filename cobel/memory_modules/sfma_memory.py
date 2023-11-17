# basic imports
import numpy as np
# framework imports
from cobel.memory_modules.memory_utils.metrics import DR
from cobel.interfaces.rl_interface import AbstractInterface
  
    
class SFMAMemory():
    
    def __init__(self, interface_OAI: AbstractInterface, number_of_actions: int, gamma: float = 0.99, decay_inhibition: float = 0.9, decay_strength: float = 1., learning_rate: float = 0.9):
        '''
        Memory module to be used with the SFMA agent.
        Experiences are stored as a static table.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        number_of_actions :                 The number of the agent's actions.
        gamma :                             The discount factor used to compute the successor representation or default representation.
        decay_inhibition :                  The factor by which inhibition is decayed.
        decay_strength :                    The factor by which the experience strengths are decayed.
        learning_rate :                     The learning rate with which experiences are updated.
        
        Returns
        ----------
        None
        '''
        # initialize variables
        self.number_of_states = interface_OAI.world['states']
        self.number_of_actions = number_of_actions
        self.decay_inhibition = decay_inhibition
        self.decay_strength = decay_strength
        self.decay_recency = 0.9
        self.learning_rate = learning_rate
        self.beta = 20
        self.rlAgent = None
        # experience strength modulation parameters
        self.reward_mod_local = False # increase during experience
        self.error_mod_local = False # increase during experience
        self.reward_mod = False # increase during experience
        self.error_mod = False # increase during experience
        self.policy_mod = False # added before replay
        self.state_mod = False # 
        # similarity metric
        self.metric = DR(interface_OAI, gamma)
        # prepare memory structures
        self.rewards = np.zeros((self.number_of_states, self.number_of_actions))
        self.states = np.tile(np.arange(self.number_of_states).reshape(self.number_of_states, 1), self.number_of_actions).astype(int)
        self.terminals = np.zeros((self.number_of_states, self.number_of_actions)).astype(int)
        # prepare replay-relevant structures
        self.C = np.zeros(self.number_of_states * self.number_of_actions) # strength
        self.T = np.zeros(self.number_of_states * self.number_of_actions) # recency
        self.I = np.zeros(self.number_of_states) # inhibition
        # increase step size
        self.C_step = 1.
        self.I_step = 1.
        # priority rating threshold
        self.R_threshold = 10**-6
        # always reactive experience with highest priority rating
        self.deterministic = False
        # consider recency of experience
        self.recency = False
        # normalize variables
        self.C_normalize = False
        self.D_normalize = False
        self.R_normalize = True
        # replay mode
        self.mode = 'default'
        # modulates reward
        self.reward_modulation = 1.
        # weighting of forward/reverse mode when using blending modes
        self.blend = 0.1
        # weightings of forward abdreverse modes when using interpolation mode
        self.interpolation_fwd, self.interpolation_rev = 0.5, 0.5
        
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
        state, action = experience['state'], experience['action']
        # update experience
        self.rewards[state][action] += self.learning_rate * (experience['reward'] - self.rewards[state][action])
        self.states[state][action] = experience['next_state']
        self.terminals[state][action] = experience['terminal']
        # update replay-relevent structures
        self.C *= self.decay_strength
        self.C[self.number_of_states * action + state] += self.C_step
        self.T *= self.decay_recency
        self.T[self.number_of_states * action + state] = 1.
        # local reward modulation (affects this experience only)
        if self.reward_mod_local:
            self.C[self.number_of_states * action + state] += experience['reward'] * self.reward_modulation
        # reward modulation (affects close experiences)
        if self.reward_mod:
            modulation = np.tile(self.metric.D[experience['state']], self.number_of_actions)
            self.C += experience['reward'] * modulation * self.reward_modulation
        # local RPE modulation (affects this experience only)
        if self.error_mod_local:
            self.C[self.number_of_states * action + state] += np.abs(experience['error'])
        # RPE modulation (affects close experiences)
        if self.error_mod:
            modulation = np.tile(self.metric.D[experience['next_state']], self.number_of_actions)
            self.C += np.abs(experience['error']) * modulation
        # additional strength increase of all experiences at current state
        if self.state_mod:
            self.C[[state + self.number_of_states * a for a in range(self.number_of_actions)]] += 1.
    
    def replay(self, replay_length: int, current_state: None | int = None, current_action: None | int = None) -> list:
        '''
        This function replays experiences.
        
        Parameters
        ----------
        replay_length :                     The number of experiences that will be replayed.
        current_state :                     The state at which replay should start.
        current_action :                    The action with which replay should start.
        
        Returns
        ----------
        experiences :                       The replay batch.
        '''
        action = current_action
        # if no action is specified pick one at random
        if current_action is None:
            action = np.random.randint(self.number_of_actions)
        # if a state is not defined, then choose an experience according to relative experience strengths
        if current_state is None:
            # we clip the strengths to catch negative values caused by rounding errors
            P = np.clip(self.C, a_min=0, a_max=None)/np.sum(np.clip(self.C, a_min=0, a_max=None))
            exp = np.random.choice(np.arange(0, P.shape[0]), p=P)
            current_state = exp % self.number_of_states
            action = int(exp/self.number_of_states)
        next_state = self.states[current_state, action]
        # reset inhibition
        self.I *= 0
        # replay
        experiences = []
        for step in range(replay_length):
            # retrieve experience strengths
            C = np.copy(self.C)
            if self.C_normalize:
                C /= np.amax(C)
            # retrieve experience similarities
            D = np.tile(self.metric.D[current_state], self.number_of_actions)
            if self.D_normalize:
                D /= np.amax(D)
            if self.mode == 'forward':
                D = np.tile(self.metric.D[next_state], self.number_of_actions)
            elif self.mode == 'reverse':
                D = D[self.states.flatten(order='F')]
            elif self.mode == 'blend_forward':
                D += self.blend * np.tile(self.metric.D[next_state], self.number_of_actions)
            elif self.mode == 'blend_reverse':
                D += self.blend * D[self.states.flatten(order='F')]
            elif self.mode == 'interpolate':
                D = self.interpolation_fwd * np.tile(self.metric.D[next_state], self.number_of_actions) + self.interpolation_rev * D[self.states.flatten(order='F')]
            elif self.mode == 'sweeping':
                D = np.tile(self.metric.D[next_state], self.number_of_actions)[self.states.flatten(order='F')]
            # retrieve inhibition
            I = np.tile(self.I, self.number_of_actions)
            # compute priority ratings
            R = C * D * (1 - I)
            if self.recency:
                R *= self.T
            # apply threshold to priority ratings
            R[R < self.R_threshold] = 0.
            # stop replay sequence if all priority ratings are all zero
            if np.sum(R) == 0.:
                break
            # determine state and action
            if self.R_normalize:
                R /= np.amax(R)
            exp = np.argmax(R)
            if not self.deterministic:
                # compute activation probabilities
                probs = self.softmax(R, -1, self.beta)
                probs = probs/np.sum(probs)
                exp = np.random.choice(np.arange(0,probs.shape[0]), p=probs)
            # determine experience tuple
            action = int(exp/self.number_of_states)
            current_state = exp - (action * self.number_of_states)
            next_state = self.states[current_state][action]
            # apply inhibition
            self.I *= self.decay_inhibition
            self.I[current_state] = min(self.I[current_state] + self.I_step, 1.)
            # "reactivate" experience
            experience = {'state': current_state, 'action': action, 'reward': self.rewards[current_state][action],
                          'next_state': next_state, 'terminal': self.terminals[current_state][action]}
            experiences += [experience]
            # stop replay at terminal states
            #if experience['terminal']:
            #    break
            
        return experiences
    
    def softmax(self, data: np.ndarray, offset: float = 0, beta: float = 5) -> np.ndarray:
        '''
        This function computes the custom softmax over the input.
        
        Parameters
        ----------
        data :                              Input of the softmax function.
        offset :                            Offset added after applying the softmax function.
        beta :                              Beta value.
        
        Returns
        ----------
        priorities :                        The softmax priorities.
        '''
        exp = np.exp(data * beta) + offset
        if np.sum(exp) == 0:
            exp.fill(1)
        else:
            exp /= np.sum(exp)
        
        return exp
    
    def retrieve_random_batch(self, number_of_experiences: int, mask: np.ndarray) -> list:
        '''
        This function retrieves a number of random experiences.
        
        Parameters
        ----------
        number_of_experiences :             The number of random experiences to be drawn.
        mask :                              Masks invalid transitions.
        
        Returns
        ----------
        experiences :                       The replay batch.
        '''
        # draw random experiences
        probs = np.ones(self.number_of_states * self.number_of_actions) * mask.astype(int)
        probs /= np.sum(probs)
        idx = np.random.choice(np.arange(self.number_of_states * self.number_of_actions), number_of_experiences, p=probs)
        # determine indeces
        idx = np.unravel_index(idx, (self.number_of_states, self.number_of_actions), order='F')
        # build experience batch
        experiences = []
        for exp in range(number_of_experiences):
            state, action = idx[0][exp], idx[1][exp]
            experiences += [{'state': state, 'action': action, 'reward': self.rewards[state][action],
                             'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}]
            
        return experiences
