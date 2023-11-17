# basic imports
import numpy as np
from collections import namedtuple
# keras-rl imports
from rl.memory import Memory

# Experiences as they are stored and returned to the agent
# To be understood as "Agent in 'state0' takes 'action', gets 'reward' and enters 'state1' which is 'terminal1'
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

class SumTree:
    
    def __init__(self, limit: int = 4096, reorder_interval: int = 1_000_000):
        '''
        This class implements a sum tree to be used by the prioritized experience replay memory module.
        
        Parameters
        ----------
        limit :                             The maximum number of experiences that can be stored.
        reorder_interval :                  The interval length after which the sum tree is reordered.
        
        Returns
        ----------
        None
        '''
        self.limit = limit
        self.sumtree_total = (2 * limit) - 1
        self.start = limit - 1
        self.length = 0
        self.entry_pos = 0
        self.reorder_counter = 0
        self.reorder_interval = reorder_interval
        # store max priorities for O(1) experience storing and importance weight sampling
        self.max_prio = 1.
        # sumtree structure storing priorities (parent = left_child + right_child)
        self.sumtree = np.zeros(self.sumtree_total) 
        # separate array storing experiences
        self.experiences = np.full(limit, None, dtype=np.object)
        # separate array storing sumtree indexes sorted by priority (ascending)
        # this array is only sorted once every 'reorder_interval' steps to avoid massive computations 
        self.order = np.arange(self.start, self.sumtree_total)

    def get_total(self) -> float:
        '''
        Return total sum of stored priorities.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        priority_sum :                      The first element of the sumtree.
        '''
        return self.sumtree[0]

    def get_experience(self, sample_value: float) -> (int, float, dict):
        '''
        Return a stored experience by sampling from the sumtree according to the given sample priority.
        
        Parameters
        ----------
        sample_value :                      Priority used to sample from the sumtree datastructure.
        
        Returns
        ----------
        idx :                               The retrieved experience's index.
        priority :                          The retrieved experience's priority.
        experience :                        The retrieved experience.
        '''
        if self.length == 0:
            raise AttributeError
        pos = self._find(0, sample_value)
        idx = pos - self.start

        return idx, self.sumtree[pos], self.experiences[idx]
    
    def get_experience_by_idx(self, idx: int) -> (int, float, dict):
        '''
        Return an experience by index.
        
        Parameters
        ----------
        idx :                               The experience's index.
        
        Returns
        ----------
        idx :                               The retrieved experience's index.
        priority :                          The retrieved experience's priority.
        experience :                        The retrieved experience.
        '''
        return idx, self.sumtree[idx+self.start], self.experiences[idx]

    def shuffle_experiences(self):
        '''
        Shuffle experiences in memory ( DOES NOT CHANGE PRIORITIES, ONLY USE THIS WHEN ALL PRIORITIES ARE EQUAL!).
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        np.random.shuffle(self.experiences)

    def put_experience(self, state0: np.ndarray, action: int, reward: float, state1: np.ndarray, terminal1: bool):
        '''
        Store an experience in memory

        # Argument
            state0 (array_like): Original state of the agent
            action (int):        Action taken in state0
            reward (float):      Reward received after taking action
            state1 (array_like): Resulting state after action 
            terminal1 (bool):    Flag describing state1 being the final state or not
        '''
        exp = Experience([state0], action, reward, [state1], terminal1)
        # if the memory is not filled, store experiences in order
        if self.length < self.limit:
            pos = self.start + self.entry_pos
        # if the memory is full, replace the stored experience with the lowest priority
        else:
            pos = self.order[0]
            self.order = np.append(self.order[1:],self.order[0])
        diff = self.max_prio - self.sumtree[pos]
        self.sumtree[pos] = self.max_prio
        self.experiences[pos - self.start] = exp
        self._update_parents(pos, diff)
        self.length = min(self.length + 1, self.limit)
        self.entry_pos = (self.entry_pos + 1) % self.limit

    def update_prios(self, idxs: list, newPrios: list):
        '''
        Updates the priorities of multiple stored experiences by index. Occasionally sorts stored indeces.
        
        Parameters
        ----------
        idxs :                              The indeces of stored experiences that will be modified.
        newPrios :                          The new experience priorities.
        
        Returns
        ----------
        None
        '''
        for i, new_prio in zip(idxs, newPrios): 
            pos = self.start + i
            diff = new_prio - self.sumtree[pos]
            self.sumtree[pos] = new_prio
            self._update_parents(pos, diff)
            self.max_prio = max(self.max_prio, new_prio)
        # if the memory module is filled
        if self.length >= self.limit:
            self.reorder_counter += 1
            # sort 'self.order'-array in the way that the stored priorities would be sorted
            if self.reorder_counter >= self.reorder_interval:
                self.reorder_counter = 0
                ordered_idxs = np.argsort(self.sumtree[self.start:])
                self.order = self.order[ordered_idxs]

    def _find(self, current_pos: int, sample_value: float) -> int:
        '''
        Return sumtree position of sample.
        
        Parameters
        ----------
        current_pos :                       Call with 0 (root node), recursively descends sumtree.
        sample_value :                      Call with value between 0 and self.getTotal() to sample an entry.
        
        Returns
        ----------
        idx :                               The index of a sampled experience in self.sumtree.
        '''
        left_pos = (2 * current_pos) + 1
        right_pos = left_pos + 1
        if left_pos >= self.sumtree_total:
            return current_pos
        if sample_value < self.sumtree[left_pos] or right_pos >= self.sumtree_total:
            # go down left node
            return self._find(left_pos, sample_value)
        else:
            # go down right node
            sample_value = sample_value - self.sumtree[left_pos]
            return self._find(right_pos, sample_value)

    def _update_parents(self, current_pos: int, diff: float):
        '''
        Propagates changes upwards to uphold the invariant "Parent = leftChild + rightChild".
        
        Parameters
        ----------
        current_pos :                       Position of previously changed node in sumtree.
        diff :                              The difference between the old and the new value.
        
        Returns
        ----------
        None
        '''
        # Could be improved by only updating every second node, since parent nodes sum up over both child nodes
        # by calling function after changing multiple values and then checking distance between indexes
        # end propagation at root node
        if current_pos == 0:
            return
        parent = (current_pos - 1) // 2
        self.sumtree[parent] += diff
        # propagate upwards
        self._update_parents(parent, diff)


class PERMemory(Memory):
    
    def __init__(self, limit: int, reorder_interval: int = 1_000_000, alpha: float = 0.6, beta: float = 0.4, output_layer_name: str = 'output', **kwargs):
        '''
        This class implements the prioritized experience replay memory module to be used with the PER DQN agent.
        
        Parameters
        ----------
        limit :                             The maximum number of experiences that can be stored.
        reorder_interval :                  The interval length after which the sum tree is reordered.
        alpha :                             Prioritized experience replay's alpha parameter.
        beta :                              Prioritized experience replay's beta parameter.
        output_layer_name :                 The name of the DQN agent's output layer.
        
        Returns
        ----------
        None
        '''
        super().__init__(**kwargs)
        self.memory = SumTree(limit, reorder_interval)
        self.alpha = alpha
        self.beta = beta
        self.beta_change = 0.001
        # needs to know output layer to build sample_weights dictionary
        self.output_layer_name = output_layer_name        
        self.sample_weights = {}
        self.prevSample = []
        self.td_errors = np.zeros(32)
        # storing previous 'append' function call's parameters, because memory storing requires following state
        self.prevState = None
        self.prevAction = None
        self.prevReward = None
        self.prevTerminal = None

    def append(self, observation: np.ndarray, action: int, reward: float, terminal1: bool, training: bool = True):
        '''
        Stores an experience in memory.
        
        Parameters
        ----------
        observation :                       The agent's observation.
        action :                            The selected action.
        reward :                            The reward received.
        terminal1 :                         True if the next state is terminal.
        training :                          True if called during in training.
        
        Returns
        ----------
        None
        '''
        # if we are not training we do have to change the memory
        if not training:
            return
        # store experience in memory
        if self.prevState is not None:
            self.memory.put_experience(self.prevState, self.prevAction, self.prevReward, observation, self.prevTerminal)
        # if we reached a terminal state the next transition cannot be stored
        if self.prevTerminal is True:
            self.prevState = None
            self.prevAction = None
            self.prevReward = None
            self.prevTerminal = None
        else:
            # store previous values to append next function call           
            self.prevState = observation 
            self.prevAction = action
            self.prevReward = reward
            self.prevTerminal = terminal1

    def sample(self, batch_size: int, batch_idxs: None | np.ndarray = None) -> list:
        '''
        Return a random sample of the memory using prioritized experience replay.
        
        Parameters
        ----------
        batch_size :                        The size of the batch to sample.
        batch_idxs :                        The indeces to retrieve.
        
        Returns
        ----------
        experiences :                       A list of experiences.
        '''
        # This method also computes importance sampling weights and  
        # uses the td-errors of sampled experiences to update their priorities
        # update priorities using the td-errors from last sample
        if self.prevSample:
            new_prios = (np.abs(self.td_errors) ** self.alpha) + 1e-5
            self.memory.update_prios(self.prevSample, new_prios)
            self.prevSample = []
        batch_samples = []
        experiences = []
        priorities = []
        segment_size = self.memory.get_total() / batch_size
        # choose indexes of experiences if none are passed 
        if batch_idxs is None:
            # choose a sample uniformly inside k (#batch_size) segments
            lower_bounds = [segment_size * i for i in range(batch_size)]
            upper_bounds = lower_bounds[1:] + [self.memory.get_total()]
            batch_samples = np.random.uniform(lower_bounds, upper_bounds, batch_size)
            # collect chosen memories
            for s in batch_samples:
                idx, prob, exp = self.memory.get_experience(s)
                self.prevSample.append(idx)
                priorities.append(prob)
                experiences.append(exp)
        # else use passed indexes to choose experiences if enough are available
        elif len(batch_idxs) > self.memory.length:
            for i in batch_idxs:
                idx, prob, exp = self.memory.get_experience_by_idx(i)
                self.prevSample.append(idx)
                priorities.append(prob)
                experiences.append(exp)
        # else choose repeated samples to fill an entire batch
        else:
            amount_missing = self.memory.length - len(batch_idxs)
            batch_idxs += [np.random.randint(0, self.memory.length, size = amount_missing)]
            for i in batch_idxs:
                idx, prob, exp = self.memory.get_experience_by_idx(i)
                
                self.prevSample.append(idx)
                priorities.append(prob)
                experiences.append(exp)
        # compute importance sampling weights
        self.set_importance_sampling_weights(priorities)

        assert len(experiences) == batch_size
        return experiences

    def set_importance_sampling_weights(self, priorities: list):
        '''
        Sets the importance sampling weights to be used by the agent during training.
        
        Parameters
        ----------
        priorities :                        Probability for each sample retrieved from memory.
        
        Returns
        ----------
        None
        '''
        if self.memory.get_total() == 0:
            return
        # calculate P(i) from the individual priorities
        probabilities = priorities / self.memory.get_total()
        # calculate scaled down importance sampling weights = (N * P(i))^-beta / max w_i
        is_weights = np.power(np.multiply(self.memory.length, probabilities), -self.beta)
        is_weights /= np.max(is_weights)
        # anneal beta towards 1 over the course of the training
        self.beta = min(self.beta + self.beta_change, 1.)
        self.sample_weights[self.output_layer_name] = np.array(is_weights)
