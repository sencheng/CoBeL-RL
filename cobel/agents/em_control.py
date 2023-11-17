# basic imports
import time
import numpy as np
from sklearn.neighbors import KDTree
# framework imports
from cobel.agents.rl_agent import AbstractRLAgent, callbacks
from cobel.policy.policy import AbstractPolicy
from cobel.interfaces.rl_interface import AbstractInterface
from cobel.networks.network import AbstractNetwork


class QEC():
    
    def __init__(self, actions: int, buffer_size: int, k: int):
        '''
        This class implements the MFEC agent's EM-based Q-function. 
        
        Parameters
        ----------
        actions :                           The number of actions available to the agent.\n
        buffer_size :                       The memory capacity for each action.\n
        k :                                 The number of nearest neighbors that will be used to estimate the Q-values.\n
        
        Returns
        ----------
        None
        '''
        self.buffers = tuple([ActionBuffer(buffer_size) for _ in range(actions)])
        self.k = k

    def estimate(self, state: np.ndarray, action: int) -> float:
        '''
        This function estimates the Q-value for a given state-action pair. 
        
        Parameters
        ----------
        state :                             The given state.\n
        action :                            The given action.\n
        
        Returns
        ----------
        q :                                 The estimated Q-value.\n
        '''
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)
        # entry in buffer
        if state_index:
            return buffer.values[state_index]
        # entry not in buffer and not enough neighbors for estimation
        if len(buffer) <= self.k:
            return 0.0
        # for estimate using the k-nearest neighbors
        value = 0.0
        neighbors = buffer.find_neighbors(state, self.k)
        for neighbor in neighbors:
            value += buffer.values[neighbor]
            
        return value / self.k

    def update(self, state: np.ndarray, action: int, value: float, time: float):
        '''
        This function updates the EM-based Q-function. 
        
        Parameters
        ----------
        state :                             The given state.\n
        action :                            The given action.\n
        value :                             The new Q-value.\n
        time :                              The current time.\n
        
        Returns
        ----------
        None
        '''
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)
        if state_index:
            max_value = max(buffer.values[state_index], value)
            max_time = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)
        else:
            buffer.add(state, value, time)

    def update_episode(self, episode: list):
        '''
        This function updates the EM-based Q-function with an episode.
        
        Parameters
        ----------
        episode :                           A list containing the single events/experiences.\n
        
        Returns
        ----------
        None
        '''
        for event in episode:
            self.update(event['state'], event['action'], event['accumulative'], event['time'])


class ActionBuffer():
    
    def __init__(self, capacity: int):
        '''
        This class implements the memory structure used by the QEC class.
        
        Parameters
        ----------
        capacity :                          The capacity of the memory structure.\n
        
        Returns
        ----------
        None
        '''
        self._tree = None
        self.capacity = capacity
        self.states = []
        self.values = []
        self.times = []

    def find_state(self, state: np.ndarray) -> None | int:
        '''
        This functions searches the memory structure for a given state.
        
        Parameters
        ----------
        state :                             The state that will be searched for.\n
        
        Returns
        ----------
        neighbor_index :                    The index of the state. Returns \'None\' if the state could not be found.\n
        '''
        if self._tree:
            neighbor_idx = self._tree.query([state])[1][0][0]
            if np.allclose(self.states[neighbor_idx], state, rtol=1e-04, atol=1e-06):
                return neighbor_idx
            
        return None

    def find_neighbors(self, state: np.ndarray, k: int) -> list | np.ndarray:
        '''
        This function looks for a given state's k-nearest neighbors. 
        
        Parameters
        ----------
        state :                             The given state.\n
        
        Returns
        ----------
        neighbors :                         A numpy array containing the k-nearest neighbors' indeces. Returns an empty list if the memory is empty.\n
        '''
        return self._tree.query([state], k)[1][0] if self._tree else []

    def add(self, state: np.ndarray, value: float, time: float):
        '''
        This function adds an entry to memory. 
        
        Parameters
        ----------
        state :                             The given state.\n
        value :                             The given state's value.\n
        time :                              The time associated with entry.\n
        
        Returns
        ----------
        None
        '''
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.times.append(time)
        else:
            min_time_idx = int(np.argmin(self.times))
            if time > self.times[min_time_idx]:
                self.replace(state, value, time, min_time_idx)
        self._tree = KDTree(self.states)

    def replace(self, state: np.ndarray, value: float, time: float, index: int):
        '''
        This function replaces an older entry with the given one.
        
        Parameters
        ----------
        state :                             The given state.\n
        value :                             The given state's value.\n
        time :                              The time associated with the new entry.\n
        index :                             The index of the entry that will be replaced.\n
        
        Returns
        ----------
        None
        '''
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time

    def __len__(self) -> int:
        return len(self.states)


class MFECAgent(AbstractRLAgent):

    class callbacks(callbacks):

        def __init__(self, rl_parent, custom_callbacks: None | dict = None):
            '''
            Callback class. Used for visualization and scenario control.
            Inherits from CoBeL-RL callback and Keras callback.
            
            Parameters
            ----------
            rl_parent :                         Reference to the RL agent.\n
            custom_callbacks :                  The custom callbacks defined by the user.\n
            
            Returns
            ----------
            None
            '''
            super().__init__(rl_parent=rl_parent, custom_callbacks=custom_callbacks)

        def on_episode_begin(self, epoch: int, logs: dict):
            '''
            The following function is called whenever an epsisode starts, and executes callbacks defined by the user.
            
            Parameters
            ----------
            epoch :                             The current trial.\n
            logs :                              A dictionary containing logs of the simulation.\n
            
            Returns
            ----------
            None
            '''
            logs['trial'] = self.rl_parent.current_trial - 1
            logs['trial_session'] = self.rl_parent.session_trial - 1
            super().on_trial_begin(logs)
            
        def on_episode_end(self, epoch: int, logs: dict):
            '''
            The following function is called whenever an epsisode ends, and executes callbacks defined by the user.
            
            Parameters
            ----------
            epoch :                             The current trial.\n
            logs :                              A dictionary containing logs of the simulation.\n
            
            Returns
            ----------
            None
            '''
            if 'nb_episode_steps' in logs:
                logs['steps'] = logs['nb_episode_steps']
            else:
                logs['steps'] = logs['nb_steps']
            # update trial count
            self.rl_parent.current_trial += 1
            self.rl_parent.session_trial += 1
            logs['trial'] = self.rl_parent.current_trial - 1
            logs['trial_session'] = self.rl_parent.session_trial - 1
            logs['trial_reward'] = logs['episode_reward']
            # stop training after the maximum number of trials was reached
            if self.rl_parent.session_trial >= self.rl_parent.max_trials:
                self.rl_parent.agent.step = self.rl_parent.max_steps + 1
            super().on_trial_end(logs)


    def __init__(self, interface_OAI: AbstractInterface, policy: AbstractPolicy, policy_test: None | AbstractPolicy = None,
                 memory_capacity: int = 2000, k: int = 3, gamma: float = 0.97, use_random_projection: bool = True,
                 pretrained_model: None | AbstractNetwork = None, custom_callbacks: None | dict = None):
        '''
        This class implements the Model Free Episodic Control agent described by Blundell et al. (2016).
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.\n
        policy :                            The agent's action selection policy.\n
        policy_test :                       The agent's action selection policy during testing. If unspecified the agent uses the train policy.\n
        memory_capacity :                   The capacity of the agent's memory.\n
        k :                                 The number of nearest neighbors that will be used to estimate Q-values.\n
        gamma :                             The discount factor used for computing the target values.\n
        use_random_projections :            A flag indicating whether observations will be transformed using a random projection.\n
        pretrained_model :                  An optional pretrained model that will be used to transform observations.\n
        custom_callbacks :                  The custom callbacks defined by the user.\n
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, custom_callbacks=custom_callbacks)
        # store the Open AI Gym interface
        self.interface_OAI = interface_OAI
        self.memory_capacity = memory_capacity
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.nb_action = self.interface_OAI.action_space.n
        self.k = k
        self.gamma = gamma
        # define the available policies
        self.policy = policy
        self.policy_test = policy_test if policy_test else policy
        # keeps track of current trial
        self.current_trial = 0 # trial count across all sessions (i.e. calls to the train/simulate method)
        self.session_trial = 0 # trial count in current seesion (i.e. current call to the train/simulate method)
        # define the maximum number of trials
        self.max_trials = 0
        # define the maximum number of steps
        self.max_steps = 10 ** 10
        self.Q_EC = QEC(self.nb_action, self.memory_capacity, self.k)
        self.current_episode = []
        self.recent_observation = None
        self.recent_action = 0
        self.use_random_projection = use_random_projection
        self.random_projection = None
        self.pretrained_model = pretrained_model
        self.state_size = 256  # size of the embedded state used by random projections
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engaged_callbacks = self.callbacks(self, custom_callbacks)
        self.training = False

    def forward(self, observation: np.ndarray) -> int:
        '''
        This function implements the agent's action selection.
        
        Parameters
        ----------
        observation :                       The current observation.\n
        
        Returns
        ----------
        action :                            The selected action.\n
        '''
        self.recent_observation = self.process_observation(observation)
        q_values = np.array([self.Q_EC.estimate(self.recent_observation, action) for action in range(self.nb_action)])
        self.recent_action = self.policy.select_action(q_values) if self.training else self.policy_test.select_action(q_values)

        return self.recent_action

    def backward(self, reward: float, terminal: bool):
        '''
        This function stores experiences of the current episode and updates the Q-function when the episode ends.
        
        Parameters
        ----------
        reward :                            The received reward.\n
        terminal :                          A flag indicating whether the episode ended.\n
        
        Returns
        ----------
        None
        '''
        # store the current event
        event = {'state': self.recent_observation, 'action': self.recent_action,
                 'reward': reward, 'time': time.time()}
        self.current_episode.append(event)
        # if the episode terminates, update the episodic memory
        if terminal:
            self.current_episode = self.accmu_reward(self.current_episode)
            self.Q_EC.update_episode(self.current_episode)
            self.current_episode = []

    def accmu_reward(self, episode: list) -> list:
        '''
        This function computes and stores the discounted episodic reward for each experience of an episode.
        
        Parameters
        ----------
        episode :                           The episode as a list of experiences.\n
        
        Returns
        ----------
        episode :                           The updated episode.\n
        '''
        R = 0.
        for exp in episode[::-1]:
            R = self.gamma * R + exp['reward']
            exp['accumulative'] = R
            
        return episode

    def process_observation(self, observation: np.ndarray) -> np.ndarray:
        '''
        This function processes an observation by either random projection or a network model.
        
        Parameters
        ----------
        observation :                       The given observation.\n
        
        Returns
        ----------
        processed_observation :             The processed observation.\n
        '''
        if self.pretrained_model:
            return self.pretrained_model.predict_on_batch(observation).flatten()
        elif self.use_random_projection:
            # if for the first time, generate a random projection
            if self.random_projection is None:
                self.random_projection = np.random.randn(np.product(observation.shape), self.state_size).astype(np.float32)
            return np.dot(observation.flatten(), self.random_projection)

        return observation.flatten()

    def compute_q_value(self, observation: np.ndarray) -> np.ndarray:
        '''
        This function computes the Q-values for a given observation.
        
        Parameters
        ----------
        observation :                       The given observation.\n
        
        Returns
        ----------
        q_values :                          The Q-values.\n
        '''
        return np.array([self.Q_EC.estimate(self.process_observation(observation), action) for action in range(self.nb_action)])

    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 100):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials that the RL agent is trained.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
        Returns
        ----------
        None
        '''
        self.training = True
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                # determine next action
                action = self.forward(state)
                # perform action
                next_state, reward, terminal, callback_value = self.interface_OAI.step(action)
                # update the agent
                self.backward(reward, terminal)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if terminal:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)

    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 50):
        '''
        This function is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials that the agent is tested.\n
        max_number_of_steps :               The maximum number of steps per trial.\n
        
        Returns
        ----------
        None
        '''
        self.training = False
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                # determine next action
                action = self.forward(state)
                # perform action
                next_state, reward, terminal, callback_value = self.interface_OAI.step(action)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if terminal:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            
    def predict_on_batch(self, batch: np.ndarray) -> np.ndarray:
        '''
        This function retrieves Q-values for a batch of states.
        
        Parameters
        ----------
        batch :                             The batch of states for which Q-values should be retrieved.\n
        
        Returns
        ----------
        predictions :                       The batch of Q-value predictions.\n
        '''
        return np.array([self.compute_q_value(state) for state in batch])
