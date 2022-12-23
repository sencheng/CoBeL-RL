import random
import time
from decimal import *
import numpy as np
from collections import deque
from sklearn.neighbors import KDTree
# keras-rl imports
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, GreedyQPolicy
# framework imports
from cobel.agents.rl_agent import AbstractRLAgent, callbacks

class QEC:
    def __init__(self, actions, buffer_size, k):
        self.buffers = tuple([ActionBuffer(buffer_size) for _ in actions])
        self.k = k

    def estimate(self, state, action):
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)

        if state_index:
            return buffer.values[state_index]

        if len(buffer) <= self.k:
            return 0.0 #float("inf")

        value = 0.0
        neighbors = buffer.find_neighbors(state, self.k)
        for neighbor in neighbors:
            value += buffer.values[neighbor]
        return value / self.k

    def update(self, state, action, value, time):
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)
        if state_index:
            max_value = max(buffer.values[state_index], value)
            max_time = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)
        else:
            buffer.add(state, value, time)

    def update_episode(self, episode):
        for event in episode:
            self.update(event['state'], event['action'], event['accumulative'], event['time'])



class ActionBuffer:
    def __init__(self, capacity):
        self._tree = None
        self.capacity = capacity
        self.states = []
        self.values = []
        self.times = []

    def find_state(self, state):
        if self._tree:
            neighbor_idx = self._tree.query([state])[1][0][0]
            #print(np.linalg.norm(state - self.states[neighbor_idx]))
            if np.allclose(self.states[neighbor_idx], state, rtol=1e-04, atol=1e-06):
                return neighbor_idx
        return None

    def find_neighbors(self, state, k):
        return self._tree.query([state], k)[1][0] if self._tree else []

    def add(self, state, value, time):
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.times.append(time)
        else:
            min_time_idx = int(np.argmin(self.times))
            if time > self.times[min_time_idx]:
                self.replace(state, value, time, min_time_idx)
        self._tree = KDTree(self.states)

    def replace(self, state, value, time, index):
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time

    def __len__(self):
        return len(self.states)


class EMControl(AbstractRLAgent):

    class callbacks(callbacks):

        def __init__(self, rl_parent, custom_callbacks={}):
            '''
            Callback class. Used for visualization and scenario control.
            Inherits from CoBeL-RL callback and Keras callback.
            
            Parameters
            ----------
            rl_parent :                         Reference to the RL agent.
            custom_callbacks :                  The custom callbacks defined by the user.
            
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
            epoch :                             The current trial.
            logs :                              A dictionary containing logs of the simulation.
            
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
            epoch :                             The current trial.
            logs :                              A dictionary containing logs of the simulation.
            
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
            


    def __init__(self, interface_OAI, memoryCapacity=2000, k=3, gamma=0.97, epsilon=0.1, use_random_projections=True, pretrained_model=None, custom_callbacks={}):
        '''
        params:
            interfaceOAI: open ai interface
            memoryCapacity: the maximal length of the memory
            k: the k-nearest method when estimating the Q function for a state-action pair
            gamma: discount factor for the rewards in rl
            epsilon: episilon greedy method in Q learning framework
        others:
            QEC: the element inside EM is a dict() which stores a state, its Q values on each action, and when
                             the state was visited last time
        '''

        super().__init__(interface_OAI, custom_callbacks=custom_callbacks)

        # store the Open AI Gym interface
        self.interface_OAI = interface_OAI
        self.memoryCapacity=memoryCapacity
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.nb_action = self.interface_OAI.action_space.n
        self.k=k
        self.gamma=gamma
        # define the available policies
        self.policy = EpsGreedyQPolicy(epsilon)
        self.test_policy = GreedyQPolicy()
        # keeps track of current trial
        self.current_trial = 0 # trial count across all sessions (i.e. calls to the train/simulate method)
        self.session_trial = 0 # trial count in current seesion (i.e. current call to the train/simulate method)
        # define the maximum number of trials
        self.max_trials = 0
        # define the maximum number of steps
        self.max_steps = 10**10

        self.Q_EC=QEC(range(self.nb_action), self.memoryCapacity, self.k)
        self.current_episode=deque()
        self.recent_observation=None
        self.recent_action=0
        self.use_random_projections=use_random_projections
        self.init_projection=False
        self.pretrained_model=pretrained_model
        self.state_size = 256  # size of the embedded state used by random projections

        # set up the visualizer for the RL agent behavior/reward outcome
        self.engaged_callbacks = self.callbacks(self, custom_callbacks)
        
        self.training = False


    def forward(self, observation):
        '''
        Perform epsilon-greedy to select an action, update the current state and action
        '''
        # process the observation and embed it to a lower-dimension
        current_state = self.process_observation(observation)
        action_values = [
            self.Q_EC.estimate(current_state, action)
            for action in range(self.nb_action)
        ]
        action_values = np.array(action_values)
        if self.training:
            action = self.policy.select_action(q_values=action_values)
        else:
            action = self.test_policy.select_action(q_values=action_values)

        self.recent_observation=current_state
        self.recent_action=action

        return action

    def book_keeping(self, observation, action):
        '''
        Record the most recent obs and action
        '''
        # process the observation and embed it to a lower-dimension
        current_state = self.process_observation(observation)
        self.recent_observation = current_state
        self.recent_action = action

    def backward(self, reward, terminal):
        '''
        The agent observe the reward and a terminal signal. If the episode is not terminal, just store the current state, action, reward tuple
        in the self.current_episode; if terminate, update the episodic memory with the current episode
        '''
        event={
            'state': self.recent_observation,
            'action': self.recent_action,
            'reward': reward,
            'time': time.time()
        }
        # store the current event
        self.current_episode.append(event)
        # if the episode terminates, update the episodic memory
        if terminal:
            self.current_episode=self.accmu_reward(self.current_episode)
            self.Q_EC.update_episode(self.current_episode)
            self.current_episode.clear()
     

    def accmu_reward(self, episode):
        episode_length = len(episode)
        for i in range(episode_length):
            accumulative=0.0
            for j in range(episode_length-1, i-1, -1):     # from the end step to the ith step
                accumulative=(accumulative+episode[j]['reward'])*self.gamma
            accumulative/=self.gamma  # divide the extra gamma
            episode[i]['accumulative']=accumulative
        return episode


    def process_observation(self, observation):
        '''
        The function for mapping the original, high-dimensional observation data to more abstract, lower-dimensional state.
        The methods can be random projection or by using a pretrained model like Variational autoencoder (VAE).
        '''
        if self.pretrained_model!=None:
            observation=self.pretrained_model.get_activations(observation).flatten()

        if self.use_random_projections:
            # flatten the observation into a single vector
            observation=observation.flatten()
            # if for the first time, generate a random projection
            if not self.init_projection:
                dim_h=len(observation)
                dim_low=self.state_size
                self.random_projections=np.float32(np.random.randn(dim_h, dim_low))
                self.init_projection=True
            processed_observation=np.dot(observation, self.random_projections)

        return processed_observation

    def compute_q_value(self, observation):
        processed_state = self.process_observation(observation)
        # extract the Q values for this state from EM
        q_value = [
            self.QEC.estimate(processed_state, action)
            for action in range(self.nb_action)
        ]
        return q_value


    def train(self, number_of_trials=100, max_number_of_steps=100):
        '''
        This function is called to train the agent.

        | **Args**
        | numberOfTrials:               The number of trials the RL agent is trained.
        | maxNumberOfSteps:             The maximum number of steps per trial.
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


    def test(self, number_of_trials=100, max_number_of_steps=50):
        '''
        This function is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials that the agent is tested.
        max_number_of_steps :               The maximum number of steps per trial.
        
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
        batch :                             The batch of states for which Q-values should be retrieved.
        
        Returns
        ----------
        predictions :                       The batch of Q-value predictions.
        '''
        Q_values = []
        for state in batch:
            Q_values.append(self.compute_q_value(state))

        return np.asarray(Q_values)
