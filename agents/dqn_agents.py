# basic imports
import numpy as np
# keras imports
from tensorflow.keras import callbacks as callback_keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
# keras-rl imports
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
# framework imports
from cobel.agents.rl_agent import AbstractRLAgent, callbacks
from cobel.memory_modules.dqn_memory import PERMemory


class DQNAgentBaseline(AbstractRLAgent):
    '''
    This class implements a DQN agent.
    The original step-based training behavior of the keras-rl2 DQN agent is overwritten to be trial-based.
    
    | **Args**
    | interface_OAI:                The interface to the Open AI Gym environment
    | memory_capacity:              The capacity of the sequential memory used in the agent.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | model:                        The network model to be used by the DQN agent. If None, a default network will be created.
    | custom_callbacks:             The custom callbacks defined by the user.
    '''
    
    class callbacksDQN(callbacks, callback_keras.Callback):
        '''
        Callback class. Used for visualization and scenario control.
        Inherits from CoBeL-RL callback and Keras callback.
        
        | **Args**
        | rl_parent:                    Reference to the RL agent.
        | custom_callbacks:             The custom callbacks defined by the user.
        '''
        
        def __init__(self, rl_parent, custom_callbacks={}):
            super().__init__(rl_parent=rl_parent, custom_callbacks=custom_callbacks)
            
        def on_episode_begin(self, epoch, logs):
            '''
            The following function is called whenever an epsisode starts.
            
            | **Args**
            | epoch:                        The current trial.
            | logs:                         A dictionary containing logs of the simulation.
            '''
            logs['trial'] = self.rl_parent.current_trial - 1
            super().on_trial_begin(logs)
            
        def on_episode_end(self, epoch, logs):
            '''
            The following function is called whenever an episode ends, and updates the reward accumulator,
            simultaneously updating the visualization of the reward function.
            
            | **Args**
            | epoch:                        The current trial.
            | logs:                         A dictionary containing logs of the simulation.
            '''
            if 'nb_episode_steps' in logs:
                logs['steps'] = logs['nb_episode_steps']
            else:
                logs['steps'] = logs['nb_steps']
            # update trial count
            self.rl_parent.current_trial += 1
            self.rl_parent.session_trial += 1
            logs['trial'] = self.rl_parent.current_trial - 1
            logs['session_trial'] = self.rl_parent.session_trial - 1
            logs['trial_reward'] = logs['episode_reward']
            # stop training after the maximum number of trials was reached
            if self.rl_parent.session_trial >= self.rl_parent.max_trials:
                self.rl_parent.agent.step = self.rl_parent.max_steps + 1
            super().on_trial_end(logs)
            
        def on_step_begin(self, step, logs):
            '''
            The following function is called whenever a step begins, and executes callbacks defined by the user.
            
            | **Args**
            | step:                         The current step.
            | logs:                         The trial log.
            '''
            super().on_step_begin(logs)
             
        def on_step_end(self, step, logs):
            '''
            The following function is called whenever a step ends, and executes callbacks defined by the user.
            
            | **Args**
            | step:                         The current step.
            | logs:                         The trial log.
            '''
            super().on_step_end(logs)
                
    def __init__(self, interface_OAI, memory_capacity=1000000, epsilon=0.3, model=None, custom_callbacks={}): 
        super().__init__(interface_OAI, custom_callbacks=custom_callbacks)
        # build model
        self.model = model
        if self.model is None:
            self.build_model()
        # prepare the memory for the RL agent
        self.memory = SequentialMemory(limit=memory_capacity, window_length=1)
        # define the available policies
        policy = EpsGreedyQPolicy(epsilon)
        # define the maximum number of steps
        self.max_steps = 10**10
        # keeps track of current trial
        self.current_trial = 0 # trial count across all sessions (i.e. calls to the train/simulate method)
        self.session_trial = 0 # trial count in current seesion (i.e. current call to the train/simulate method)
        # define the maximum number of trials
        self.max_trials = 0
        # construct the agent
        self.agent = DQNAgent(model=self.model, nb_actions=self.number_of_actions, memory=self.memory, gamma=0.8, nb_steps_warmup=100, enable_dueling_network=False,
                            dueling_type='avg', target_model_update=1e-2, policy=policy, batch_size=32)
        # compile the agent
        self.agent.compile(Adam(lr=.001,), metrics=['mse'])
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engaged_callbacks = self.callbacksDQN(self, custom_callbacks)
        
    def build_model(self):
        '''
        This function builds a default network model for the DQN agent.
        '''
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,) + self.interface_OAI.observation_space.shape))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=self.number_of_actions, activation='linear', name='output'))

    def train(self, number_of_trials=100, max_number_of_steps=100, replay_batch_size=32):
        '''
        This function is called to train the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the RL agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
        | replay_batch_size:            The number of random that will be replayed.
        '''
        self.max_trials = number_of_trials
        self.session_trial = 0
        self.agent.batch_size = replay_batch_size
        # call the fit method to start the RL learning process
        self.agent.fit(self.interface_OAI, nb_steps=self.max_steps, verbose=0, callbacks=[self.engaged_callbacks], nb_max_episode_steps=max_number_of_steps, visualize=False)
    
    def test(self, number_of_trials=100, max_number_of_steps=100):
        '''
        This function is called to test the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the RL agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
        '''
        self.max_trials = number_of_trials
        self.session_trial = 0
        # call the fit method to start the RL learning process
        self.agent.test(self.interface_OAI, nb_episodes=number_of_trials, verbose=0, callbacks=[self.engaged_callbacks], nb_max_episode_steps=max_number_of_steps, visualize=False)
    
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''
        return self.agent.model.predict_on_batch(batch)


class DQNAgentSTR(DQNAgentBaseline):
    '''
    This class implements a DQN agent.
    The DQN agent implemented by this class learns state-action transition and reward models which it can utilize to simulate experiences.
    The original step-based training behavior of the keras-rl2 DQN agent is overwritten to be trial-based.
    
    | **Args**
    | interface_OAI:                The interface to the Open AI Gym environment
    | memory_capacity:              The capacity of the sequential memory used in the agent.
    | epsilon:                      The epsilon value for the epsilon greedy policy..
    | model:                        The network model to be used by the DQN agent. If None, a default network will be created.
    | custom_callbacks:             The custom callbacks defined by the user.
    '''
    
    class callbacksSTR(DQNAgentBaseline.callbacksDQN):
        '''
        Callback class. Used for visualization and scenario control.
        Provides the RL agent's internal model with experiences.
        
        | **Args**
        | rl_parent:                    Reference to the RL agent.
        | custom_callbacks:             The custom callbacks defined by the user.
        '''
        
        def __init__(self, rl_parent, custom_callbacks={}):        
            super().__init__(rl_parent, custom_callbacks=custom_callbacks)        
            # store recent initial_state
            self.initial_state = None

        def on_step_begin(self, step, logs):
            '''
            The following function is called whenever a step begins and stores the current trial's initial state.
            
            | **Args**
            | step:                         The current trial step.
            | logs:                         A dictionary containing logs of the simulation.
            '''
            self.initial_state = self.rl_parent.interface_OAI.observation
            super().on_step_begin(step, logs)
        
        def on_step_end(self, step, logs):
            '''
            The following function is called whenever a step ends and sends the current experience to the RL agent's internal model.
            
            | **Args**
            | step:                         The current trial step.
            | logs:                         A dictionary containing logs of the simulation.
            '''
            if not self.rl_parent.simulating:
                logs['initial_state'] = self.initial_state
                self.rl_parent.internal_OAI.store_and_learn(logs)
            super().on_step_end(step, logs)
            
    class internal_model():
        '''
        Interface for the environmental model of the agent.
        Keeps track of the most recent experiences and learns a model of the environment.
        
        | **Args**
        | rlParent:                     Reference to the RL agent.
        | window:                       Window length of the running batch.
        '''
        
        def __init__(self, rl_parent=None, window=128):
            self.rl_parent = rl_parent
            # possible initial observations
            self.initial_observations = None
            # number of recent experiences stored
            self.window = window
            # (short-term) memory structures
            self.states, self.actions, self.rewards, self.follow_up_states, self.terminals = [], [], [], [], []
            # current simulated observation
            self.observation = None
            # the threshold at which a state is considered terminal
            self.terminal_threshold = .5
            # the number of epochs that the environmental model is trained for on each step
            self.epochs = 1
            # build environmental models
            self.build_transition_model()
            self.build_reward_model()
            self.build_terminal_model()
            
        def store_and_learn(self, logs):
            '''
            The following function stores the experience contained in the logs dictionary
            and trains the environmental model on the most recent experiences.
            
            | **Args**
            | logs:                         A dictionary containing logs of the simulation.
            '''
            # leep track of running batch
            self.states = self.states[-(self.window - 1):] + [logs['initial_state']]
            action = np.zeros(self.rl_parent.number_of_actions)
            action[logs['action']] = 1.
            self.actions = self.actions[-(self.window - 1):] + [action]
            self.rewards = self.rewards[-(self.window - 1):] + [logs['reward']]
            self.follow_up_states = self.follow_up_states[-(self.window - 1):] + [logs['observation']]
            self.terminals = self.terminals[-(self.window - 1):] + [float(logs['reward'] > 0.)]
            # update environmental models
            for epoch in range(self.epochs):
                self.model_transition.train_on_batch(np.concatenate((np.array(self.states), np.array(self.actions)), axis=1), np.array(self.follow_up_states))
                self.model_reward.train_on_batch(np.array(self.follow_up_states), np.array(self.rewards))
                self.model_terminal.train_on_batch(np.array(self.follow_up_states), np.array(self.terminals))
            
        def build_transition_model(self):
            '''
            This function builds a default state-action transition model.
            '''
            self.model_transition = Sequential()
            self.model_transition.add(Flatten(input_shape=(np.product(self.rl_parent.interface_OAI.observation_space.shape) + self.rl_parent.number_of_actions, )))
            self.model_transition.add(Dense(units=64, activation='tanh'))
            self.model_transition.add(Dense(units=64, activation='tanh'))
            self.model_transition.add(Dense(units=np.product(self.rl_parent.interface_OAI.observation_space.shape), activation='linear'))
            self.model_transition.compile(optimizer='adam', loss='mse')
        
        def build_reward_model(self):
            '''
            This function builds a default reward model.
            '''
            self.model_reward = Sequential()
            self.model_reward.add(Flatten(input_shape=(np.product(self.rl_parent.interface_OAI.observation_space.shape), )))
            self.model_reward.add(Dense(units=64, activation='tanh'))
            self.model_reward.add(Dense(units=64, activation='tanh'))
            self.model_reward.add(Dense(units=1, activation='linear'))
            self.model_reward.compile(optimizer='adam', loss='mse')
            
        def build_terminal_model(self):
            '''
            This function builds a default terminal model.
            '''
            self.model_terminal = Sequential()
            self.model_terminal.add(Flatten(input_shape=(np.product(self.rl_parent.interface_OAI.observation_space.shape), )))
            self.model_terminal.add(Dense(units=64, activation='tanh'))
            self.model_terminal.add(Dense(units=64, activation='tanh'))
            self.model_terminal.add(Dense(units=1, activation='sigmoid'))
            self.model_terminal.compile(optimizer='adam', loss='mse')
        
        def step(self, action):
            '''
            The interal model's step function.
            Experiences are simulated using the environmental model.
            
            | **Args**
            | action:                       The action selected by the agent.
            '''
            # one-hot encoding of the selected action
            action_enc = np.zeros((1, self.rl_parent.number_of_actions))
            action_enc[0, action] = 1.
            # simulate experience
            self.observation = self.model_transition.predict_on_batch(np.concatenate((self.observation, action_enc), axis=1))
            reward = self.model_reward.predict_on_batch(self.observation)[0][0]
            terminal = self.model_terminal.predict_on_batch(self.observation)[0][0]
            
            return self.observation[0], reward, terminal > self.terminal_threshold, {}
        
        def reset(self):
            '''
            The internal model's reset function which resets the intermodal to one of the possible initial observations.
            
            | **Args**
            | step:                         The current trial step.
            | logs:                         A dictionary containing logs of the simulation.
            '''
            self.observation = np.array([self.initial_observations[np.random.randint(self.initial_observations.shape[0])]])
            
            return self.observation[0]
                
    def __init__(self, interface_OAI, memory_capacity=1000000, epsilon=0.3, model=None, custom_callbacks={}):   
        super().__init__(interface_OAI, memory_capacity, epsilon, model=model, custom_callbacks=custom_callbacks)
        # initialize interal model interface
        self.internal_OAI = self.internal_model(self)
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engaged_callbacks = self.callbacksSTR(self, custom_callbacks)
        # is the agent simulating experiences
        self.simulating = False
        
    def simulate(self, number_of_trials=100, max_number_of_steps=100, replay_batch_size=32, initial_observations=None):
        '''
        This function is called to train the agent using experiences simulated by the agent's interal model of the environment.
        
        | **Args**
        | number_of_trials:             The number of trials the RL agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
        | replay_batch_size:            The number of random that will be replayed.
        | initial_observations:         The initial observation when simulating.
        '''
        self.simulating = True
        self.max_trials = number_of_trials
        self.session_trial = 0
        self.agent.batch_size = replay_batch_size
        self.internal_OAI.initial_observations = initial_observations
        # call the fit method to start the RL learning process
        self.agent.fit(self.internal_OAI, nb_steps=self.max_steps, verbose=0, callbacks=[self.engaged_callbacks], nb_max_episode_steps=max_number_of_steps, visualize=False)
        self.simulating = False
        
        
class PERDQNAgent(DQNAgent):
    '''
    This class extends Keras-RL's DQN agent to use prioritized experience replay.
    Has to be used in conjunction with the appropriate memory module.
    '''
    def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=False, enable_dueling_network=False, dueling_type='avg', *args, **kwargs):
        super().__init__(model, policy, test_policy, enable_double_dqn, enable_dueling_network, dueling_type, *args, **kwargs)
        
    def backward(self, reward, terminal):
        '''
        Backward method of Keras-RL's DQN agent extended to compute and pass over temporal difference errors to its memory module.
        Furthermore, the network is updated using sample weights.
        '''
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal, training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch, reward_batch, action_batch, terminal1_batch, state1_batch = [], [], [], [], []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch, state1_batch = self.process_state_batch(state0_batch), self.process_state_batch(state1_batch)
            terminal1_batch, reward_batch = np.array(terminal1_batch), np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch


            # passing the temporal difference errors to the memory module
            Qs = self.model.predict_on_batch(state0_batch)
            Qs = [q[a] for q, a in zip(Qs, action_batch)]
            self.memory.td_errors = [r - q for q, r in zip(Qs, Rs)]


            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets], sample_weight=self.memory.sample_weights)
            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics
    

class PERDQNAgentBaseline(DQNAgentBaseline):
    '''
    This class implements a baseline agent which uses prioritized experience replay.
    
    | **Args**
    | interface_OAI:                The interface to the Open AI Gym environment
    | memory_capacity:              The capacity of the sequential memory used in the agent.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | model:                        The network model to be used by the DQN agent. If None, a default network will be created. The name of the ouput layer must be "output".
    | custom_callbacks:             The custom callbacks defined by the user.
    '''
    def __init__(self, interface_OAI, memory_capacity=1000000, epsilon=0.3, model=None, custom_callbacks={}):
        # store the Open AI Gym interface
        self.interface_OAI = interface_OAI
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.number_of_actions = self.interface_OAI.action_space.n
        # build model
        self.model = model
        if self.model is None:
            self.build_model()
        # prepare the memory for the RL agent
        self.memory = PERMemory(limit=memory_capacity, window_length=1)
         # define the available policies
        policy = EpsGreedyQPolicy(epsilon)
        # define the maximum number of steps
        self.max_steps = 10**10
        # keeps track of current trial
        self.current_trial = 0 # trial count across all sessions (i.e. calls to the train/simulate method)
        self.session_trial = 0 # trial count in current seesion (i.e. current call to the train/simulate method)
        # define the maximum number of trials
        self.max_trials = 0
        # construct the agent
        self.agent = PERDQNAgent(model=self.model, nb_actions=self.number_of_actions, memory=self.memory, nb_steps_warmup=35, enable_dueling_network=False,
                            dueling_type='avg', target_model_update=1e-2, policy=policy, batch_size=32)
        # compile the agent
        self.agent.compile(Adam(lr=.001,), metrics=['mse'])
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engaged_callbacks = self.callbacksDQN(self, custom_callbacks)