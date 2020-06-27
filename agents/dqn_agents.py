import numpy as np
import tensorflow as tf
import os
from keras import callbacks
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Add
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory



### The reinforcement learing class. It wraps all functionality required to set up a RL agent.
class DQNAgentBaseline():
    ### The nested visualization class that is required by 'KERAS-RL' to visualize the training success (by means of episode reward)
    ### at the end of each episode, and update the policy visualization.
    class callbacks(callbacks.Callback):

        # The constructor.
        # 
        # rlParent:     the ACT_ReinforcementLearningModule that hosts this class
        # trialBeginFcn:the callback function called in the beginning of each trial, defined for more flexibility in scenario control
        # trialEndFcn:  the callback function called at the end of each trial, defined for more flexibility in scenario control
        def __init__(self, rlParent, trialBeginFcn=None, trialEndFcn=None):

            super(DQNAgentBaseline.callbacks, self).__init__()

            # store the hosting class
            self.rlParent = rlParent

            # store the trial end callback function
            self.trialBeginFcn = trialBeginFcn

            # store the trial end callback function
            self.trialEndFcn = trialEndFcn

        # The following function is called whenever an epsisode starts,
        # and updates the visual output in the plotted reward graphs.
        def on_episode_begin(self, epoch, logs):

            # retrieve the Open AI Gym interface
            interfaceOAI = self.rlParent.interfaceOAI

            if self.trialBeginFcn is not None:
                self.trialBeginFcn(epoch, self.rlParent)

        # The following function is called whenever an episode ends, and updates the reward accumulator,
        # simultaneously updating the visualization of the reward function
        def on_episode_end(self, epoch, logs):

            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch, self.rlParent, logs)

    # The constructor.
    # 
    # guiParent:        the widget that shows necessary visualization
    # interfaceOAI:     the interface to the Open AI Gym environment
    # agentParams:      the parameters of the agent to be used, provided as a dictionary
    # visualOutput:     true if the module has to visualize the results
    # maxEpochs:        the maximum number of epochs to be logged
    # memoryCapacity:   the capacity of the sequential memory used in the agent
    # epsilon:          the epsilon value for the epsilon greedy policy
    # trialBeginFcn:    the callback function called at the beginning of each trial, defined for more flexibility in scenario control
    # trialEndFcn:      the callback function called at the end of each trial, defined for more flexibility in scenario control
    def __init__(self, interfaceOAI, memoryCapacity=10000, epsilon=0.3, trialBeginFcn=None, trialEndFcn=None):

        # store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI

        # prepare the model used in the reinforcement learner

        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.nb_actions = self.interfaceOAI.action_space.n
        # a sequential model is standardly used here, this model is subject to changes
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,) + self.interfaceOAI.observation_space.shape))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=64, activation='tanh'))

        self.model.add(Dense(units=self.nb_actions, activation='linear'))

        # prepare the memory for the RL agent
        self.memory = SequentialMemory(limit=memoryCapacity, window_length=1)

        # define the available policies
        policyEpsGreedy = EpsGreedyQPolicy(epsilon)
        # construct the agent

        # Retrieve the agent's parameters from the agentParams dictionary
        self.agent = DQNAgent(model=self.model, nb_actions=self.nb_actions, memory=self.memory, nb_steps_warmup=100,
                              enable_dueling_network=False, dueling_type='avg', target_model_update=1e-2,
                              policy=policyEpsGreedy, batch_size=32)

        # compile the agent
        self.agent.compile(Adam(lr=.001, ), metrics=['mse'])

        # set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks = self.callbacks(self, trialBeginFcn, trialEndFcn)

    ### The following function is called to train the agent.
    def train(self, steps):
        # call the fit method to start the RL learning process
        self.maxSteps = steps
        self.agent.fit(self.interfaceOAI, nb_steps=steps, verbose=0, callbacks=[self.engagedCallbacks],
                       nb_max_episode_steps=100, visualize=False)


class ModularDQNAgentBaseline:
    """
    Wrapper for a keras DQNAgent.
    All parameters including the memory and model structure can be defined by the constructor.
    """

    # The nested visualization class that is required by 'KERAS-RL' to visualize the training success
    # (by means of episode reward)
    # at the end of each episode, and update the policy visualization.
    class callbacks(callbacks.Callback):

        # The constructor.
        #
        # rlParent:     the ACT_ReinforcementLearningModule that hosts this class trialBeginFcn:the callback function
        # called in the beginning of each trial, defined for more flexibility in scenario control trialEndFcn:  the
        # callback function called at the end of each trial, defined for more flexibility in scenario control
        def __init__(self, rlParent, trialBeginFcn=None, trialEndFcn=None):

            super(ModularDQNAgentBaseline.callbacks, self).__init__()

            # store the hosting class
            self.rlParent = rlParent

            # store the trial end callback function
            self.trialBeginFcn = trialBeginFcn

            # store the trial end callback function
            self.trialEndFcn = trialEndFcn

        # The following function is called whenever an epsisode starts,
        # and updates the visual output in the plotted reward graphs.
        def on_episode_begin(self, epoch, logs):

            # retrieve the Open AI Gym interface
            interfaceOAI = self.rlParent.interfaceOAI

            if self.trialBeginFcn is not None:
                self.trialBeginFcn(epoch, self.rlParent)

        # The following function is called whenever an episode ends, and updates the reward accumulator,
        # simultaneously updating the visualization of the reward function
        def on_episode_end(self, epoch, logs):

            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch, self.rlParent, logs)

    def __init__(self, oai_env=None,
                 policy=EpsGreedyQPolicy(eps=.3), create_memory_fcn=None, create_model_fcn=None,
                 nb_steps_warmup=50000, nb_max_episode_steps=None,
                 nb_max_start_steps=0, start_step_policy=None,
                 action_repetition=1, train_interval=1, memory_interval=1, memory_window=1,
                 batch_size=32, target_model_update=10000, learning_rate=0.00025, gamma=0.99, metrics=["mse"],
                 enable_double_dqn=True, enable_dueling_network=False, dueling_type="avg",
                 trial_begin_fcn=None, trial_end_fcn=None, other_callbacks=None):
        """
        Constructor
        :param oai_env:            the env for the agent to act in
        :param policy:                  the agents policy
        :param create_memory_fcn:       the memory modul func
        :param create_model_fcn:        the model modul func
        :param nb_steps_warmup:         the number of steps to act randomly before starting the rl process
        :param nb_max_episode_steps:    the max steps per episode
        :param nb_max_start_steps:      the max steps to use the start policy at the beginning of each episode
        :param start_step_policy:       the policy to use at the start of every episode
        :param action_repetition:       how often to repeat an action
        :param train_interval:          how often to do training
        :param memory_interval:         how often store experiences
        :param memory_window:           how many observations will be concatenated to a single input signal
        :param batch_size:              the size of training batches
        :param target_model_update:     how often to update the target model
        :param learning_rate:           defines how fast the model adapts
        :param gamma:                   discount for future rewards
        :param metrics:                 the metrics used for error calculation
        :param enable_double_dqn:       enables the double dqn feature
        :param enable_dueling_network:  enables the dueling network feature
        :param dueling_type:            dueling type
        :param trial_begin_fcn:         the function that is called at the beginning of every episode
        :param trial_end_fcn:           the function that is called at the end of every episode
        :param other_callbacks:         additional callback to pass to the keras agent
        """

        # check needed parameters
        assert oai_env is not None
        assert create_memory_fcn is not None
        assert create_model_fcn is not None
        assert policy is not None

        # save parameters
        self.policy = policy
        self.nb_steps_warmup = nb_steps_warmup
        self.nb_max_episode_steps = nb_max_episode_steps
        self.nb_max_start_steps = nb_max_start_steps
        self.start_step_policy = start_step_policy
        self.action_repetition = action_repetition
        self.gamma = gamma
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.batch_size = batch_size
        self.memory_window = memory_window
        self.target_model_update = target_model_update
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        self.learning_rate = learning_rate
        self.metrics = metrics

        # store the Open AI Gym interface
        self.interfaceOAI = oai_env

        # prepare the model used in the reinforcement learner
        # the number of actions, retrieved from the Open AI Gym interface
        nb_actions = self.interfaceOAI.action_space.n
        observation_shape = self.interfaceOAI.observation_space.shape

        # construct the agent.
        self.agent = DQNAgent(nb_actions=nb_actions,
                              policy=self.policy,
                              memory=create_memory_fcn(self.memory_window),
                              model=create_model_fcn(observation_shape, nb_actions, self.memory_window),
                              gamma=self.gamma,
                              memory_interval=self.memory_interval,
                              train_interval=self.train_interval,
                              nb_steps_warmup=self.nb_steps_warmup,
                              target_model_update=self.target_model_update,
                              batch_size=self.batch_size,
                              enable_double_dqn=self.enable_double_dqn,
                              enable_dueling_network=self.enable_dueling_network,
                              dueling_type=self.dueling_type)

        # compile the agent
        self.agent.compile(Adam(lr=self.learning_rate, ), metrics=self.metrics)

        # add the callbacks
        cobel_callback = self.callbacks(self, trial_begin_fcn, trial_end_fcn)
        other_callbacks.append(cobel_callback)
        self.callbacks = other_callbacks

    def save(self, name):
        """
        saves the agents weights to the folder models/
        :param name: name of the file
        """
        path = "models/" + name
        if os.path.exists(path):
            self.agent.model.save_weights(path)
            print("Model saved to ", path)
        else:
            print("Model not saved.", path, "not found.")

    def load(self, path):
        """
        loads the agents weights from the given path
        :param path: the path
        """
        if os.path.exists(path):
            self.agent.model.load_weights(path)
            print("Model loaded from ", path)
        else:
            print("Model not loaded.", path, "not found.")

    def train(self, nb_steps):
        """
        trains the agent
        :param nb_steps: how many steps to train
        """
        # call the fit method to start the RL learning process
        self.agent.fit(self.interfaceOAI,
                       nb_steps=nb_steps, nb_max_episode_steps=self.nb_max_episode_steps,
                       action_repetition=self.action_repetition,
                       nb_max_start_steps=self.nb_max_start_steps, start_step_policy=self.start_step_policy,
                       callbacks=self.callbacks, visualize=False, verbose=0)

    def test(self, nb_episodes):
        """
        tests the agent
        :param nb_episodes: how many episodes to test
        """
        #
        self.agent.test(self.interfaceOAI,
                        nb_episodes=nb_episodes, nb_max_episode_steps=self.nb_max_episode_steps,
                        action_repetition=self.action_repetition,
                        nb_max_start_steps=self.nb_max_start_steps, start_step_policy=self.start_step_policy,
                        callbacks=self.callbacks, visualize=False, verbose=0)


def sequential_memory_modul(limit=10000):
    """
    returns a function that returns a memory for given parameters
    """

    def get_memory(memory_window):
        """
        this function is called by the ModularDQNAgent to instantiate it's network memory.
        :memory_window: number of observations that are processed as a single input
        """
        return SequentialMemory(limit=limit, window_length=memory_window)

    return get_memory


def sequential_model_modul(nb_units=64, nb_layers=4, activation="tanh"):
    """
    returns a function that returns a model for given parameters
    """

    def get_model(observation_shapes, nb_actions, memory_window):
        """
        this function is called by the ModularDQNAgent to instantiate it's network model.
        :param observation_shapes:  the shape of the observation space
        TODO: extend to multiple observation shapes to support multiple sensors
        :param nb_actions:          the size of the action space
        :param memory_window:       number of observations that are processed as a single input
        """

        print("configured model input shape: ", observation_shapes)

        # the input layer is a tensor with dimensions MEMORY_WINDOW x observation shape.
        network_input = Input((memory_window,) + observation_shapes)

        # this input tensor is flattened to a 1 x n vector.
        network = Flatten()(network_input)

        # then the hidden layers are added on top.
        for i in range(nb_layers):
            network = Dense(units=nb_units, activation=activation)(network)

        # finally the output layer is added.
        output = Dense(units=nb_actions, activation='linear')(network)

        return Model(inputs=network_input, outputs=output)

    return get_model
