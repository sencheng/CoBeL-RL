


import numpy     as np
import tensorflow as tf
import datetime

from keras import callbacks
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Add
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.agents import DDPGAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from pathlib import Path

# Wraps an Keras DDPGAgent.
class DDPGAgentBaseline():
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
    def __init__(self, interfaceOAI, memoryCapacity=10000, learning_rate=0.001, trialBeginFcn=None, trialEndFcn=None):

        # store the AI Gym interface
        self.interfaceOAI=interfaceOAI

        # the number of actions, retrieved from the Open AI Gym interface
        self.nb_actions = self.interfaceOAI.action_space.n

        # actor model
        actor_hidden1_units = 512

        # a sequential model is standardly used here, this model is subject to changes
        self.actor = Sequential()
        self.actor.add(Flatten(input_shape=(1,)+self.interfaceOAI.observation_space.shape))
        self.actor.add(Dense(units=actor_hidden1_units, activation='tanh'))
        self.actor.add(Dense(units=actor_hidden1_units, activation='tanh'))
        self.actor.add(Dense(units=actor_hidden1_units, activation='tanh'))
        self.actor.add(Dense(units=self.nb_actions, activation='tanh'))

        # prepare the memory for the RL agent
        self.memory=SequentialMemory(limit=memoryCapacity,window_length=1)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=.15, mu=0., sigma=.3)

        # critic network.
        critic_hidden1_units = 300
        critic_hidden2_units = 600
        # observation input
        self.observation_input = Input(shape=(1,) + self.interfaceOAI.observation_space.shape, name='observation_input')
        self.flattened_observation = Flatten()(self.observation_input)
        w1 = Dense(critic_hidden1_units, activation='relu')(self.flattened_observation)
        h1 = Dense(critic_hidden2_units, activation='linear')(w1)
        # action input
        self.action_input = Input(shape=(self.nb_actions,), name='action_input')
        a1 = Dense(critic_hidden2_units, activation='linear')(self.action_input)
        # merge both
        h2 = Add()([h1,a1])
        h3 = Dense(critic_hidden2_units, activation='relu')(h2)
        V = Dense(1, activation='linear')(h3)
        # create critics model
        self.critic = Model(input=[self.observation_input, self.action_input], output=V)
        '''
        x = Concatenate()([self.action_input, self.flattened_observation])
        x = Dense(16)(x)
        x = Activation('relu')(x)
        x = Dense(16)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        self.critic = Model(inputs=[self.action_input, self.observation_input], outputs=x)
        '''
        # construct the agent
        self.agent = DDPGAgent(nb_actions=self.nb_actions,
                          actor=self.actor,
                          critic=self.critic,
                          critic_action_input=self.action_input,
                          memory=self.memory,
                          nb_steps_warmup_critic=198,
                          nb_steps_warmup_actor=198,
                          random_process=self.random_process,
                          gamma=.99,
                          target_model_update=1e-3
                          )
        # compile agent
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

        # set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks=self.callbacks(self,trialBeginFcn,trialEndFcn)

    ### The following function is called to train the agent.
    def train(self,steps):
        self.maxSteps=steps

        # setup Tensorboard for logging
        log_dir = Path("logs/fit/" + datetime.datetime.now().strftime("%Y.%m.%d-%H-%M-%S"))  # create OS-agnostic path
        log_dir = str(log_dir)                                                               # extract as string
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

        # call the fit method to start the RL learning process

        self.agent.fit(self.interfaceOAI,
                       nb_steps=steps,
                       verbose=0,
                       callbacks=[self.engagedCallbacks], #, tensorboard_callback],
                       nb_max_episode_steps=198, # don't know if this is in sync with unity episodes.
                       visualize=False,
                       )

    ### The nested visualization class that is required by 'KERAS-RL' to visualize the training success (by means of episode reward)
    ### at the end of each episode, and update the policy visualization.
    class callbacks(callbacks.Callback):


        # The constructor.
        #
        # rlParent:     the ACT_ReinforcementLearningModule that hosts this class
        # trialBeginFcn:the callback function called in the beginning of each trial, defined for more flexibility in scenario control
        # trialEndFcn:  the callback function called at the end of each trial, defined for more flexibility in scenario control
        def __init__(self,rlParent,trialBeginFcn=None,trialEndFcn=None):

            super(DDPGAgentBaseline.callbacks,self).__init__()

            # store the hosting class
            self.rlParent=rlParent

            # store the trial end callback function
            self.trialBeginFcn=trialBeginFcn

            # store the trial end callback function
            self.trialEndFcn=trialEndFcn



        # The following function is called whenever an epsisode starts,
        # and updates the visual output in the plotted reward graphs.
        def on_episode_begin(self,epoch,logs):

            # retrieve the Open AI Gym interface
            interfaceOAI=self.rlParent.interfaceOAI

            if self.trialBeginFcn is not None:
                self.trialBeginFcn(epoch,self.rlParent)


        # The following function is called whenever an episode ends, and updates the reward accumulator,
        # simultaneously updating the visualization of the reward function
        def on_episode_end(self,epoch,logs):

            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch,self.rlParent,logs)


### The reinforcement learing class. It wraps all functionality required to set up a RL agent.
class RobotDQNAgent():


    ### The nested visualization class that is required by 'KERAS-RL' to visualize the training success (by means of episode reward)
    ### at the end of each episode, and update the policy visualization.
    class callbacks(callbacks.Callback):


        # The constructor.
        #
        # rlParent:     the ACT_ReinforcementLearningModule that hosts this class
        # trialBeginFcn:the callback function called in the beginning of each trial, defined for more flexibility in scenario control
        # trialEndFcn:  the callback function called at the end of each trial, defined for more flexibility in scenario control
        def __init__(self,rlParent,trialBeginFcn=None,trialEndFcn=None):

            super(RobotDQNAgent.callbacks,self).__init__()

            # store the hosting class
            self.rlParent=rlParent

            # store the trial end callback function
            self.trialBeginFcn=trialBeginFcn

            # store the trial end callback function
            self.trialEndFcn=trialEndFcn



        # The following function is called whenever an epsisode starts,
        # and updates the visual output in the plotted reward graphs.
        def on_episode_begin(self,epoch,logs):

            # retrieve the Open AI Gym interface
            interfaceOAI=self.rlParent.interfaceOAI

            if self.trialBeginFcn is not None:
                self.trialBeginFcn(epoch,self.rlParent)


        # The following function is called whenever an episode ends, and updates the reward accumulator,
        # simultaneously updating the visualization of the reward function
        def on_episode_end(self,epoch,logs):

            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch,self.rlParent,logs)




    # The constructor.
    def __init__(self, interfaceOAI,         # the interface to the environment.
                       memoryCapacity=1000000, # total size of the replay buffer (how many experiences to collect before updating).
                       sampleSize=32,      # number of experiences uses for one iteration of gradient decent.
                       memoryWindow=5,       # number of observation to concatenate.
                       learningRate=0.00025,   # learning factor determines how fast weights change.
                       epsilon=0.3,          # epsilon factor determines how random the agent will act.
                       trialBeginFcn=None,   # episode begin callback (used for plotting).
                       trialEndFcn=None      # episode end callback (used for plotting).
                       ):

        # store the Open AI Gym interface
        self.interfaceOAI=interfaceOAI
        # prepare the model used in the reinforcement learner
        # the number of actions, retrieved from the Open AI Gym interface
        self.nb_actions = self.interfaceOAI.action_space.n
        self.observation_shape = self.interfaceOAI.observation_space.shape
        # prepare the memory for the RL agent
        memory = SequentialMemory(limit=memoryCapacity, window_length=memoryWindow)
        # additional network parameters
        MODEL_HIDDEN_UNITS = 1024
        MODEL_HIDDEN_LAYERS = 3
        WARMUP_STEPS = 50000
        POLICY = EpsGreedyQPolicy(epsilon)

        #
        # MODEL FOR VECTOR OBSERVATIONS ########################################
        #

        # the input layer is a tensor with dimensions MEMORY_WINDOW x observation_vector inputs.
        input = Input((memoryWindow,) + self.observation_shape)
        # this input tensor is flattened to a 1 x n vector.
        network = Flatten()(input)
        # then the hidden layers are added on top.
        for i in range(MODEL_HIDDEN_LAYERS):
            network = Dense(units=MODEL_HIDDEN_UNITS, activation='tanh')(network)
        # finally the output layer is added.
        output = Dense(units=self.nb_actions, activation='linear')(network)
        # create model
        self.model = Model(inputs=input, outputs=output)
        # debug printing model info.
        print(self.model.summary())

        #
        # AGENT ################################################################
        #

        # construct the agent.
        self.agent=DQNAgent(model=self.model,
                            nb_actions=self.nb_actions,
                            memory=memory,
                            nb_steps_warmup=WARMUP_STEPS,
                            target_model_update=10000,
                            train_interval=5,
                            policy=POLICY,
                            batch_size=sampleSize)

        # compile the agent
        self.agent.compile(Adam(lr=learningRate,), metrics=['mae'])

        #
        # CALLBACKS ############################################################
        #

        # setup Tensorboard for logging
        log_dir = Path("logs/fit/" + datetime.datetime.now().strftime("%Y.%m.%d-%H-%M-%S"))  # create OS-agnostic path
        log_dir = str(log_dir)                                                               # extract as string
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

        # set up the visualizer for the RL agent behavior/reward outcome
        visualizer_callback = self.callbacks(self,trialBeginFcn,trialEndFcn)

        # store callbacks
        self.engagedCallbacks=[tensorboard_callback, visualizer_callback]

    ### The following function is called to train the agent.
    def train(self,steps):
        self.maxSteps=steps

        # call the fit method to start the RL learning process
        self.agent.fit(self.interfaceOAI, nb_steps=steps, verbose=0, nb_max_episode_steps=1000,
                       callbacks=self.engagedCallbacks,visualize=False)
