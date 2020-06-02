


import numpy     as np
import tensorflow as tf
import datetime

from keras import callbacks
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
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
    def __init__(self, interfaceOAI, memoryCapacity=10000, trialBeginFcn=None, trialEndFcn=None):

        # store the AI Gym interface
        self.interfaceOAI=interfaceOAI

        # prepare the model used in the reinforcement learner

        # the number of actions, retrieved from the Open AI Gym interface
        self.nb_actions = self.interfaceOAI.action_space.n

        # a sequential model is standardly used here, this model is subject to changes
        self.actor = Sequential()
        self.actor.add(Flatten(input_shape=(1,)+self.interfaceOAI.observation_space.shape))
        self.actor.add(Dense(units=64,activation='tanh'))
        self.actor.add(Dense(units=64,activation='tanh'))
        self.actor.add(Dense(units=64,activation='tanh'))
        self.actor.add(Dense(units=64,activation='tanh'))
        self.actor.add(Dense(units=self.nb_actions,activation='tanh'))

        # prepare the memory for the RL agent
        self.memory=SequentialMemory(limit=memoryCapacity,window_length=1)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=.15, mu=0., sigma=.3)

        # DDPGAgent pendulum example critic network.
        self.action_input = Input(shape=(self.nb_actions,), name='action_input')
        self.observation_input = Input(shape=(1,) + self.interfaceOAI.observation_space.shape, name='observation_input')
        self.flattened_observation = Flatten()(self.observation_input)
        x = Concatenate()([self.action_input, self.flattened_observation])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        self.critic = Model(inputs=[self.action_input, self.observation_input], outputs=x)

        print("nb_actions", self.nb_actions)
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
class DQNAgentBaseline():


    ### The nested visualization class that is required by 'KERAS-RL' to visualize the training success (by means of episode reward)
    ### at the end of each episode, and update the policy visualization.
    class callbacks(callbacks.Callback):


        # The constructor.
        #
        # rlParent:     the ACT_ReinforcementLearningModule that hosts this class
        # trialBeginFcn:the callback function called in the beginning of each trial, defined for more flexibility in scenario control
        # trialEndFcn:  the callback function called at the end of each trial, defined for more flexibility in scenario control
        def __init__(self,rlParent,trialBeginFcn=None,trialEndFcn=None):

            super(DQNAgentBaseline.callbacks,self).__init__()

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
    def __init__(self, interfaceOAI,memoryCapacity=10000,epsilon=0.3,trialBeginFcn=None,trialEndFcn=None):

        # store the Open AI Gym interface
        self.interfaceOAI=interfaceOAI



        # prepare the model used in the reinforcement learner

        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.nb_actions = self.interfaceOAI.action_space.n
        # a sequential model is standardly used here, this model is subject to changes
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,)+self.interfaceOAI.observation_space.shape))
        self.model.add(Dense(units=64,activation='tanh'))
        self.model.add(Dense(units=64,activation='tanh'))
        self.model.add(Dense(units=64,activation='tanh'))
        self.model.add(Dense(units=64,activation='tanh'))
        self.model.add(Dense(units=self.nb_actions,activation='linear'))

        # prepare the memory for the RL agent
        self.memory=SequentialMemory(limit=memoryCapacity,window_length=1)

        # define the available policies
        policyEpsGreedy=EpsGreedyQPolicy(epsilon)

        # construct the agent

        # Retrieve the agent's parameters from the agentParams dictionary
        self.agent=DQNAgent(model=self.model,
                            nb_actions=self.nb_actions,
                            memory=self.memory,
                            nb_steps_warmup=198,
                            enable_dueling_network=False,
                            dueling_type='avg',
                            target_model_update=1e-2,
                            policy=policyEpsGreedy,
                            batch_size=32)

        # compile the agent
        self.agent.compile(Adam(lr=.001,), metrics=['mse'])

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

        self.agent.fit(self.interfaceOAI, nb_steps=steps, verbose=0,
                       callbacks=[tensorboard_callback,self.engagedCallbacks],nb_max_episode_steps=10000,visualize=False,
                       )
