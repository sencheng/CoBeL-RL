import os
from keras import callbacks
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


class ModularDDPGAgentBaseline:
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

            super(ModularDDPGAgentBaseline.callbacks, self).__init__()

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
                 create_memory_fcn=None, create_actor_fcn=None, create_critic_fcn=None,
                 random_process=None,
                 nb_max_episode_steps=None,
                 train_interval=1, memory_interval=1, memory_window=1,
                 batch_size=32, target_model_update=10000, learning_rate=0.00025, gamma=0.99, metrics=["mse"],
                 trial_begin_fcn=None, trial_end_fcn=None, other_callbacks=None):
        """
        Constructor

        :param oai_env:                 the env for the agent to act in
        :param create_memory_fcn:       the memory modul func
        :param nb_max_episode_steps:    the max steps per episode. note that this parameters can also be set in the env
        :param train_interval:          how often to do training
        :param memory_interval:         how often store experiences
        :param memory_window:           how many observations will be concatenated to a single input signal
        :param batch_size:              the size of training batches
        :param target_model_update:     how often to update the target model
        :param learning_rate:           defines how fast the model adapts
        :param gamma:                   discount for future rewards
        :param metrics:                 the metrics used for error calculation
        :param trial_begin_fcn:         the function that is called at the beginning of every episode
        :param trial_end_fcn:           the function that is called at the end of every episode
        :param other_callbacks:         additional callback to pass to the keras agent
        """

        # check needed parameters
        assert oai_env is not None
        assert create_memory_fcn is not None
        assert create_actor_fcn is not None
        assert create_critic_fcn is not None
        assert random_process is not None

        # save parameters
        self.nb_max_episode_steps = nb_max_episode_steps
        self.gamma = gamma
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.batch_size = batch_size
        self.memory_window = memory_window
        self.target_model_update = target_model_update
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.random_process = random_process

        # store the Open AI Gym interface
        self.interfaceOAI = oai_env

        # prepare the model parameters
        nb_actions = self.interfaceOAI.action_space.n
        observation_shape = self.interfaceOAI.observation_shape

        # construct the agent
        self.agent = DDPGAgent(nb_actions=nb_actions,
                               actor=create_actor_fcn(),
                               critic=create_critic_fcn(),
                               critic_action_input=self.critic_action_input,
                               memory=create_memory_fcn(self.memory_window),
                               gamma=self.gamma,
                               batch_size=self.batch_size,
                               nb_steps_warmup_actor=self.nb_steps_warmup_actor,
                               nb_steps_warmup_critic=self.nb_steps_warmup_critic,
                               train_interval=self.train_interval,
                               memory_interval=self.memory_interval,
                               delta_range=self.delta_range,
                               delta_clip=self.delta_clip,
                               random_process=self.random_process,
                               custom_model_objects=self.custom_model_objects,
                               target_model_update=self.target_model_update)

        # compile the agent
        self.agent.compile(Adam(lr=self.learning_rate, ), metrics=self.metrics)

        # merge the callbacks
        cobel_callback = self.callbacks(self, trial_begin_fcn, trial_end_fcn)
        other_callbacks.append(cobel_callback)
        self.callbacks = other_callbacks

    def save(self, path):
        """
        saves the agents weights to the folder models/

        :param path: the path
        """
        if os.path.exists(path):
            self.agent.model.save_weights(path)
            print(">>> Model saved to ", path)
        else:
            print(">>> Model not saved.", path, "not found.")

    def load(self, path):
        """
        loads the agents weights from the given path

        :param path: the path
        """
        if os.path.exists(path):
            self.agent.model.load_weights(path)
            print(">>> Model loaded from ", path)
        else:
            print(">>> Model not loaded.", path, "not found.")

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

    :param limit:   max number of stored experiences
    :return:        function that creates a keras sequential memory
    """

    def get_memory(memory_window):
        """
        this function is called by the ModularDQNAgent to instantiate it's network memory.

        :param memory_window: number of observations that are processed as a single input
        :return:        keras sequential memory
        """
        return SequentialMemory(limit=limit, window_length=memory_window)

    return get_memory