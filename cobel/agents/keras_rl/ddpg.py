# basic imports
import numpy as np
# keras imports
from tensorflow.keras import callbacks as callback_keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input, Activation
from tensorflow.keras.optimizers import Adam
# keras-rl imports
from rl.agents import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory
# framework imports
from cobel.agents.rl_agent import AbstractRLAgent, callbacks


class DDPGAgentBaseline(AbstractRLAgent):
    
    class callbacksDDPG(callbacks, callback_keras.Callback):
        
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
            logs['session_trial'] = self.rl_parent.session_trial - 1
            logs['trial_reward'] = logs['episode_reward']
            # stop training after the maximum number of trials was reached
            if self.rl_parent.session_trial >= self.rl_parent.max_trials:
                self.rl_parent.agent.step = self.rl_parent.max_steps + 1
            super().on_trial_end(logs)
            
        def on_step_begin(self, step: int, logs: dict):
            '''
            The following function is called whenever a step begins, and executes callbacks defined by the user.
            
            Parameters
            ----------
            step :                              The current trial step.
            logs :                              A dictionary containing logs of the simulation.
            
            Returns
            ----------
            None
            '''
            super().on_step_begin(logs)
             
        def on_step_end(self, step: int, logs: dict):
            '''
            The following function is called whenever a step ends, and executes callbacks defined by the user.
            
            Parameters
            ----------
            step :                              The current trial step.
            logs :                              A dictionary containing logs of the simulation.
            
            Returns
            ----------
            None
            '''
            super().on_step_end(logs)
                
    def __init__(self, interface_OAI, memory_capacity=1000000, model_actor=None, model_critic=None, action_input=None, custom_callbacks={}):
        '''
        This class implements a DDPG agent.
        The original step-based training behavior of the keras-rl2 DDPG agent is overwritten to be trial-based.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        memory_capacity :                   The capacity of the sequential memory used in the agent.
        model_actor :                       The network actor model to be used by the DDPG agent.
        model_critic :                      The network critic model to be used by the DDPG agent.
        action_input :                      The network critic model's action input.
        custom_callbacks :                  The custom callbacks defined by the user.
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, custom_callbacks=custom_callbacks)
        # build models
        self.build_models(model_actor, model_critic, action_input)
        # prepare the memory for the RL agent
        self.memory = SequentialMemory(limit=memory_capacity, window_length=1)
        # define the available policies
        random_process = OrnsteinUhlenbeckProcess(size=self.number_of_actions, theta=.15, mu=0., sigma=.3)
        # define the maximum number of steps
        self.max_steps = 10**10
        # keeps track of current trial
        self.current_trial = 0 # trial count across all sessions (i.e. calls to the train/simulate method)
        self.session_trial = 0 # trial count in current seesion (i.e. current call to the train/simulate method)
        # define the maximum number of trials
        self.max_trials = 0
        # construct the agent
        self.agent = DDPGAgent(nb_actions=self.number_of_actions, actor=self.model_actor, critic=self.model_critic, critic_action_input=self.action_input, memory=self.memory,
                               nb_steps_warmup_critic=100, nb_steps_warmup_actor=100, random_process=random_process, gamma=.8, target_model_update=1e-2, batch_size=32)
        # compile the agent
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engaged_callbacks = self.callbacksDDPG(self, custom_callbacks)
        
    def build_models(self, model_actor=None, model_critic=None, action_input=None):
        '''
        This function sets actor and critic models of the DDPG agent.
        If none are specified then default actor and critic models are build.
        
        Parameters
        ----------
        model_actor :                       The network actor model to be used by the DDPG agent.
        model_critic :                      The network critic model to be used by the DDPG agent.
        action_input :                      The network critic model's action input.
        
        Returns
        ----------
        None
        '''
        # actor model
        self.model_actor = model_actor
        if self.model_actor is None:
            self.model_actor = Sequential()
            self.model_actor.add(Flatten(input_shape=(1,) + self.interface_OAI.observation_space.shape))
            self.model_actor.add(Dense(units=64, activation='tanh'))
            self.model_actor.add(Dense(units=64, activation='tanh'))
            self.model_actor.add(Dense(units=64, activation='tanh'))
            self.model_actor.add(Dense(units=64, activation='tanh'))
            self.model_actor.add(Dense(units=self.number_of_actions, activation='linear', name='output'))
        # critic model
        self.model_critic = model_critic
        self.action_input = action_input
        if self.model_critic is None:
            self.observation_input = Input(shape=(1,) + self.interface_OAI.observation_space.shape, name='observation_input')
            self.observation_flattened = Flatten()(self.observation_input)
            self.action_input = Input(shape=(self.number_of_actions), name='action_input')
            self.x = Concatenate()([self.action_input, self.observation_flattened])
            self.x = Dense(64)(self.x)
            self.x = Activation('tanh')(self.x)
            self.x = Dense(64)(self.x)
            self.x = Activation('tanh')(self.x)
            self.x = Dense(64)(self.x)
            self.x = Activation('tanh')(self.x)
            self.x = Dense(64)(self.x)
            self.x = Activation('tanh')(self.x)
            self.x = Dense(1)(self.x)
            self.x = Activation('linear')(self.x)
            self.model_critic = Model(inputs=[self.action_input, self.observation_input], outputs=self.x)

    def train(self, number_of_trials=100, max_number_of_steps=100, replay_batch_size=32):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the RL agent is trained.
        max_number_of_steps :               The maximum number of steps per trial.
        replay_batch_size :                 The number of random experiences that will be replayed.
        
        Returns
        ----------
        None
        '''
        self.max_trials = number_of_trials
        self.session_trial = 0
        self.agent.batch_size = replay_batch_size
        # call the fit method to start the RL learning process
        self.agent.fit(self.interface_OAI, nb_steps=self.max_steps, verbose=0, callbacks=[self.engaged_callbacks], nb_max_episode_steps=max_number_of_steps, visualize=False)
    
    def test(self, number_of_trials=100, max_number_of_steps=100):
        '''
        This function is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the RL agent is trained.
        max_number_of_steps :               The maximum number of steps per trial.
        
        Returns
        ----------
        None
        '''
        self.max_trials = number_of_trials
        self.session_trial = 0
        # call the fit method to start the RL learning process
        self.agent.test(self.interface_OAI, nb_episodes=number_of_trials, verbose=0, callbacks=[self.engaged_callbacks], nb_max_episode_steps=max_number_of_steps, visualize=False)
    
    def predict_on_batch(self, batch: np.ndarray) -> np.ndarray:
        '''
        This function retrieves Q-values for a batch of observations.
        
        Parameters
        ----------
        batch :                             The batch of observations.
        
        Returns
        ----------
        predictions :                       The batch of predicted Q-values.
        '''
        return self.agent.model.predict_on_batch(batch)
