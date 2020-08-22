import os
import time
from interfaces.oai_gym_interface import UnityInterface, get_cobel_path, get_env_path
from random import randrange
from tensorflow.keras import backend
import numpy as np

visualOutput = True
backend.set_image_data_format(data_format='channels_last')

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import collections
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from agents.DDPG.noise import OUActionNoise
from agents.DDPG.buffer import Buffer
from agents.utils.mish import Mish
from agents.utils.ringbuffer import RingBuffer

class DDPG_Agent:
    def __init__(self,env: UnityInterface,frameskip=4):
        self.u_env = env
        self.num_states = env.observation_space.shape + (4,)
        self.num_actions = env.action_space.n
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]

        self.tau = 0.005
        self.actor_lr  = 0.0001
        self.critic_lr = 0.0001
        self.gamma = 0.99
        self.hid_act = "mish"
        self.mem_size = 25000
        self.batch_size = 128
        self.episodes = 10000
        
        self.std_dev = 0.3
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))
        
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr,clipnorm=1.,clipvalue=0.5)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr,clipnorm=1.,clipvalue=0.5)
    
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic(self.hid_act)

        self.target_actor = self.get_actor(self.num_states,self.hid_act)
        self.target_critic = self.get_critic(self.hid_act)
        
        #Load All Weights
        self.actor_model.load_weights('/home/wkst/Desktop/actor.h5')
        self.critic_model.load_weights('/home/wkst/Desktop/critic.h5')
        
        #Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        
        #Buffer
        self.buffer = Buffer(self.num_states,self.num_actions, self.mem_size, self.batch_size)
        self.ringbuffer = RingBuffer(4)
        
    def get_actor(self):
        last_init = tf.random_uniform_initializer(-0.003,0.003)
        state_input = layers.Input(shape=self.num_states)
    
        out = layers.Conv2D(16,3, activation=self.hidden_act,padding="same", kernel_initializer=tf.keras.initializers.HeNormal)(state_input)
        out = layers.MaxPool2D(2)(out)

        out = layers.Conv2D(32,3, activation=self.hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(out)
        out = layers.MaxPool2D(2)(out)
        out = layers.Flatten()(out)

        out = layers.Dense(128, activation=self.hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(out)

        outputs = layers.Dense(self.num_actions, activation="tanh", kernel_initializer=last_init)(out)
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(state_input, outputs)
        return model

    def get_critic(self):
        state_input = layers.Input(shape=self.num_states)
        action_input = layers.Input(shape=(self.num_actions))
    
        out = layers.Conv2D(16,3, activation=self.hidden_act,padding="same", kernel_initializer=tf.keras.initializers.HeNormal)(state_input)
        out = layers.MaxPool2D(2)(out)

        out = layers.Conv2D(32,3, activation=self.hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(out)
        out = layers.MaxPool2D(2)(out)

        out = layers.Flatten()(out)
        out = layers.Dense(128, activation=self.hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(out)

        # Action as input
        action_out = layers.Dense(128, activation=self.hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(action_input)

        #Concatenate Both Layers
        concat = layers.Concatenate()([out, action_out])
        concat_out = layers.Dense(64, activation=self.hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(concat)

        outputs = layers.Dense(1)(concat_out)
        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    def train_model(self,state_batch,action_batch,reward_batch,next_state_batch):
        state_batch,action_batch,reward_batch,next_state_batch = self.buffer.sample_batch()
        
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions])
            critic_value = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))
        
        with tf.GradientTape() as tape:
            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))
    
    def update_target(self):
        new_weights = []
        target_variables = self.target_critic.weights
        for i, variable in enumerate(self.critic_model.weights):
            new_weights.append(variable * tau + target_variables[i] * (1 - tau))
        self.target_critic.set_weights(new_weights)

        new_weights = []
        target_variables = self.target_actor.weights
        for i, variable in enumerate(self.actor_model.weights):
            new_weights.append(variable * tau + target_variables[i] * (1 - tau))
        self.target_actor.set_weights(new_weights)

    def sample_action(self,state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = self.ou_noise()
        
        # Adding noise to action
        #sampled_actions = sampled_actions.numpy() + noise

        #Without noise
        sampled_actions = sampled_actions.numpy()
        
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)]
    
    def train(self):
        for ep in range(self.episodes):
            score = 0.0
            prev_state = env._reset()

            self.ringbuffer.insert_obs(prev_state[0][:,:,0])
            self.ringbuffer.insert_obs(prev_state[0][:,:,1])
            self.ringbuffer.insert_obs(prev_state[0][:,:,2])
            self.ringbuffer.insert_obs(prev_state[0][:,:,3])
            prev_state = ringbuffer.generate_arr()
            
            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = np.array(self.sample_action(tf_prev_state))
                state, reward, done, info = env._step(action)
                score += reward
                
                #Get Each third Frame as workaround
                if state[0].shape == self.num_states:
                    self.ringbuffer.insert_obs(state[0][:,:,2])
                else:
                    self.ringbuffer.insert_obs(state[0][0][:,:,2])
                
                state = self.ringbuffer.generate_arr()
                self.buffer.record((prev_state, action[0], reward, state))
                self.train_model()
                update_target(tau)
                if done:
                    break
            prev_state = state
            print("ep" , ep, ": ", score)
            if score >= 4:
                print("solved!")
                actor_model.save_weights("actor.h5")
                critic_model.save_weights("critic.h5")