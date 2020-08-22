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
from agents.utils.mish import Mish
from agents.utils.ringbuffer import RingBuffer

project = get_cobel_path()
environment_path = get_env_path()
SCENE_NAME = "ContinuousRobotMaze"

env = UnityInterface(env_path=environment_path, scene_name=SCENE_NAME,nb_max_episode_steps=10000000, decision_interval=4,
                               agent_action_type="continuous", use_gray_scale_images=True)

num_states = env.observation_space.shape + (4,)
num_actions = env.action_space.n
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Size of State Space ->  {}".format(num_states))
print("Size of Action Space ->  {}".format(num_actions))
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

class Buffer:
    def __init__(self, num_states,num_actions,buffer_capacity=100000, batch_size=64):

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        BUFSIZE_STATE = (self.buffer_capacity,) + num_states
        self.state_buffer = np.zeros(BUFSIZE_STATE)
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros(BUFSIZE_STATE)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions])
            critic_value = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

def update_target(tau):
    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(critic_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_critic.set_weights(new_weights)

    new_weights = []
    target_variables = target_actor.weights
    for i, variable in enumerate(actor_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_actor.set_weights(new_weights)

hidden_act = "Mish"

def get_actor():
    last_init = tf.random_uniform_initializer(-0.003,0.003)
    state_input = layers.Input(shape=num_states)
    out = layers.Conv2D(16,3, activation=hidden_act,padding="same", kernel_initializer=tf.keras.initializers.HeNormal)(state_input)
    #out = layers.BatchNormalization()(out)
    out = layers.MaxPool2D(2)(out)

    out = layers.Conv2D(32,3, activation=hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(out)
    #out = layers.BatchNormalization()(out)
    out = layers.MaxPool2D(2)(out)

    # out = layers.Conv2D(32,3, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal)(out)
    # out = layers.MaxPool2D(2)(out)

    out = layers.Flatten()(out)

    out = layers.Dense(128, activation=hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(out)

    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    outputs = outputs * upper_bound
    model = tf.keras.Model(state_input, outputs)
    return model

def get_critic():

    # State as input
    #First Conv + MaxPool2D
    state_input = layers.Input(shape=num_states)

    out = layers.Conv2D(16,3, activation=hidden_act,padding="same", kernel_initializer=tf.keras.initializers.HeNormal)(state_input)
    #out = layers.BatchNormalization()(out)
    out = layers.MaxPool2D(2)(out)

    out = layers.Conv2D(32,3, activation=hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(out)
    #out = layers.BatchNormalization()(out)
    out = layers.MaxPool2D(2)(out)

    out = layers.Flatten()(out)
    out = layers.Dense(128, activation=hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(128, activation=hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(action_input)

    #Concatenate Both Layers
    concat = layers.Concatenate()([out, action_out])
    concat_out = layers.Dense(64, activation=hidden_act, kernel_initializer=tf.keras.initializers.HeNormal)(concat)

    outputs = layers.Dense(1)(concat_out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)
    #print(model.summary())
    return model

def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    
    # Adding noise to action
    #sampled_actions = sampled_actions.numpy() + noise

    #Without noise
    sampled_actions = sampled_actions.numpy()
    
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]

std_dev = 0.3
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

#Load All Weights
actor_model.load_weights('/home/wkst/Desktop/actor.h5')
critic_model.load_weights('/home/wkst/Desktop/critic.h5')

#Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
actor_lr  = 0.0001
critic_lr = 0.0001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr,clipnorm=1.,clipvalue=0.5)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr,clipnorm=1.,clipvalue=0.5)

total_episodes = 10000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(num_states,num_actions, 25000, 128)
ringbuffer = RingBuffer(4)

for ep in range(total_episodes):
    score = 0.0
    prev_state = env._reset()

    ringbuffer.insert_obs(prev_state[0][:,:,0])
    ringbuffer.insert_obs(prev_state[0][:,:,1])
    ringbuffer.insert_obs(prev_state[0][:,:,2])
    ringbuffer.insert_obs(prev_state[0][:,:,3])

    prev_state = ringbuffer.generate_arr()

    while True:
        #ringbuffer.print_arr()
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = np.array(policy(tf_prev_state, ou_noise))
        state, reward, done, info = env._step(action)
        #Get Each third Frame as workaround
        score += reward
        if state[0].shape == num_states:
            ringbuffer.insert_obs(state[0][:,:,2])
        else:
            ringbuffer.insert_obs(state[0][0][:,:,2])
        
        #Fill Whole 4 Frames in Ringbuffer
        #ringbuffer.insert_obs(state[0][:,:,0])
        #ringbuffer.insert_obs(state[0][:,:,1])
        #ringbuffer.insert_obs(state[0][:,:,2])
        #ringbuffer.insert_obs(state[0][:,:,3])
        state = ringbuffer.generate_arr()

        #buffer.record((prev_state, action[0], reward, state))

        #buffer.learn()
        #update_target(tau)

        if done:
            break
        prev_state = state
    print("ep" , ep, ": ", score)
    if score >= 4:
        print("solved!")
        actor_model.save_weights("actor.h5")
        critic_model.save_weights("critic.h5")
            

backend.clear_session()
unity_env.close()