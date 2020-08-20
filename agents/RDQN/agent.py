import sys
import math
import os
import random
import collections
from collections import deque
from typing import Deque, Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras.layers import Dense, Input, GaussianNoise, Lambda, Reshape, RepeatVector, Softmax, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import initializers

from agents.RDQN.custom_layers import NoisyDense

from agents.RDQN.buffers import ReplayBuffer, PrioritizedReplayBuffer
from interfaces.oai_gym_interface import UnityInterface

class RDQNAgent:
    #PASS
    def __init__(
        self, 
        env: UnityInterface,
        num_frames: int = 10000,
        memory_size: int = 3000,
        batch_size: int = 64,
        target_update: int = 100,
        gamma: float = 0.9,
        # PER parameters
        alpha: float = 0.4,
        beta: float = 0.4,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 20.0,
        atom_size: int = 51
    ):
        self.u_env = env
        self.obs_dim = env.observation_space.shape
        self.action_dim = env.action_space.n
        
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma

        #Multistep DQN parameters
        self.n_step = 3
        
        # Prioritized Experience Replay
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(self.obs_dim, memory_size, batch_size,alpha,self.n_step,self.gamma)
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = np.linspace(self.v_min, self.v_max, self.atom_size)

        # networks: dqn, dqn_target
        self.dqn = self.build_model()
        self.dqn_target = self.build_model()

        # transition to store in memory
        self.transition = list()

    def aggregate_layers(self,layers):
        layerlist = []
        v = layers[0]
        adv = layers[1]
        adv = Reshape((self.action_dim,self.atom_size))(adv)
        adv_mean = K.mean(adv,axis=1)
        for i in range(self.action_dim):
            layerlist.append(v + adv[:,i] - adv_mean)
        return layerlist

    def build_model(self):
        input_layer = Input(shape=self.obs_dim)

        c1 = Conv2D(8, (3, 3), activation='relu')(input_layer)
        mp1 = MaxPooling2D((2, 2))(c1)
        
        c2 = Conv2D(16, (3, 3), activation='relu')(mp1)
        mp2 = MaxPooling2D((2, 2))(c2)
        
        fl = Flatten()(mp2)

        #Value Stream
        v_layer = NoisyDense(128,activation='relu')(fl)
        v = NoisyDense(self.atom_size)(v_layer)

        #Advantage Stream
        adv_layer = NoisyDense(128,activation='relu')(fl)
        adv = NoisyDense(self.atom_size * self.action_dim)(adv_layer)

        agg = Lambda(self.aggregate_layers)([v,adv])

        distribution_list = []

        for i in range(self.action_dim):
            distribution_list.append(Softmax(axis=1)(agg[i]))

        model = Model(input_layer, distribution_list)
        model.compile(optimizer=Adam(lr=0.0025), loss='categorical_crossentropy')
        
        return model

    def select_action(self, state):
        state = np.array(state)
        state = np.expand_dims(state,0)
        z = self.dqn.predict(state)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.support)), axis=1) 
        q_opt = np.argmax(q)
        
        self.transition = [state, q_opt]
        return q_opt

    def step(self, action):
        next_state, reward, done, _ = self.u_env._step(np.array([[action]]))
        if next_state[0].shape == self.obs_dim:
          next_state = next_state[0]
        else:
          next_state = next_state[0][0]
        self.transition += [reward, next_state, done]
        one_step_transition = self.transition
        self.memory.store(*one_step_transition)
            
        return next_state, reward, done

    def get_optimal_actions(self,next_state : np.array):
        dqn_next_action = self.dqn.predict(next_state)
        optimal_action_idxs = []
        z_concat = np.vstack(dqn_next_action)
        q = np.sum(np.multiply(z_concat, np.array(self.support)), axis=1)
        q = q.reshape((self.batch_size, self.action_dim), order='F')
        next_action = np.argmax(q, axis=1)
        next_action = np.array(next_action,dtype=np.int32)
        return next_action

    def update_model(self):
        samples = self.memory.sample_batch(self.beta)
        weights = np.array(samples["weights"])
        indices = np.array(samples["indices"])

        state = np.array(samples["obs"])
        next_state = np.array(samples["next_obs"])
        action = np.array(samples["acts"],dtype=np.int32)
        reward = np.array(samples["rews"])
        done = np.array(samples["done"])

        m_prob = [np.zeros((samples["obs"].shape[0], self.atom_size)) for i in range(self.action_dim)]
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        next_action = self.get_optimal_actions(next_state)
        next_dist = self.dqn_target.predict(next_state)

        #Iterate over whole batch
        for i in range(samples["obs"].shape[0]):
            for j in range(self.atom_size):
                t_z = reward[i] + (1 - done[i]) * self.gamma**self.n_step * self.support[j]
                t_z = self.clamp(t_z,self.v_min,self.v_max)
                b = (t_z - self.v_min) / delta_z
                l = math.floor(b)
                u = math.ceil(b)
                if u != l:
                    m_prob[action[i]][i][l] += next_dist[next_action[i]][i][j] * (u - b)
                    m_prob[action[i]][i][u] += next_dist[next_action[i]][i][j] * (b - l)
                else:
                    m_prob[action[i]][i][l] += next_dist[next_action[i]][i][j]

        #KL Divergence Calc
        dist = self.dqn.predict(state)
        x,y = np.zeros((self.batch_size,self.atom_size)), np.zeros((self.batch_size,self.atom_size))
        for i in range (self.batch_size):
            x[i] = dist[action[i]][i]
            y[i] = m_prob[action[i]][i]
        
        log_x = np.log(x + 0.01)
        elementwise_loss = -(y * log_x).sum(1)

        self.dqn.fit(state,m_prob, batch_size=self.batch_size,epochs=1, verbose=0, sample_weight=[weights,weights,weights,weights])
        
        # PER: importance sampling before average
        loss = np.mean(elementwise_loss * weights)

        # PER: update priorities
        new_priorities = elementwise_loss + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        return sum(elementwise_loss)
    
    def Average(self,lst): 
        return sum(lst) / len(lst) 

    def train(self, num_frames: int):
        state = self.u_env._reset()
        if state[0].shape == self.obs_dim:
          state = state[0]
        else:
          state = state[0][0]
        
        score = collections.deque([0,0,0,0,0],maxlen=5)
        currentRew = 0
        ep = 0
        updateCount = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            
            next_state, reward, done = self.step(action)
                
            state = next_state
            currentRew += reward
            
            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                score.append(currentRew)
                currentRew = 0
                ep += 1
                avg = self.Average(list(score))
                print("Ep:",ep," Current Average Reward: ", avg)
                if avg >= 9:
                    print("Save Model")
                    self.dqn.save_weights("/home/wkst/Desktop/RDQN.h5")
                state = self.u_env._reset()
                if state[0].shape == self.obs_dim:
                    state = state[0]
                else:
                    state = state[0][0]
                
            # if training is ready
            if len(self.memory) >= self.batch_size:
                updateCount += 1
                self.update_model()
                
            # if hard update is needed
            if updateCount % self.target_update == 0:
                self._target_hard_update()

    def clamp(self, n, smallest, largest): 
        return max(smallest, min(n, largest))

    def _target_hard_update(self):
        self.dqn_target.set_weights(self.dqn.get_weights())