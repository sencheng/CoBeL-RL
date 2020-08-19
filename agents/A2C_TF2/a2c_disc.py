import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gym
import logging
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
from interfaces.oai_gym_interface import UnityInterface
from agents.A2C_TF2.mish import Mish
import collections

#Solving Benchmark
#32x32|relu -> ~140 Episodes
#16x16|relu -> ~170 Episodes
#16x16|mish -> ~160 Episodes

class ProbabilityDistribution(tf.keras.Model):
  def call(self, logits, **kwargs):
    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

hidden_act = "Mish"

class Model(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__('mlp_policy')

    self.c1 = kl.Conv2D(8, (3, 3), activation=hidden_act,kernel_initializer=tf.keras.initializers.HeNormal)
    self.mp1 = kl.MaxPooling2D((2, 2))
    
    self.c2 = kl.Conv2D(16, (3, 3), activation=hidden_act,kernel_initializer=tf.keras.initializers.HeNormal)
    self.mp2 = kl.MaxPooling2D((2, 2))
    
    #self.c3 = kl.Conv2D(64, (3, 3), activation='relu',kernel_initializer=tf.keras.initializers.HeNormal)
    self.fl = kl.Flatten()
    
    self.d1 = kl.Dense(128,activation=hidden_act,kernel_initializer=tf.keras.initializers.HeNormal)

    self.hidden1 = kl.Dense(64, activation=hidden_act,kernel_initializer=tf.keras.initializers.HeNormal)
    self.hidden2 = kl.Dense(64, activation=hidden_act,kernel_initializer=tf.keras.initializers.HeNormal)
    self.value = kl.Dense(1, name='value')
    
    self.logits = kl.Dense(num_actions, name='policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    x = self.c1(x)
    x = self.mp1(x)
    x = self.c2(x)
    x = self.mp2(x)
    #x = self.c3(x)
    x = self.fl(x)
    x = self.d1(x)

    hidden_logs = self.hidden1(x)
    hidden_vals = self.hidden2(x)
    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs):
    logits, value = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


#Working Configuration
#lr : 7e-4
#entropy_c : 1e-4
#Solved after ~1000 steps

class A2CAgent:
  def __init__(self, env: UnityInterface, lr=7e-4, gamma=0.99, value_c=0.5, entropy_c=1e-4):
    self.u_env = env
    self.obs_dim = env.observation_space.shape
    self.action_dim = env.action_space.n

    self.gamma = gamma
    self.value_c = value_c
    self.entropy_c = entropy_c
    self.batchsize = 64

    self.model = Model(num_actions=self.action_dim)
    self.model.compile(optimizer=ko.Adam(lr=lr,clipnorm=1.,clipvalue=0.5),loss=[self._logits_loss, self._value_loss])
    #self.model.compile(optimizer=ko.Adam(lr=lr),loss=[self._logits_loss, self._value_loss])

  def Average(self,lst): 
    return sum(lst) / len(lst) 
  
  def train(self,num_frames: int):
    #Last 5 Episode rewards
    score = collections.deque([0,0,0,0,0],maxlen=5)
    currentRew = 0
    ep = 0
    actions = np.empty((self.batchsize,), dtype=np.int32)
    rewards, dones, values = np.empty((3, self.batchsize))
    obs_shape = np.insert(self.obs_dim, 0, self.batchsize)
    observations = np.empty(obs_shape,dtype=np.float32)
    next_obs = self.u_env._reset()
    for update in range(int(num_frames / self.batchsize)):
      for step in range(self.batchsize):
        if next_obs[0].shape == self.obs_dim:
          observations[step] = next_obs[0].copy()
        else:
          observations[step] = next_obs[0][0].copy()
        actions[step], values[step] = self.model.action_value(observations[step][None, :])
        next_obs, rewards[step], dones[step], _ = self.u_env._step(np.array([[actions[step]]]))
        currentRew += rewards[step]
        if dones[step]:
          score.append(currentRew)
          currentRew = 0
          ep += 1
          next_obs = self.u_env._reset()
          avg = self.Average(list(score))
          print("Ep:",ep," Current Average Reward: ", avg)
          if avg >= 9:
            print("Save Model")
            self.model.save_weights("/home/wkst/Desktop/A2C.h5")
      _, next_value = self.model.action_value(next_obs[0][None, :])
      returns, advs = self._returns_advantages(rewards, dones, values, next_value)
      acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
      losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
    
  def _returns_advantages(self, rewards, dones, values, next_value):
    returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
    for t in reversed(range(rewards.shape[0])):
      returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
    returns = returns[:-1]
    advantages = returns - values
    return returns, advantages

  def _value_loss(self, returns, value):
    return self.value_c * kls.mean_squared_error(returns, value)

  def _logits_loss(self, actions_and_advantages, logits):
    actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
    weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
    actions = tf.cast(actions, tf.int32)
    policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
    probs = tf.nn.softmax(logits)
    entropy_loss = kls.categorical_crossentropy(probs, probs)
    return policy_loss - self.entropy_c * entropy_loss