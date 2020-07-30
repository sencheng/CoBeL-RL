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

class ProbabilityDistribution(tf.keras.Model):
  def call(self, logits, **kwargs):
    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__('mlp_policy')

    self.conv1 = kl.Conv2D(256, kernel_size=3, activation='relu')
    self.mp1 = kl.MaxPooling2D(pool_size=(2,2))
    self.conv2 = kl.Conv2D(128, kernel_size=3, activation='relu')
    self.mp2 = kl.MaxPooling2D(pool_size=(2,2))
    self.flatten = kl.Flatten()

    self.hidden1 = kl.Dense(512, activation='relu')
    self.hidden2 = kl.Dense(512, activation='relu')
    self.value = kl.Dense(1, name='value')
    
    self.logits = kl.Dense(num_actions, name='policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    x = self.conv1(x)
    x = self.mp1(x)
    x = self.conv2(x)
    x = self.mp2(x)
    x = self.flatten(x)

    hidden_logs = self.hidden1(x)
    hidden_vals = self.hidden2(x)
    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs):
    logits, value = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

class A2CAgent:
  def __init__(self, env: UnityInterface, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
    self.u_env = env
    self.obs_dim = env.observation_space.shape
    self.action_dim = env.action_space.n

    self.gamma = gamma
    self.value_c = value_c
    self.entropy_c = entropy_c
    self.batchsize = 64

    self.model = Model(num_actions=self.action_dim)
    self.model.compile(optimizer=ko.RMSprop(lr=lr),loss=[self._logits_loss, self._value_loss])

  def train(self,num_frames: int):
    actions = np.empty((self.batchsize,), dtype=np.int32)
    rewards, dones, values = np.empty((3, self.batchsize))
    obs_shape = np.insert(self.obs_dim, 0, self.batchsize)
    observations = np.empty(obs_shape,dtype=np.float32)
    ep_rewards = [0.0]
    next_obs = self.u_env._reset()
    for update in range(num_frames):
      for step in range(self.batchsize):
        observations[step] = next_obs[0].copy()
        actions[step], values[step] = self.model.action_value(next_obs[0][None, :])
        next_obs, rewards[step], dones[step], _ = self.u_env._step(np.array([[actions[step]]]))
        ep_rewards[-1] += rewards[step]
        if dones[step]:
          ep_rewards.append(0.0)
          next_obs = self.u_env.reset()
          print("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))
      if update % 50 == 0:
        print("Save Model")
        self.model.save("/home/wkst/Desktop/a2c_model")
      _, next_value = self.model.action_value(next_obs[0][None, :])
      returns, advs = self._returns_advantages(rewards, dones, values, next_value)
      acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
      losses = self.model.train_on_batch(observations, [acts_and_advs, returns])

    return ep_rewards

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