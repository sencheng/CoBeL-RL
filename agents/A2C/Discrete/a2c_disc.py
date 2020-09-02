import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls

from interfaces.oai_gym_interface import UnityInterface

from agents.utils.misc import Average, Get_Single_Input
from collections import deque
from agents.A2C.Discrete.models import Model

class A2CAgent:
  def __init__(self, env: UnityInterface, lr=7e-4, gamma=0.99, value_c=0.5, entropy_c=1e-4):
    self.modelpath = "/home/wkst/Desktop/A2C.h5"
    self.u_env = env
    self.obs_dim = env.observation_space.shape
    self.action_dim = env.action_space.n
    self.hid_act = "relu"
    self.gamma = gamma
    self.value_c = value_c
    self.entropy_c = entropy_c
    self.batchsize = 200

    self.model = Model(self.obs_dim,self.action_dim,self.hid_act)
    self.model.compile(optimizer=Adam(lr=lr,clipnorm=1.,clipvalue=0.5),loss=[self._logits_loss, self._value_loss])
    self.model.action_value(np.zeros((1,self.obs_dim[0],self.obs_dim[1],self.obs_dim[2])))
    self.model.load_weights(self.modelpath)
    
  def train(self,num_frames: int):
    score = deque([0,0,0,0,0],maxlen=5)
    currentRew = 0
    ep = 0
    actions = np.empty((self.batchsize,), dtype=np.int32)
    rewards, dones, values = np.empty((3, self.batchsize))
    obs_shape = np.insert(self.obs_dim, 0, self.batchsize)
    observations = np.empty(obs_shape,dtype=np.float32)
    next_obs = self.u_env._reset()
    
    for update in range(int(num_frames / self.batchsize)):
      for step in range(self.batchsize):
        next_obs = Get_Single_Input(next_obs,self.obs_dim)
        observations[step] = next_obs.copy()
        
        actions[step], values[step] = self.model.action_value(observations[step][None, :])
        next_obs, rewards[step], dones[step], _ = self.u_env._step(np.array([[actions[step]]]))
        currentRew += rewards[step]
        if dones[step]:
          score.append(currentRew)
          currentRew = 0
          ep += 1
          next_obs = self.u_env._reset()
          avg = Average(list(score))
          print("Ep:",ep," Avg:", avg," Current:", score[4])
          if avg >= 9:
            print("Save Model")
            self.model.save_weights(self.modelpath)
      #Training Step
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