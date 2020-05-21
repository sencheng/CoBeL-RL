import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.initializers import RandomUniform
import numpy as np
import scipy.stats

from tensorflow.math import log,exp,tanh
import tensorflow_probability as tfp
tfd = tfp.distributions

class Actor_Net(tf.keras.Model):
    def __init__(self,state_size,hidden_size,action_size,log_std_min=-20,log_std_max=2,init_w=3e-3):
        super(Actor_Net,self).__init__(name = 'actor_net')
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        lim = 1. / np.sqrt(hidden_size)
        hid_init = RandomUniform(minval=-lim, maxval=lim, seed=None)
        out_init = RandomUniform(minval=log_std_min, maxval=log_std_max, seed=None)

        self.fc1 = layers.Dense(hidden_size,activation = 'relu',kernel_initializer=hid_init, input_dim=state_size,dtype='float32')
        self.fc2 = layers.Dense(hidden_size,activation='relu',kernel_initializer=hid_init)
        self.mu = layers.Dense(action_size,kernel_initializer=out_init)
        self.log_std = layers.Dense(action_size,kernel_initializer=out_init)
    def call(self, X):
        x = self.fc1(X)
        x = self.fc2(x)
        mu = self.mu(x)
        log = self.mu(x)
        log_clamp = tf.clip_by_value(log,self.log_std_min,self.log_std_max)
        return mu, log_clamp
    def get_action(self,X):
        mu, log_std = self.call(X)
        std = exp(log_std)
        e = np.random.normal(0,1)
        action = tf.squeeze(tanh(mu + e * std))
        return action
    def evaluate(self, X, epsilon=1e-6):
        mu, log_std = self.call(X)
        std = exp(log_std)
        e = np.random.normal(0,1)
        action = tanh(mu + e * std)
        dist = tfd.Normal(loc=mu, scale=std)
        log_prob = log(dist.prob(mu + e * std)) - log(1 - action**2 + epsilon)
        return action, log_prob