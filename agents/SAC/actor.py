import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomUniform
import numpy as np
import scipy.stats

from tensorflow.math import log,exp,tanh, reduce_sum
import tensorflow_probability as tfp
tfd = tfp.distributions

from agents.utils.mish import Mish

tf.keras.backend.set_floatx('float32')

class Actor_Net(tf.keras.Model):
    def __init__(self,state_size,hidden_size,action_size,log_std_min=-20,log_std_max=2,init_w=3e-3):
        super(Actor_Net,self).__init__(name = 'actor_net')
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        lim = 1. / np.sqrt(hidden_size)
        hid_init = RandomUniform(minval=-lim, maxval=lim, seed=None)
        out_init = RandomUniform(minval=-3e-3, maxval=3e-3, seed=None)

        #Conv Head
        self.conv1 = layers.Conv2D(16, kernel_size=3, activation='relu',padding="same",kernel_initializer=hid_init,input_shape=state_size)
        self.mp1 = layers.MaxPooling2D(pool_size=(2,2))

        self.conv2 = layers.Conv2D(32, kernel_size=3, activation='relu',kernel_initializer=hid_init)
        self.mp2 = layers.MaxPooling2D(pool_size=(2,2))
        
        self.flatten = layers.Flatten()

        self.dense = layers.Dense(128,activation='relu',kernel_initializer=hid_init)

        self.mu = layers.Dense(action_size,kernel_initializer=out_init)
        self.log_std = layers.Dense(action_size,kernel_initializer=out_init)
        
    def call(self, X):
        x = self.conv1(X)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.flatten(x)
        x = self.dense(x)
        mu = self.mu(x)
        log = self.log_std(x)
        log_dev = tf.clip_by_value(log,self.log_std_min,self.log_std_max)
        return mu, log_dev

    def get_action(self,X):
        mu, log_std = self.call(X)
        std = exp(log_std)
        dist = tfd.Normal(loc=mu, scale=std)
        z = dist.sample()
        action = tf.squeeze(tanh(z))
        return action

    def evaluate(self, X, epsilon=1e-6):
        mu, log_std = self.call(X)
        std = exp(log_std)
        dist = tfd.Normal(loc=mu, scale=std)
        z = dist.sample()
        action = tf.squeeze(tanh(z))
        log_prob = dist.log_prob(z) - log(1 - action**2 + epsilon)
        log_prob = reduce_sum(log_prob,axis=1,keepdims=True)
        return action, log_prob