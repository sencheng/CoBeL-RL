import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input,Dense, Conv2D, MaxPooling2D, Flatten, Concatenate, BatchNormalization
from tensorflow.keras.initializers import HeNormal, RandomUniform
from tensorflow.keras import Model
from tensorflow.math import log, exp, reduce_sum, tanh
from agents.RDQN.custom_layers import NoisyDense

import numpy as np
rand_uniform_init = RandomUniform(-0.003,0.003)

#Distributions
tf_dists = tfp.distributions
noise_dist = tf_dists.Normal(0,0.2,False,False)
normal_dist = tf_dists.Normal(0,1,False,False)

log_std_min = -20
log_std_max = 2

def create_lenet5(input_layer):
    out = Conv2D(12,5,1,padding="same", activation="tanh")(input_layer)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid")(out)
    out = Conv2D(32,5,1,padding="valid",activation="tanh")(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid")(out)
    out = Flatten()(out)
    out = Dense(240,activation="tanh")(out)
    out = BatchNormalization()(out)
    out = Dense(168,activation="tanh")(out)
    out = BatchNormalization()(out)
    return out

def create_critic(state_dim,action_dim,hidden_dim = 256):
    input_state = Input(shape=(state_dim[0],state_dim[1],state_dim[2]))
    input_action = Input(shape=(action_dim,))
    
    out = create_lenet5(input_state)

    act_out = Dense(82,activation='tanh')(input_action)
    act_out = BatchNormalization()(act_out)

    concatenated_layers = Concatenate()([out,act_out])
    concat_out = Dense(64,activation='tanh')(concatenated_layers)
    
    v = Dense(1,kernel_initializer=rand_uniform_init)(concat_out)
    return Model([input_state,input_action], v)

class GaussianPolicy(tf.keras.Model):
    def __init__(self,state_dim,action_dim,hidden_dim = 256):
        super(GaussianPolicy,self).__init__(name = "Policy Network")
        
        self.conv1 = Conv2D(16, kernel_size=3, activation='relu',kernel_initializer=HeNormal)
        self.mp1 = MaxPooling2D(pool_size=(2,2))
        
        self.conv2 = Conv2D(32, kernel_size=3, activation='relu',kernel_initializer=HeNormal)
        self.mp2 = MaxPooling2D(pool_size=(2,2))
        
        self.flatten = Flatten()
        self.cnn_dense = Dense(128,activation='relu',kernel_initializer=HeNormal)

        self.mu_dense = Dense(action_dim,kernel_initializer=rand_uniform_init,bias_initializer=rand_uniform_init)
        self.log_std_dense = Dense(action_dim,kernel_initializer=rand_uniform_init,bias_initializer=rand_uniform_init)
        
    def call(self, X):
        x = self.conv1(X)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.mp2(x)

        x = self.flatten(x)
        x = self.cnn_dense(x)

        mu = self.mu_dense(x)
        log = self.log_std_dense(x)
        
        log_std = tf.clip_by_value(log,log_std_min,log_std_max)
        return mu, log_std

    def sample(self, X, epsilon=1e-6):
        mu, log_std = self.call(X)
        std = exp(log_std)
        
        dist = tf_dists.Normal(loc=mu, scale=std)
        e = normal_dist.sample()
        action = tanh(mu + e * std)
        log_prob = dist.log_prob(mu + e * std) - log(1 - tf.math.pow(action,2) + epsilon)
        log_prob = reduce_sum(log_prob,axis=1)
        return action, log_prob , 0

class DeterministicPolicy(tf.keras.Model):
    def __init__(self,state_dim,action_dim,hidden_dim = 256):
        super(DeterministicPolicy,self).__init__(name = "Deterministic Policy Network")
        self.actions = action_dim

        self.c1 = Conv2D(12,5,1,padding="same", activation="tanh")
        self.bn1 = BatchNormalization()
        self.mp1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid")
        
        self.c2 = Conv2D(32,5,1,padding="valid",activation="tanh")
        self.bn2 = BatchNormalization()
        self.mp2 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid")
        
        self.fl = Flatten()
        
        self.d3 = Dense(240,activation="tanh")
        self.bn3 = BatchNormalization()
        
        self.d4 = Dense(168,activation="tanh")
        self.bn4 = BatchNormalization()

        self.mean = Dense(action_dim,activation='tanh',kernel_initializer=rand_uniform_init,bias_initializer=rand_uniform_init)

    def call(self, X):
        x = self.c1(X)
        x = self.bn1(x)
        x = self.mp1(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = self.mp2(x)

        x = self.fl(x)

        x = self.d3(x)
        x = self.bn3(x)

        x = self.d4(x)
        x = self.bn4(x)

        return self.mean(x)

    def sample(self, X, epsilon=1e-6):
        mean = self.call(X)
        noise = noise_dist.sample((self.actions))
        #With Noise
        #action = mean + noise

        #Without Noise
        action = mean
        
        #action for exploration
        #mean for deterministic evaluation
        #Logpi = 0
        return action, tf.constant(0.0),  mean
