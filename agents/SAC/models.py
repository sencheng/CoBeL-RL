import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input,Dense, Conv2D, MaxPooling2D, Flatten, Concatenate
from tensorflow.keras.initializers import HeNormal, RandomUniform
from tensorflow.keras import Model
from tensorflow.math import log, exp, reduce_sum, tanh
import numpy as np
rand_uniform_init = RandomUniform(-0.003,0.003)

#Distributions
tf_dists = tfp.distributions
noise_dist = tf_dists.Normal(0,0.1,False,False)
normal_dist = tf_dists.Normal(0,1,False,False)

log_std_min = -20
log_std_max = 2
        
def create_critic(state_dim,action_dim,hidden_dim = 256):
    input_state = Input(shape=(state_dim[0],state_dim[1],state_dim[2]))
    input_action = Input(shape=(action_dim,))
    
    #Feature Extraction
    conv1 = Conv2D(16, kernel_size=3, activation='relu',kernel_initializer=HeNormal)(input_state)
    mp1 = MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2 = Conv2D(32, kernel_size=3, activation='relu',kernel_initializer=HeNormal)(mp1)
    mp2 = MaxPooling2D(pool_size=(2,2))(conv2)

    flatten = Flatten()(mp2)
    
    cnn_dense = Dense(128,activation='relu',kernel_initializer=HeNormal)(flatten)
    act_dense = Dense(64,activation='relu',kernel_initializer=HeNormal)(input_action)
    
    concatenated_layers = Concatenate()([cnn_dense,act_dense])
    pre_dense = Dense(64,activation='relu',kernel_initializer=HeNormal)(concatenated_layers)
    
    v = Dense(1,kernel_initializer=rand_uniform_init)(pre_dense)
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

        self.conv1 = Conv2D(16, kernel_size=3, activation='relu',kernel_initializer=HeNormal)
        self.mp1 = MaxPooling2D(pool_size=(2,2))
        
        self.conv2 = Conv2D(32, kernel_size=3, activation='relu',kernel_initializer=HeNormal)
        self.mp2 = MaxPooling2D(pool_size=(2,2))
        
        self.flatten = Flatten()
        self.cnn_dense = Dense(128,activation='relu',kernel_initializer=HeNormal)

        self.mean = Dense(action_dim,activation='tanh',kernel_initializer=rand_uniform_init,bias_initializer=rand_uniform_init)

    def call(self, X):
        x = self.conv1(X)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.mp2(x)

        x = self.flatten(x)
        x = self.cnn_dense(x)

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
