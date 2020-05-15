from keras.models import Sequential
from keras.layers import Dense, Lambda, Input
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.losses import mean_squared_error
from keras.backend import clip
from keras import Model
import numpy as np
import scipy.stats

class Actor:
    def __init__(self,state_size, action_size, seed, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.hidden_size = hidden_size
        self.init_w = init_w
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.LR_CRITIC = float(5e-4)
        self.model = self.build_actor_network()

    def clamp_layer(self,input):
        return clip(input,self.log_std_min,self.log_std_max)

    def build_actor_network(self):
        lim = 1. / np.sqrt(self.hidden_size)

        hid_init = RandomUniform(minval=-lim, maxval=lim, seed=None)
        out_init = RandomUniform(minval=self.log_std_min, maxval=self.log_std_max, seed=None)
    
        input_layer = Input(shape=(self.state_size,))
        d1 = Dense(self.hidden_size,kernel_initializer=hid_init,activation="relu")(input_layer)
        d2 = Dense(self.hidden_size,kernel_initializer=hid_init,activation="relu")(d1)

        mu_d = Dense(self.hidden_size)(d2)
        mu_out = Dense(self.action_size)(mu_d)

        log_std_d = Dense(self.hidden_size)(d2)
        log_std = Dense(self.action_size)(log_std_d)
        log_std_out = Lambda(self.clamp_layer)(log_std)

        model = Model(input_layer, [mu_out,log_std_out])
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.LR_CRITIC))
        #model.summary()
        return model

    def get_action(self, state):
        mu, log_std = self.model.predict(state)
        std = np.exp(log_std)
        e = np.random.normal(0,1)
        action = np.tanh(mu + e * std)
        return action[0]
    
    def evaluate(self, state):
        mu, log_std = self.model.predict(state)
        std = np.exp(log_std)
        e = np.random.normal(0,1)
        action = np.tanh(mu + e * std)
        
        log_prob = np.log(scipy.stats.norm.pdf(mu + e * std,mu,std)) - np.log(1 - action[0]**2 + epsilon)
        return action[0], log_prob