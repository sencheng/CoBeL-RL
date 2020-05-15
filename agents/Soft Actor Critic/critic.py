from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.losses import mean_squared_error
import numpy as np

LR_CRITIC = float(5e-4)

#PASS

class Critic:
    def __init__(self,state_size, action_size, seed, hidden_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.hidden_size = hidden_size

        self.network = self.build_critic_network()

    def build_critic_network(self):
        lim = 1. / np.sqrt(self.hidden_size)

        hid_init = RandomUniform(minval=-lim, maxval=lim, seed=None)
        out_init = RandomUniform(minval=-3e-3, maxval=3e-3, seed=None)
    
        model = Sequential()
        model.add(Dense(input_shape=(self.state_size+self.action_size,),units=self.hidden_size,kernel_initializer=hid_init,activation="relu"))
        model.add(Dense(units=self.hidden_size,kernel_initializer=hid_init,activation="relu"))
        model.add(Dense(units=1))
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=LR_CRITIC))
        return model

    def set_weights(self,weights):
        self.network.set_weights(weights)

    def get_weights(self):
        return self.network.get_weights()

    def predict(self,states,actions):
        in_ = np.concatenate((states,actions),axis=1)
        return self.network.predict(in_)

