from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.losses import mean_squared_error
import numpy as np

LR_CRITIC = float(5e-4)

def build_critic_network(state_size, action_size, seed, hidden_size=32):
    lim = 1. / np.sqrt(hidden_size)

    hid_init = RandomUniform(minval=-lim, maxval=lim, seed=None)
    out_init = RandomUniform(minval=-3e-3, maxval=3e-3, seed=None)
    
    model = Sequential()
    model.add(Dense(input_shape=(state_size+action_size,),units=hidden_size,kernel_initializer=hid_init,activation="relu"))
    model.add(Dense(units=hidden_size,kernel_initializer=hid_init,activation="relu"))
    model.add(Dense(units=1))
    model.compile(loss=mean_squared_error, optimizer=Adam(lr=LR_CRITIC))
    return model


