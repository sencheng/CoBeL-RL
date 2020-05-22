import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from keras.initializers import RandomUniform

class Critic_Net(tf.keras.Model):
    def __init__(self,model_name,state_size,hidden_size,action_size,init_w=3e-3):
        super(Critic_Net,self).__init__(name = model_name)
        lim = 1. / np.sqrt(hidden_size)
        hid_init = RandomUniform(minval=-lim, maxval=lim, seed=None)
        out_init = RandomUniform(minval=-init_w, maxval=init_w, seed=None)

        self.fc1 = layers.Dense(hidden_size,activation = 'relu',kernel_initializer=hid_init, input_dim=state_size+action_size,dtype='float32')
        self.fc2 = layers.Dense(hidden_size,activation='relu',kernel_initializer=hid_init)
        self.v = layers.Dense(units=1)
    def call(self, state, action):
        x = tf.concat([state,action], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.v(x)


