import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomUniform

tf.keras.backend.set_floatx('float32')

class Critic_Net(tf.keras.Model):
    def __init__(self,model_name,state_size,hidden_size,action_size,init_w=3e-3):
        super(Critic_Net,self).__init__(name = model_name)
        lim = 1. / np.sqrt(hidden_size)
        hid_init = RandomUniform(minval=-lim, maxval=lim, seed=None)
        out_init = RandomUniform(minval=-init_w, maxval=init_w, seed=None)

        #Conv Head
        self.conv1 = layers.Conv2D(128, kernel_size=3, activation='relu',input_shape=state_size)
        self.mp1 = layers.MaxPooling2D(pool_size=(2,2))
        self.conv2 = layers.Conv2D(128, kernel_size=3, activation='relu')
        self.mp2 = layers.MaxPooling2D(pool_size=(2,2))
        self.flatten = layers.Flatten()

        self.dense = layers.Dense(hidden_size,activation='relu',kernel_initializer=hid_init)
        self.v = layers.Dense(units=1,kernel_initializer=out_init)
    def call(self,X):
        x = self.conv1(X)
        x = self.mp1(X)
        x = self.conv2(X)
        x = self.mp2(X)
        x = self.flatten(x)
        x = self.dense(x)
        return self.v(x)


