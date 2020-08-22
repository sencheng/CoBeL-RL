import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomUniform
from agents.utils.mish import Mish
tf.keras.backend.set_floatx('float32')

class Critic_Net(tf.keras.Model):
    def __init__(self,model_name,state_size,hidden_size,action_size,init_w=3e-3):
        super(Critic_Net,self).__init__(name = model_name)
        lim = 1. / np.sqrt(hidden_size)
        hid_init = RandomUniform(minval=-lim, maxval=lim, seed=None)
        out_init = RandomUniform(minval=-init_w, maxval=init_w, seed=None)

        #Conv Head
        self.conv1 = layers.Conv2D(16, kernel_size=3, activation='relu',kernel_initializer=hid_init,input_shape=state_size)
        self.mp1 = layers.MaxPooling2D(pool_size=(2,2))
        self.conv2 = layers.Conv2D(32, kernel_size=3, activation='relu',kernel_initializer=hid_init)
        self.mp2 = layers.MaxPooling2D(pool_size=(2,2))
        self.flatten = layers.Flatten()

        self.cnn_dense = layers.Dense(128,activation='relu',kernel_initializer=hid_init,input_shape=(action_size,))
        self.act_dense = layers.Dense(64,activation='relu',kernel_initializer=hid_init)

        self.concat = layers.Dense(64,activation='relu',kernel_initializer=hid_init)
        
        self.v = layers.Dense(units=1,kernel_initializer=out_init)
    def call(self,X):
        x = self.conv1(X[0])
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.flatten(x)
        x = self.cnn_dense(x)

        y = self.act_dense(X[1])
        
        z = layers.concatenate([x,y])
        z = self.concat(z)
        return self.v(z)


