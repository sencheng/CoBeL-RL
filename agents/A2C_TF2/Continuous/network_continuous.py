import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizer

class Model(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__('mlp_policy')
    
    #Vectorized Input or ConvNet...
    self.base = layers.Dense(128, activation='relu')

    #Output
    self.mu = layers.Dense(num_actions, name='mu')
    self.logstd = layers.Dense(num_actions, name='logstd')
    self.value = layers.Dense(1, name='value')

  def call(self, inputs, **kwargs):
    # Inputs is a numpy array, convert to a tensor.
    x = tf.convert_to_tensor(inputs)
    x = self.base(x)
    #Output
    return self.mu(x),self.logstd(x),self.value(x)

  def action_value(self, obs):
    mu,log_std,value = self.predict_on_batch(obs)
    std = np.exp(log_std)
    #Sample one more more action from the processed input data
    action = np.random.normal(mu,std)
    return action, np.squeeze(value, axis=-1)

