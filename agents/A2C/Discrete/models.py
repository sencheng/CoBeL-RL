import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
from agents.utils.mish import Mish

class ProbabilityDistribution(tf.keras.Model):
  def call(self, logits, **kwargs):
    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
  def __init__(self, num_states, num_actions, hid_act):
    super().__init__('mlp_policy')

    self.c1 = kl.Conv2D(8, (3, 3), activation=hid_act,kernel_initializer=tf.keras.initializers.HeNormal, input_shape=num_states)
    self.mp1 = kl.MaxPooling2D((2, 2))
    self.c2 = kl.Conv2D(16, (3, 3), activation=hid_act,kernel_initializer=tf.keras.initializers.HeNormal)
    self.mp2 = kl.MaxPooling2D((2, 2))
    
    self.fl = kl.Flatten()
    
    self.d1 = kl.Dense(128,activation=hid_act,kernel_initializer=tf.keras.initializers.HeNormal)

    self.hidden1 = kl.Dense(64, activation=hid_act,kernel_initializer=tf.keras.initializers.HeNormal)
    self.hidden2 = kl.Dense(64, activation=hid_act,kernel_initializer=tf.keras.initializers.HeNormal)
    self.value = kl.Dense(1, name='value')
    
    self.logits = kl.Dense(num_actions, name='policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    x = self.c1(x)
    x = self.mp1(x)
    x = self.c2(x)
    x = self.mp2(x)
    x = self.fl(x)
    x = self.d1(x)

    hidden_logs = self.hidden1(x)
    hidden_vals = self.hidden2(x)
    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs):
    logits, value = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)