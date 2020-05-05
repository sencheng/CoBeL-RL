import keras.backend as K
import numpy as np

from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam, RMSprop
from keras.models import Model


class Agent:
    def __init__(self, inp_dim, out_dim, lr):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.rms_optimizer =  RMSprop(lr=lr, epsilon=0.1, rho=0.99)

    def fit(self, inp, targ):
        self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)

    def predict(self, inp):
        return self.model.predict(self.reshape(inp))

    def reshape(self, x):
        if len(x.shape) < 2: return np.expand_dims(x, axis=0)
        else: return x

class Critic(Agent):
    def __init__(self, inp_dim, out_dim, network, lr):
        Agent.__init__(self, inp_dim, out_dim, lr)
        self.model = self.addHead(network)
        self.discounted_r = K.placeholder(shape=(None,))

    def addHead(self, network):
        x = Dense(128, activation='relu')(network.output)
        out = Dense(1, activation='linear')(x)
        return Model(network.input, out)

    def optimizer(self):
        critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        return K.function([self.model.input, self.discounted_r], [], updates=updates)

class Actor(Agent):
    def __init__(self, inp_dim, out_dim, network, lr):
        Agent.__init__(self, inp_dim, out_dim, lr)
        self.model = self.addHead(network)
        self.action_pl = K.placeholder(shape=(None, self.out_dim))
        self.advantages_pl = K.placeholder(shape=(None,))

    def addHead(self, network):
        x = Dense(128, activation='relu')(network.output)
        out = Dense(self.out_dim, activation='softmax')(x)
        return Model(network.input, out)

    def optimizer(self):
        weighted_actions = K.sum(self.action_pl * self.model.output, axis=1)
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(self.advantages_pl)
        entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
        loss = 0.001 * entropy - K.sum(eligibility)

        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], loss)
        return K.function([self.model.input, self.action_pl, self.advantages_pl], [], updates=updates)
