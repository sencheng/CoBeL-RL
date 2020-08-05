import tensorflow as tf
import sys
import gym
import numpy as np
from scipy.stats import norm
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from interfaces.oai_gym_interface import UnityInterface
EPISODES = 3000

# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, env: UnityInterface, FrameSkip = 4):
        self.u_env = env
        self.state_size = env.observation_space.shape + (FrameSkip,)
        self.action_size = env.action_space.n

        self.value_size = 1

        #Hyperparameters
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.discount_factor = .9

        #Models
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

    def build_model(self):
        #Input => (Batch,Width,Height,Frames)
        state = kl.Input(batch_shape=(None, self.state_size[0],self.state_size[1],self.state_size[2]))

        #Conv Head
        conv1 = kl.Conv2D(128, kernel_size=3, activation='relu')(state)
        mp1 = kl.MaxPooling2D(pool_size=(2,2))(conv1)
        conv2 = kl.Conv2D(128, kernel_size=3, activation='relu')(mp1)
        mp2 = kl.MaxPooling2D(pool_size=(2,2))(conv2)
        flatten = kl.Flatten()(mp2)

        #Actor Tail
        actor_hid = kl.Dense(256, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(flatten)
        mu = kl.Dense(self.action_size, activation='tanh', kernel_initializer='he_uniform')(actor_hid)
        sigma = kl.Dense(self.action_size, activation='softplus', kernel_initializer='he_uniform')(actor_hid)

        #Critic Tail
        critic_hid = kl.Dense(256, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(flatten)
        state_value = kl.Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')(critic_hid)

        actor = Model(inputs=state, outputs=(mu, sigma))
        critic = Model(inputs=state, outputs=state_value)

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, 1))
        advantages = K.placeholder(shape=(None, 1))

        mu, sigma_sq = self.actor.output

        pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))

        exp_v = log_pdf * advantages

        exp_v = K.sum(exp_v + 0.01 * entropy)
        actor_loss = -exp_v

        optimizer = Adam(lr=self.actor_lr)
        with actor_loss.graph.as_default():
            updates = optimizer.get_updates(loss=actor_loss,params=self.actor.trainable_weights)

        train = K.function([self.actor.input,action, advantages],tf.constant(0), updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, 1))

        value = self.critic.output
        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        with loss.graph.as_default():
            updates = optimizer.get_updates(loss=loss,params=self.critic.trainable_weights)

        train = K.function([self.critic.input,discounted_reward],tf.constant(0), updates=updates)
        return train

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        mu, sigma = self.actor.predict(state)
        epsilon = np.random.randn(self.action_size)
        action = mu + np.sqrt(sigma) * epsilon
        action = np.clip(action, -1, 1)
        return action

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0] = reward - value
            target[0][0] = reward
        else:
            advantages[0] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.optimizer[0]([state, action, advantages])
        self.optimizer[1]([state, target])

    def train(self,num_frames: int):
        state = self.u_env._reset()
        state = np.expand_dims(state[0],axis=0)
        for frame_idx in range(num_frames):
            action = self.get_action(state)
            next_state, reward, done, info = self.u_env._step(np.array(action))
            next_state = np.expand_dims(next_state[0],axis=0)
            self.train_model(state, action, reward, next_state, done)
            state = next_state

    

    