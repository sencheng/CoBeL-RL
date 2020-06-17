import random
import numpy as np
import gym

from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical

from keras.layers import Input, Dense
import keras.backend as K

from network import Agent, Critic, Actor

class A2C:
    def __init__(self, env, gamma = 0.99, lr = 0.0001):
        self.env = env

        # Environment and A2C parameters
        self.act_dim = env.action_space.n
        self.env_dim = env.observation_space.shape[0]
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = Actor(self.env_dim, self.act_dim, self.shared, lr)
        self.critic = Critic(self.env_dim, self.act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        inp = Input(shape=(self.env_dim,))
        x = Dense(64, activation='relu')(inp)
        x = Dense(128, activation='relu')(inp)
        return Model(inp, x)

    def policy_action(self, s):
        return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

    def discount(self, r):
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards, done):
        discounted_rewards = self.discount(rewards)
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization
        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])

    def train(self):
        results = []
        EPISODES = 300

        for i in range(EPISODES):
            time, cumul_reward, done = 0, 0, False
            old_state = self.env.reset()
            actions, states, rewards = [], [], []

            while not done:
                self.env.render()

                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)

                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = self.env.step(a)

                # Memorize (s, a, r) for training
                actions.append(to_categorical(a, self.act_dim))
                rewards.append(r)
                states.append(old_state)

                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1

            # Train using discounted rewards ie. compute updates
            self.train_models(states, actions, rewards, done)
            print(cumul_reward)

        return results

env = gym.make('CartPole-v0')
net = A2C(env, gamma = 0.99, lr = 0.001)
net.train()


