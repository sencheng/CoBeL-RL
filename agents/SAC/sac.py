import numpy as np
import random
from collections import namedtuple, deque
import gym

import time
import argparse

from agent import Agent
import matplotlib.pyplot as plt

def SAC(n_episodes=200, max_t=500):
    scores_deque = deque(maxlen=100)
    average_100_scores = []

    for i_episode in range(0, n_episodes):
        state = env.reset()
        state = state.reshape((1,state_size))
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            action_v = np.clip(action*action_high, action_low, action_high)
            next_state, reward, done, info = env.step([action_v])

            next_state = next_state.reshape((1,state_size))
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
            score += reward

            if done:
                break 
        
        scores_deque.append(score)
        average_100_scores.append(np.mean(scores_deque))
        
        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)),end="")
    x = np.arange(n_episodes)
    plt.plot(x,scores_deque)
    plt.ylabel('some numbers')
    plt.show()
    
if __name__ == "__main__":
    env = gym.make("Pendulum-v0")

    episodes = 50
    max_t = 500
    HIDDEN_SIZE = 256
    
    action_size = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]

    state_size = env.observation_space.shape[0]
    
    agent = Agent(state_size=state_size, action_size=action_size, hidden_size=HIDDEN_SIZE)

    SAC(episodes,max_t)    
    env.close()