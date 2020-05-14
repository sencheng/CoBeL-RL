import numpy as np
import random
from collections import namedtuple, deque
import gym

import time
import argparse

from agent import Agent

def SAC(n_episodes=200, max_t=500, print_every=10):
    scores_deque = deque(maxlen=100)
    average_100_scores = []

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state = state.reshape((1,state_size))
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            action_v = np.clip(action*action_high, action_low, action_high)
            next_state, reward, done, info = env.step(action_v)
            env.render()
            next_state = next_state.reshape((1,state_size))
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
            score += reward

            if done:
                break 
        
        scores_deque.append(score)
        average_100_scores.append(np.mean(scores_deque))
        
        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}  Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)))
    
if __name__ == "__main__":
    env_name = "Pendulum-v0"
    seed = int(0)
    episodes = int(100)
    HIDDEN_SIZE = int(256)
    
    t0 = time.time()
    env = gym.make(env_name)
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    env.seed(seed)
    np.random.seed(seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed,hidden_size=HIDDEN_SIZE, action_prior="uniform") #"normal"

    SAC(n_episodes=episodes, max_t=500, print_every=int(100))    
        
    t1 = time.time()
    env.close()
    print("training took {} min!".format((t1-t0)/60))