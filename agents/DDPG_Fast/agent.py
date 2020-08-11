import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from agents.DDPG.noise import OUNoise
from agents.DDPG.actor import ActorNetwork
from agents.DDPG.critic import CriticNetwork
from agents.DDPG.replay_buffer import ReplayBuffer

from interfaces.oai_gym_interface import UnityInterface

class DDPG_Agent:
    def __init__(self,session, env: UnityInterface, FrameSkip = 4, NumEpisodes = 1000):
        self.sess = session
        self.u_env = env
        self.state_dim = env.observation_space.shape + (FrameSkip,)
        self.action_dim = env.action_space.n
        self.num_episodes = NumEpisodes
        self.tau = 0.001
        self.gamma = 0.99
        self.min_batch = 64
        self.actor_lr = 0.00005
        self.critic_lr = 0.0005
        self.buffer_size = 1000000

        self.actor_noise = OUNoise(mu=np.zeros(self.action_dim))
        self.actor = ActorNetwork(self.sess,self.state_dim, self.action_dim, 1, self.actor_lr, self.tau, self.min_batch)
        self.critic = CriticNetwork(self.sess, self.state_dim, self.action_dim, self.critic_lr, self.tau, self.gamma, self.actor.get_num_trainable_vars())

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()

        # Initialize replay memory
        replay_buffer = ReplayBuffer(self.buffer_size, 0)

        max_episodes = self.num_episodes
        max_steps = 3000
        score_list = []


        for i in range(max_episodes):
            state = self.u_env._reset()
            score = 0
            for j in range(max_steps):
                action = self.actor.predict(np.reshape(state, (1, self.actor.s_dim[0],self.actor.s_dim[1],self.actor.s_dim[2]))) + self.actor_noise()
                action = np.clip(action,-1,1)
                print(action)
                next_state, reward, done, info = self.u_env._step(action)
                replay_buffer.add(np.reshape(state, (self.actor.s_dim[0],self.actor.s_dim[1],self.actor.s_dim[2])), np.reshape(action, (self.actor.a_dim,)), reward,
                                done, np.reshape(next_state, (self.actor.s_dim[0],self.actor.s_dim[1],self.actor.s_dim[2],)))

                # updating the network in batch
                if replay_buffer.size() < self.min_batch:
                    continue

                states, actions, rewards, dones, next_states = replay_buffer.sample_batch(self.min_batch)
                target_q = self.critic.predict_target(next_states, self.actor.predict_target(next_states))

                y = []
                for k in range(self.min_batch):
                    y.append(rewards[k] + self.critic.gamma * target_q[k] * (1-dones[k]))

                # Update the critic given the targets
                predicted_q_value, _ = self.critic.train(states, actions, np.reshape(y, (self.min_batch, 1)))

                # Update the actor policy using the sampled gradient
                a_outs = self.actor.predict(states)
                grads = self.critic.action_gradients(states, a_outs)
                self.actor.train(states, grads[0])

                # Update target networks
                self.actor.update_target_network()
                self.critic.update_target_network()

                state = next_state
                score += reward

                if done:
                    print('Reward: {} | Episode: {}/{}'.format(int(score), i, max_episodes))
                    break
            score_list.append(score)
            avg = np.mean(score_list[-100:])
            print("Average of last 100 episodes: {0:.2f} \n".format(avg))
        return score_list

