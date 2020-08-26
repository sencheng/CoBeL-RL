import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.math import exp

from agents.SAC_Fixed.models import create_critic,PolicyNetwork
from agents.utils.basic_buffer import BasicBuffer
from agents.utils.ringbuffer import RingBuffer

class SACAgent:
    def __init__(self, env, gamma = 0.99, tau = 0.01, alpha = 0.2, q_lr = 3e-4, policy_lr = 3e-4, a_lr = 3e-4, buffer_maxlen = 1000000):
        self.u_env = env
        
        self.action_range = [env.action_space.low, env.action_space.high]
        self.obs_dim = env.observation_space.shape + (4,)
        self.action_dim = env.action_space.n
        self.batch_size = 256
        
        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2
        
        # initialize networks 
        self.q_net1 = create_critic(self.obs_dim,self.action_dim,256)
        self.q_net2 = create_critic(self.obs_dim,self.action_dim,256)
        self.target_q_net1 = create_critic(self.obs_dim,self.action_dim,256)
        self.target_q_net2 = create_critic(self.obs_dim,self.action_dim,256)
        self.policy_net = PolicyNetwork(self.obs_dim,self.action_dim,256)

        self.target_q_net1.set_weights(self.q_net1.get_weights())
        self.target_q_net2.set_weights(self.q_net2.get_weights())
        
        # initialize optimizers 
        self.q1_optimizer = Adam(learning_rate=q_lr,epsilon=1e-8)
        self.q2_optimizer = Adam(learning_rate=q_lr,epsilon=1e-8)
        self.policy_optimizer = Adam(learning_rate=policy_lr,epsilon=1e-8)
        
        # entropy temperature
        self.alpha = alpha
        self.target_entropy = -env.action_space.shape[0]
        self.log_alpha = tf.Variable([0.0])
        self.alpha_optim = Adam(lr=a_lr,)

        #Memory
        self.replay_buffer = BasicBuffer(buffer_maxlen)
        self.buffer = RingBuffer(4)
    
    def get_action(self, state):
        action, _ = self.policy_net.sample(state)
        action = np.clip(action*self.action_range[1], self.action_range[0], self.action_range[1])
        return action
    
    @tf.function
    def train_q_networks(self,X,y_true):
        with tf.GradientTape() as tape:
            critic_value = self.q_net1(X)
            critic_value = tf.squeeze(critic_value)
            critic_loss = tf.math.reduce_mean(tf.math.square(y_true - critic_value))
        critic_grad = tape.gradient(critic_loss, self.q_net1.trainable_variables)
        self.q1_optimizer.apply_gradients(zip(critic_grad, self.q_net1.trainable_variables))
        
        with tf.GradientTape() as tape:
            critic_value = self.q_net2(X)
            critic_value = tf.squeeze(critic_value)
            critic_loss = tf.math.reduce_mean(tf.math.square(y_true - critic_value))
        critic_grad = tape.gradient(critic_loss, self.q_net2.trainable_variables)
        self.q2_optimizer.apply_gradients(zip(critic_grad, self.q_net2.trainable_variables))
        
    @tf.function
    def train_policy_network(self,X):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.policy_net.trainable_variables)
            policy_actions, log_pi = self.policy_net.sample(X)
            state_action_policy = tf.concat([X, policy_actions], axis=1)
            q_min = tf.math.minimum(
                self.target_q_net1(state_action_policy),
                self.target_q_net2(state_action_policy))
            loss = tf.math.reduce_mean(self.alpha * log_pi - tf.squeeze(q_min))
        grads = tape.gradient(loss,self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
    
    @tf.function
    def train_alpha(self,X):
        _ , log_pis = self.policy_net.sample(X)
        with tf.GradientTape() as tape:
            tape.watch(self.log_alpha)
            alpha_loss = tf.reduce_mean((self.log_alpha.read_value()[0] * (-log_pis - self.target_entropy)))
        grads = tape.gradient(alpha_loss,[self.log_alpha])
        self.alpha_optim.apply_gradients(zip(grads, [self.log_alpha]))
        
    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        rewards = tf.squeeze(rewards)
        next_states = tf.convert_to_tensor(next_states)
        dones = np.array(dones)
        
        next_actions, next_log_pi = self.policy_net.sample(next_states)
        nstate_naction = tf.concat([next_states, next_actions], axis=1)
        q_min = tf.math.minimum(
            self.target_q_net1(nstate_naction),
            self.target_q_net2(nstate_naction)
        )
        q_min = tf.squeeze(q_min)
        
        next_q_target = q_min - self.alpha * next_log_pi
        state_action = tf.concat([states, actions], axis=1)
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target
        
        self.train_q_networks(state_action,expected_q)
        
        if self.update_step % self.delay_step == 0:
            self.train_policy_network(states)
            self.soft_update(self.q_net1, self.target_q_net1, self.tau)
            self.soft_update(self.q_net2, self.target_q_net2, self.tau)

        self.train_alpha(states)
        self.alpha = tf.math.exp(self.log_alpha)
        self.update_step += 1
    
    def soft_update(self, local_model, target_model, tau):
        a = np.array(local_model.get_weights()) #local
        b = np.array(target_model.get_weights()) #target
        target_model.set_weights(tau*a + (1.0-tau)*b)
    
    def train(self,numEpisodes = 10000):
        for episode in range(numEpisodes):
            state = self.u_env._reset()
            self.buffer.insert_obs(state[0][:,:,0])
            self.buffer.insert_obs(state[0][:,:,1])
            self.buffer.insert_obs(state[0][:,:,2])
            self.buffer.insert_obs(state[0][:,:,3])
            state = self.buffer.generate_arr()
            state = np.expand_dims(state,axis=0)
            
            while True:
                action = self.get_action(np.float32(state))
                next_state, reward, done, _ = self.u_env._step(action)
                if next_state[0].shape == self.obs_dim:
                    self.buffer.insert_obs(next_state[0][:,:,2])
                else:
                    self.buffer.insert_obs(next_state[0][0][:,:,2])
                self.replay_buffer.push(np.float32(state), np.float32(action), np.float32(reward), np.float32(next_state), done)

                if len(self.replay_buffer) > self.batch_size:
                    self.update(self.batch_size)   
                
                if done:
                    break
                state = next_state