import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.math import exp

from agents.SAC.models import create_critic,GaussianPolicy, DeterministicPolicy
from agents.SAC.buffer import Buffer
from agents.utils.ringbuffer import RingBuffer

class SACAgent:
    def __init__(self, env, gamma = 0.99, tau = 0.01, alpha = 0.2, buffer_maxlen = 50000, deterministic = True):
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
        self.deterministic = deterministic

        #Learning Rate
        # self.q_lr = 3e-4 
        # self.policy_lr = 3e-4
        # self.a_lr = 3e-4
        
        self.q_lr       = 0.001
        self.policy_lr  = 0.00001
        self.a_lr       = 0.001
        
        # initialize networks 
        self.q_net1 = create_critic(self.obs_dim,self.action_dim,256)
        self.q_net2 = create_critic(self.obs_dim,self.action_dim,256)
        self.target_q_net1 = create_critic(self.obs_dim,self.action_dim,256)
        self.target_q_net2 = create_critic(self.obs_dim,self.action_dim,256)
        
        if deterministic == False:
            self.policy_net = GaussianPolicy(self.obs_dim,self.action_dim,256)
        else:
            self.policy_net = DeterministicPolicy(self.obs_dim,self.action_dim,256)

        self.policy_net(np.expand_dims(np.zeros(self.obs_dim),axis=0))
        self.policy_net.load_weights("/home/wkst/Desktop/det_actor.h5")
        self.q_net1.load_weights("/home/wkst/Desktop/critic1.h5")
        self.q_net2.load_weights("/home/wkst/Desktop/critic2.h5")
        
        self.target_q_net1.set_weights(self.q_net1.get_weights())
        self.target_q_net2.set_weights(self.q_net2.get_weights())
        
        # initialize optimizers 
        self.q1_optimizer = Adam(learning_rate=self.q_lr,epsilon=1e-8,clipnorm=1.,clipvalue=0.5)
        self.q2_optimizer = Adam(learning_rate=self.q_lr,epsilon=1e-8,clipnorm=1.,clipvalue=0.5)
        self.policy_optimizer = Adam(learning_rate=self.policy_lr,epsilon=1e-8,clipnorm=1.,clipvalue=0.5)
        
        # entropy temperature
        self.alpha = alpha
        if deterministic == False:
            self.target_entropy = -env.action_space.shape[0]
            self.log_alpha = tf.Variable([0.0])
            self.alpha_optim = Adam(lr=self.a_lr)

        #Memory
        self.replay_buffer = Buffer(self.obs_dim,self.action_dim, buffer_maxlen, self.batch_size)
        self.buffer = RingBuffer(4)
    
    def get_action(self, state):
        action, _, _ = self.policy_net.sample(state)
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
            policy_actions, log_pi, _ = self.policy_net.sample(X)
            q_min = tf.math.minimum(
                self.target_q_net1([X, policy_actions]),
                self.target_q_net2([X, policy_actions]))
            loss = tf.math.reduce_mean(self.alpha * log_pi - tf.squeeze(q_min))
        grads = tape.gradient(loss,self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
    
    @tf.function
    def train_alpha(self,X):
        _ , log_pis, _ = self.policy_net.sample(X)
        with tf.GradientTape() as tape:
            tape.watch(self.log_alpha)
            alpha_loss = tf.reduce_mean((self.log_alpha.read_value()[0] * (-log_pis - self.target_entropy)))
        grads = tape.gradient(alpha_loss,[self.log_alpha])
        self.alpha_optim.apply_gradients(zip(grads, [self.log_alpha]))
        
    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch()
        
        next_actions, next_log_pi, _ = self.policy_net.sample(next_states)
        q_min = tf.math.minimum(
            self.target_q_net1([next_states, next_actions]),
            self.target_q_net2([next_states, next_actions])
        )
        q_min = tf.squeeze(q_min)
        
        next_q_target = q_min - self.alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target
        
        self.train_q_networks([states, actions],expected_q)
        
        if self.update_step % self.delay_step == 0:
            self.train_policy_network(states)
            self.soft_update(self.q_net1, self.target_q_net1, self.tau)
            self.soft_update(self.q_net2, self.target_q_net2, self.tau)

        if self.deterministic == False:
            self.train_alpha(states)
            self.alpha = tf.math.exp(self.log_alpha)
        self.update_step += 1
    
    def soft_update(self, local_model, target_model, tau):
        a = np.array(local_model.get_weights()) #local
        b = np.array(target_model.get_weights()) #target
        target_model.set_weights(tau*a + (1.0-tau)*b)
    
    def train(self,numEpisodes = 10000):
        for episode in range(numEpisodes):
            score = 0.0
            state = self.u_env._reset()
            self.buffer.insert_obs(state[0][:,:,0])
            self.buffer.insert_obs(state[0][:,:,1])
            self.buffer.insert_obs(state[0][:,:,2])
            self.buffer.insert_obs(state[0][:,:,2])
            state = self.buffer.generate_arr()
            state = np.expand_dims(state,axis=0)
            
            while True:
                action = self.get_action(np.float32(state))
                next_state, reward, done, _ = self.u_env._step(action)
                if(reward > 0):
                    score += reward
                #print(reward)
                if next_state[0].shape == self.obs_dim:
                    self.buffer.insert_obs(next_state[0][:,:,2])
                else:
                    self.buffer.insert_obs(next_state[0][0][:,:,2])
                
                next_state = self.buffer.generate_arr()
                next_state = np.expand_dims(next_state,axis=0)
                self.replay_buffer.record((np.float32(state[0]), np.float32(action[0]), np.float32(reward), np.float32(next_state[0]), done))
                # if self.replay_buffer.buffer_counter > self.batch_size:
                #     self.update(self.batch_size)   
                
                if done:
                    break
                state = next_state
            print("ep" , episode, ": ", score)
            if score >= 5:
                print("solved!")
                self.policy_net.save_weights("det_actor.h5")
                self.q_net1.save_weights("critic1.h5")
                self.q_net2.save_weights("critic2.h5")
