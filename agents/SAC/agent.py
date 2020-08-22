import random
import numpy as np

from agents.SAC.actor import Actor_Net
from agents.SAC.critic import Critic_Net
from agents.SAC.buffers import ReplayBuffer

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from interfaces.oai_gym_interface import UnityInterface

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from agents.utils.ringbuffer import RingBuffer



class SACAgent():
    def __init__(self, env : UnityInterface, FrameSkip = 4, hid_size = 128, buffer_size = int(1e6), batch_size = 128):
        self.u_env = env

        self.state_size = env.observation_space.shape + (4,)
        self.action_size = env.action_space.n

        self.gamma = 0.99
        self.tau = 0.01
        self.hidden_size = hid_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        self.actor_lr  = 0.005
        self.critic_lr = 0.005

        self.target_entropy = -self.action_size
        self.alpha = 1
        self.log_alpha = tf.Variable([0.0])
        self.alpha_opt = Adam(lr=self.actor_lr)
        
        # Actor Network 
        self.actor_local = Actor_Net(self.state_size,self.hidden_size,self.action_size)
        self.actor_opt = Adam(lr=self.actor_lr)

        # Critic Network (w/ Target Network)
        self.critic1 = Critic_Net("critic1",self.state_size,self.hidden_size,self.action_size)
        self.critic1.compile(optimizer=Adam(lr=self.critic_lr),loss=tf.keras.losses.MSE,metrics=['accuracy'])

        self.critic2 = Critic_Net("critic2",self.state_size,self.hidden_size,self.action_size)
        self.critic2.compile(optimizer=Adam(lr=self.critic_lr),loss=tf.keras.losses.MSE,metrics=['accuracy'])

        self.critic1_target = Critic_Net("critic1_target",self.state_size,self.hidden_size,self.action_size)
        self.critic2_target = Critic_Net("critic2_target",self.state_size,self.hidden_size,self.action_size)

        self.critic1_target.set_weights(self.critic1.get_weights())
        self.critic2_target.set_weights(self.critic2.get_weights())

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size)
        self.buffer = RingBuffer(4)
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)
            
    def act(self, state):
        action = self.actor_local.get_action(state)
        return action

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        next_action, next_log_pis = self.actor_local.evaluate(next_states)

        #Clipped double Q Trick
        next_q1 = self.critic1_target.call([next_states, next_action])
        next_q2 = self.critic2_target.call([next_states, next_action])
        
        next_q_target = tf.squeeze(tf.math.minimum(next_q1,next_q2))
        next_q_target = next_q_target - self.alpha * tf.squeeze(next_log_pis)
        Q_targets = rewards + (1 - dones) * self.gamma * next_q_target
        #######################
        
        #Fit
        self.critic1.fit([states,actions],Q_targets,verbose=0)
        self.critic2.fit([states,actions],Q_targets, verbose=0)
        ####
        
        alpha = np.exp(self.log_alpha.read_value()[0])
        # Compute alpha loss
        converted_states = tf.convert_to_tensor(states,dtype=np.float32)

        with tf.GradientTape() as tape:
            tape.watch(self.log_alpha)
            actions_pred, log_pis = self.actor_local.evaluate(converted_states)
            alpha_loss = -K.mean(self.log_alpha.read_value()[0] * (log_pis + self.target_entropy))
        grads = tape.gradient(alpha_loss,self.log_alpha)
        grads = tf.expand_dims(grads,0)
        self.alpha_opt.apply_gradients(zip(grads, [self.log_alpha]))
        self.alpha = alpha

        #Fit Actor Step
        with tf.GradientTape() as tape:
            actions_pred , log_pis = self.actor_local.evaluate(converted_states)
            log_pis = tf.squeeze(log_pis)
            c1_in = tf.squeeze(self.critic1.call([converted_states, actions_pred]))
            loss = (alpha * log_pis - c1_in)
            mean = tf.math.reduce_mean(loss, axis=0)
        grads = tape.gradient(mean,self.actor_local.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor_local.trainable_variables))
                
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target, self.tau)
        self.soft_update(self.critic2, self.critic2_target, self.tau)
                     
    def soft_update(self, local_model, target_model, tau):
        a = np.array(local_model.get_weights()) #local
        b = np.array(target_model.get_weights()) #target
        target_model.set_weights(tau*a + (1.0-tau)*b)
    
    def train(self, numEpisodes = 10000):
        for ep in range(numEpisodes):
            score = 0
            state = self.u_env._reset()

            self.buffer.insert_obs(state[0][:,:,0])
            self.buffer.insert_obs(state[0][:,:,1])
            self.buffer.insert_obs(state[0][:,:,2])
            self.buffer.insert_obs(state[0][:,:,3])
            state = self.buffer.generate_arr()
            state = np.expand_dims(state,axis=0)

            while True:
                action = self.act(state)
                action_v = tf.expand_dims(tf.clip_by_value(action*1, -1, 1),axis=0)
                action_v = tf.keras.backend.eval(action_v)
                #print(action_v)
                next_state, reward, done, info = self.u_env._step(action_v)
                score += reward
                if next_state[0].shape == self.state_size:
                    self.buffer.insert_obs(next_state[0][:,:,2])
                else:
                    self.buffer.insert_obs(next_state[0][0][:,:,2])
            
                next_state = self.buffer.generate_arr()
                next_state = np.expand_dims(next_state,axis=0)

                self.step(state, action_v, reward, next_state, done)
                state = next_state
                if done:
                    break
            print("ep" , ep, ": ", score)
            if score >= 4:
                print("solved!")
                self.actor_local.save_weights("actor_sac.h5")
                self.critic1.save_weights("critic1_sac.h5")
                self.critic2.save_weights("critic2_sac.h5")

