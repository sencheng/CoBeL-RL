import random
import numpy as np

from actor import Actor
from critic import Critic
from buffers import ReplayBuffer

import autograd.numpy as np
from autograd import grad

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow.keras.backend as K

LR_ACTOR = float(5e-4)
LR_CRITIC = float(5e-4)
BUFFER_SIZE = int(1e6)
BATCH_SIZE = int(256)
GAMMA = float(0.99)
TAU = float(1e-2)
FIXED_ALPHA = None

class Agent():
    def __init__(self, state_size, action_size, random_seed, hidden_size, action_prior="uniform"):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.target_entropy = -action_size  # -dim(A)
        
        self.alpha = 1
        self.log_alpha = tf.Variable([0.0])
        self.alpha_opt = tf.keras.optimizers.Adam(lr=LR_ACTOR)
        
        self._action_prior = action_prior
        
        # Actor Network 
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size)
        self.actor_opt = tf.keras.optimizers.Adam(lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed, hidden_size)
        self.critic2 = Critic(state_size, action_size, random_seed, hidden_size)
        
        self.critic1_target = Critic(state_size, action_size, random_seed,hidden_size)
        self.critic2_target = Critic(state_size, action_size, random_seed,hidden_size)

        self.critic1_target.set_weights(self.critic1.get_weights())
        self.critic2_target.set_weights(self.critic2.get_weights())

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.halving = np.arange(256)
        self.halving = self.halving.fill(0.5)
        
    def step(self, state, action, reward, next_state, done, step):
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(step, experiences, GAMMA)
            
    def act(self, state):
        action = self.actor_local.get_action(state)
        return action

    def alpha_loss(self, x, log_pis):
        loss = -(x * (log_pis + self.target_entropy)).mean()
        return loss

    def actor_step(self,X,alpha,policy_prior_log_probs):
        with tf.GradientTape() as tape:
            actions_pred , log_pis = self.actor_local.model(X)
            c1_in = self.critic1.predict_learn(X, actions_pred)
            loss = K.mean((alpha * log_pis - c1_in - policy_prior_log_probs))
        grads = tape.gradient(loss,self.actor_local.model.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor_local.model.trainable_variables))

    def alpha_step(self,X):
        with tf.GradientTape() as tape:
            tape.watch(self.log_alpha)
            actions_pred, log_pis = self.actor_local.model(X)
            alpha_loss = - K.mean(self.log_alpha.read_value()[0] * (log_pis + self.target_entropy))
        grads = tape.gradient(alpha_loss,self.log_alpha)
        grads = tf.expand_dims(grads,0)
        self.alpha_opt.apply_gradients(zip(grads, [self.log_alpha]))

    def learn(self, step, experiences, gamma, d=1):
        states, actions, rewards, next_states, dones = experiences
        

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target.predict(next_states, next_action)
        Q_target2_next = self.critic2_target.predict(next_states, next_action)

        # take the min of both critics for updating
        Q_target_next = np.minimum(Q_target1_next.squeeze(), Q_target2_next.squeeze())

        #next_action = next_action.squeeze()
        log_pis_next = log_pis_next.squeeze()

        if FIXED_ALPHA == None:
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * (Q_target_next - self.alpha * log_pis_next))
        else:
            Q_targets = rewards + (gamma * (1 - dones) * (Q_target_next - FIXED_ALPHA * log_pis_next))
        
        x = np.concatenate((states,actions),axis=1)
        y = np.expand_dims(Q_targets,axis=1)

        loss1 = self.critic1.network.fit(x,y, verbose=False, sample_weight=self.halving)
        loss2 = self.critic2.network.fit(x,y, verbose=False, sample_weight=self.halving)

        #Actor Learning Step
        if step % d == 0:
            ###########################
            alpha = np.exp(self.log_alpha.read_value()[0])

            # Compute alpha loss
            converted_states = tf.convert_to_tensor(states,dtype=np.float32)
            actions_pred, log_pis = self.actor_local.evaluate(converted_states)

            self.alpha_step(X=converted_states)

            self.alpha = alpha
            #print(self.alpha)
            ############################

            #Fit
            self.actor_step(X=converted_states,alpha=alpha,policy_prior_log_probs=0.0)
                
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1.network, self.critic1_target.network, TAU)
        self.soft_update(self.critic2.network, self.critic2_target.network, TAU)
                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        a = np.array(local_model.get_weights()) 
        b = np.array(target_model.get_weights()) 
        target_model.set_weights(b + (1-tau)*a)
