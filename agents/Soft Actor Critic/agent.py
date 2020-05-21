import random
import numpy as np

from actor import Actor_Net
from critic import Critic_Net
from buffers import ReplayBuffer

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

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
        self.actor_local = Actor_Net(self.state_size,hidden_size,self.action_size)
        self.actor_opt = tf.keras.optimizers.Adam(lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic1 = Critic_Net("critic1",self.state_size,hidden_size,self.action_size)
        self.critic1_opt = Adam(lr=LR_CRITIC)

        self.critic2 = Critic_Net("critic2",self.state_size,hidden_size,self.action_size)
        self.critic2_opt = Adam(lr=LR_CRITIC)

        self.critic1_target = Critic_Net("critic1_target",self.state_size,hidden_size,self.action_size)
        self.critic2_target = Critic_Net("critic2_target",self.state_size,hidden_size,self.action_size)

        self.critic1_target.set_weights(self.critic1.get_weights())
        self.critic2_target.set_weights(self.critic2.get_weights())

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.halving = np.full((BATCH_SIZE,), 0.5, dtype=float)
        
    def step(self, state, action, reward, next_state, done, step):
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(step, experiences, GAMMA)
            
    def act(self, state):
        action = self.actor_local.get_action(state)
        return action

    def actor_step(self,X,alpha,policy_prior_log_probs):
        with tf.GradientTape() as tape:
            actions_pred , log_pis = self.actor_local.evaluate(X)
            log_pis = tf.squeeze(log_pis)
            c1_in = tf.squeeze(self.critic1.call(X,actions_pred))
            loss = (alpha * log_pis - c1_in - policy_prior_log_probs)
            mean = tf.math.reduce_mean(loss, axis=0)
        grads = tape.gradient(mean,self.actor_local.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor_local.trainable_variables))

    def alpha_step(self,X):
        with tf.GradientTape() as tape:
            tape.watch(self.log_alpha)
            actions_pred, log_pis = self.actor_local.evaluate(X)
            alpha_loss = - K.mean(self.log_alpha.read_value()[0] * (log_pis + self.target_entropy))
        grads = tape.gradient(alpha_loss,self.log_alpha)
        grads = tf.expand_dims(grads,0)
        self.alpha_opt.apply_gradients(zip(grads, [self.log_alpha]))

    def learn(self, step, experiences, gamma, d=1):
        states, actions, rewards, next_states, dones = experiences
        

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor_local.evaluate(next_states)
        Q_target1_next = self.critic1_target.call(next_states, next_action)
        Q_target2_next = self.critic2_target.call(next_states, next_action)

        # take the min of both critics for updating
        Q_target_next = tf.squeeze(tf.math.minimum(Q_target1_next,Q_target2_next))
        
        log_pis_next = tf.squeeze(log_pis_next)

        if FIXED_ALPHA == None:
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * (Q_target_next - self.alpha * log_pis_next))
        else:
            Q_targets = rewards + (gamma * (1 - dones) * (Q_target_next - FIXED_ALPHA * log_pis_next))

        #Critic1
        with tf.GradientTape() as tape:
            v_s = self.critic1.call(states,actions)
            loss = 0.5 * tf.keras.losses.mean_squared_error(Q_targets,v_s)
        grads = tape.gradient(loss,self.critic1.trainable_variables)
        self.critic1_opt.apply_gradients(zip(grads, self.critic1.trainable_variables))

        #Critic2
        with tf.GradientTape() as tape:
            v_s = self.critic2.call(states,actions)
            loss = 0.5 * tf.keras.losses.mean_squared_error(Q_targets,v_s)
        grads = tape.gradient(loss,self.critic2.trainable_variables)
        self.critic2_opt.apply_gradients(zip(grads, self.critic2.trainable_variables))

        #Actor Learning Step
        if step % d == 0:
            ###########################
            alpha = np.exp(self.log_alpha.read_value()[0])

            # Compute alpha loss
            converted_states = tf.convert_to_tensor(states,dtype=np.float32)
            actions_pred, log_pis = self.actor_local.evaluate(converted_states)

            self.alpha_step(X=converted_states)

            self.alpha = alpha
            ############################

            #Fit
            self.actor_step(X=converted_states,alpha=alpha,policy_prior_log_probs=0.0)
                
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target, TAU)
        self.soft_update(self.critic2, self.critic2_target, TAU)
                     
    def soft_update(self, local_model, target_model, tau):
        a = np.array(local_model.get_weights()) #local
        b = np.array(target_model.get_weights()) #target
        #98% of target + 2% local
        target_model.set_weights(tau*a + (1.0-tau)*b)
