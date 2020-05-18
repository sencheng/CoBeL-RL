import torch
import random
import numpy as np

from actor import Actor
from critic import Critic
from buffers import ReplayBuffer
from adam import Adam

import autograd.numpy as np
from autograd import grad

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
        self._action_prior = action_prior
        
        # Actor Network 
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size)
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed, hidden_size)
        self.critic2 = Critic(state_size, action_size, random_seed, hidden_size)
        
        self.critic1_target = Critic(state_size, action_size, random_seed,hidden_size)
        self.critic2_target = Critic(state_size, action_size, random_seed,hidden_size)

        self.critic1_target.set_weights(self.critic1.get_weights())
        self.critic2_target.set_weights(self.critic2.get_weights())

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Adam Optimizer
        self.adam = Adam(0.0)
        
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

    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
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

        loss1 = self.critic1.network.fit(x,y, verbose=False)
        loss2 = self.critic2.network.fit(x,y, verbose=False)

        #Actor Learning Step
        if step % d == 0:
            alpha = np.exp(self.adam.param)
            # Compute alpha loss
            actions_pred, log_pis = self.actor_local.evaluate(states)

            #Forward Pass
            a_loss = grad(self.alpha_loss)
            a_grad = a_loss(self.adam.param,log_pis) 

            #Backward Pass
            self.adam.backward_pass(a_grad)
            self.alpha = alpha

            # policy_prior_log_probs = 0.0
            # actor_loss = (alpha * log_pis.squeeze - self.critic1(states, actions_pred.squeeze(0)) - policy_prior_log_probs).mean()

            # # Minimize the loss
            # self.actor_optimizer.zero_grad()
            # actor_loss.backward()
            # self.actor_optimizer.step()
                
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
