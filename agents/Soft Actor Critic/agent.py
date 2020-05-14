import torch
import random

from actor import Actor
from critic import build_critic_network
from buffers import ReplayBuffer

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
        #self.log_alpha = torch.tensor([0.0], requires_grad=True)
        #self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=LR_ACTOR) 
        self._action_prior = action_prior
        
        # Actor Network 
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size)
        
        # Critic Network (w/ Target Network)
        self.critic1 = build_critic_network(state_size, action_size, random_seed, hidden_size)
        self.critic2 = build_critic_network(state_size, action_size, random_seed, hidden_size)
        
        self.critic1_target = build_critic_network(state_size, action_size, random_seed,hidden_size)
        self.critic2_target = build_critic_network(state_size, action_size, random_seed,hidden_size)

        self.critic1_target.set_weights(self.critic1.get_weights())
        self.critic2_target.set_weights(self.critic2.get_weights())

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        

    def step(self, state, action, reward, next_state, done, step):
        self.memory.add(state, action, reward, next_state, done)

        # # Learn, if enough samples are available in memory
        # if len(self.memory) > BATCH_SIZE:
        #     experiences = self.memory.sample()
        #     self.learn(step, experiences, GAMMA)
            
    def act(self, state):
        action = self.actor_local.get_action(state)
        return action

    # def learn(self, step, experiences, gamma, d=1):
    #     """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
    #     Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
    #     Critic_loss = MSE(Q, Q_target)
    #     Actor_loss = α * log_pi(a|s) - Q(s,a)
    #     where:
    #         actor_target(state) -> action
    #         critic_target(state, action) -> Q-value
    #     Params
    #     ======
    #         experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
    #         gamma (float): discount factor
    #     """
    #     states, actions, rewards, next_states, dones = experiences
        

    #     # ---------------------------- update critic ---------------------------- #
    #     # Get predicted next-state actions and Q values from target models
    #     next_action, log_pis_next = self.actor_local.evaluate(next_states)

    #     Q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
    #     Q_target2_next = self.critic2_target(next_states.to(device), next_action.squeeze(0).to(device))

    #     # take the mean of both critics for updating
    #     Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        
    #     if FIXED_ALPHA == None:
    #         # Compute Q targets for current states (y_i)
    #         Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
    #     else:
    #         Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - FIXED_ALPHA * log_pis_next.squeeze(0).cpu()))
    #     # Compute critic loss
    #     Q_1 = self.critic1(states, actions).cpu()
    #     Q_2 = self.critic2(states, actions).cpu()
    #     critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
    #     critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets.detach())
    #     # Update critics
    #     # critic 1
    #     self.critic1_optimizer.zero_grad()
    #     critic1_loss.backward()
    #     self.critic1_optimizer.step()
    #     # critic 2
    #     self.critic2_optimizer.zero_grad()
    #     critic2_loss.backward()
    #     self.critic2_optimizer.step()
    #     if step % d == 0:
    #     # ---------------------------- update actor ---------------------------- #
    #         if FIXED_ALPHA == None:
    #             alpha = torch.exp(self.log_alpha)
    #             # Compute alpha loss
    #             actions_pred, log_pis = self.actor_local.evaluate(states)
    #             alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
    #             self.alpha_optimizer.zero_grad()
    #             alpha_loss.backward()
    #             self.alpha_optimizer.step()
                
    #             self.alpha = alpha
    #             # Compute actor loss
    #             if self._action_prior == "normal":
    #                 policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
    #                 policy_prior_log_probs = policy_prior.log_prob(actions_pred)
    #             elif self._action_prior == "uniform":
    #                 policy_prior_log_probs = 0.0
    
    #             actor_loss = (alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs ).mean()
    #         else:
                
    #             actions_pred, log_pis = self.actor_local.evaluate(states)
    #             if self._action_prior == "normal":
    #                 policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
    #                 policy_prior_log_probs = policy_prior.log_prob(actions_pred)
    #             elif self._action_prior == "uniform":
    #                 policy_prior_log_probs = 0.0
    
    #             actor_loss = (FIXED_ALPHA * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu()- policy_prior_log_probs ).mean()
    #         # Minimize the loss
    #         self.actor_optimizer.zero_grad()
    #         actor_loss.backward()
    #         self.actor_optimizer.step()

    #         # ----------------------- update target networks ----------------------- #
    #         self.soft_update(self.critic1, self.critic1_target, TAU)
    #         self.soft_update(self.critic2, self.critic2_target, TAU)
                     

    
    # def soft_update(self, local_model, target_model, tau):
    #     """Soft update model parameters.
    #     θ_target = τ*θ_local + (1 - τ)*θ_target
    #     Params
    #     ======
    #         local_model: PyTorch model (weights will be copied from)
    #         target_model: PyTorch model (weights will be copied to)
    #         tau (float): interpolation parameter 
    #     """
    #     for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #         target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
   