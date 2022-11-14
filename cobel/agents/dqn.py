# basic imports
import numpy as np
# framework imports
from cobel.agents.rl_agent import AbstractRLAgent
from cobel.memory_modules.dqn_memory import SimpleMemory


class SimpleDQN(AbstractRLAgent):

    def __init__(self, interface_OAI, epsilon=0.3, beta=5, gamma=0.99, model=None, custom_callbacks={}):
        '''
        This class implements a simple deep-Q network. 
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        epsilon :                           The epsilon value for the epsilon greedy policy.
        beta :                              The inverse temperature parameter for the softmax policy.
        gamma :                             The discount factor used for computing the target values.
        model :                             The network model to be used by the agent.
        custom_callbacks :                  The custom callbacks defined by the user.
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, custom_callbacks=custom_callbacks)
        # build target and online models
        self.prepare_models(model)
        # memory module
        self.M = SimpleMemory()
        # the rate at which the target model is updated (for values < 1 the target model is blended with the online model)
        self.target_model_update = 10**-2
        # count the steps since the last update of the target model
        self.steps_since_last_update = 0
        # the current trial across training and test sessions
        self.current_trial = 0
        # policy parameters
        self.policy = 'greedy'
        self.epsilon = epsilon
        self.beta = beta
        # the discount factor
        self.gamma = gamma
        # double Q-learning
        self.DDQN_mode = False
        
    def prepare_models(self, model=None):
        '''
        This functions prepares target and online models. 
        
        Parameters
        ----------
        model :                             The network model to be used by the agent.
        
        Returns
        ----------
        None
        '''
        # build target model
        self.model_target = model
        # build online model by cloning the target model
        self.model_online = self.model_target.clone_model()
        
    def train(self, number_of_trials=100, max_number_of_steps=50, replay_batch_size=100, no_replay=False):
        '''
        This function is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials that the agent is trained.
        max_number_of_steps :               The maximum number of steps per trial.
        replay_batch_size :                 The number of random experiences that will be replayed.
        no_replay :                         If true, experiences are not replayed.
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                self.steps_since_last_update += 1
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # store experience
                self.M.store(experience)
                # update current state
                state = next_state
                # perform experience replay
                if not no_replay:
                    self.replay(replay_batch_size)
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            
    def test(self, number_of_trials=100, max_number_of_steps=50):
        '''
        This function is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials that the agent is tested.
        max_number_of_steps :               The maximum number of steps per trial.
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                # determine next action
                action = np.argmax(self.retrieve_Q(state))
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
        
    def replay(self, replay_batch_size=200):
        '''
        This function replay experiences to update the Q-function.
        
        Parameters
        ----------
        replay_batch_size :                 The number of random experiences that will be replayed.
        
        Returns
        ----------
        None
        '''
        # sample random batch of experiences
        states, actions, rewards, next_states, terminals = self.M.retrieve_batch(replay_batch_size)
        # compute update targets
        targets = self.model_online.predict_on_batch(states)
        bootstraps, bootstraps_online = self.model_target.predict_on_batch(next_states), None
        if self.DDQN_mode:
            bootstraps_online = self.model_online.predict_on_batch(next_states)
        for sample in range(replay_batch_size):
            bootstrap_estimate = np.amax(bootstraps[sample])
            if self.DDQN_mode:
                bootstrap_estimate = bootstraps[sample][np.argmax(bootstraps_online[sample])]
            targets[sample, actions[sample]] = rewards[sample] + self.gamma * bootstrap_estimate * terminals[sample]
        # update online model
        self.model_online.train_on_batch(states, targets)
        # update target model
        if self.target_model_update < 1.:
            weights_target = np.array(self.model_target.get_weights(), dtype=object)
            weights_online = np.array(self.model_online.get_weights(), dtype=object)
            weights_target += self.target_model_update * (weights_online - weights_target)
            self.model_target.set_weights(weights_target)
        elif self.steps_since_last_update >= self.target_model_update:
            self.model_target.set_weights(self.model_online.get_weights())
            self.steps_since_last_update = 0
    
    def retrieve_Q(self, state: np.ndarray) -> np.ndarray:
        '''
        This function retrieves Q-values for a given state.
        
        Parameters
        ----------
        state :                             The state for which Q-values should be retrieved.
        
        Returns
        ----------
        q_values :                          The state's Q-values.
        '''
        return self.model_online.predict_on_batch(np.array([state]))[0]
    
    def predict_on_batch(self, batch: np.ndarray) -> np.ndarray:
        '''
        This function retrieves Q-values for a batch of states.
        
        Parameters
        ----------
        batch :                             The batch of states for which Q-values should be retrieved.
        
        Returns
        ----------
        predictions :                       The batch of Q-value predictions.
        '''
        return self.model_online.predict_on_batch(batch)
    
    def select_action(self, state: np.ndarray, epsilon=0.3, beta=5, test_mode=False) -> int:
        '''
        This function selects an action according to the Q-values of the current state.
        
        Parameters
        ----------
        state :                             The state for which Q-values should be retrieved.
        epsilon :                           The epsilon parameter used under greedy action selection.
        beta :                              The temperature parameter used when applying the softmax function to the Q-values.
        test_mode :                         If true, then the action with the maximum Q-value is selected irrespective of the policy.
        
        Returns
        ----------
        action :                            The action that was selected.
        '''
        # revert to 'greedy' in case that the method name is not valid
        if not self.policy in ['greedy', 'softmax']:
            self.policy = 'greedy'
        # retrieve Q-values
        qVals = self.retrieve_Q(state)
        actions = np.arange(qVals.shape[0])
        # select action with highest value
        if self.policy == 'greedy' or test_mode:
            # act greedily and break ties
            action = np.argmax(qVals)
            ties = np.arange(qVals.shape[0])[(qVals == qVals[action])]
            action = ties[np.random.randint(ties.shape[0])]
            # in case that Q-values are equal select a random action
            if np.random.rand() < epsilon and not test_mode:
                action = np.random.randint(qVals.shape[0])
            return actions[action]
        # select action probabilistically
        elif self.policy == 'softmax':
            qVals -= np.amax(qVals)
            probs = np.exp(beta * qVals)/np.sum(np.exp(beta * qVals))
            action = np.random.choice(qVals.shape[0], p=probs)
            return actions[action]
