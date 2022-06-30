

class callbacks():
    '''
    Callback class. Used for visualization and scenario control.
    
    | **Args**
    | rl_parent:                    Reference to the RL agent.
    | custom_callbacks:             The custom callbacks defined by the user.
    '''
    
    def __init__(self, rl_parent, custom_callbacks={}):
        # store the hosting class
        self.rl_parent = rl_parent
        # store the trial end callback function
        self.custom_callbacks = custom_callbacks
    
    def on_trial_begin(self, logs):
        '''
        The following function is called whenever a trial begins, and executes callbacks defined by the user.
        
        | **Args**
        | logs:                         The trial log.
        '''
        logs['rl_parent'] = self.rl_parent
        if 'on_trial_begin' in self.custom_callbacks:
            for custom_callback in self.custom_callbacks['on_trial_begin']:
                custom_callback(logs)
                
    def on_trial_end(self, logs):
        '''
        The following function is called whenever a trial ends, and executes callbacks defined by the user.
        
        | **Args**
        | logs:                         The trial log.
        '''
        logs['rl_parent'] = self.rl_parent
        if 'on_trial_end' in self.custom_callbacks:
            for custom_callback in self.custom_callbacks['on_trial_end']:
                custom_callback(logs)
                
    def on_step_begin(self, logs):
        '''
        The following function is called whenever a step begins, and executes callbacks defined by the user.
        
        | **Args**
        | logs:                         The trial log.
        '''
        logs['rl_parent'] = self.rl_parent
        if 'on_step_begin' in self.custom_callbacks:
            for custom_callback in self.custom_callbacks['on_step_begin']:
                custom_callback(logs)
                
    def on_step_end(self, logs):
        '''
        The following function is called whenever a step, and executes callbacks defined by the user.
        
        | **Args**
        | logs:                         The trial log.
        '''
        logs['rl_parent'] = self.rl_parent
        if 'on_step_end' in self.custom_callbacks:
            for custom_callback in self.custom_callbacks['on_step_end']:
                custom_callback(logs)
                

class AbstractRLAgent():
    '''
    Abstract class of an RL agent.
    
    | **Args**
    | interface_OAI:                The interface to the Open AI Gym environment.
    | custom_callbacks:             The custom callbacks defined by the user.
    '''
    
    def __init__(self, interface_OAI, custom_callbacks={}):
        # store the Open AI Gym interface
        self.interface_OAI = interface_OAI        
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.number_of_actions = self.interface_OAI.action_space.n
        # initialze callbacks class with customized callbacks
        self.engaged_callbacks = callbacks(self, custom_callbacks)

                
    def train(self, number_of_trials=100, max_number_of_steps=50):
        '''
        This function is called to train the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the RL agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
        '''
        raise NotImplementedError('.train() function not implemented!')
        
    def test(self, number_of_trials=100, max_number_of_steps=50):
        '''
        This function is called to test the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the RL agent is tested.
        | max_number_of_steps:          The maximum number of steps per trial.
        '''
        raise NotImplementedError('.test() function not implemented!')
        
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states/observations.
        
        | **Args**
        | batch:                        The batch of states/observations.
        '''
        raise NotImplementedError('.predict_on_batch() function not implemented!')
