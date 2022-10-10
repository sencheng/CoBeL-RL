# basic imports
import numpy as np
import pyqtgraph as qg
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input, Activation
# CoBel-RL framework
from cobel.agents.keras_rl.ddpg import DDPGAgentBaseline
from cobel.interfaces.move_2d import InterfaceMove2D
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True


def build_models():
        '''
        This function builds the actor and critic models of the DDPG agent.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        model_actor :                       The network actor model to be used by the DDPG agent.
        model_critic :                      The network critic model to be used by the DDPG agent.
        '''
        # actor model
        model_actor = Sequential()
        model_actor.add(Flatten(input_shape=(1,) + (4,)))
        model_actor.add(Dense(units=64, activation='tanh'))
        model_actor.add(Dense(units=64, activation='tanh'))
        model_actor.add(Dense(units=2, activation='tanh', name='output'))
        # critic model
        observation_input = Input(shape=(1,) + (4,), name='observation_input')
        observation_flattened = Flatten()(observation_input)
        action_input = Input(shape=(2), name='action_input')
        x = Concatenate()([action_input, observation_flattened])
        x = Dense(64)(x)
        x = Activation('tanh')(x)
        x = Dense(64)(x)
        x = Activation('tanh')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        model_critic = Model(inputs=[action_input, observation_input], outputs=x)
        
        return model_actor, model_critic, action_input

def single_run():
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Demo: DDPG & Move 2D Interface')
    
    # build models
    model_actor, model_critic, action_input = build_models()
    
    # a dictionary that contains all employed modules
    modules = dict()
    modules['rl_interface'] = InterfaceMove2D(modules, visual_output, main_window)
    #modules['rl_interface'].static_goal = np.array([.5, .5])
    
    # amount of train and test trials
    trials_train, trials_test = 250, 250
    # amount of trials
    number_of_trials = trials_train + trials_test
    # maximum steos per trial
    max_steps = 100
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [-1.2 * max_steps, 1])
    
    # initialize RL agent
    rl_agent = DDPGAgentBaseline(modules['rl_interface'], 1000000, model_actor, model_critic, action_input, {'on_trial_end': [reward_monitor.update]})
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(trials_train, max_steps)
    
    # test the agent learn
    rl_agent.test(trials_test, max_steps)
    
    # clear keras session (for performance)
    K.clear_session()
    
    # stop visualization
    if visual_output:
        main_window.close()


if __name__ == '__main__':
    single_run()
