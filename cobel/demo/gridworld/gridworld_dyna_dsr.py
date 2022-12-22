# basic imports
import numpy as np
import pyqtgraph as qg
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# CoBel-RL framework
from cobel.agents.dyna_dqn import DynaDSR
from cobel.networks.network_tensorflow import SequentialKerasNetwork
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor, EscapeLatencyMonitor
from cobel.misc.gridworld_tools import make_gridworld

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True


def build_model(input_shape, output_units):
    '''
    This function builds a simple network model. 
    
    Parameters
    ----------
    input_shape :                       The network model's input shape.
    output_units :                      The network model's number of output units.
    
    Returns
    ----------
    model :                             The built network model.
    '''
    model = Sequential()
    model.add(Dense(units=64, input_shape=input_shape, activation='tanh'))
    model.add(Dense(units=64, activation='tanh'))
    model.add(Dense(units=output_units, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model

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
        main_window = qg.GraphicsWindow(title='Demo: Dyna-DSR Agent')
    
    # define environmental barriers
    invalid_transitions = [(3, 4), (4, 3), (8, 9), (9, 8), (13, 14), (14, 13), (18, 19), (19, 18)]
    invalid_transitions = []
    
    # initialize world
    world = make_gridworld(5, 5, terminals=[4], rewards=np.array([[4, 10]]), goals=[4], invalid_transitions=invalid_transitions)
    world['starting_states'] = np.array([12])
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, visual_output, main_window)
    
    # amount of trials
    number_of_trials = 50
    # maximum steps per trial
    max_steps = 50
    
    # initialize monitors
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [0, 10])
    el_monitor = EscapeLatencyMonitor(number_of_trials, max_steps, main_window, visual_output)
    main_window.setGeometry(50, 50, 1600, 450)
    
    # build models
    model_SR = SequentialKerasNetwork(build_model((25,), 25))
    model_reward = SequentialKerasNetwork(build_model((25,), 1))
        
    # initialize RL agent
    rl_agent = DynaDSR(interface_OAI=modules['rl_interface'], epsilon=0.3, beta=5, gamma=0.9,
                       model_SR=model_SR, model_reward=model_reward, custom_callbacks={'on_trial_end': [reward_monitor.update, el_monitor.update]})
    #rl_agent.mask_actions = True
    #rl_agent.ignore_terminality = False
    #rl_agent.policy = 'softmax'
    #rl_agent.use_Deep_DR = True
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps, replay_batch_size=64)
    
    # clear keras session (for performance)
    K.clear_session()
    
    # and also stop visualization
    if visual_output:
        main_window.close()


if __name__ == "__main__":
    # run demo
    single_run()
