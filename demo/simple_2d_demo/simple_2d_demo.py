# basic imports
import numpy as np
import PyQt5 as qt
import pyqtgraph as qg
# tensorflow
from tensorflow.keras import backend as K
# CoBel-RL framework
from cobel.agents.dqn_agents import DQNAgentBaseline
from cobel.interfaces.oai_gym_simple_2d import OAIGymInterfaceSimple2D
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True


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
        main_window = qg.GraphicsWindow(title='Demo: DQN & Simple 2D Interface')
    
    # define reward locations
    rewards = np.array([[0.75, 0.75, 10]])
    # define discount factor for moving 0.1 units of distance (assuming the agent moves straight)
    gamma_base = 0.9
    # define step size
    step_size = 0.015
    # determine step gamma
    gamma = np.power(gamma_base, 1.0/(0.1/step_size))
    
    # a dictionary that contains all employed modules
    modules = dict()
    modules['rl_interface'] = OAIGymInterfaceSimple2D(modules, 'step', rewards, visual_output, main_window, None)
    modules['rl_interface'].step_size = step_size
    
    # amount of trials
    number_of_trials = 300
    # maximum steos per trial
    max_steps = 250
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [0, 10])
    
    # initialize RL agent
    rl_agent = DQNAgentBaseline(modules['rl_interface'], 100000, 0.3, None, custom_callbacks={'on_trial_end': [reward_monitor.update]})
    rl_agent.agent.gamma = gamma
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps)
    
    # clear keras session (for performance)
    K.clear_session()
    
    # stop visualization
    if visual_output:
        main_window.close()


if __name__ == '__main__':
    single_run()
