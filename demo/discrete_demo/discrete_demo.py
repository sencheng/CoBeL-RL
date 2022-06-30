# basic imports
import numpy as np
import PyQt5 as qt
import pyqtgraph as qg
# tensorflow
from tensorflow.keras import backend as K
# CoBel-RL framework
from cobel.agents.dqn_agents import DQNAgentBaseline
from cobel.interfaces.oai_gym_discrete import OAIGymInterfaceDiscrete
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor
from cobel.misc.gridworld_tools import makeGridworld

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
        main_window = qg.GraphicsWindow(title='Demo: DQN & Discrete Interface')
    
    # initialize world (we use a grid world as an example)
    world = makeGridworld(5, 5, terminals=[4], rewards=np.array([[4, 10]]), goals=[4])
    # let the agent always start at the lower left corner
    world['startingStates'] = np.array([20])
    # we use a one-hot encoding for the states
    observations = np.eye(25)
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = OAIGymInterfaceDiscrete(modules, world['sas'], observations, world['rewards'], world['terminals'],
                                                      world['startingStates'], world['coordinates'], world['goals'], visual_output, main_window, None)

    # amount of trials
    number_of_trials = 100
    # maximum steos per trial
    max_steps = 30
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [0, 10])
    
    # initialize RL agent
    rl_agent = DQNAgentBaseline(modules['rl_interface'], 10000, 0.3, None, custom_callbacks={'on_trial_end': [reward_monitor.update]})
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps)
    
    # clear keras session (for performance)
    K.clear_session()
    
    # and also stop visualization
    if visual_output:
        main_window.close()


if __name__ == '__main__':
    single_run()
