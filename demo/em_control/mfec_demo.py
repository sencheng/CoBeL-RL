# basic imports
import numpy as np
import pyqtgraph as qg
# CoBel-RL framework
from cobel.agents.em_control import MFECAgent
from cobel.policy.greedy import EpsilonGreedy
from cobel.interfaces.discrete import InterfaceDiscrete
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor
from cobel.misc.gridworld_tools import make_gridworld


def single_run(visual_output: bool = True):
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
        main_window = qg.GraphicsLayoutWidget(title='Demo: MFEC & Discrete Interface')
        main_window.show()
    
    # initialize world (we use a grid world as an example)
    world = make_gridworld(5, 5, terminals=[4], rewards=np.array([[4, 10]]), goals=[4])
    # let the agent always start at the lower left corner
    world['starting_states'] = np.array([20])
    # we use a one-hot encoding for the states
    observations = np.eye(25)
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceDiscrete(modules, world['sas'], observations, world['rewards'], world['terminals'],
                                                world['starting_states'], world['coordinates'], world['goals'], visual_output, main_window)

    # amount of trials
    number_of_trials = 500
    # maximum steos per trial
    max_steps = 30
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [0, 10])
    
    # define policy
    policy = EpsilonGreedy(0.001)
    
    # initialize RL agent
    rl_agent = MFECAgent(modules['rl_interface'], policy, custom_callbacks={'on_trial_end': [reward_monitor.update]})
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps)
    
    # and also stop visualization
    if visual_output:
        main_window.close()


if __name__ == '__main__':
    single_run()
