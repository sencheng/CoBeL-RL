# basic imports
import numpy as np
import pyqtgraph as qg
# CoBel-RL framework
from cobel.agents.dyna_q import DynaQAgent
from cobel.policy.greedy import EpsilonGreedy
from cobel.interfaces.gridworld import InterfaceGridworld
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
        main_window = qg.GraphicsLayoutWidget(title='Demo: Dyna-Q Agent')
        main_window.show()
    
    # define environmental barriers
    invalid_transitions = [(3, 4), (4, 3), (8, 9), (9, 8), (13, 14), (14, 13), (18, 19), (19, 18)]
    
    # initialize world
    world = make_gridworld(5, 5, terminals=[4], rewards=np.array([[4, 10]]), goals=[4], invalid_transitions=invalid_transitions)
    world['starting_states'] = np.array([20])
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, visual_output, main_window)
    
    # amount of trials
    number_of_trials = 200
    # maximum steps per trial
    max_steps = 25
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [0, 10])
    
    # define policy
    policy = EpsilonGreedy(0.1)
    
    # initialize RL agent
    rl_agent = DynaQAgent(interface_OAI=modules['rl_interface'], policy=policy, policy_test=policy, learning_rate=0.9,
                          gamma=0.9, custom_callbacks={'on_trial_end': [reward_monitor.update]})
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps, replay_batch_size=5)
    
    # and also stop visualization
    if visual_output:
        main_window.close()


if __name__ == '__main__':
    single_run()
