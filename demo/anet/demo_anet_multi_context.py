# basic imports
import numpy as np
import matplotlib.pyplot as plt
# framework imports
from cobel.agents.anet import AssociativeNetworkAgent
from cobel.interfaces.sessions import InterfaceSessionsCompContext

if __name__ == '__main__':
    # training params
    original_model = False
    number_of_sessions = 12
    saturation_level = 10 ** 6
    phase_trials = {'acq': 600, 'ext': 600, 'rec': 240}
    phase_requirements = {'acq': 0.85, 'ext': 0.15, 'rec': 0.15}
    contexts = {'spatial': 8, 'local': 7, 'global': 7}
    context_schedule = {0: {'acq': [0, 0, 0], 'ext': [6, 2, 4], 'rec': [0, 2, 4], 'type': 0},
                        1: {'acq': [2, 0, 0], 'ext': [5, 1, 5], 'rec': [5, 0, 5], 'type': 1},
                        2: {'acq': [7, 0, 0], 'ext': [1, 3, 6], 'rec': [1, 3, 0], 'type': 2},
                        3: {'acq': [4, 0, 0], 'ext': [3, 4, 1], 'rec': [4, 4, 1], 'type': 0},
                        4: {'acq': [1, 0, 0], 'ext': [5, 6, 2], 'rec': [5, 0, 2], 'type': 1},
                        5: {'acq': [6, 0, 0], 'ext': [2, 5, 3], 'rec': [2, 5, 0], 'type': 2},
                        6: {'acq': [4, 0, 0], 'ext': [0, 1, 6], 'rec': [4, 1, 6], 'type': 0},
                        7: {'acq': [7, 0, 0], 'ext': [3, 3, 1], 'rec': [3, 0, 1], 'type': 1},
                        8: {'acq': [1, 0, 0], 'ext': [2, 2, 5], 'rec': [2, 2, 0], 'type': 2},
                        9: {'acq': [6, 0, 0], 'ext': [5, 4, 3], 'rec': [6, 4, 3], 'type': 0},
                        10: {'acq': [3, 0, 0], 'ext': [0, 5, 2], 'rec': [0, 0, 2], 'type': 1},
                        11: {'acq': [4, 0, 0], 'ext': [7, 6, 4], 'rec': [7, 6, 0], 'type': 2}}
    
    # initialize interface    
    modules = {}
    modules['rl_interface'] = InterfaceSessionsCompContext(modules, number_of_sessions, phase_trials,
                                                           phase_requirements, contexts, context_schedule, not original_model)
    modules['rl_interface'].phase_trials_min = 40
    
    # prepare callback
    def stop(logs: dict):
        logs['rl_parent'].stop = logs['rl_parent'].interface_OAI.finished
    
    # initialize agent
    agent = AssociativeNetworkAgent(modules['rl_interface'], custom_callbacks={'on_trial_end': [stop]})
    # set learning parameters
    agent.linear_update = False
    agent.learning_rates['excitatory'].fill(0.02)
    agent.learning_rates['inhibitory'].fill(0.01)
    agent.saturation['excitatory'].fill(saturation_level)
    agent.saturation['inhibitory'].fill(saturation_level)
    
    # train the agent
    agent.train(modules['rl_interface'].trials_max)
    
    responses = np.array(modules['rl_interface'].responses)
    
    for session in range(number_of_sessions):
        plt.figure(1, figsize=(10, 3))
        plt.suptitle('Session %d' % (session + 1), position=(0.5, 1.1), fontsize=15)
        session_trials = responses[:, 0] == session
        for phase in range(3):
            plt.subplot(1, 3, phase + 1)
            plt.title({0: 'Acquisition', 1: 'Extinction', 2: 'Recall'}[phase])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            phase_trials = responses[:, 1] == phase
            for stimulus in [0, 1]:
                stimulus_trials = responses[:, -2] == stimulus
                trials = np.arange(responses.shape[0])[session_trials * phase_trials * stimulus_trials]
                trials -= np.amin(np.arange(responses.shape[0])[session_trials])
                trials += 1
                color = {0: 'green', 1: 'red'}[stimulus]
                plt.plot(trials, np.cumsum(responses[session_trials * phase_trials * stimulus_trials][:, 4]), color=color)
        plt.savefig('multi_context_demo_session_%d.png' % (session + 1), dpi=200, bbox_inches='tight')
        plt.close('all')
    