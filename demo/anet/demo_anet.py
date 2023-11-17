# basic imports
import numpy as np
import matplotlib.pyplot as plt
# framework imports
from cobel.agents.anet import AssociativeNetworkAgent
from cobel.interfaces.sessions import InterfaceSessions


if __name__ == '__main__':
    # training params
    number_of_sessions = 5
    saturation_level = 20
    phase_trials = {'acq': 600, 'ext': 600, 'rec': 68 * 4}
    phase_requirements = {'acq': 0.85, 'ext': 0.33, 'rec': 0.33}
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceSessions(modules, number_of_sessions, phase_trials, phase_requirements)
    modules['rl_interface'].phase_trials_min = 38
    
    # prepare callback
    def stop(logs: dict):
        logs['rl_parent'].stop = logs['rl_parent'].interface_OAI.finished
    
    # initialize agent
    agent = AssociativeNetworkAgent(modules['rl_interface'], custom_callbacks={'on_trial_end': [stop]})
    # set learning parameters
    agent.linear_update = False
    agent.learning_rates['excitatory'].fill(0.02)
    agent.learning_rates['inhibitory'].fill(0.02)
    agent.saturation['excitatory'].fill(saturation_level)
    agent.saturation['inhibitory'].fill(saturation_level)
    # set saturation to max for familiar stimuli
    agent.weights['excitatory'][0, 0] = saturation_level
    agent.weights['excitatory'][1, 1] = saturation_level
    
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
            for stimulus in [0, 1, (session + 1) * 2, (session + 1) * 2 + 1]:
                stimulus_trials = responses[:, 2] == stimulus
                trials = np.arange(responses.shape[0])[session_trials * phase_trials * stimulus_trials]
                trials -= np.amin(np.arange(responses.shape[0])[session_trials])
                trials += 1
                color = 'green'
                if stimulus == modules['rl_interface'].sessions[session]['extinction_stimulus']:
                    color = 'red'
                elif stimulus < 2:
                    color = ['cyan', 'orange'][stimulus]
                plt.plot(trials, np.cumsum(responses[session_trials * phase_trials * stimulus_trials][:, 3]), color=color)
        plt.savefig('demo_session_%d.png' % (session + 1), dpi=200, bbox_inches='tight')
        plt.close('all')
    