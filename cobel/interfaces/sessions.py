# basic imports
import gymnasium as gym
import random
import numpy as np
# framework imports
from cobel.interfaces.rl_interface import AbstractInterface


class InterfaceSessions(AbstractInterface):
    
    def __init__(self, modules: dict, number_of_sessions: int = 1, phase_trials: None | dict = None, phase_requirements: None | dict = None):
        '''
        This class implements a multi-session ABA extinction paradigm.
        
        Parameters
        ----------
        modules :                           Contains framework modules.\n
        number_of_sessions :                The number of sessions that will be generated.\n
        phase_trials :                      The maximum number of phase trials per stimulus.\n
        phase_requirements :                The performance that the extinction stimulus has to reach in each phase.\n
        
        Returns
        ----------
        None
        '''
        super().__init__(modules=modules, with_GUI=False)
        self.number_of_sessions = number_of_sessions
        self.phase_trials = {'acq': 1, 'ext': 1, 'rec': 1} if phase_trials is None else phase_trials
        self.phase_requirements = {'acq': 0.8, 'ext': 0.3, 'rec': 0.3} if phase_requirements is None else phase_requirements
        self.number_of_contexts = 3
        # left and right peck + omission
        self.action_space = gym.spaces.Discrete(3)
        # n stimuli + 2 contexts
        self.observation_size = self.number_of_contexts + 2 * (self.number_of_sessions + 1)
        self.observation_space = np.ones(self.observation_size)
        self.trials_max = self.number_of_sessions * np.product(list(self.phase_trials.values()))
        self.sessions = self.prepare_sessions()
        self.current_session = 0
        self.current_context = 'acq'
        self.current_phase_trial = 0
        self.performance = []
        self.performance = {i: [] for i in range(4)}
        self.phase_trials_min = 150
        self.responses = []
        self.finished = False
        
    def prepare_sessions(self) -> list:
        '''
        This function prepares the training sessions.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        sessions :                          The training sessions.\n
        '''
        sessions = []
        for session in range(self.number_of_sessions):
            extinction_stimulus = np.random.randint(2)
            phases = {'extinction_stimulus': extinction_stimulus + 2 * (session + 1), 'extinction_context': np.random.randint(self.number_of_contexts - 1) + 1}
            non_extinction_context = [i for i in range(self.number_of_contexts)]
            non_extinction_context.remove(0)
            non_extinction_context.remove(phases['extinction_context'])
            non_extinction_context = non_extinction_context[np.random.randint(len(non_extinction_context))]
            for phase in self.phase_trials:
                phases[phase] = []
                for trial in range(self.phase_trials[phase]):
                    phases[phase].append([0, 1., 0])
                    if phase == 'ext':
                        phases[phase][-1][2] = phases['extinction_context'] if extinction_stimulus != 0 else non_extinction_context
                    
                    phases[phase].append([1, 1., 0])
                    if phase == 'ext':
                        phases[phase][-1][2] = phases['extinction_context'] if extinction_stimulus != 1 else non_extinction_context
                    
                    phases[phase].append([session * 2 + 2, float(extinction_stimulus != 0 or phase == 'acq') * 2 - 1, 0])
                    if phase == 'ext':
                        phases[phase][-1][2] = phases['extinction_context'] if phases['extinction_stimulus'] == 2 * session + 2 else non_extinction_context
                    
                    phases[phase].append([session * 2 + 3, float(extinction_stimulus != 1 or phase == 'acq') * 2 - 1, 0])
                    if phase == 'ext':
                        phases[phase][-1][2] = phases['extinction_context'] if phases['extinction_stimulus'] == 2 * session + 3 else non_extinction_context
                        
                random.shuffle(phases[phase])
            sessions.append(phases)
            
        return sessions
    
    def check_performance(self) -> bool:
        '''
        This function checks if the performance criterium for the extinction stimulus was reached.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        end_phase :                         A flag indicating whether the criterium was reached.\n
        '''
        end_phase = False
        if np.product(np.array([len(self.performance[i]) > self.phase_trials_min for i in self.performance])) > 0:
            performance = {i: np.mean(np.clip(self.performance[i][-20:], a_min=0, a_max=1)) for i in self.performance}
            if self.current_context == 'acq':
                performance = np.amin(np.array(list(performance.values())))
                end_phase = performance >= self.phase_requirements['acq']
            elif self.current_context == 'ext':
                end_phase = performance[self.sessions[self.current_session]['extinction_stimulus']] <= self.phase_requirements['ext']
            else:
                end_phase = performance[self.sessions[self.current_session]['extinction_stimulus']] <= self.phase_requirements['rec']
                
        return end_phase
    
    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        '''
        Gymnasium's step function.
        Executes the agent's action and propels the simulation.
        
        Parameters
        ----------
        action :                            The action selected by the agent.
        
        Returns
        ----------
        observation :                       The observation of the new current state.\n
        reward :                            The reward received.\n
        end_trial :                         Flag indicating whether the trial ended.\n
        logs :                              The (empty) logs dictionary.\n
        '''
        observation = np.zeros(self.observation_size)
        stimulus, reward, context = self.sessions[self.current_session][self.current_context][self.current_phase_trial]
        #if stimulus == self.sessions[self.current_session]['extinction_stimulus']:
        #    self.performance.append(stimulus % 2 == action)
        self.performance[stimulus].append(stimulus % 2 == action)
        reward *= stimulus % 2 == action
        if reward == 0:
            reward = -1.
        if action == 2:
            reward = 0.
        response = 0
        if action != 2:
            response = (stimulus % 2 == action) * 2 - 1
        self.responses.append([self.current_session, {'acq': 0, 'ext': 1, 'rec': 2}[self.current_context], stimulus,
                               response, action, stimulus == self.sessions[self.current_session]['extinction_stimulus']])
        self.current_phase_trial += 1
        # check if session/phase changed
        if self.current_phase_trial >= self.phase_trials[self.current_context] or self.check_performance():
            print(self.current_session, self.current_context)
            self.current_context = {'acq': 'ext', 'ext': 'rec', 'rec': 'acq'}[self.current_context]
            self.current_session += int(self.current_context == 'acq')
            self.current_phase_trial = 0
            self.performance = []
            self.performance = {0: [], 1: [], self.current_session * 2 + 2: [], self.current_session * 2 + 3: []}
            if self.current_session >= self.number_of_sessions:
                self.current_session -= 1
                self.current_phase = 'rec'
                self.finished = True
        
        return observation, reward, True, {}
    
    def reset(self) -> np.ndarray:
        '''
        Gymnasium's reset function.
        Resets the environment and the agent's state.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        observation :                       The observation of the new current state.\n
        '''
        # prepare observation
        observation = np.zeros(self.observation_size)
        observation[self.sessions[self.current_session][self.current_context][self.current_phase_trial][0]] = 1. 
        observation[-self.number_of_contexts + self.sessions[self.current_session][self.current_context][self.current_phase_trial][2]] = 1.
        
        return observation
    
    
class InterfaceSessionsCompContext(AbstractInterface):
    
    def __init__(self, modules: dict, number_of_sessions: int = 1, phase_trials: None | dict = None, phase_requirements: None | dict = None,
                 contexts: None | dict = None, schedule: None | dict = None, separate_stimuli: bool = False):
        '''
        This class implements a multi-session compositional context ABC extinction paradigm.
        
        Parameters
        ----------
        modules :                           Contains framework modules.\n
        number_of_sessions :                The number of sessions that will be generated. If a schedule was provided this value will be overwritten with the sessions defined in the schedule.\n
        phase_trials :                      The maximum number of phase trials per stimulus.\n
        phase_requirements :                The performance that the extinction stimulus has to reach in each phase.\n
        contexts :                          The number of contexts per context type (i.e., spatial, local and global). Must include the neutral context.\n
        schedule :                          A dictionary containing the context schedule for each session.\n
        separate_stimuli :                  Flag indicating whether there are separate inputs for stimuli presented on left/right.\n
        
        Returns
        ----------
        None
        '''
        super().__init__(modules=modules, with_GUI=False)
        self.number_of_sessions = number_of_sessions if schedule is None else len(schedule)
        self.stimuli = 2 * (number_of_sessions + 1)
        self.phase_trials = {'acq': 1, 'ext': 1, 'rec': 1} if phase_trials is None else phase_trials
        self.phase_requirements = {'acq': 0.85, 'ext': 0.15, 'rec': 0.15} if phase_requirements is None else phase_requirements
        self.contexts = {'spatial': 8, 'local': 7, 'global': 7} if contexts is None else contexts
        self.separate_stimuli = separate_stimuli
        # left and right peck + omission
        self.action_space = gym.spaces.Discrete(3)
        # n stimuli + 2 contexts
        self.observation_size = (1 + int(self.separate_stimuli)) * 2 * (self.number_of_sessions + 1) + np.product(list(self.contexts.values()))
        self.observation_space = np.ones(self.observation_size)
        # setup environment
        self.trials_max = self.number_of_sessions * np.product(list(self.phase_trials.values()))
        self.schedule = self.prepare_schedule() if schedule is None else schedule
        self.sessions = self.prepare_sessions()
        # environmental state
        self.current_session = 0
        self.current_context = 'acq'
        self.current_phase_trial = 0
        self.performance = {'familiar': [], 'novel': []}
        self.phase_trials_min = 150
        self.responses = []
        self.finished = False
        
    def prepare_schedule(self) -> dict:
        '''
        This function prepares the context schedule.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        schedule :                          The context schedule.\n
        '''
        context_schedule = {}
        for session in range(self.number_of_sessions):
            # random extinction context and neutral acquisition context
            context_ext = np.random.randint([0, 1, 1], [self.contexts[c] for c in ['spatial', 'local', 'global']])
            context_acq = np.zeros(3)
            # ensure that the spatial context is different during acquisition
            valid = list(np.arange(self.contexts['spatial']))
            valid.remove(context_ext[0])
            context_acq[0] = valid[np.random.randint(len(valid))]
            # randomly pick a context type for recall
            recall_type = np.random.rand(3)
            context_rec = np.copy(context_ext)
            context_rec[recall_type] = context_acq[recall_type]
            # add to schedule
            context_schedule[session] = {'acq': context_acq, 'ext': context_ext, 'rec': context_rec, 'type': recall_type}
            
        return context_schedule
        
    def prepare_sessions(self) -> list:
        '''
        This function prepares the training sessions.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        sessions :                          The training sessions.\n
        '''
        # prepare session trials
        sessions = []
        for session in range(self.number_of_sessions):
            # randomly select extinction stimulus
            extinction_stimulus = np.random.randint(2)
            non_extinction_stimulus = (1 - extinction_stimulus) + 2 * (session + 1)
            extinction_stimulus += + 2 * (session + 1)
            phases = {'extinction_stimulus': extinction_stimulus, 'non_extinction_stimulus': non_extinction_stimulus, 'renewal_type': self.schedule[session]['type']}
            for phase in self.phase_trials:
                phases[phase] = []
                for trial in range(self.phase_trials[phase]):
                    # randomly select stimuli positions
                    correct_familiar, correct_novel = np.random.randint(2, size=2)
                    phases[phase].append([0, 1, 1, correct_familiar, self.schedule[session][phase]])
                    phases[phase].append([extinction_stimulus, non_extinction_stimulus, 1 - 2 * int(phase != 'acq'), correct_novel, self.schedule[session][phase]])
                random.shuffle(phases[phase])
            sessions.append(phases)
            
        return sessions
    
    def check_performance(self) -> bool:
        '''
        This function checks if the performance criterium for the extinction stimulus was reached.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        end_phase :                         A flag indicating whether the criterium was reached.\n
        '''
        end_phase = False
        if np.product(np.array([len(self.performance[i]) > self.phase_trials_min for i in self.performance])) > 0:
            performance = {i: np.mean(np.clip(self.performance[i][-20:], a_min=0, a_max=1)) for i in self.performance}
            if self.current_context == 'acq':
                end_phase = performance['familiar'] >= 0.85 and performance['novel'] >= 0.85
            elif self.current_context == 'ext':
                end_phase = performance['familiar'] >= 0.85 and performance['novel'] <= 0.15
            elif self.current_context == 'rec':
                end_phase = performance['novel'] <= 0.15
                
        return end_phase
    
    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        '''
        Gymnasium's step function.
        Executes the agent's action and propels the simulation.
        
        Parameters
        ----------
        action :                            The action selected by the agent.
        
        Returns
        ----------
        observation :                       The observation of the new current state.\n
        reward :                            The reward received.\n
        end_trial :                         Flag indicating whether the trial ended.\n
        logs :                              The (empty) logs dictionary.\n
        '''
        observation = np.zeros(self.observation_size)
        stimulus_p, stimulus_m, reward, position, contexts = self.sessions[self.current_session][self.current_context][self.current_phase_trial]
        # store performance
        self.performance['familiar' if stimulus_p < 2 else 'novel'].append(position == action)
        # compute reward
        reward *= position == action
        if reward == 0:
            reward = -1.
        if action == 2:
            reward = 0.
        # record response
        response = 0
        if action != 2:
            response = (position == action) * 2 - 1
        self.responses.append([self.current_session, {'acq': 0, 'ext': 1, 'rec': 2}[self.current_context], stimulus_p, stimulus_m,
                               response, position, action, 0 if stimulus_p < 2 else 1, self.sessions[self.current_session]['renewal_type']])
        self.current_phase_trial += 1
        # check if session/phase changed
        if self.current_phase_trial >= self.phase_trials[self.current_context] or self.check_performance():
            self.current_context = {'acq': 'ext', 'ext': 'rec', 'rec': 'acq'}[self.current_context]
            self.current_session += int(self.current_context == 'acq')
            self.current_phase_trial = 0
            self.performance = []
            self.performance = {'familiar': [], 'novel': []}
            if self.current_session >= self.number_of_sessions:
                self.current_session -= 1
                self.current_phase = 'rec'
                self.finished = True
        
        return observation, reward, True, {}
    
    def reset(self) -> np.ndarray:
        '''
        Gymnasium's reset function.
        Resets the environment and the agent's state.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        observation :                       The observation of the new current state.\n
        '''
        # prepare observation
        observation = np.zeros(self.observation_size)
        stimulus_p, stimulus_m, reward, position, contexts = self.sessions[self.current_session][self.current_context][self.current_phase_trial]
        if self.separate_stimuli:
            observation[stimulus_p + self.stimuli * position] = 1.
            observation[stimulus_m + self.stimuli * (1 - position)] = 1.
        else:
            observation[stimulus_p] = 1.
            observation[stimulus_m] = 1.
        # context cues
        offsets = np.cumsum([self.contexts[c] for c in ['global', 'local', 'spatial']])
        observation[-(offsets[2] - contexts[0])] = 1.
        observation[-(offsets[1] - contexts[1])] = 1.
        observation[-(offsets[0] - contexts[2])] = 1.
        
        return observation
    