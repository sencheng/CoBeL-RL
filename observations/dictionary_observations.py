import gym

class DictionaryObservations():
    """
    Creates a dictionary of observations from individual observation
    modules.
    """
    def __init__(self, obs_modules=None):
        self.obs_modules = obs_modules
        self.keys = list(self.obs_modules.keys())
        self.observation = dict.fromkeys(self.keys)

    def update(self):
        for key in self.keys:
            self.obs_modules[key].update()
            self.observation[key] = self.obs_modules[key].observation

    
    def getObservationSpace(self):
        observation_space = dict.fromkeys(self.keys)
        for key in self.keys:
            observation_space[key] = self.obs_modules[key].getObservationSpace()

        return gym.spaces.Dict(observation_space)
