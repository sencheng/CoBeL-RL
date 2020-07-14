import os
import time
import numpy as np
import gym
import subprocess
import signal
from gym import spaces
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from rl.core import Processor


### This is the Open AI gym interface class. The interface wraps the control path and ensures communication
### between the agent and the environment. The class descends from gym.Env, and is designed to be minimalistic (currently!).
class OAIGymInterface(gym.Env):

    # The constructor.
    # modules:          the dict of all available modules in the system
    # withGUI:          if True, the module provides GUI control
    # rewardCallback:   this callback function is invoked in the step routine in order to get the appropriate reward w.r.t. the experimental design

    def __init__(self, modules, withGUI=True, rewardCallback=None):
        # store the modules
        self.modules = modules

        # store visual output variable
        self.withGUI = withGUI

        # memorize the reward callback function
        self.rewardCallback = rewardCallback

        self.world = self.modules['world']
        self.observations = self.modules['observation']

        # second: action space
        self.action_space = modules['spatial_representation'].get_action_space()
        
        
        # third: observation space
        self.observation_space = modules['observation'].getObservationSpace()

        # all OAI spaces have been initialized!

        # this observation variable is filled by the OBS modules
        self.observation = None

        # required for the analysis of the agent's behavior
        self.forbiddenZoneHit = False
        self.finalNode = -1

        # a variable that allows the OAI class to access the robotic agent class
        self.rlAgent = None

    # This function (slot) updates the observation provided by the environment
    #
    # observation:  the observation used to perform the update
    def updateObservation(self,observation):
        self.observation=observation
    
    
    
    
    
    
        
    
    # The step function that propels the simulation.
    # This function is called by the .fit function of the RL agent whenever a novel action has been computed.
    # The action is used to decide on the next topology node to run to, and step then triggers the control path (including 'Blender')
    # by making use of direct calls and signal/slot methods.
    #
    # action:   the action to be executed

    def _step(self, action):
        
        
        callbackValue=self.modules['spatial_representation'].generate_behavior_from_action(action)
        callbackValue['rlAgent']=self.rlAgent
        callbackValue['modules']=self.modules
        
        reward,stopEpisode=self.rewardCallback(callbackValue)
        
        return self.modules['observation'].observation, reward, stopEpisode, {}

    # This function restarts the RL agent's learning cycle by initiating a new episode.
    #
    def _reset(self):

        self.modules['spatial_representation'].generate_behavior_from_action('reset')

        # return the observation
        return self.modules['observation'].observation


def unity_decorater(func):
    """
    wraps on internal errors raised by the unity python api
    results in more readable error messages
    """

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            print(f"Unity crashed. {e}")

    return wrapper


def get_cobel_path():
    """
    returns the cobel project path
    TODO move this to some kind of utility class?
    """

    paths = os.environ['PYTHONPATH'].split(os.pathsep)
    path = None
    for p in paths:
        if 'CoBeL-RL' in p:
            full_path = p
            base_folder = full_path.split(sep='CoBeL-RL')[0]
            path = base_folder + 'CoBeL-RL'
            break
    return path


def get_env_path():
    """
    returns the unity env path
    TODO move this to some kind of utility class?
    """
    if 'UNITY_ENVIRONMENT_EXECUTABLE' in os.environ.keys():
        return os.environ['UNITY_ENVIRONMENT_EXECUTABLE']
    else:
        return None


class UnityInterface(gym.Env):
    """
    Wrapper for Unity 3D with ML-agents
    """

    class EmptyClass:
        """
        this class is used as an info dictionary
        """
        pass

    class UnityProcessor(Processor):
        """
        keras processor for the unity interface
        """
        def __init__(self, env_agent_specs,
                     agent_action_type):

            self.agent_action_type = agent_action_type
            self.observation_space = self.get_observation_space(env_agent_specs)
            self.action_space, self.action_shape, self.env_action_type = self.get_action_space(env_agent_specs,
                                                                                               agent_action_type)

        def get_observation_space(self, env_agent_specs):
            """
            Extract the information about the observation space from ml-agents group_spec.

            :return:                    observation space
            """

            # list of the sensor observation shapes
            observation_shapes = env_agent_specs.observation_shapes

            # this means we have multiple sensors attached
            if len(observation_shapes) > 1:

                # so we calculate the size of the concatenated flattened sensors observations.
                observation_shape = (sum([np.prod(shape) for shape in observation_shapes]),)

                print(">>> Warning! multiple sensor observations are only supported when flattened.\n"
                      ">>> flattening the observations!!!")

            else:

                # select the single sensors observation shape
                observation_shape = observation_shapes[0]

            observation_space = gym.spaces.Box(low=0, high=1, shape=observation_shape)

            return observation_space

        def get_action_space(self, env_agent_specs, agent_action_type):
            """
            Extract the information's about the action space from ml-agents group_spec

            :param env_agent_specs:     the group_spec object for the agent transmitted by ml agents env.
            :param agent_action_type:   the agent_action_type string
            :return:                    tuple (action_shape, action_space, action_type) used by CoBeL-RL.
            """
            # extract action specs.
            # to get the action_shape. fetch the action_shape from the spec object.
            action_shape = env_agent_specs.action_shape

            # get the action type by examining the action_type string of the spec object.
            action_type = "discrete" if 'DISCRETE' in str(env_agent_specs.action_type) else "continuous"

            # instantiate the action space
            #
            # depends on the type of agent you are using and the action space of the environment.
            #
            # if you are using an agent with a natively discrete space like a DQNAgent, you are not able to run
            # a continuous example. For compatibility reasons, we set up a mapping to discretize the continuous space.
            # see: make_continuous
            #
            # we also need to process the action_space for discrete actions, since Unity uses branches to structure
            # the actions.
            # see: make_discrete
            #
            if action_type == "discrete" and agent_action_type == "discrete":
                # Unity uses branches of discrete actions so we use all possible combinations as action_space.
                action_space = gym.spaces.Discrete(n=np.prod(action_shape))

            elif action_type == "continuous" and agent_action_type == "discrete":
                action_space = gym.spaces.Box(low=-1 * np.ones(shape=action_shape), high=np.ones(shape=action_shape))
                # continuous actions in Unity are bidirectional, so we double the action space.
                action_space.n = action_shape * 2
                print(">>> Warning!!! the environment requires a continuous action space\n"
                      ">>> and you configured a discrete agent action space! You will not reach optimal precision!")

            elif action_type == "continuous" and agent_action_type == "continuous":
                action_space = gym.spaces.Box(low=-1 * np.ones(shape=action_shape), high=np.ones(shape=action_shape))
                action_space.n = action_shape

            else:
                raise NotImplementedError(
                    'This combination of action and agent type is not supported. Check the definitions')

            return action_space, action_shape, action_type

        def process_observation(self, observation):
            """Processes the observation as obtained from the environment for use in an agent and
            returns it.
            # Arguments
                observation (object): An observation as obtained by the environment
            # Returns
                Observation obtained by the environment processed
            """
            observation = self.format_observations(observation)

            # WORKAROUND for extra observations:
            #
            # some unity envs send two observations when an agent is done with it's episode.
            #
            # Most envs have been modified to achieve that this is prevented,
            # but some will still send two observations.
            #
            # by getting the observation at index 0, we get the last observation of the episode.
            if not self.observation_space.shape == observation.shape:
                print(f'double obs! expected: {self.observation_space.shape} !=  received: {observation.shape}')
                observation = observation[0]

            return observation

        def format_observations(self, observations):
            """
            Format the received observation to work with cobel.

            :param observations:    the sensor observations received from ml-agents
            :return:                the formatted observation
            """
            # this means we have multiple sensors attached
            if len(observations) > 1:
                # flatten all sensor observations into a single vector
                formatted_observations = np.concatenate([np.ravel(o) for o in observations])
            else:
                # use the single observation
                formatted_observations = observations[0]

            return formatted_observations

        def process_action(self, action):
            """Processes an action predicted by an agent but before execution in an environment.
            # Arguments
                action (int): Action given to the environment
            # Returns
                Processed action given to the environment
            """
            action = self.format_action(action)
            return action

        def format_action(self, action):
            """
            This is a wrapper for the action / agent_action_type logic.

            :param action: the action received from the agent.
            :return:       the action formatted for unity.

            it's not possible to use a discrete agent like DQN with a
            continuous env from out of the box. So for compatibility reasons,
            we set up a mapping.

            there are four possible combinations:

            action_type = continuous, agent_action_type = discrete
            -> we map the discrete action to a continuous action
            see: make_continuous

            action_type = continuous, agent_action_type = continuous
            -> nothing to do, just wrap the action.

            action_type = discrete, agent = discrete
            -> Unity uses branches to structure the actions,
            so we map our one-hot-vector accordingly.
            see: make_discrete

            action_type = discrete, agent = continuous
            -> not supported at the moment.
            """

            # if action is a single int we wrap it in a list for compatibility
            if isinstance(action, np.integer):
                action = [action]

            if isinstance(action[0], np.floating):
                assert self.agent_action_type is 'continuous', f'the agent_action_type is set to {self.agent_action_type}' \
                                                               f', but the action is {type(action[0])}'

            if isinstance(action[0], np.integer):
                assert self.agent_action_type is 'discrete', f'the agent_action_type is set to {self.agent_action_type}' \
                                                             f', but the action is {type(action[0])}'

            if self.env_action_type == 'continuous' and self.agent_action_type == 'discrete':
                action = self.make_continuous(action[0])

            elif self.env_action_type == 'continuous' and self.agent_action_type == 'continuous':
                action = np.array([action])

            elif self.env_action_type == 'discrete' and self.agent_action_type == 'discrete':
                action = self.make_discrete(action[0])

            else:
                raise NotImplementedError(
                    'This combination of action and agent type is not supported.')

            return action

        def make_continuous(self, action):
            """
            Takes an action represented by a positive integer and turns it into a representation suitable for continuous
            unity environments.

            :param action:   a positive value integer from 0 to N
            :return:            an array with the correct format and range to be used by the ML-Agents framework

            :Reason of existence: The DQN outputs values that are integers. However, the ML-agents framework takes as
            inputs actions that also have negative values. Take for example an action space of 4 and an integer action id,
            for example the DQN outputs action 0 from possible actions [0,1,2,3].
            This would normally correspond to an one-hot array
            a=[1,0,0,0]
            But since the ML-agents framework makes use of negative values in continuous inputs, the correct array should be
            a=[1,0]
            (and for action 1 -> [-1,0], for action 2 -> [0,1], and for action 3 -> [0,-1]

            :High-level view of implementation: Take an integer alpha. If alpha is odd, it has a negative value. If it
             is even, the value is positive. Its index in the actions array is alpha divided by two and rounded down.

            :Future: This method will likely need to be extended, and perhaps moved inside the individual agent scripts
            inside Unity. Also, if you can think of an easier way to do this, you should probably rewrite this function.
            """

            assert action >= 0, 'This function assumes that actions are enumerated in the domain of positive integers.'
            assert int(action) == action, 'Unexpected input. Expected integer, received {}'.format(type(action))

            # check if odd
            value = action % 2

            # spread range from 0,1 to 0,2
            value = value * 2

            # adjust range to -1, 1
            value -= 1

            # flip signs so that evens corresponds to positive and odds corresponds to negative
            value = -value

            # get the correct bin by rounding down via the old python division
            index = action // 2

            # make new action
            new_action = np.zeros(self.action_shape)

            # put the new action in the correct bin
            new_action[index] = value

            return np.array([new_action])

        def make_discrete(self, action):
            """
            Encodes positive one hot integer into Unity acceptable format

            :param action:      a positive integer in the range of 0, N
            :return:            correctly formatted action.

            Adaptation to Unity branching system.

            In Unity you can specify branches for an discrete actions.
            f.e. a moving branch where you got the options
            0 = stay, 1 = left, 2 = right
            and maybe another action
            0 = no jump, 1 = jump

            but from our dqn agent we only get out a one hot vector.

            in order to map this we set the action space to
            the product of the action shape value.
            f.e. action_shape = (3,2) in Unity => (1,6) one hot vector
            by doing so we have all possible combinations of actions covered.
            see get_action_specs

            the last step is to map this one hot vector back to the branches.
            for this we resize the one hot vector to the shape of the initial
            action space ...
            f.e. [0, 0, 1, 0, 0, 0] => [[0, 0], [1, 0], [0, 0]]]
            and the calculate the indices where it is one.
            in this case: x=1, y=0 and use them as the values for the branches.

            the output is then [1, 0] corresponding to branch0 action1, branch1 action0
            """

            assert action >= 0, 'This function assumes that actions are enumerated in the domain of positive integers.'
            assert int(action) == action, 'Unexpected input. Expected integer, received {}'.format(type(action))

            # setup a one-hot vector with all possible combinations of actions.
            one_hot_vector = np.zeros(self.action_space.n)

            # set the chosen action to 1.
            one_hot_vector[action] = 1

            # resize to be a branch matrix.
            one_hot_vector.resize(self.action_shape)

            # get the coordinate where the 1 was stored
            coordinates = np.where(one_hot_vector == 1)

            # store the coordinates as values in the branches.
            branches = []
            for arr in coordinates:
                branches.append(arr[0])

            # wrap and return.
            return np.array([branches])

        def process_reward(self, reward):
            """Processes the reward as obtained from the environment for use in an agent and
            returns it.
            # Arguments
                reward (float): A reward as obtained by the environment
            # Returns
                Reward obtained by the environment processed
            """
            return reward

    def __init__(self, env_path, scene_name=None,
                 time_scale=2.0, nb_max_episode_steps=0, decision_interval=5, agent_action_type='discrete',
                 modules=None,
                 seed=42, timeout_wait=60, side_channels=None,
                 performance_monitor=None, with_gui=True):

        """
        Constructor

        connects to a given environment executable or to the unity editor and acts as a gym for keras agents.

        :param env_path:                full path to compiled unity executable, if None mlagents waits for the editor
                                        to connect.
        :param modules:                 the old CoBeL-RL modules. Currently unnecessary.
        :param seed:                    Random seed. Keep at 42 for good luck.
        :param timeout_wait:            Time until Unity is declared ded.
        :param side_channels:           possible channels to talk with the academy.
        :param time_scale:              Speed of the simulation.
        :param nb_max_episode_steps:    the number of maximum steps per episode.
        :param decision_interval:       the number of simulation steps before entering the next rl cycle.
        :param agent_action_type:       the native action type of the agent.
        :param performance_monitor:     the monitor used for visualizing the learning process.
        :param with_gui:                whether or not show the performance monitor and the environment gui.
        """

        # storage for editor process
        self.editor_process = None

        # setup side channels
        if side_channels is None:
            side_channels = []

        # setup engine channel
        self.engine_configuration_channel = EngineConfigurationChannel()

        # add engine config channel
        side_channels.append(self.engine_configuration_channel)

        # step parameters
        self.env_configuration_channel = FloatPropertiesChannel()
        self.env_configuration_channel.set_property("max_step", nb_max_episode_steps)
        self.env_configuration_channel.set_property("decision_interval", decision_interval)

        # add env config channel
        side_channels.append(self.env_configuration_channel)

        # command line args
        args = []

        # when env is an executable mlagents can load a given scene.
        if scene_name is not None:
            args = ["--mlagents-scene-name", scene_name]

        # when no env_path is given mlagents waits for a editor instance to connect on port 5004
        if env_path is None:
            # set port to 5004
            worker_id = 0
            print(">>> waiting for editor <<<")
        else:
            # select random worker id.
            # There is an issue in Linux where the worker id becomes available only after some time has passed since the
            # last usage. In order to mitigate the issue, a (hopefully) new worker id is automatically selected unless
            # specifically instructed not to.
            # Implementation note: two consecutive samples between 0 and 1200 have an immediate 1/1200 chance of
            # being the same. By using the modulo of unix time we arrive to that likelihood only after an hour, by when
            # the port has hopefully been released.
            # Additional notes: The ML-agents framework adds 5004 to the worker_id internally, so no need to worry about
            # port collision with the OS.
            worker_id = round(time.time()) % 1200

        # try to start the communicator
        try:
            # connect python to executable environment
            env = UnityEnvironment(file_name=env_path, worker_id=worker_id, seed=seed, base_port=5004,
                                   timeout_wait=timeout_wait, side_channels=side_channels, no_graphics=not with_gui,
                                   args=args)

            # reset the environment
            env.reset()

            # Set the time scale of the engine
            self.engine_configuration_channel.set_configuration_parameters(time_scale=time_scale, width=400, height=400)

            # receive environment information from environment
            group_name = env.get_agent_groups()[0]  # get agent ID
            group_spec = env.get_agent_group_spec(group_name)  # get agent specifications

            print("Specs received:", group_spec)

            # save environment variables
            self.env = env
            self.group_name = group_name

            # setup processor
            self.processor = self.UnityProcessor(env_agent_specs=group_spec,
                                                 agent_action_type=agent_action_type)

            # get the spaces from processor
            self.observation_space = self.processor.observation_space
            self.action_space = self.processor.action_space

            # plotting variables
            self.n_step = 0
            self.nb_episode = 0
            self.nb_episode_steps = 0
            self.cumulative_episode_reward = 0

            self.performance_monitor = performance_monitor

            if not with_gui:
                self.performance_monitor = None

            # initialize the observation plots with the original observation shapes
            if self.performance_monitor is not None:
                self.performance_monitor.instantiate_observation_plots(group_spec.observation_shapes)

        except UnityWorkerInUseException as e:
            print("the desired port is still in use. please retry after a few seconds.")

    def _step(self, action, *args, **kwargs):
        """
        Make the simulation move forward until the next decision is requested.

        :param action:  integer corresponding to the index of a one-hot-vector that is one.
        :return:        (observation, reward, done, info), necessary to function as a gym
        """

        # step the env with the provided action.
        self.step_env(action)

        # accumulate steps
        self.n_step += 1

        # get results
        # at the moment only one agent in the environment is supported.
        observation, reward, done = self.get_step_results()

        # Instantiate info var
        # This is currently unused, but required by gym/core.py. At some point, useful information could be stored here
        # and passed to the callbacks to increase CoBeL-RL's interoperability with other ML frameworks
        info = self.EmptyClass()
        info.items = lambda: iter({})  # don't ask why :/

        """
        #### PLOT SECTION #####################################################
        """

        # accumulate episode data
        self.cumulative_episode_reward += reward
        self.nb_episode_steps += 1

        # update step plotting data
        if self.performance_monitor is not None:
            self.performance_monitor.set_step_data(self.n_step)

        if done:
            # accumulate episodes
            self.nb_episode += 1

            # plot learning data
            if self.performance_monitor is not None:
                self.performance_monitor.set_episode_data(self.nb_episode, self.nb_episode_steps,
                                                          self.cumulative_episode_reward)

            # reset episode data
            self.cumulative_episode_reward = 0
            self.nb_episode_steps = 0

        if self.performance_monitor is not None:
            self.performance_monitor.update(nb_step=self.n_step)

        return observation, reward, done, info

    def step_env(self, action):
        """
        Wrapper for the step functionality of the Unity env.
        We format the action, set it and step the env.

        :param action:  the action provided by an agent.
        :return:        
        """
        # display the action
        if self.performance_monitor is not None:
            self.performance_monitor.display_actions(action)

        # setup action in the Unity environment
        self.env.set_actions(self.group_name, action)

        # forward the simulation by a tick (and execute action)
        self.env.step()

    def get_step_results(self):
        """
        Wrapper for the get_step_result function of Unity.
        We only use the first result of each type, since we only support one agent.

        :return: tuple observation, reward, done
        """

        # get the step result for our agent
        step_result = self.env.get_step_result(self.group_name)

        # get the sensor observations
        observations = step_result.obs

        # remove the singleton dimensions
        observations = [o.squeeze() for o in observations]

        # this displays the sensor observations
        # if multiple sensors are attached it displays a plot for each one.
        if self.performance_monitor is not None:
            self.performance_monitor.display_observations(observations)

        reward = step_result.reward[0]
        done = step_result.done[0]

        return observations, reward, done

    def _reset(self):
        """
        Resets the environment and returns the initial observation

        :return: initial observation
        """
        # resets the ml-agents academy.
        self.env.reset()

        # get the initial observation from the env.
        observation, _, _ = self.get_step_results()

        if self.performance_monitor is not None:
            self.performance_monitor.set_step_data(self.n_step)
            self.performance_monitor.update()

        return observation

    @unity_decorater
    def _close(self):
        """
        Closes the environment

        :return:
        """
        self.env.close()
        self.kill_editor_process()

    def start_editor_process(self, resource_path, scene_path, scene_name):
        """
        starts the unity editor by calling the executable at 'UNITY_EXECUTABLE_PATH'
        """
        assert resource_path is not None
        assert scene_path is not None
        assert scene_name is not None

        print(f'>>> starting editor process <<<"\nResources at: {resource_path}\nScene {scene_name} at: {scene_path}')
        self.editor_process = subprocess.Popen([os.environ['UNITY_EXECUTABLE_PATH'],
                                                '-createProject',
                                                '/home/philip/dev/unity_folder/projects/temp',
                                                '-executeMethod', 'PackageImporter.Import',
                                                '-resourcePath', resource_path,
                                                '-scenePath', scene_path,
                                                '-sceneName', scene_name])

    def kill_editor_process(self):
        """
        stops the editor process.
        """
        if self.editor_process is not None:
            os.killpg(os.getpgid(self.editor_process.pid), signal.SIGINT)


