import numpy as np
import gym
import PyQt5 as qt
from gym import spaces
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
import time
from matplotlib import pyplot as plt

### This is the Open AI gym interface class. The interface wraps the control path and ensures communication
### between the agent and the environment. The class descends from gym.Env, and is designed to be minimalistic (currently!).
class OAIGymInterface(gym.Env):


    # The constructor.
    # modules:          the dict of all available modules in the system
    # withGUI:          if True, the module provides GUI control
    # rewardCallback:   this callback function is invoked in the step routine in order to get the appropriate reward w.r.t. the experimental design

    def __init__(self, modules,withGUI=True,rewardCallback=None):

        # store the modules
        self.modules=modules

        # store visual output variable
        self.withGUI=withGUI

        # memorize the reward callback function
        self.rewardCallback=rewardCallback

        self.world=self.modules['world']
        self.observations=self.modules['observation']


        # second: action space
        self.action_space = gym.spaces.Discrete(modules['topologyGraph'].cliqueSize)


        # third: observation space
        self.observation_space=modules['observation'].getObservationSpace()

        # all OAI spaces have been initialized!

        # this observation variable is filled by the OBS modules
        self.observation=None

        # required for the analysis of the agent's behavior
        self.forbiddenZoneHit=False
        self.finalNode=-1

        # a variable that allows the OAI class to access the robotic agent class
        self.rlAgent=None


    # This function (slot) updates the observation provided by the environment
    #
    # observation:  the observation used to perform the update
    def updateObservation(self,observation):
        self.observation=observation




    # This function moves the robot/agent to a goal position
    #
    # goalPosition: the goal position to move the robot/agent to
    #
    def moveToPosition(self,goalPosition):
        # set the new goal position
        # wait until the agent reaches the designated next node (when teleporting, this happens instantaneously)

        self.modules['world'].actuateRobot(np.array([goalPosition[0],goalPosition[1],90.0]))
        self.modules['world'].actuateRobot(np.array([goalPosition[0],goalPosition[1],90.0]))

        if self.withGUI:
            qt.QtGui.QApplication.instance().processEvents()



        # reset the 'goal reached' indicator of the environment

        self.modules['observation'].update()
        self.modules['topologyGraph'].updateRobotPose([goalPosition[0],goalPosition[1],0.0,1.0])



    # The step function that propels the simulation.
    # This function is called by the .fit function of the RL agent whenever a novel action has been computed.
    # The action is used to decide on the next topology node to run to, and step then triggers the control path (including 'Blender')
    # by making use of direct calls and signal/slot methods.
    #
    # action:   the action to be executed

    def _step(self, action):

        previousNode=self.modules['topologyGraph'].currentNode
        # with action given, the next node can be computed
        self.modules['topologyGraph'].nextNode=self.modules['topologyGraph'].nodes[self.modules['topologyGraph'].currentNode].neighbors[action].index
        # array to store the next node's coordinates
        nextNodePos=np.array([0.0,0.0])

        if self.modules['topologyGraph'].nextNode!=-1:
            # compute the next node's coordinates
            nextNodePos=np.array([self.modules['topologyGraph'].nodes[self.modules['topologyGraph'].nextNode].x,self.modules['topologyGraph'].nodes[self.modules['topologyGraph'].nextNode].y])
        else:
            # if the next node corresponds to an invalid node, the agent stays in place
            self.modules['topologyGraph'].nextNode=self.modules['topologyGraph'].currentNode
            # prevent the agent from starting any motion pattern
            self.modules['world'].goalReached=True
            nextNodePos=np.array([self.modules['topologyGraph'].nodes[self.modules['topologyGraph'].currentNode].x,self.modules['topologyGraph'].nodes[self.modules['topologyGraph'].currentNode].y])

        print('step to node: %d' % self.modules['topologyGraph'].nextNode)

        # actually move the robot to the node
        self.moveToPosition(nextNodePos)




        # make the current node the one the agent travelled to
        self.modules['topologyGraph'].currentNode=self.modules['topologyGraph'].nextNode




        callbackValue=dict()
        callbackValue['rlAgent']=self.rlAgent
        callbackValue['modules']=self.modules
        callbackValue['currentNode']=self.modules['topologyGraph'].nodes[self.modules['topologyGraph'].currentNode]
        callbackValue['previousNode']=self.modules['topologyGraph'].nodes[previousNode]
        reward,stopEpisode=self.rewardCallback(callbackValue)

        return self.modules['observation'].observation, reward, stopEpisode, {}


    # This function restarts the RL agent's learning cycle by initiating a new episode.
    #
    def _reset(self):

        # a random node is chosen to place the agent at (this node MUST NOT be the global goal node!)
        nextNode=-1
        while True:
            nrNodes=len(self.modules['topologyGraph'].nodes)
            nextNode=np.random.random_integers(0,nrNodes-1)
            if self.modules['topologyGraph'].nodes[nextNode].startNode:
                break

        nextNodePos=np.array([self.modules['topologyGraph'].nodes[nextNode].x,self.modules['topologyGraph'].nodes[nextNode].y])

        print('reset to node: %d' % nextNode)

        # actually move the robot to the node
        self.moveToPosition(nextNodePos)
        # make the current node the one the agent travelled to
        self.modules['topologyGraph'].currentNode=nextNode

        # return the observation
        return self.modules['observation'].observation

def unity_decorater(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            print(f'Unity crashed. {e}')
    return wrapper


class unity_wrapper(gym.Env):
    """
    Wrapper for Unity 3D with ML-agents
    """
    class EmptyClass:
        pass

    def __init__(self, env_path, modules=None, withGUI=True, worker_id=None,
                 seed=42, timeout_wait=60, side_channels=None, time_scale=5.0, 
                 agent_action_type='discrete'):
        """
        :param env_path: full path to compiled unity executable
        :param modules: the old CoBeL-RL modules. Currently unnecessary.
        :param withGUI: graphics, bool
        :param rewardCallback: TODO: implement if needed
        :param worker_id: Port used to communicate with Unity
        :param seed: Random seed. Keep at 42 for good luck.
        :param timeout_wait: Time until Unity is declared ded
        :param side_channels: possible channels to talk with the academy (to adjust environmental settings, e.g. light)
        :param time_scale: Speed of the simulation
        :param agent_action_type: the native action type of the agent.
        """

        # setup communication port
        if worker_id is None:
            # There is an issue in Linux where the worker id becomes available only after some time has passed since the
            # last usage. In order to mitigate the issue, a (hopefully) new worker id is automatically selected unless
            # specifically instructed not to.
            # Implementation note: two consecutive samples between 0 and 1200 have an immediate 1/1200 chance of
            # being the same. By using the modulo of unix time we arrive to that likelihood only after an hour, by when
            # the port has hopefully been released.
            # Additional notes: The ML-agents framework adds 5005 to the worker_id internally, so no need to worry about
            # port collision with the OS.
            worker_id = round(time.time()) % 1200

        # setup engine channel
        self.engine_configuration_channel = EngineConfigurationChannel()

        # setup side channels
        if side_channels is None:
            side_channels = []

        side_channels.append(self.engine_configuration_channel)

        # connect python to executable environment
        env = UnityEnvironment(file_name=env_path, worker_id=worker_id, seed=seed, timeout_wait=timeout_wait,
                               side_channels=side_channels, no_graphics=not withGUI)

        # Reset the environment
        env.reset()

        # Set the time scale of the engine
        self.engine_configuration_channel.set_configuration_parameters(time_scale=time_scale, width=400, height=400)

        # receive environment information from environment
        group_name = env.get_agent_groups()[0]             # get agent ID
        group_spec = env.get_agent_group_spec(group_name)  # get agent specifications

        # save environment variables
        self.env = env
        self.group_name = group_name
        self.agent_action_type = agent_action_type
        self.observation_shape, self.observation_space = self.get_observation_specs(group_spec)
        self.action_shape, self.action_space, self.action_type = self.get_action_specs(group_spec, agent_action_type)

        # debug stuff
        self.n_step = 0
        self.nb_episode = 0
        self.episode_steps = 0
        self.cumulative_reward = 0
        self.reward_plot = np.zeros(shape=(1000, 1000))

    def get_observation_specs(self, env_agent_specs):
        """
        Extract the information about the observation space from mlagents group_spec.

        :param env_agent_specs:     the group_spec object for the agent transmitted by ml agents env.
        :return:                    tuple (observation_shape, observation_space)

        Just getting the shape of the observation of agent 0 ;)
        and constructing the action space from the shape.
        """

        observation_shape = env_agent_specs.observation_shapes[0]
        observation_space = np.zeros(shape=observation_shape)

        return (observation_shape, observation_space)

    def get_action_specs(self, env_agent_specs, agent_action_type):
        """
        Extract the informations about the action space from mlagents group_spec

        :param env_agent_specs:     the group_spec object for the agent transmitted by ml agents env.
        :param agent_action_type:   the agent_action_type string 
        :return:                    tuple (action_shape, action_space, action_type) used by CoBeL-RL.
        """
        # extract action specs.
        # to get the action_shape just fetch the action_shape from the spec object.
        action_shape = env_agent_specs.action_shape

        # get the action type by examinating the action_type string of the spec object.
        action_type = "discrete" if 'DISCRETE' in str(env_agent_specs.action_type) else "continuous"

        # instantiate the action space
        #
        # depends on the type of agent you are using and the action space of the
        # environment.
        #
        # if you are using an agent with a natively discrete space like a DQNAgent, you are not able to run
        # a continuous example. For compability reasons, we set up a mapping to discretize the continuous space.
        # see: make_continuous 
        #
        # we also need to process the action_space for discrete actions, since Unity uses branches to structure
        # the actions.
        # see: make_discrete
        #
        if action_type is "discrete" and agent_action_type is "discrete":
            action_space = gym.spaces.Discrete(n=np.prod(action_shape)) # Unity uses branches of discrete action so we use all possible 
                                                                        # combinations of them as action_space for the DQN.


        elif action_type is "continuous" and agent_action_type is "discrete":
            action_space = gym.spaces.Box(low=-1*np.ones(shape=action_shape), high=np.ones(shape=action_shape))
            action_space.n = action_shape * 2                           # continuous actions in Unity are bidirectional, 
                                                                        # so we double the action space.

        elif action_type is "continuous" and agent_action_type is "continuous":
            action_space = gym.spaces.Box(low=-1*np.ones(shape=action_shape), high=np.ones(shape=action_shape))
            action_space.n = action_shape                               

        else:
            raise NotImplementedError('This combination of action and agent type is not supported. Check the definitions')

        return (action_shape, action_space, action_type)

    def _step(self, action, *args, **kwargs):
        """
        Make the simulation move forward one tick.

        :param action:  integer corresponding to the index of a one-hot-vector that is one.
        :return:        (observation, reward, done, info), necessary to function as a gym
        """

        # accumulate steps
        self.n_step += 1

        # step the env with the provided action.
        self.step_env(action)

        # get results
        # at the moment only one agent in the environment is supported.
        observation, reward, done = self.get_step_results()

        # Instantiate info var
        # This is currently unused, but required by gym/core.py. At some point, useful information could be stored here
        # and passed to the callbacks to increase CoBeL-RL's interoperability with other ML frameworks
        info = self.EmptyClass()
        info.items = lambda: iter({})  # don't ask why :/

        #
        #### DEBUG SECTION #####################################################
        #

        # WORKAROUND for extra observations:
        #
        # some unity envs send two observations when an agent is done with it's episode.
        # this is due to instantly requesting a decision after the episode ends.
        # resulting in adding two observations in one step to the agents data.
        #
        # 3DBall and Robot env have been changed to achieve that requesting a decision
        # and ending the episode is exclusive, but other demos, f.e. the ones which make
        # use of the 'DecisionRequester' script in unity will still send two observations,
        # when they are configured to request a decision at every step.
        #
        # by getting the observation at index 0, we get the last observation of the previous episode.
        # a possible problem is that we lose an observation of the next episode.
        #
        # by getting the observation at index 1 we get the first observation of the next episode.
        # the problem that we loose the last observation seemed smaller, so we keep the first.
        #
        # an easy workaround is to set the parameter in the environment such that it not requests
        # a decision in every step (but in every 2,3,... step).
        #
        # currently if we need an observation every step, the unity example env should to be modified.
        #
        double_obs_error = False
        if not self.observation_shape == observation.shape:
            double_obs_error = True
            print(f'double obs received {observation}')
            observation = observation[1]

        # DEBUG: used to check if the double obs occure only together with 'done'.
        #
        # TODO: remove since always true.
        #
        if double_obs_error and not done:
            raise Exception("Double observation didn't occured as assumed.")

        # update debug vars.
        self.cumulative_reward += reward
        self.episode_steps += 1

        # print episode info
        print('total step = {0}, episode_step = {1}, cumulative reward = {2}'.format(
            self.n_step, self.episode_steps, self.cumulative_reward))

        # plot episode info.
        #self.reward_plot[self.nb_episode-1, self.episode_steps-1] = self.cumulative_reward

        # reset debug vars, when done.
        if done:
            self.episode_steps = 0
            self.cumulative_reward = 0
            self.nb_episode += 1

        return observation, reward, done, info

    @unity_decorater
    def step_env(self, action):
        """
        Wrapper for the step functionality of the Unity env.
        We format the action, set it and step the env.

        :param action:  the action provided by an agent.
        :return:        
        """
        # format the action for unity.
        formatted_action = self.format_action(action)
        
        # setup action in the Unity environment
        self.env.set_actions(self.group_name, formatted_action)

        # forward the simulation by a tick (and execute action)
        self.env.step()

    @unity_decorater
    def get_step_results(self):
        """
        Wrapper for the get_step_result function of Unity.
        We only use the first result of each type, since we only support one agent.

        :return: tuple observation, reward, done
        """
        step_result = self.env.get_step_result(self.group_name)
        observation = self.format_observation(step_result.obs[0])
        reward = step_result.reward[0]
        done = step_result.done[0]
        return (observation, reward, done)

    @unity_decorater
    def _reset(self):
        """
        Resets the environment to prepare for the start of a new episode (if environment calls for it)

        :return: the agents initial observation
        """
        # resets the mlagents academy.
        self.env.reset()

        # get the intial observation from the env.
        observation, _ , _ = self.get_step_results()

        return observation

    @unity_decorater
    def _close(self):
        """
        Closes the environment

        :return:
        """
        self.env.close()

        # debug plot
        plt.matshow(self.reward_plot)
        plt.show()

    def format_observation(self, obs):
        """
        Formate the received observation to work with cobel.
        At the moment we just remove the singleton dimensions.

        :param obs: the observation received from env.step or env.reset
        :return:    the formated observation
        """
        return obs.squeeze() # remove singleton dimensions.

    def format_action(self, action):
        """
        This is a wrapper for the action / agent_action_type logic.

        :param action: the action received from the agent.
        :return:       the action formatted for unity.

        it's not possible to use a discrete agent like DQN with a
        continuous env from out of the box. So for compability reasons,
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
        if self.action_type is 'continuous' and self.agent_action_type is 'discrete':
            action = self.make_continuous(action)

        elif self.action_type is 'continuous' and self.agent_action_type is 'continuous':
            action = np.array([action])

        elif self.action_type is 'discrete' and self.agent_action_type is 'discrete':
            action = self.make_discrete(action)

        else:
            raise NotImplementedError('This combination of action and agent type is not supported. Check the definitions.')

        return action

    def make_continuous(self, action_id):
        """
        Takes an action represented by a positive integer and turns it into a representation suitable for continuous
        unity environments.
        
        :param action_id:   a positive value integer from 0 to N
        :return:            an array with the correct format and range to be used by the ML-Agents framework

        TODO: Find a dumber way to do this

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
        assert action_id >= 0, 'This function assumes that actions are enumerated in the domain of positive integers.'
        assert int(action_id) == action_id, 'Unexpected input. Expected integer, received {}'.format(type(action_id))

        # check if odd
        value = action_id % 2

        # spread range from 0,1 to 0,2
        value = value * 2

        # adjust range to -1, 1
        value -= 1

        # flip signs so that evens corresponds to positive and odds corresponds to negative
        value = -value

        # get the correct bin by rounding down via the old python division
        index = action_id//2

        # make new action
        new_action = np.zeros(self.action_shape)

        # put the new action in the correct bin
        new_action[index] = value

        return np.array([new_action])

    def make_discrete(self, action_id):
        """
        Encodes positive one hot integer into Unity-acceptable format

        :param action_id:   a positive integer in the range of 0, N
        :return:            correctly formatted action.

        Adaptation to Unity branching system.

        In Unity you can specify branches for an discrete actions.
        f.e. a moving branch where you got the options
        0 = stay, 1 = left, 2 = right
        and maybe another action 
        0 = no jump, 1 = jump

        but from our dqn agent we only get out a one hot vector.

        in order to map this correctly we set the action space to
        the product of the action shape value.
        f.e. action_shape = (3,2) in Unity => (1,6) one hot vector
        by doing so we have all possible combinations of actions covered.

        the last step is to map this one hot vector back to the branches.
        for this we resize the one hot vector to the shape of the initial
        action space ...
        f.e. [0, 0, 1, 0, 0, 0] => [[0, 0], [1, 0], [0, 0]]]
        and the calculate the indicies where it is one.
        in this case: x=1, y=0 and use them as the values for the branchs ;)

        the output is then [1, 0] correnponding to branch0 action1, branch1 action0

        this should work for all cases, n branches with m actions.
        """
        assert action_id >= 0, 'This function assumes that actions are enumerated in the domain of positive integers.'
        assert int(action_id) == action_id, 'Unexpected input. Expected integer, received {}'.format(type(action_id))

        # setup a one-hot vector with all possible combinations of actions.
        one_hot_vector = np.zeros(self.action_space.n)

        # set the chosen action to 1.
        one_hot_vector[action_id] = 1

        # resize to be a branch matrix.
        one_hot_vector.resize(self.action_shape)

        # get the coordinate where the 1 was stored (multidimensional)
        coords = np.where(one_hot_vector == 1)

        # store the coordinates as values in the branches.
        branches = []
        for arr in coords:
            branches.append(arr[0])

        # wrap and return.
        return np.array([branches])    
