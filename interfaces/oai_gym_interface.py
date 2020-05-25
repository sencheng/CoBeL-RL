import numpy as np
import gym
import PyQt5 as qt
from gym import spaces
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
import time

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

        print(action)
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


class unity2cobelRL(gym.Env):
    """
    Wrapper for Unity with ML-agents
    """
    class EmptyClass:
        pass

    def __init__(self, env_path, modules, withGUI=True, worker_id=None,
                 seed=42, timeout_wait=60, side_channels=None, time_scale=5.0):
        """
        :param env_path: full path to compiled unity executable
        :param modules: the old CoBeL-RL modules. Currently unnecessary
        :param withGUI: graphics, bool
        :param rewardCallback: TODO: implement if needed
        :param worker_id: Port used to communicate with Unity
        :param seed: Random seed. Keep at 42 for good luck.
        :param timeout_wait: Time until Unity is declared ded
        :param side_channels: possible channels to talk with the academy (to adjust environmental settings, e.g. light)
        :param time_scale: Speed of the simulation
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
        self.engine_configuration_channel.set_configuration_parameters(time_scale=time_scale, width=800, height=800)

        # receive environment information from environment
        group_name = env.get_agent_groups()[0]             # get agent ID
        group_spec = env.get_agent_group_spec(group_name)  # get agent specifications

        # refine information
        observation_space = group_spec.observation_shapes[0]
        action_shape = group_spec.action_shape
        action_type = "discrete" if 'DISCRETE' in str(group_spec.action_type) else "continuous"

        # instantiate action space
        if action_type is "discrete":
            self.action_space = gym.spaces.Discrete(n=action_shape[0])
        elif action_type is "continuous":
            self.action_space = gym.spaces.Box(low=-1*np.ones(shape=action_shape), high=np.ones(shape=action_shape))
            self.action_space.n = action_shape*2 # continuous actions in Unity are bidirectional
        else:
            raise NotImplementedError('Action type is not recognized. Check the self.action_type definition')

        # save environment variables
        self.env = env
        self.group_name = group_name
        self.group_spec = group_spec

        self.observation_space = np.zeros(shape=observation_space)

        self.action_shape = action_shape
        self.action_type = action_type

        # debugging stuff
        self.env.reset()
        step_result = self.env.get_step_result(self.group_name)
        observation = step_result.obs[0].squeeze()  # remove singleton dimensions
        self.observation_shape = observation.shape
        self.n_step = 0
        self.episode_steps = 0
        self.cumulative_reward = 0

        print(f'group_name: {group_name}')
        print(f'group_spec: {group_spec}')

    def _step(self, action, *args, **kwargs):
        """
        Make the simulation move forward one tick.
        :param action: integer
        :return: (observation, reward, done, info), necessary to function as a gym
        """

        self.episode_steps += 1;

        # format action
        if self.action_type is 'continuous':
            action = self.id2continuous(action)
        elif self.action_type is 'discrete':
            action = self.id2discrete(action)
        else:
            raise NotImplementedError('Action type is not recognized. Check the self.action_type definition')

        # accumulate steps
        self.n_step +=1

        # setup action in the Unity environment
        self.env.set_actions(self.group_name, action)

        # forward the simulation by a tick (and execute action)
        self.env.step()

        # get results
        step_result = self.env.get_step_result(self.group_name)
        observation = step_result.obs[0].squeeze()  # removes singleton dimensions
        reward = step_result.reward[0]
        done = step_result.done[0]

        # Instantiate info var
        # This is currently unused, but required by gym/core.py. At some point, useful information could be stored here
        # and passed to the callbacks to increase CoBeL-RL's interoperability with other ML frameworks
        info = self.EmptyClass()
        info.items = lambda: iter({})  # don't ask why :/

        # WORKAROUND for extra observations:
        #
        # some unity envs send two observations when an agent is done with it's episode.
        # this is due to requesting a decision before the episode ends.
        # resulting in adding two observations in one step to the agents data.
        #
        # 3DBall and Robot env have been changed to achieve that requesting a decision
        # and ending the episode is exclusive, but other demos, f.e. the ones which make
        # use of the 'DecisionRequester' script in unity will still send two observations,
        # when they are configured to request a decision every step.
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
        # currently if we need an observation every step, the unity example env need to be modified.
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


        # add current reward for debugging.
        self.cumulative_reward += reward

        # current WORKAROUND for _reset behavior:
        #
        # when an agent is done and resetted: reset the academy, too.
        # this syncs the local episodes of an agent with the episodes of the academy.
        # TODO: think about if we want to allow the agents to have local episodes.
        # See: _reset
        if done:
            #self.env.reset()
            # print episode debug info
            self.episode_steps = 0
            self.cumulative_reward = 0
            self.env.reset()

        # print episode debug info
        print('total step = {0}, episode_step = {1}, cumulative reward = {2}'.format(
            self.n_step, self.episode_steps, self.cumulative_reward))

        return observation, reward, done, info

    def _reset(self):
        """
        Resets the environment to prepare for the start of a new episode (if environment calls for it)
        :return: the agent observation
        """
        # self.env.reset() resets the mlagents academy.
        #
        # in mlagents the agents have a local maxstep value to reset multiple times in
        # an episode without being addressed by python.
        # so the agents can have multiple runs in an academy episode.
        #
        # it is disabled at the moment, because the _reset method is called very frequently
        # by a module and the agents can't make any progress.
        #
        # TODO: search for a better place to call it.
        #
        #self.env.reset()
        step_result = self.env.get_step_result(self.group_name)
        observation = step_result.obs[0].squeeze()  # remove singleton dimensions

        return observation

    def _close(self):
        """
        Closes the environment
        :return:
        """
        self.env.close()

    def format_observation(self, obs):
        """
        Currently unused. Can be used to perform manipulations on the observation.
        :param obs: the observation received from env.step or env.reset
        :return:
        """
        raise NotImplementedError

    def id2continuous(self, action_id):
        """
        Takes an action represented by a positive integer and turns it into a representation suitable for continuous
        unity environments

        TODO: Find a dumber way to do this
        :param action_id: a positive value integer from 0 to N
        :return: an array with the correct format and range to be used by the ML-Agents framework

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

    def id2discrete(self, action_id):
        # """
        # Encodes positive integers into Unity-acceptable format
        # :param action_id: a positive integer in the range of 0, N
        # :return: correctly formatted action.
        # """
        assert action_id >= 0, 'This function assumes that actions are enumerated in the domain of positive integers.'
        assert int(action_id) == action_id, 'Unexpected input. Expected integer, received {}'.format(type(action_id))

        new_action = np.array([[action_id]])

        return new_action
