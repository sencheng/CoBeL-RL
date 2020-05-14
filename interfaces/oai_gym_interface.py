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

    def __init__(self, env_path, modules, withGUI=True, rewardCallback=None, worker_id=None,
                 seed=42, timeout_wait=60, side_channels=None, time_scale=1.0):
        '''
        :param env_path: full path to compiled unity executable
        :param modules: the old CoBeL-RL modules. Currently unnecessary
        :param withGUI: graphics, bool
        :param rewardCallback: TODO: implement
        :param worker_id: Port used to communicate with Unity
        :param seed: Random seed. Keep at 42 for good luck. 
        :param timeout_wait: Time until Unity is declared ded
        :param side_channels: possible channels to talk with the academy (to adjust environmental settings, e.g. light)
        :param time_scale: Speed of the simulation
        '''''
        # store the modules
        #self.modules = modules

        if worker_id is None:
            # There is an issue in Linux where the worker id becomes available only after some time has passed since the
            # last usage. In order to mitigate the issue, a (hopefully) new worker id is automatically selected unless
            # specifically instructed not to.
            # Implementation note: two consecutive samples between 0 and 1200 have an immediate 1/1200 chance of
            # being the same. By using the modulo of unix time we arrive to that likelihood only after an hour, by when
            # the port has hopefully been released.
            # Additional notes: The ML-agents framework adds 5005 to the worker_id internally.
            worker_id = round(time.time()) % 1200 + 5005  # np.random.randint(0, 1200)

        # memorize the reward callback function
        self.rewardCallback = rewardCallback

        # The world and observations modules are deprecated. Interfacing is handled by Unity. If you need to
        # perform changes on the world that are not RL-agent actions now you can use a side channel.

        # setup engine channel
        self.engine_configuration_channel = EngineConfigurationChannel()
        if side_channels is None:
            side_channels = []

        side_channels.append(self.engine_configuration_channel)

        # connect python to executable environment
        env = UnityEnvironment(file_name=env_path, worker_id=worker_id, seed=seed, timeout_wait=timeout_wait,
                               side_channels=side_channels, no_graphics=not withGUI)

        # Reset the environment
        env.reset()

        # Set the default "brain" to work with
        group_name = env.get_agent_groups()[0]
        group_spec = env.get_agent_group_spec(group_name)

        # Set the time scale of the engine
        self.engine_configuration_channel.set_configuration_parameters(time_scale=time_scale)

        # Action and observation spaces are determined inside the Unity environment. See the
        # "Making a Unity environment" tutorial.

        # self.action_space = gym.spaces.Discrete(modules['topologyGraph'].cliqueSize)
        # self.observation_space = modules['observation'].getObservationSpace()

        # all OAI spaces have been initialized!

        # this observation variable is now fulfilled by the Unity interfacing
        self.observation = None

        # a variable that allows the OAI class to access the robotic agent class
        self.rlAgent = None

        # save environment variables
        self.env = env
        self.group_name=group_name
        self.group_spec=group_spec

        # extract environment information
        observation_space = group_spec.observation_shapes[0]
        action_shape = group_spec.action_shape
        action_type = "discrete" if 'DISCRETE' in str(group_spec.action_type) else "continuous"

        if action_type is "discrete":
            self.action_space = gym.spaces.Discrete(n=action_shape[0])
        elif action_type is "continuous":
            self.action_space = gym.spaces.Box(low=-1*np.ones(shape=action_shape), high=np.ones(shape=action_shape))
            self.action_space.n = action_shape*2
        else:
            raise NotImplementedError('Action type is not recognized. Check the self.action_type definition')
        # make gym
        #  self.EmptyClass = self.EmptyClass
        # self.action_space = self.EmptyClass()  # a hack to give this variable the ability to hold other variables
        # continuous actions in Unity also take negative values, so if we are using a softmax activation we have to
        # double the action space to account for them

        self.observation_space = np.zeros(shape=observation_space)

        self.action_shape = action_shape
        self.action_type = action_type

        print('action shape is {}'.format(action_shape))
        print('action type is {}'.format(action_type))
        print('action space is {}'.format(group_spec.action_size))
        print('action size is {}'.format(self.action_space.n))

        # debugging stuff
        self.env.reset()
        step_result = self.env.get_step_result(self.group_name)
        observation = step_result.obs[0].squeeze()  # remove singleton dimensions
        self.observation_shape = observation.shape
        self.n_step = 0

    # The step function that propels the simulation.
    def _step(self, action, *args, **kwargs):
        """
        :param action: integer
        :return: (observation, reward, done, info), necessary to function as a gym
        """

        if self.action_type is 'continuous':
            action = self.id2continuous(action)
        elif self.action_type is 'discrete':
            action = self.id2discrete(action)
        else:
            raise NotImplementedError('Action type is not recognized. Check the self.action_type definition')

        # print('Received action of shape {0} and value {1} for step {2}'.format(action.shape, action,str(self.n_step)))
        self.n_step +=1

        # setup action in the Unity environment
        self.env.set_actions(self.group_name, action)

        # forward the simulation by a timestep (and execute action)
        self.env.step()

        # get results
        step_result = self.env.get_step_result(self.group_name)
        observation = step_result.obs[0].squeeze()  # remove singleton dimensions
        reward = step_result.reward[0]
        done = step_result.done[0]

        # currently unused, but required by gym/core.py. At some point, useful information could be stored here and
        # passed to the callbacks to increase CoBeL-RL's interoperability with other ML frameworks
        info = self.EmptyClass()
        info.items = lambda : iter({})

        # print("step obs shape = {}".format(observation.shape))
        if not self.observation_shape == observation.shape:
            # Unity seems to throw extra sets of observations in a seemingly random fashion. When this happens,
            # only the first observation is taken into account.
            # ATTENTION: If you want to implement multi-agent RL this is going to need to be fixed.
            observation=observation[0]

        # print('Received action of shape {0} and value {1} for step {2}'.format(action.shape, action, str(self.n_step)))
        if done:
            print('Reward = {0}, step obs shape = {1}, step = {2}'.format(
                round(reward,2),observation.shape, self.n_step))

        return observation, reward, done, info

    # This function restarts the RL agent's learning cycle by initiating a new episode.
    def _reset(self):
        self.env.reset()
        step_result = self.env.get_step_result(self.group_name)
        observation = step_result.obs[0].squeeze()  # remove singleton dimensions
        # print('_reset obs shape = {}'.format(observation.shape))
        return observation

    def _close(self):
        self.env.close()

    def format_observation(self, obs):
        """
        :param obs:
        :return:
        """
        raise NotImplementedError

    def id2continuous(self, action_id):
        """
        Take an action represented by a positive integer and turn it into a representation suitable for continuous
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
        # :return:
        # """
        # assert action_id >= 0, 'This function assumes that actions are enumerated in the domain of positive integers.'
        # assert int(action_id) == action_id, 'Unexpected input. Expected integer, received {}'.format(type(action_id))
        #
        # new_action = np.zeros(self.action_shape)
        # index = action_id
        # new_action[index] = 1
        new_action = np.array([[action_id]])
        # print(new_action)
        return new_action
