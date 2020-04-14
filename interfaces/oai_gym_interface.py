


import numpy as np
import gym
import PyQt5 as qt


from gym import spaces



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
