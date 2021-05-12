# basic imports
import numpy as np
import gym
import pyqtgraph as pg
import PyQt5 as qt
# CoBel-RL framework
from misc.gridworld_visualization import CogArrow


class OAIGymInterface(gym.Env):
    '''
    Open AI interface for use with gridworld environments.
    
    | **Args**
    | modules:                      Contains framework modules.
    | world:                        The gridworld.
    | withGUI:                      If true, observations and policy will be visualized.
    | guiParent:                    The main window for visualization.
    | rewardCallback:               The callback function used to compute the reward.
    '''
    
    def __init__(self, modules, world, withGUI=True, guiParent=None, rewardCallback=None):
        # store the modules
        self.modules = modules
        # store visual output variable
        self.withGUI = withGUI       
        # memorize the reward callback function
        self.rewardCallback = rewardCallback       
        self.world = world
        # a variable that allows the OAI class to access the robotic agent class
        self.rlAgent = None
        self.guiParent = guiParent
        # prepare observation and action spaces
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,))
        self.action_space = gym.spaces.Discrete(4)
        # initialize visualization
        self.initVisualization()
        # execute initial environment reset
        self.reset()
        
    def step(self, action):
        '''
        AI Gym's step function.
        Executes the agent's action and propels the simulation.
        
        | **Args**
        | action:                       The action selected by the agent.
        '''
        # execute action
        transitionProbabilities = self.world['sas'][self.currentState][action]
        if self.world['deterministic']:
            self.currentState = np.argmax(transitionProbabilities)
        else:
            self.currentState = np.random.choice(np.arange(transitionProbabilities.shape[0]), p=transitionProbabilities)
        # determine current coordinates
        self.currentCoordinates = self.world['coordinates'][self.currentState]
        # determine reward and whether the episode should end
        reward = self.world['rewards'][self.currentState]
        stopEpisode = self.world['terminals'][self.currentState]
        # update visualization
        self.updateVisualization()
        
        return self.currentState, reward, stopEpisode, {}
    
    def reset(self):
        '''
        AI Gym's reset function.
        Resets the environment and the agent's state.
        '''
        # select randomly from possible starting states
        self.currentState = self.world['startingStates'][np.random.randint(self.world['startingStates'].shape[0])]
        # determine current coordinates
        self.currentCoordinates = self.world['coordinates'][self.currentState]
        
        return self.currentState
    
    def updateTransitions(self, invalidTransitions):
        '''
        This function updates the gridworld's transitions.
        
        | **Args**
        | invalidTransitions:           The gridworld's invalid transitions as a list of 2-tuples.
        '''
        # update the list of invalid transitions
        self.world['invalidTransitions'] = invalidTransitions
        # recompute the state-action-state transitions
        self.world['sas'] = np.zeros((self.world['states'], 4, self.world['states']))
        for state in range(self.world['states']):
            for action in range(4):
                h = int(state/self.world['width'])
                w = state - h * self.world['width']
                # left
                if action == 0:
                    w = max(0, w-1)
                # up
                elif action == 1:
                    h = max(0, h-1)
                # right
                elif  action == 2:
                    w = min(self.world['width'] - 1, w+1)
                # down
                else:
                    h = min(self.world['height'] - 1, h+1)
                # apply wind
                # currently walls are not taken into account!
                h += self.world['wind'][state][0]
                w += self.world['wind'][state][1]
                h = min(max(0, h), self.world['height'] - 1)
                w = min(max(0, w), self.world['width'] - 1)
                # determine next state
                nextState = int(h * self.world['width'] + w)
                if nextState in self.world['invalidStates'] or (state, nextState) in self.world['invalidTransitions']:
                    nextState = state
                self.world['sas'][state][action][nextState] = 1
    
    def initVisualization(self):
        '''
        This function initializes visualization if visualization enabled.
        '''
        if self.withGUI:
            # prepare observation plot
            self.observationPlot = self.guiParent.addPlot( title="Observation" )
            self.observationPlot.hideAxis('bottom')
            self.observationPlot.hideAxis('left')
            self.observationPlot.setXRange(-0.01, 0.01)
            self.observationPlot.setYRange(-0.1, 0.1)
            self.observationPlot.setAspectLocked()
            self.stateText = pg.TextItem(text='-1', anchor=(0,0))
            self.coordText = pg.TextItem(text='(-1, -1)', anchor=(0.25,-1))
            self.observationPlot.addItem(self.stateText)
            self.observationPlot.addItem(self.coordText)
            # prepare grid world plot
            self.gridPlot = self.guiParent.addPlot( title="Grid World" )
            self.gridPlot.hideAxis('bottom')
            self.gridPlot.hideAxis('left')
            self.gridPlot.getViewBox().setBackgroundColor((255,255,255))
            self.gridPlot.setXRange(-1, self.world['width'] + 1)
            self.gridPlot.setYRange(-1, self.world['height'] + 1)
            # build graph for the grid world's background
            self.gridBackground = []
            for j in range(self.world['height'] + 1):
                for i in range(self.world['width'] + 1):
                    node = [i, j, []]
                    if i - 1 >= 0:
                        node[2] += [j * (self.world['width'] + 1) + i - 1]
                    if i + 1 < self.world['width'] + 1:
                        node[2] += [j * (self.world['width'] + 1) + i + 1]
                    if j - 1 >= 0:
                        node[2] += [(j - 1) * (self.world['width'] + 1) + i]
                    if j + 1 < self.world['height'] + 1:
                        node[2] += [(j + 1) * (self.world['width'] + 1) + i]
                        
                    self.gridBackground += [node]
            # determine node coordinates and edges
            self.gridNodes, self.gridEdges = [], []
            for n, node in enumerate(self.gridBackground):
                self.gridNodes += [node[:2]]
                for neighbor in node[2]:
                    self.gridEdges += [[n, neighbor]]
            # add graph item
            self.gridNodes, self.gridEdges = np.array(self.gridNodes), np.array(self.gridEdges)
            self.grid = pg.GraphItem(pos=self.gridNodes, adj=self.gridEdges, pen=pg.mkPen(width=2), symbolPen=None, symbolBrush=None)
            self.gridPlot.addItem(self.grid)
            # make hard outline
            self.outlineNodes = np.array([[-0.05, -0.05],
                                          [-0.05, self.world['height'] + 0.05],
                                          [self.world['width'] + 0.05, -0.05],
                                          [self.world['width'] + 0.05, self.world['height'] + 0.05]])
            self.outlineEdges = np.array([[0,1],[0,2],[1,3],[2,3]])
            self.outline = pg.GraphItem(pos=self.outlineNodes, adj=self.outlineEdges, pen=pg.mkPen(color=(0, 0, 0), width=5), symbolPen=None, symbolBrush=None)
            self.gridPlot.addItem(self.outline)
            # mark goal states
            self.goals = []
            for goal in self.world['goals']:
                coordinates = self.world['coordinates'][goal] + 0.05
                nodes = np.array([coordinates, coordinates + np.array([0, 0.9]), coordinates + np.array([0.9, 0]), coordinates + 0.9])
                edges = np.array([[0,1],[0,2],[1,3],[2,3]])
                self.goals += [pg.GraphItem(pos=nodes, adj=edges, pen=pg.mkPen(color=(0, 255, 0), width=5), symbolPen=None, symbolBrush=None)]
                self.gridPlot.addItem(self.goals[-1])
            # draw walls
            self.walls = []
            for transition in self.world['invalidTransitions']:
                first, second = np.amin(transition), np.amax(transition)
                diff = np.abs(self.world['coordinates'][first] - self.world['coordinates'][second])
                coord = self.world['coordinates'][first]
                nodes = np.array([[0, 0], [0, 0]])
                if diff[0] > 0:
                    nodes = np.array([coord + np.array([1, 0]), coord + np.array([1, 1])])
                else:
                    nodes = np.array([coord + np.array([0, 0]), coord + np.array([1, 0])])
                edges =  np.array([[0, 1]])
                self.walls += [pg.GraphItem(pos=nodes, adj=edges, pen=pg.mkPen(color=(0, 0, 0), width=5), symbolPen=None, symbolBrush=None)]
                self.gridPlot.addItem(self.walls[-1])
            # make arrows for policy visualization
            self.arrows = []
            for state in self.world['coordinates']:
                self.arrows += [CogArrow(angle=0.0,headLen=20.0,tipAngle=25.0,tailLen=0.0,brush=(255,255,0))]
                self.arrows[-1].setData(state[0] + 0.5, state[1] + 0.5, 0.)
                self.gridPlot.addItem(self.arrows[-1])
               
    def updateVisualization(self):
        '''
        This function updates visualization if visualization uenabled.
        '''
        if self.withGUI:
            # update observation
            self.stateText.setText(str(self.currentState))
            self.coordText.setText('('+ str(self.currentCoordinates[1]) + ', ' + str(self.currentCoordinates[0]) + ')')
            # update arrows for policy visualization
            angleTable = {0: 0., 1: 90., 2: 180., 3: 270.}
            predictions = self.rlAgent.predict_on_batch(np.arange(self.world['states']))
            for p, prediction in enumerate(predictions):
                self.arrows[p].setData(self.world['coordinates'][p][0] + 0.5, self.world['coordinates'][p][1] + 0.5, angleTable[np.argmax(prediction)])
            
            #if qt.QtGui.QApplication.instance() is not None:
            qt.QtGui.QApplication.instance().processEvents()