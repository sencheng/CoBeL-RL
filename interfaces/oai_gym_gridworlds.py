# basic imports
import numpy as np
import gym
import pyqtgraph as pg
import PyQt5 as qt
# CoBel-RL framework
from cobel.misc.gridworld_visualization import CogArrow


class OAIGymInterface(gym.Env):
    '''
    Open AI interface for use with gridworld environments.
    
    | **Args**
    | modules:                      Contains framework modules.
    | world:                        The gridworld.
    | with_GUI:                     If true, observations and policy will be visualized.
    | gui_parent:                   The main window for visualization.
    '''
    
    def __init__(self, modules, world, with_GUI=True, gui_parent=None):
        # store the modules
        self.modules = modules
        # store visual output variable
        self.with_GUI = with_GUI       
        # store gridworld
        self.world = world
        # a variable that allows the OAI class to access the agent class
        self.rl_agent = None
        self.gui_parent = gui_parent
        # prepare observation and action spaces
        self.action_space = gym.spaces.Discrete(4)
        # initialize visualization
        self.init_visualization()
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
        transition_probabilities = self.world['sas'][self.current_state][action]
        if self.world['deterministic']:
            self.current_state = np.argmax(transition_probabilities)
        else:
            self.current_state = np.random.choice(np.arange(transition_probabilities.shape[0]), p=transition_probabilities)
        # determine current coordinates
        self.current_coordinates = self.world['coordinates'][self.current_state]
        # determine reward and whether the episode should end
        reward = self.world['rewards'][self.current_state]
        end_trial = self.world['terminals'][self.current_state]
        # update visualization
        self.update_visualization()
        
        return self.current_state, reward, end_trial, {}
    
    def reset(self):
        '''
        AI Gym's reset function.
        Resets the environment and the agent's state.
        '''
        # select randomly from possible starting states
        self.current_state = self.world['startingStates'][np.random.randint(self.world['startingStates'].shape[0])]
        # determine current coordinates
        self.current_coordinates = self.world['coordinates'][self.current_state]
        
        return self.current_state
    
    def update_transitions(self, invalid_transitions):
        '''
        This function updates the gridworld's transitions.
        
        | **Args**
        | invalidTransitions:           The gridworld's invalid transitions as a list of 2-tuples.
        '''
        # update the list of invalid transitions
        self.world['invalidTransitions'] = invalid_transitions
        # recompute the state-action-state transitions
        self.world['sas'] = np.zeros((self.world['states'], 4, self.world['states']))
        for state in range(self.world['states']):
            for action in range(4):
                h = int(state/self.world['width'])
                w = state - h * self.world['width']
                # left
                if action == 0:
                    w = max(0, w - 1)
                # up
                elif action == 1:
                    h = max(0, h - 1)
                # right
                elif  action == 2:
                    w = min(self.world['width'] - 1, w + 1)
                # down
                else:
                    h = min(self.world['height'] - 1, h + 1)
                # apply wind
                # currently walls are not taken into account!
                h += self.world['wind'][state][0]
                w += self.world['wind'][state][1]
                h = min(max(0, h), self.world['height'] - 1)
                w = min(max(0, w), self.world['width'] - 1)
                # determine next state
                next_state = int(h * self.world['width'] + w)
                if next_state in self.world['invalidStates'] or (state, next_state) in self.world['invalidTransitions']:
                    next_state = state
                self.world['sas'][state][action][next_state] = 1
    
    def init_visualization(self):
        '''
        This function initializes visualization if visualization enabled.
        '''
        if self.with_GUI:
            # prepare observation plot
            self.observation_plot = self.gui_parent.addPlot(title='Observation')
            self.observation_plot.hideAxis('bottom')
            self.observation_plot.hideAxis('left')
            self.observation_plot.setXRange(-0.01, 0.01)
            self.observation_plot.setYRange(-0.1, 0.1)
            self.observation_plot.setAspectLocked()
            self.state_text = pg.TextItem(text='-1', anchor=(0,0))
            self.coord_text = pg.TextItem(text='(-1, -1)', anchor=(0.25,-1))
            self.observation_plot.addItem(self.state_text)
            self.observation_plot.addItem(self.coord_text)
            # prepare grid world plot
            self.grid_plot = self.gui_parent.addPlot(title='Grid World')
            self.grid_plot.hideAxis('bottom')
            self.grid_plot.hideAxis('left')
            self.grid_plot.getViewBox().setBackgroundColor((255,255,255))
            self.grid_plot.setXRange(-1, self.world['width'] + 1)
            self.grid_plot.setYRange(-1, self.world['height'] + 1)
            # build graph for the grid world's background
            self.grid_background = []
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
                    self.grid_background += [node]
            # determine node coordinates and edges
            self.grid_nodes, self.grid_edges = [], []
            for n, node in enumerate(self.grid_background):
                self.grid_nodes += [node[:2]]
                for neighbor in node[2]:
                    self.grid_edges += [[n, neighbor]]
            # add graph item
            self.grid_nodes, self.grid_edges = np.array(self.grid_nodes), np.array(self.grid_edges)
            self.grid = pg.GraphItem(pos=self.grid_nodes, adj=self.grid_edges, pen=pg.mkPen(width=2), symbolPen=None, symbolBrush=None)
            self.grid_plot.addItem(self.grid)
            # make hard outline
            self.outline_nodes = np.array([[-0.05, -0.05],
                                          [-0.05, self.world['height'] + 0.05],
                                          [self.world['width'] + 0.05, -0.05],
                                          [self.world['width'] + 0.05, self.world['height'] + 0.05]])
            self.outline_edges = np.array([[0,1],[0,2],[1,3],[2,3]])
            self.outline = pg.GraphItem(pos=self.outline_nodes, adj=self.outline_edges, pen=pg.mkPen(color=(0, 0, 0), width=5), symbolPen=None, symbolBrush=None)
            self.grid_plot.addItem(self.outline)
            # mark goal states
            self.goals = []
            for goal in self.world['goals']:
                coordinates = self.world['coordinates'][goal] + 0.05
                nodes = np.array([coordinates, coordinates + np.array([0, 0.9]), coordinates + np.array([0.9, 0]), coordinates + 0.9])
                edges = np.array([[0,1],[0,2],[1,3],[2,3]])
                self.goals += [pg.GraphItem(pos=nodes, adj=edges, pen=pg.mkPen(color=(0, 255, 0), width=5), symbolPen=None, symbolBrush=None)]
                self.grid_plot.addItem(self.goals[-1])
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
                self.grid_plot.addItem(self.walls[-1])
            # make arrows for policy visualization
            self.arrows = []
            for state in self.world['coordinates']:
                self.arrows += [CogArrow(angle=0.0,headLen=20.0,tipAngle=25.0,tailLen=0.0,brush=(255,255,0))]
                self.arrows[-1].setData(state[0] + 0.5, state[1] + 0.5, 0.)
                self.grid_plot.addItem(self.arrows[-1])
            # update Qt visuals
            if hasattr(qt.QtGui, 'QApplication'):
                qt.QtGui.QApplication.instance().processEvents()
            else:
                qt.QtWidgets.QApplication.instance().processEvents()
               
    def update_visualization(self):
        '''
        This function updates visualization if visualization enabled.
        '''
        if self.with_GUI:
            # update observation
            self.state_text.setText(str(self.current_state))
            self.coord_text.setText('('+ str(self.current_coordinates[1]) + ', ' + str(self.current_coordinates[0]) + ')')
            # update arrows for policy visualization
            angle_table = {0: 0., 1: 90., 2: 180., 3: 270.}
            predictions = self.rl_agent.predict_on_batch(np.arange(self.world['states']))
            for p, prediction in enumerate(predictions):
                self.arrows[p].setData(self.world['coordinates'][p][0] + 0.5, self.world['coordinates'][p][1] + 0.5, angle_table[np.argmax(prediction)])
            # update Qt visuals
            if hasattr(qt.QtGui, 'QApplication'):
                qt.QtGui.QApplication.instance().processEvents()
            else:
                qt.QtWidgets.QApplication.instance().processEvents()