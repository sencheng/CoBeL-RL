# basic imports
import numpy as np
import gym
import pyqtgraph as pg
import PyQt5 as qt


class InterfaceMove2D(gym.Env):
    
    def __init__(self, modules: dict, with_GUI=True, gui_parent=None):
        '''
        Open AI interface implementing a simple continuous environment
        in which the agent has to move itself to an arbitrary goal location.
        The agent receives both its current position and the goal position as input.
        
        Parameters
        ----------
        modules :                           Contains framework modules.
        with_GUI :                          If true, observations and policy will be visualized.
        gui_parent :                        The main window used for visualization.
        
        Returns
        ----------
        None
        '''
        # store the modules
        self.modules = modules
        # store visual output variable
        self.with_GUI = with_GUI      
        # limits of the environments the agent is allowed to move in
        self.limits = np.array([0., 1., 0., 1.])
        # the agent's current state in the environment (current position + goal position)
        self.state = np.random.rand(4)
        # a variable that allows the OAI class to access the robotic agent class
        self.rl_agent = None
        self.gui_parent = gui_parent
        # prepare observation and action spaces
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4, ))
        self.action_space = gym.spaces.Discrete(2)
        # only provide reward when close to the goal
        self.sparse_reward = False
        # set one static goal
        self.static_goal = None
        # the goal radius
        self.goal_radius = 0.1
        # initialize visualization
        self.initialize_visualization()
        # execute initial environment reset
        self.current_step = 0
        self.reset()
        
    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        '''
        AI Gym's step function.
        Executes the agent's action and propels the simulation.
        
        Parameters
        ----------
        action :                            The action selected by the agent.
        
        Returns
        ----------
        observation :                       The observation of the new current state.
        reward :                            The reward received.
        end_trial :                         Flag indicating whether the trial ended.
        logs :                              The (empty) logs dictionary.
        '''
        reward, end_trial = 0, False
        # execute action
        self.state[:2] += action
        self.state[0] = np.clip(self.state[0], a_min=self.limits[0], a_max=self.limits[1])
        self.state[1] = np.clip(self.state[1], a_min=self.limits[2], a_max=self.limits[3])
        # determine reward and whether the episode should end
        reward = -np.sqrt(np.sum((self.state[:2] - self.state[2:])**2))
        end_trial = (-reward < self.goal_radius)
        if self.sparse_reward:
            reward = float(end_trial)
        self.observation = np.copy(self.state)
        # update visualization
        self.update_visualization()
        self.current_step += 1
        
        return self.observation, reward, end_trial, {}
    
    def reset(self) -> np.ndarray:
        '''
        AI Gym's reset function.
        Resets the environment and the agent's state.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        observation :                       The observation of the new current state.
        '''
        # reset the agent to a random position with a random orientation
        self.state = np.random.rand(4)
        if self.static_goal is not None:
            self.state[2:] = self.static_goal
        self.observation = np.copy(self.state)
        self.current_step = 0
        
        return self.observation
    
    def initialize_visualization(self):
        '''
        This function initializes the elements required for visualization.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        if self.with_GUI:
            # determine minimum and maximum coordinates
            self.coord_min, self.coord_max = min(self.limits[0], self.limits[2]), max(self.limits[1], self.limits[3])
            # state information panel
            self.state_information_panel = self.gui_parent.addPlot(title='State Information')
            self.state_information_panel.hideAxis('bottom')
            self.state_information_panel.hideAxis('left')
            self.state_information_panel.setXRange(0, 1)
            self.state_information_panel.setYRange(0, 1)
            self.state_information_panel.setAspectLocked()
            self.coord_info = pg.TextItem(text='(-1, -1)')
            self.coord_label = pg.TextItem(text='Current Coordinates:')
            self.font = pg.Qt.QtGui.QFont()
            self.font.setPixelSize(20)
            self.coord_info.setFont(self.font)
            self.coord_label.setFont(self.font)
            self.coord_info.setPos(0.1, 0.8)
            self.coord_label.setPos(0.1, 0.85)
            self.state_information_panel.addItem(self.coord_info)
            self.state_information_panel.addItem(self.coord_label)
            # behavior panel
            self.behavior_panel = self.gui_parent.addPlot(title='Behavior')
            width, height = (self.limits[1] - self.limits[0]), (self.limits[3] - self.limits[2])
            self.behavior_panel.setXRange(self.limits[0] - width * 0.05, self.limits[1] + width * 0.05)
            self.behavior_panel.setYRange(self.limits[2] - height * 0.05, self.limits[3] + height * 0.05)
            self.behavior_panel.setAspectLocked()
            self.markers = pg.ScatterPlotItem()
            coords = np.concatenate((self.state[2:].reshape((1, 2)), self.state[:2].reshape((1, 2))))
            brushes = [pg.mkBrush(color=(0, 255, 0)), pg.mkBrush(color=(128, 128, 128))]
            self.markers.setData(pos=coords, brush=brushes, size=10)
            self.walls = pg.GraphItem()
            nodes = np.array([[self.limits[0], self.limits[2]], [self.limits[1], self.limits[2]],
                              [self.limits[0], self.limits[3]], [self.limits[1], self.limits[3]]])
            edges = np.array([[0, 1], [1, 3], [3, 2], [2, 0]])
            self.walls.setData(pos=nodes, adj=edges, pen=(255, 0, 0), symbolPen=(0, 0, 0, 0), symbolBrush=(0, 0, 0, 0))
            self.behavior_panel.addItem(self.markers)
            self.behavior_panel.addItem(self.walls)
            
    def update_visualization(self):
        '''
        This function updates the visualization.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        if self.with_GUI:
            # update state information panel
            self.coord_info.setText(str(self.state[:2]))
            # update behavior panel
            coords = np.concatenate((self.state[2:].reshape((1, 2)), self.state[:2].reshape((1, 2))))
            brushes = [pg.mkBrush(color=(0, 255, 0)), pg.mkBrush(color=(128, 128, 128))]
            self.markers.setData(pos=coords, brush=brushes, size=10)
            # process changes
            if hasattr(qt.QtGui, 'QApplication'):
                qt.QtGui.QApplication.instance().processEvents()
            else:
                qt.QtWidgets.QApplication.instance().processEvents()
