# basic imports
import numpy as np
import gym
import pyqtgraph as pg
import PyQt5 as qt


class OAIGymInterfaceSimple2D(gym.Env):
    '''
    Open AI interface implementing a simple continuous 2D environment.
    Two types of virtual agent are supported:
    1. An agent which moves in the four cardinal directions with a fixed step size.
    2. A differential wheeled robot which can move either or both of its wheels with a fixed step size.
    
    | **Args**
    | modules:                      Contains framework modules.
    | robot_type:                   A string identifying the robot type.
    | rewards:                      The reward function.
    | withGUI:                      If true, observations and policy will be visualized.
    | guiParent:                    The main window for visualization.
    | rewardCallback:               The callback function used to compute the reward.
    '''
    
    def __init__(self, modules, robot_type, rewards, withGUI=True, guiParent=None, rewardCallback=None):
        # store the modules
        self.modules = modules
        # store visual output variable
        self.withGUI = withGUI       
        # memorize the reward callback function
        self.rewardCallback = rewardCallback       
        # list of reward locations (along with reward magnitude)
        self.R = rewards
        # limits of the environments the agent is allowed to move in
        self.limits = np.array([0., 1., 0., 1.])
        # the agent's current state in the environment (position + orientation)
        self.state = np.array([0., 0., 0.])
        # if true, the agent receives a punishment for running into the walls
        self.punish_wall = False
        # robot type
        self.type = robot_type
        # a variable that allows the OAI class to access the robotic agent class
        self.rlAgent = None
        self.guiParent = guiParent
        # prepare observation and action spaces
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2, ))
        self.action_space = gym.spaces.Discrete(4)
        if self.type == 'wheel':
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3, ))
            self.action_space = gym.spaces.Discrete(3)
        # robot params
        self.body_radius = 0.05
        self.wheel_radius = 0.02
        self.wheel_distance = 0.1
        self.step_size = 0.015
        # initialize visualization
        self.initialize_visualization()
        # execute initial environment reset
        self.currentStep = 0
        self.reset()
        
    def step(self, action):
        '''
        AI Gym's step function.
        Executes the agent's action and propels the simulation.
        
        | **Args**
        | action:                       The action selected by the agent.
        '''
        reward, stopEpisode, wall_hit = 0, False, False
        # execute action
        if self.type == 'step':
            self.state[:2] += np.array([[-1., 0.], [0., 1.], [1., 0.], [0., -1.]])[action] * self.step_size
            prev_state = np.copy(self.state)
            self.state[0] = np.clip(self.state[0], a_min=self.limits[0], a_max=self.limits[1])
            self.state[1] = np.clip(self.state[1], a_min=self.limits[2], a_max=self.limits[3])
            self.observation = np.copy(self.state[:2])
            wall_hit = not np.array_equal(self.state, prev_state)
        elif self.type == 'wheel':
            # in case that the agent moves straight we can simplify the computation of the next state
            if action == 2:
                self.state += np.array([np.cos(self.state[2]), np.sin(self.state[2]), 0.]) * self.step_size
            # in case the wheels traveled different distances
            else:
                v = np.array([[0., 1], [1., 0.]])[action] * self.step_size
                omega = (v[1] - v[0])/self.wheel_distance
                R = 0.5 *  self.wheel_distance * (np.sum(v)/(v[1] - v[0]))
                ICC = self.state[:2] + np.array([-R * np.sin(self.state[2]), R * np.sin(self.state[2])])
                M = np.array([[np.cos(omega), -np.sin(omega), 0.], [np.sin(omega), np.cos(omega), 0.], [0., 0., 1.]])
                state = np.copy(self.state)
                state[:2] -= ICC
                self.state = np.matmul(M, state.reshape((3, 1))) + np.array([[ICC[0]], [ICC[1]], [self.state[2]]])
                self.state = self.state.reshape((3, ))
            prev_state = np.copy(self.state)
            self.state[0] = np.clip(self.state[0], a_min=self.limits[0], a_max=self.limits[1])
            self.state[1] = np.clip(self.state[1], a_min=self.limits[2], a_max=self.limits[3])
            self.state[2] %= 2 * np.pi
            self.observation = np.copy(self.state)
            wall_hit = not np.array_equal(self.state, prev_state)
        # determine reward and whether the episode should end
        distance_to_reward = np.sqrt(np.sum((self.R[:, :2] - self.state[:2])**2, axis=1))
        if np.sum(distance_to_reward <= self.body_radius * 2) != 0:
            reward = self.R[np.argmax(distance_to_reward <= self.body_radius * 2), 2]
            stopEpisode = True
        if wall_hit and self.punish_wall:
            reward, stopEpisode = -10, False
        # update visualization
        self.update_visualization()
        self.currentStep += 1
        
        return self.observation, reward, stopEpisode, {}
    
    def reset(self):
        '''
        AI Gym's reset function.
        Resets the environment and the agent's state.
        '''
        # reset the agent to a random position with a random orientation
        self.state[0] = np.random.uniform(self.limits[0], self.limits[1])
        self.state[1] = np.random.uniform(self.limits[2], self.limits[3])
        self.state[2] = np.random.uniform(0, 2 *  np.pi)
        self.observation = np.copy(self.state)
        # we omit  orientation when using the "step" type
        if self.type == 'step':
            self.state[2] = 0.
            self.observation = self.observation[:2]
        self.currentStep = 0
        
        return self.observation
    
    def initialize_visualization(self):
        '''
        This function initializes the elements required for visualization.
        '''
        if self.withGUI:
            # determine minimum and maximum coordinates
            self.coord_min, self.coord_max = min(self.limits[0], self.limits[2]), max(self.limits[1], self.limits[3])
            # state information panel
            self.state_information_panel = self.guiParent.addPlot(title='State Information')
            self.state_information_panel.hideAxis('bottom')
            self.state_information_panel.hideAxis('left')
            self.state_information_panel.setXRange(0, 1)
            self.state_information_panel.setYRange(0, 1)
            self.state_information_panel.setAspectLocked()
            self.coord_info = pg.TextItem(text='(-1, -1)')
            self.coord_label = pg.TextItem(text='Current Coordinates:')
            self.orientation_info = pg.TextItem(text='0')
            self.orientation_label = pg.TextItem(text='Current Orientation:')
            self.font = pg.Qt.QtGui.QFont()
            self.font.setPixelSize(20)
            self.coord_info.setFont(self.font)
            self.coord_label.setFont(self.font)
            self.coord_info.setPos(0.1, 0.8)
            self.coord_label.setPos(0.1, 0.85)
            self.orientation_info.setFont(self.font)
            self.orientation_label.setFont(self.font)
            self.orientation_info.setPos(0.1, 0.6)
            self.orientation_label.setPos(0.1, 0.65)
            self.state_information_panel.addItem(self.coord_info)
            self.state_information_panel.addItem(self.coord_label)
            self.state_information_panel.addItem(self.orientation_info)
            self.state_information_panel.addItem(self.orientation_label)
            # behavior panel
            self.behavior_panel = self.guiParent.addPlot(title='Behavior')
            width, height = (self.limits[1] - self.limits[0]), (self.limits[3] - self.limits[2])
            self.behavior_panel.setXRange(self.limits[0] - width * 0.05, self.limits[1] + width * 0.05)
            self.behavior_panel.setYRange(self.limits[2] - height * 0.05, self.limits[3] + height * 0.05)
            self.behavior_panel.setAspectLocked()
            self.markers = pg.ScatterPlotItem()
            coords = np.concatenate((self.R[:,:2], self.state[:2].reshape((1, 2))))
            brushes = [pg.mkBrush(color=(0, 255, 0)) for r in self.R] + [pg.mkBrush(color=(128, 128, 128))]
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
        '''
        if self.withGUI:
            # update state information panel
            self.coord_info.setText(str(self.state[:2]))
            self.orientation_info.setText(str(np.rad2deg(self.state[2])))
            # update behavior panel
            coords = np.concatenate((self.R[:,:2], self.state[:2].reshape((1, 2))))
            brushes = [pg.mkBrush(color=(0, 255, 0)) for r in self.R] + [pg.mkBrush(color=(128, 128, 128))]
            self.markers.setData(pos=coords, brush=brushes, size=10)
            # process changes
            if hasattr(qt.QtGui, 'QApplication'):
                qt.QtGui.QApplication.instance().processEvents()
            else:
                qt.QtWidgets.QApplication.instance().processEvents()