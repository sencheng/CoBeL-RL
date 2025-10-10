# basic imports
import copy
import numpy as np
import shapely as sh  # type: ignore
from shapely.ops import nearest_points  # type: ignore
import gymnasium as gym
import pyqtgraph as pg  # type: ignore
import PyQt6 as qt
# framework imports
from .interface import Interface
from ..misc.visualization import CogArrow
# typing
from .interface import Action, StepTuple, ResetTuple
from .simulator.simulator import Simulator
from typing import Literal
from numpy.typing import NDArray


class Continuous2D(Interface):
    """
    Interface implementing a simple continuous 2D environment.
    Two types of virtual agent are supported:
    1. An agent which moves in the four cardinal directions with a fixed step size.
    2. A differential wheeled robot which can move either
       or both of its wheels with a fixed step size.

    Parameters
    ----------
    robot_type : str
        A string defining the robot type.
        Can be either 'step' or 'wheel'.
    room : sh.Polygon
        A Shapely polygon representing the environment's borders.
    spawn : sh.Polygon
        A Shapely polygon representing the agent's starting area.
    obstacles : list of sh.Polygon or None
        An optional list of Shapely polygons representing
        environmental obstacles.
    rewards : NDArray
        The reward function as a (N, 3) NumPy array.
        The first two columns represent the reward x and y positions,
        the third column represent the reward magnitude.
    simulator : Simulator or None, optional
        An optional simulator which overwrites observations for all positions.
    widget : pg.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the environment will be visualized.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    R : NDArray
        The reward function as a (N, 3) NumPy array.
        The first two columns represent the reward x and y positions,
        the third column represent the reward magnitude.
    room : sh.Polygon
        A Shapely polygon representing the environment's borders.
    spawn : sh.Polygon
        A Shapely polygon representing the agent's starting area.
    obstacles : list of sh.Polygon or None
        An optional list of Shapely polygons representing
        environmental obstacles.
    environment : sh.Polygon
        A Shapely polygon representing the final environment,
        i.e., room - obstacles.
    buffer : float
        Buffer distance used when computing collisions with
        the environment.
    state: NDArray
        The agent's current state (position and orientation).
    punish_wall: bool
        Flag indicating whether the agent receives a reward
        penalty for hitting a wall. False by default.
    robot_type: str
        The robot type that will be used.
        Can be either 'step' for which the agent can move
        in the four cardinal directions for a specified step size,
        or 'wheel' for which the agent will behave like a
        differential wheel robot (each wheel can be moved for
        a specified step size).
    observation_space : gym.Space
        The observation space. Per default it is a
        gym.spaces.Box containing the 2D position (`robot_type` is 'step')
        or 2D position and orientation (`robot_type` is 'wheel').
        When a simulator was provided `observation_space` is overwritten.
    simulator : Simulator or None
        An optional simulator which overwrites observations for all positions.
    action_space : gym.spaces.Discrete
        The action space.
    body_radius: float
        The radius of the agent.
    wheel_radius: float
        The radius of the agent's wheels.
    wheel_distance: float
        The distance for which each wheel travles per step.
    step_size: float
        The agent's step size.
    current_step: int
        Tracks the agent current step.
    rng : numpy.random.Generator
        A random number generator instance used for
        probablistic action selection.

    Examples
    --------

    Continuous 2D environment can be easily set up using Shapely
    and template function provided by CoBeL-RL. ::

        >>> import shapely as sh
        >>> from cobel.interface import Continuous2D
        >>> from cobel3.misc.continuous_tools import make_circle
        >>> room = sh.Polygon([(0., 0.), (1., 0.), (1., 1.),
        ...                    (0., 1.), (0., 0.)])
        >>> spawn = None
        >>> obstacles = [make_circle(np.array(0.5, 0.5), 0.05)]
        >>> rewards = np.array([[0.9, 0.9, 10.]])
        >>> env = Continuous2D('step', room, spawn,
        ...                     obstacles, rewards)

    """

    def __init__(
        self,
        robot_type: Literal['step', 'wheel'],
        room: sh.Polygon,
        spawn: None | sh.Polygon,
        obstacles: None | list,
        rewards: NDArray,
        simulator: None | Simulator = None,
        widget: None | pg.GraphicsLayoutWidget = None,
        rng: None | np.random.Generator = None,
    ) -> None:
        super().__init__(widget)
        self.rng = np.random.default_rng() if rng is None else rng
        # list of reward locations (along with reward magnitude)
        self.R = rewards
        # environmental variables
        self.room = room
        self.obstacles = [] if obstacles is None else obstacles
        self.environment = copy.deepcopy(self.room)
        for obstacle in self.obstacles:
            self.environment = self.environment.difference(obstacle)
        self.coords_env = np.array(self.environment.exterior.coords)
        self.coords_obs = [
            np.array(interior.coords) for interior in self.environment.interiors
        ]
        self.limits = np.array(self.environment.bounds)
        self.spawn: sh.Polygon
        if spawn is not None and self.environment.intersects(spawn):
            self.spawn = self.environment.intersection(spawn)
        else:
            self.spawn = self.environment
        self.buffer = -(10**-6)
        # the agent's current state in the environment (position + orientation)
        self.state = np.array([0.0, 0.0, 0.0])
        # if true, the agent receives a punishment for running into the walls
        self.punish_wall = False
        # robot type
        self.type = robot_type
        self.simulator = simulator
        # prepare observation and action spaces
        self.observation_space: gym.spaces.Space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,)
        )
        self.action_space = gym.spaces.Discrete(4)
        if self.type == 'wheel':
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))
            self.action_space = gym.spaces.Discrete(3)
        if self.simulator is not None:
            pass  # implement later
            # self.observation_space = self.simulator.observation_space
        # robot params
        self.body_radius = 0.05
        self.wheel_radius = 0.02
        self.wheel_distance = 0.1
        self.step_size = 0.015
        # initialize visualization
        self.initialize_visualization()
        # execute initial environment reset
        self.current_step = 0
        self.reset()

    def step(self, action: Action) -> StepTuple:
        """
        The interface's step function (compatible with Gymnasium's step function).

        Parameters
        ----------
        action : Action
            The action selected by the agent.

        Returns
        -------
        observation : Observation
            The observation of the new current state.
        reward : float
            The reward received.
        end_trial : bool
            A flag indicating whether the trial ended.
        truncated : bool
            A flag required by Gymnasium (not used).
        logs : dict
            The (empty) logs dictionary.
        """
        reward, end_trial, wall_hit = 0, False, False
        target_state = np.copy(self.state)
        # execute action
        if self.type == 'step':
            target_state[:2] += (
                np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])[
                    int(action)
                ]
                * self.step_size
            )
            self.state[:2] = self.clip_movement(self.state, target_state)
            self.observation = np.copy(self.state[:2])
        elif self.type == 'wheel':
            # in case that the agent moves straight we can
            # simplify the computation of the next state
            if action == 2:
                target_state += (
                    np.array([np.cos(target_state[2]), np.sin(target_state[2]), 0.0])
                    * self.step_size
                )
            # in case the wheels traveled different distances
            else:
                v = np.array([[0.0, 1], [1.0, 0.0]])[int(action)] * self.step_size
                omega = (v[1] - v[0]) / self.wheel_distance
                R = 0.5 * self.wheel_distance * (np.sum(v) / (v[1] - v[0]))  # noqa: N806
                ICC = target_state[:2] + np.array(  # noqa: N806
                    [-R * np.sin(target_state[2]), R * np.sin(target_state[2])]
                )
                M = np.array( # noqa: N806
                    [
                        [np.cos(omega), -np.sin(omega), 0.0],
                        [np.sin(omega), np.cos(omega), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                state = np.copy(target_state)
                state[:2] -= ICC
                target_state = np.matmul(M, state.reshape((3, 1))) + np.array(
                    [[ICC[0]], [ICC[1]], [target_state[2]]]
                )
                target_state = target_state.reshape((3,))
            self.state[:2] = self.clip_movement(self.state, target_state)
            self.state[2] = target_state[2] % (2 * np.pi)
            self.observation = np.copy(self.state)
        wall_hit = not np.array_equal(self.state[:2], target_state[:2])
        # determine reward and whether the episode should end
        distance_to_reward = np.sqrt(
            np.sum((self.R[:, :2] - self.state[:2]) ** 2, axis=1)
        )
        if np.sum(distance_to_reward <= self.body_radius * 2) != 0:
            reward = self.R[np.argmax(distance_to_reward <= self.body_radius * 2), 2]
            end_trial = True
        elif wall_hit and self.punish_wall:
            reward, end_trial = -10, False
        # update visualization
        self.update_visualization()
        self.current_step += 1

        return self.observation, reward, end_trial, end_trial, {}

    def reset(self) -> ResetTuple:
        """
        The interface's reset function (compatible with Gymnasium's reset function).

        Returns
        -------
        observation : Observation
            The observation of the new current state.
        logs : dict
            The (empty) logs dictionary.
        """
        # reset the agent to a random position with a random orientation
        limits = np.array(self.spawn.bounds)
        candidate = self.rng.uniform(low=limits[[0, 1]], high=limits[[2, 3]])
        while not self.spawn.contains(sh.Point(candidate)):
            candidate = self.rng.uniform(low=limits[[0, 1]], high=limits[[2, 3]])
        self.state[:2] = candidate
        self.state[2] = self.rng.uniform(0, 2 * np.pi)
        self.observation = np.copy(self.state)
        # we omit  orientation when using the "step" type
        if self.type == 'step':
            self.state[2] = 0.0
            self.observation = self.observation[:2]
        self.current_step = 0

        return self.observation, {}

    def clip_movement(self, current_state: NDArray, target_state: NDArray) -> tuple:
        """
        This function clips the movement so that the target state lies
        within the environment.

        Parameters
        ----------
        current_state : NDArray
            The current state.
        target_state : NDArray
            The target state after movement.

        Returns
        -------
        clipped_target : tuple
            The new target state after clipping.
        """
        # ensure that the current state lies within the environment
        # (necessary due to precision errors)
        current = sh.Point(current_state[:2])
        if not self.environment.contains(current) or not self.environment.touches(
            current
        ):
            current_state[:2] = nearest_points(
                self.environment.buffer(self.buffer), current
            )[0].coords[0]
        # compute intersections
        intersections = self.environment.intersection(
            sh.LineString([current_state[:2], target_state[:2]])
        )
        # handle edge cases with complex intersections
        if type(intersections) in [sh.MultiLineString, sh.GeometryCollection]:
            intersections = intersections.geoms[0]
        intersections = list(intersections.coords)

        return intersections[min(len(intersections) - 1, 1)]

    def initialize_visualization(self) -> None:
        """
        This function initializes the elements required for visualization.
        """
        if self.widget:
            # determine minimum and maximum coordinates
            self.coord_min = min(self.limits[0], self.limits[2])
            self.coord_max = max(self.limits[1], self.limits[3])
            # state information panel
            self.state_information_panel = self.widget.addPlot(
                title='State Information'
            )
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
            self.behavior_panel = self.widget.addPlot(title='Behavior')
            width = self.limits[2] - self.limits[0]
            height = self.limits[3] - self.limits[1]
            self.behavior_panel.setXRange(
                self.limits[0] - width * 0.05, self.limits[2] + width * 0.05
            )
            self.behavior_panel.setYRange(
                self.limits[1] - height * 0.05, self.limits[3] + height * 0.05
            )
            self.behavior_panel.setAspectLocked()
            self.markers = pg.ScatterPlotItem()
            coords = np.concatenate((self.R[:, :2], self.state[:2].reshape((1, 2))))
            brushes = [pg.mkBrush(color=(0, 255, 0)) for r in self.R] + [
                pg.mkBrush(color=(128, 128, 128))
            ]
            self.markers.setData(pos=coords, brush=brushes, size=10)
            self.behavior_panel.plot(
                self.coords_env[:, 0], self.coords_env[:, 1], pen=(255, 0, 0)
            )
            for coords in self.coords_obs:
                self.behavior_panel.plot(
                    coords[:, 0], coords[:, 1], pen=(255, 255, 255)
                )
            self.behavior_panel.addItem(self.markers)
            self.arrow: None | CogArrow = None
            if self.type == 'wheel':
                angle = np.rad2deg(self.state[2])
                if 180 > angle > 0:
                    angle += 2 * (90 - angle)
                elif 360 > angle > 180:
                    angle += 2 * (270 - angle)
                self.arrow = CogArrow(
                    angle=angle,
                    headLen=25,
                    tipAngle=30,
                    pen=pg.mkPen(color=(128, 128, 128)),
                    brush=pg.mkBrush(color=(128, 128, 128)),
                )
                self.arrow.set_data(self.state[0], self.state[1], angle)
                self.behavior_panel.addItem(self.arrow)

    def update_visualization(self) -> None:
        """
        This function updates the visualization.
        """
        if self.widget:
            # update state information panel
            self.coord_info.setText(str(self.state[:2]))
            self.orientation_info.setText(str(np.rad2deg(self.state[2])))
            # update behavior panel
            coords = np.concatenate((self.R[:, :2], self.state[:2].reshape((1, 2))))
            brushes = [pg.mkBrush(color=(0, 255, 0)) for r in self.R] + [
                pg.mkBrush(color=(128, 128, 128))
            ]
            self.markers.setData(pos=coords, brush=brushes, size=10)
            if type(self.arrow) is CogArrow:
                angle = np.rad2deg(self.state[2])
                if 180 >= angle >= 0:
                    angle += 2 * (90 - angle)
                elif 360 > angle > 180:
                    angle += 2 * (270 - angle)
                self.arrow.set_data(self.state[0], self.state[1], angle)
            # process changes
            if hasattr(qt, 'QtWidgets'):
                qt.QtWidgets.QApplication.instance().processEvents()  # type: ignore

    def get_position(self) -> NDArray:
        """
        This function returns the agent's position in the environment.

        Returns
        -------
        position : NDArray
            Numpy array containing the agent's position.
        """
        return np.copy(self.state[:2])
