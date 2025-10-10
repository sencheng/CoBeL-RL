# basic imports
import numpy as np
import gymnasium as gym
import pyqtgraph as pg  # type: ignore
import PyQt6 as qt
# framework imports
from .interface import Interface
from ..misc.visualization import CogArrow
# typing
from typing import Any, TypedDict
from numpy.typing import NDArray
from .interface import Action, StepTuple, ResetTuple


class WorldDict(TypedDict):
    width: int
    height: int
    states: int
    rewards: NDArray
    terminals: NDArray
    sas: NDArray
    starting_states: NDArray
    invalid_transitions: list[tuple[int, int]]
    invalid_states: list[int]
    wind: NDArray
    goals: list[int]
    coordinates: NDArray
    deterministic: bool


class Gridworld(Interface):
    """
    This class implements a gridworld environment.

    Parameters
    ----------
    world : dict
        A dictionary containing the gridworld's definition.
    widget : pg.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the environment will be visualized.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    world : dict
        A dictionary containing the gridworld's definition.
    observation_space : gym.spaces.Discrete
        The gridworld's observation space.
    action_space : gym.spaces.Discrete
        The action space associated with the gridworld.
    current_state : Observation
        The current state of the gridworld.
    current_coordinates : NDArray
        the coordinates of associated with `current_state`.
    rng : numpy.random.Generator
        A random number generator instance used for
        probablistic action selection.

    Examples
    --------

    Gridworld environments can be easily set up using
    the various template functions provided by CoBeL-RL. ::

        >>> from cobel.interface import Gridworld
        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> env = Gridworld(make_open_field(4, 4, 0, 1))

    """

    def __init__(
        self,
        world: WorldDict,
        widget: None | pg.GraphicsLayoutWidget = None,
        rng: None | np.random.Generator = None,
    ) -> None:
        super().__init__(widget)
        self.rng = np.random.default_rng() if rng is None else rng
        self.world = world
        self.observation_space = gym.spaces.Discrete(self.world['states'])
        self.action_space = gym.spaces.Discrete(4)
        self.current_state: int = 0
        self.init_visualization()
        self.reset()
        self.current_coordinates = self.world['coordinates'][self.current_state]

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
        transition_probabilities = self.world['sas'][self.current_state][action]
        if self.world['deterministic']:
            self.current_state = int(np.argmax(transition_probabilities))
        else:
            self.current_state = int(
                self.rng.choice(
                    np.arange(self.world['states']), p=transition_probabilities
                )
            )
        self.current_coordinates = self.world['coordinates'][self.current_state]
        reward = self.world['rewards'][self.current_state]
        end_trial = bool(self.world['terminals'][self.current_state])
        self.update_visualization()

        return self.current_state, reward, end_trial, False, {}

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
        self.current_state = int(self.rng.choice(self.world['starting_states']))
        self.current_coordinates = self.world['coordinates'][self.current_state]

        return self.current_state, {}

    def get_position(self) -> NDArray:
        """
        This function returns the agent's position in the environment.

        Returns
        -------
        position : NDArray
            A numpy array containing the agent's position.
        """
        return np.copy(self.current_coordinates)

    def init_visualization(self) -> None:
        """
        This function initializes the visualization of the gridworld environment.
        """
        if self.widget is not None:
            # prepare observation plot
            self.plot_observation = self.widget.addPlot(title='Observation')
            self.plot_observation.hideAxis('bottom')
            self.plot_observation.hideAxis('left')
            self.plot_observation.setXRange(-0.01, 0.01)
            self.plot_observation.setYRange(-0.1, 0.1)
            self.plot_observation.setAspectLocked()
            self.text_state = pg.TextItem(text='-1', anchor=(0, 0))
            self.text_coordinates = pg.TextItem(text='(-1, -1)', anchor=(0.25, -1))
            self.plot_observation.addItem(self.text_state)
            self.plot_observation.addItem(self.text_coordinates)
            # prepare gridworld plot
            self.plot_grid = self.widget.addPlot(title='Gridworld')
            self.plot_grid.hideAxis('bottom')
            self.plot_grid.hideAxis('left')
            self.plot_grid.getViewBox().setBackgroundColor((255, 255, 255))
            self.plot_grid.setXRange(-1, self.world['width'] + 1)
            self.plot_grid.setYRange(-1, self.world['height'] + 1)
            # build graph for the gridworld's background
            self.grid_background = []
            for j in range(self.world['height'] + 1):
                for i in range(self.world['width'] + 1):
                    node: list = [i, j, []]
                    if i - 1 >= 0:
                        node[2].append(j * (self.world['width'] + 1) + i - 1)
                    if i + 1 <= self.world['width']:
                        node[2].append(j * (self.world['width'] + 1) + i + 1)
                    if j - 1 >= 0:
                        node[2].append((j - 1) * (self.world['width'] + 1) + i)
                    if j + 1 <= self.world['height']:
                        node[2].append((j + 1) * (self.world['width'] + 1) + i)
                    self.grid_background.append(node)
            # determine node coordinates and edges
            self.grid_nodes, self.grid_edges = [], []
            for n, node in enumerate(self.grid_background):
                self.grid_nodes.append(node[:2])
                for neighbor in node[2]:
                    self.grid_edges.append([n, neighbor])
            # add graph item
            self.grid = pg.GraphItem(
                pos=np.array(self.grid_nodes),
                adj=np.array(self.grid_edges),
                pen=pg.mkPen(width=2),
                symbolPen=None,
                symbolBrush=None,
            )
            self.plot_grid.addItem(self.grid)
            # make hard outline
            self.outline_nodes = np.array(
                [
                    [-0.05, -0.05],
                    [-0.05, self.world['height'] + 0.05],
                    [self.world['width'] + 0.05, -0.05],
                    [self.world['width'] + 0.05, self.world['height'] + 0.05],
                ]
            )
            self.outline_edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
            self.outline = pg.GraphItem(
                pos=np.array(self.outline_nodes),
                adj=np.array(self.outline_edges),
                pen=pg.mkPen(color=(0, 0, 0), width=5),
                symbolPen=None,
                symbolBrush=None,
            )
            self.plot_grid.addItem(self.outline)
            # mark goal states
            self.goals = []
            for goal in self.world['goals']:
                goal_coordinates = self.world['coordinates'][goal] + 0.05
                goal_nodes = np.array(
                    [
                        goal_coordinates,
                        goal_coordinates + np.array([0, 0.9]),
                        goal_coordinates + np.array([0.9, 0]),
                        goal_coordinates + 0.9,
                    ]
                )
                goal_edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
                self.goals.append(
                    pg.GraphItem(
                        pos=goal_nodes,
                        adj=goal_edges,
                        pen=pg.mkPen(color=(0, 255, 0), width=5),
                        symbolPen=None,
                        symbolBrush=None,
                    )
                )
                self.plot_grid.addItem(self.goals[-1])
            # mark current state
            mark_coordinates = self.world['coordinates'][self.current_state]
            mark_nodes = np.array(
                [
                    mark_coordinates,
                    mark_coordinates + np.array([0, 0.9]),
                    mark_coordinates + np.array([0.9, 0]),
                    mark_coordinates + 0.9,
                ]
            )
            mark_edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
            self.mark_current = pg.GraphItem(
                pos=mark_nodes,
                adj=mark_edges,
                pen=pg.mkPen(color=(255, 0, 0), width=5),
                symbolPen=None,
                symbolBrush=None,
            )
            self.plot_grid.addItem(self.mark_current)
            # draw walls
            self.walls = []
            for transition in self.world['invalid_transitions']:
                first, second = np.amin(transition), np.amax(transition)
                coord = self.world['coordinates'][first]
                diff = np.abs(coord - self.world['coordinates'][second])
                nodes: np.typing.NDArray
                if diff[0] > 0:
                    nodes = np.array(
                        [coord + np.array([1, 0]), coord + np.array([1, 1])]
                    )
                else:
                    nodes = np.array(
                        [coord + np.array([0, 0]), coord + np.array([1, 0])]
                    )
                edges = np.array([[0, 1]])
                self.walls.append(
                    pg.GraphItem(
                        pos=nodes,
                        adj=edges,
                        pen=pg.mkPen(color=(0, 0, 0), width=5),
                        symbolPen=None,
                        symbolBrush=None,
                    )
                )
                self.plot_grid.addItem(self.walls[-1])
            # make arros for policy visualization
            self.arrows = []
            for state in self.world['coordinates']:
                self.arrows.append(
                    CogArrow(
                        angle=0.0,
                        headLen=20.0,
                        tipAngle=25.0,
                        tailLen=0.0,
                        brush=(255, 255, 0),
                    )
                )
                self.arrows[-1].set_data(state[0] + 0.5, state[1] + 0.5, 0.0)
                self.plot_grid.addItem(self.arrows[-1])
            # update visuals
            qt.QtWidgets.QApplication.instance().processEvents()  # type: ignore

    def update_visualization(self, logs: None | dict[str, Any] = None) -> None:
        """
        This function initializes the visualization of the gridworld environment.

        Parameters
        ----------
        logs : dict or None, optional
            The log dictionary.
        """
        if self.widget is not None:
            # update observation plot
            self.text_state.setText(str(self.current_state))
            self.text_coordinates.setText(
                '(%d, %d)' % tuple(np.flip(self.current_coordinates.astype(int)))
            )
            # update goal states
            for goal in self.goals:
                self.plot_grid.removeItem(goal)
            self.goals = []
            for goal in self.world['goals']:
                goal_coordinates = self.world['coordinates'][goal] + 0.05
                goal_nodes = np.array(
                    [
                        goal_coordinates,
                        goal_coordinates + np.array([0, 0.9]),
                        goal_coordinates + np.array([0.9, 0]),
                        goal_coordinates + 0.9,
                    ]
                )
                goal_edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
                self.goals.append(
                    pg.GraphItem(
                        pos=goal_nodes,
                        adj=goal_edges,
                        pen=pg.mkPen(color=(0, 255, 0), width=5),
                        symbolPen=None,
                        symbolBrush=None,
                    )
                )
                self.plot_grid.addItem(self.goals[-1])
            # update current state
            coords = self.world['coordinates'][self.current_state]
            self.mark_current.setPos(coords[0], coords[1] - self.world['height'] + 1)
            # update policy visualization
            if logs is not None:
                orientation_map = {0: 0.0, 1: 90.0, 2: 180.0, 3: 270.0}
                predictions = logs['agent'].predict_on_batch(
                    np.arange(self.world['states'])
                )
                for p, prediction in enumerate(predictions):
                    self.arrows[p].set_data(
                        self.world['coordinates'][p][0] + 0.5,
                        self.world['coordinates'][p][1] + 0.5,
                        orientation_map[int(np.argmax(prediction))],
                    )
            # update visuals
            qt.QtWidgets.QApplication.instance().processEvents()  # type: ignore
