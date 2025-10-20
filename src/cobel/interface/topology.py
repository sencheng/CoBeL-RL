# basic imports
import copy
import numpy as np
import PyQt6 as qt
import pyqtgraph as pg  # type: ignore
import gymnasium as gym
# framework imports
from .interface import Interface
# typing
from typing import TypedDict
from numpy.typing import NDArray
from .interface import Action, Observation, StepTuple, ResetTuple
from .simulator.simulator import Simulator

Pose = tuple[float, float, float, float, float, float]
NodeID = str


class Node(TypedDict):
    id: NodeID
    pose: Pose
    reward: float
    terminal: bool
    neighbors: list[NodeID]


class Topology(Interface):
    """
    This class implements a generic topology interface.

    Parameters
    ----------
    nodes : dict of Node
        A dictionary containing the topology nodes.
    starting_nodes : list of Node or None, optional
        An optional list containing the starting nodes.
    simulator : Simulator or None, optional
        An optional simulator which overwrites observations for all nodes.
    widget : pg.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the environment will be visualized.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    nodes : dict of Node
        A dictionary containing the topology nodes.
    starting_nodes : list of Node
        An optional list containing the starting nodes.
        If none were provided in `starting_nodes` all non-terminal
        nodes become starting nodes.
    current_node : NodeID
        The ID of the current node.
    observation_space : gym.Space
        The observation space associated with the topology environment.
        Per default `observation_space` is gym.spaces.Box and represents
        a node's pose, i.e., position + rotation.
        If a simulator was provided then the `observation_space` will
        be overwritten.
    action_space : gym.spaces.Discrete
        The action space associated with the topology environment.
    simulator : Simulator or None
        An optional simulator which overwrites observations for all nodes.
    observation : Observation
        The current observation.
    rng : numpy.random.Generator
        A random number generator instance used for
        probablistic action selection.

    Examples
    --------

    Topology environments can be easily set up using
    the various template functions provided by CoBeL-RL. ::

        >>> from cobel.interface import Topology
        >>> from cobel.misc.topology_tools import linear_track
        >>> env = Topology(linear_track(10, 2))

    """

    def __init__(
        self,
        nodes: dict[NodeID, Node],
        starting_nodes: None | list[NodeID] = None,
        simulator: Simulator | None = None,
        widget: None | pg.GraphicsLayoutWidget = None,
        rng: None | np.random.Generator = None,
    ) -> None:
        super().__init__(widget)
        self.rng = np.random.default_rng() if rng is None else rng
        self.nodes = nodes
        self.starting_nodes: list[NodeID]
        # if no starting nodes were defined all non-terminal nodes
        # are considered starting nodes
        if starting_nodes is None:
            self.starting_nodes = [
                node_id
                for node_id, node in self.nodes.items()
                if (not node['terminal'])
            ]
        else:
            self.starting_nodes = starting_nodes
        self.current_node: NodeID
        self.current_node = self.rng.choice(self.starting_nodes)
        self.action_space = gym.spaces.Discrete(
            len(self.nodes[self.current_node]['neighbors'])
        )
        self.simulator = simulator
        self.observation_space: gym.spaces.Space
        if self.simulator is None:
            self.observation_space = gym.spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.inf, 0.0, 0.0, 0.0]),
                high=np.array([np.inf, np.inf, np.inf, 360.0, 360.0, 360.0]),
                dtype=np.float64,
            )
        else:
            self.observation_space = self.simulator.observation_space
        self.observation: Observation
        self.init_visualization()

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
        action = int(action)
        assert type(action) is int, 'Invalid action type!'
        self.current_node = self.nodes[self.current_node]['neighbors'][action]
        pose = self.nodes[self.current_node]['pose']
        reward = self.nodes[self.current_node]['reward']
        end_trial = self.nodes[self.current_node]['terminal']
        self.update_visualization()

        return self.get_observation(pose), reward, end_trial, end_trial, {}

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
        self.current_node = self.rng.choice(self.starting_nodes)

        return self.get_observation(self.nodes[self.current_node]['pose']), {}

    def get_observation(self, pose: Pose) -> Observation:
        """
        This function retrieves the observation for the current node's pose.

        Parameters
        ----------
        pose : Pose
            The current node's pose.

        Returns
        -------
        observation : Observation
            The observation at the current node.
        """
        if self.simulator is None:
            self.observation = np.array(pose)
        else:
            self.observation = self.simulator.get_observation(pose)

        return copy.deepcopy(self.observation)

    def get_position(self) -> NDArray:
        """
        This function returns the agent's position in the environment.

        Returns
        -------
        position : NDArray
            A numpy array containing the agent's position.
        """
        return np.array(self.nodes[self.current_node]['pose'])

    def init_visualization(self) -> None:
        """
        This function initializes the visualization of the interface.
        """
        if self.widget is not None:
            # check if observations can be visualized
            self.observation_type = 'unknown'
            if type(self.observation_space) is gym.spaces.Box:
                if len(self.observation_space.shape) == 1:
                    self.observation_type = 'sensor'
                elif len(self.observation_space.shape) == 2:
                    self.observation_type = 'grey'
                elif len(self.observation_space.shape) == 3:
                    self.observation_type = 'color'
            # determine minimum and maximum coordinates
            self.coordinates_min = [10.0**6, 10.0**6]
            self.coordinates_max = [-(10.0**6), -(10.0**6)]
            self.coordinates: list[tuple[float, float]] = []
            self.id_map: dict[NodeID, int] = {}
            # state information panel
            self.panel_information = self.widget.addPlot(title='State Information')
            self.panel_information.hideAxis('bottom')
            self.panel_information.hideAxis('left')
            self.panel_information.setXRange(0, 1)
            self.panel_information.setYRange(0, 1)
            self.panel_information.setAspectLocked()
            self.info_state = pg.TextItem(text='-1')
            self.label_state = pg.TextItem(text='Current Node:')
            self.info_pose = pg.TextItem(text='(0, 0, 0, 0, 0, 0)')
            self.label_pose = pg.TextItem(text='Node Pose:')
            self.font = pg.Qt.QtGui.QFont()
            self.font.setPixelSize(20)
            self.info_state.setFont(self.font)
            self.info_pose.setFont(self.font)
            self.label_state.setFont(self.font)
            self.label_pose.setFont(self.font)
            self.info_state.setPos(0.1, 0.95)
            self.label_state.setPos(0.1, 1.0)
            self.info_pose.setPos(0.1, 0.8)
            self.label_pose.setPos(0.1, 0.85)
            self.info_observation: pg.TexItem | pg.ImageItem
            if self.observation_type == 'unknown':
                self.info_observation = pg.TextItem('unknown')
                self.info_observation.setFont(self.font)
                self.info_observation.setPos(0.1, 0.55)
            else:
                self.info_observation = pg.ImageItem()
                self.info_observation.setOpts(axisOrder='row-major')
            self.label_observation = pg.TextItem('Current Observation:')
            self.label_observation.setFont(self.font)
            self.label_observation.setPos(0.1, 0.6)
            self.panel_information.addItem(self.info_state)
            self.panel_information.addItem(self.label_state)
            self.panel_information.addItem(self.info_pose)
            self.panel_information.addItem(self.label_pose)
            self.panel_information.addItem(self.info_observation)
            self.panel_information.addItem(self.label_observation)
            # determine connections between nodes
            self.connections: list[tuple[int, int]] = []
            self.goals: list[int] = []
            for n, (id_1, node_1) in enumerate(self.nodes.items()):
                self.id_map[id_1] = n
                if node_1['terminal']:
                    self.goals.append(n)
                self.coordinates.append(node_1['pose'][:2])
                self.coordinates_min[0] = min(
                    self.coordinates_min[0], node_1['pose'][0]
                )
                self.coordinates_min[1] = min(
                    self.coordinates_min[1], node_1['pose'][1]
                )
                self.coordinates_max[0] = max(
                    self.coordinates_max[0], node_1['pose'][0]
                )
                self.coordinates_max[1] = max(
                    self.coordinates_max[1], node_1['pose'][1]
                )
                for m, (id_2, _) in enumerate(self.nodes.items()):
                    if id_2 in node_1['neighbors'] and (n, m) not in self.connections:
                        self.connections.append((n, m))
            # behavioral panel
            self.panel_behavior = self.widget.addPlot(title='Behavior')
            width = self.coordinates_max[0] - self.coordinates_min[0]
            self.panel_behavior.setXRange(
                self.coordinates_min[0] - width * 0.05,
                self.coordinates_max[0] + width * 0.05,
            )
            self.panel_behavior.setYRange(
                self.coordinates_min[1] - width * 0.05,
                self.coordinates_max[1] + width * 0.05,
            )
            self.panel_behavior.setAspectLocked()
            self.graph = pg.GraphItem()
            brushes = [pg.mkBrush(color=(128, 128, 128))] * len(self.nodes)
            for goal in self.goals:
                brushes[goal] = pg.mkBrush(color=(0, 255, 0))
            self.graph.setData(
                pos=np.array(self.coordinates),
                adj=np.array(self.connections),
                symbolBrush=brushes,
            )
            self.panel_behavior.addItem(self.graph)

    def update_visualization(self) -> None:
        """
        This function updates the visualization of the interface.
        """
        if self.widget is not None:
            # update state information panel
            self.info_state.setText(self.current_node)
            self.info_pose.setText(
                '(%.03f, %.03f, %.03f, %.03f, %.03f, %.03f)'
                % self.nodes[self.current_node]['pose']
            )
            if self.observation_type == 'unknown':
                self.info_observation.setText('Unknown format.')
            else:
                assert type(self.observation) is np.ndarray
                obs_data: NDArray = np.copy(self.observation)
                if self.observation_type == 'sensor':
                    obs_data = np.tile(obs_data, 3).reshape(
                        (1, obs_data.shape[0], 3), order='F'
                    )
                elif self.observation_type == 'grey':
                    obs_data = np.tile(
                        obs_data.reshape(obs_data.shape[0], obs_data.shape[1], 1), 3
                    )
                self.info_observation.setImage(np.flip(obs_data, axis=0))
                self.info_observation.setRect(
                    qt.QtCore.QRectF(
                        0.0,
                        0.52 - obs_data.shape[0] / obs_data.shape[1],
                        1.0,
                        obs_data.shape[0] / obs_data.shape[1],
                    )
                )
            # update behavioral panel
            brushes = [pg.mkBrush(color=(128, 128, 128))] * len(self.nodes)
            for goal in self.goals:
                brushes[goal] = pg.mkBrush(color=(0, 255, 0))
            brushes[self.id_map[self.current_node]] = pg.mkBrush(color=(255, 0, 0))
            self.graph.setData(
                pos=np.array(self.coordinates),
                adj=np.array(self.connections),
                symbolBrush=brushes,
            )
            if hasattr(qt, 'QtWidgets'):
                instance = qt.QtWidgets.QApplication.instance()
                assert type(instance) is qt.QtWidgets.QApplication
                instance.processEvents()
