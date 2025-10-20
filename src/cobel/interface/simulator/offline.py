# basic import
import socket
import numpy as np
import gymnasium as gym
# framework imports
from .simulator import Simulator, ImageInfo
from ..topology import Pose
from ..interface import Observation
# typing
from typing import Any
from numpy.typing import NDArray


class OfflineSimulator(Simulator):
    """
    This class implements a drop-in simulator replacement which
    provides pre-rendered observations.

    Parameters
    ----------
    scene : str
        The name of the scene that should loaded initially.
    observations : dict of Pose
        The dictionary containing pre-rendered observations
        for different agent poses.
    observation_space : gym.Space
        The observation space.
    executable : str of Node or None, optional
        The path to the simulator executable.

    Attributes
    ----------
    observations : dict of Pose
        The dictionary containing pre-rendered observations
        for different agent poses.

    Examples
    --------

    The offline simulator can be used with either pre-rendered
    observations or manually defined obersvations. ::

        >>> from cobel.interface.simulator.offline import OfflineSimulator
        >>> observations = {(0., 0., 0., 0., 0., 0.): np.eye(2)[0],
        ...                 (1., 0., 0., 0., 0., 0.): np.eye(2)[1]}
        >>> sim = OfflineSimulator(observations, gym.spaces.Box(0., 1., (2, )))

    """

    def __init__(
        self, observations: dict[Pose, Observation], observation_space: gym.spaces.Space
    ) -> None:
        self.agent_pose: Pose = list(observations)[0]
        self.observations = observations
        self.observation_space = observation_space

    def get_observation(self, pose: Pose) -> Observation:
        """
        This function returns the observation at a given pose.

        Parameters
        ----------
        pose : Pose
            The global pose that the agent is moved to.

        Returns
        -------
        observation : Observation
            The observation at the given pose.
        """
        return self.observations[pose]

    def connect_socket(self, connection_socket: socket.socket, port: int) -> None:
        """
        Dummy method required by parent class.
        """
        pass

    def receive(self, connection_socket: socket.socket, data_size: int) -> bytes:
        """
        Dummy method required by parent class.
        """
        return b'dummy'

    def receive_in_chunks(self, socket: socket.socket, chunk_size: int) -> bytes:
        """
        Dummy method required by parent class.
        """
        return b'dummy'

    def move_agent(self, x: float, y: float, yaw: float) -> tuple[NDArray, NDArray]:
        """
        Dummy method required by parent class.
        """
        return np.array(self.agent_pose), np.ones(1)

    def move_object(self, object_id: str, pose: Pose) -> None:
        """
        Dummy method required by parent class.
        """
        pass

    def set_illumination(self, light_source: str, color: NDArray) -> None:
        """
        Dummy method required by parent class.
        """
        pass

    def get_objects(self) -> dict[str, Any]:
        """
        Dummy method required by parent class.
        """
        return {'dummy': 'dummy'}

    def get_illumination(self, light_source: str) -> NDArray:
        """
        Dummy method required by parent class.
        """
        return np.ones(3)

    def get_image_info(self) -> ImageInfo:
        """
        Dummy method required by parent class.
        """
        return {'width': 1, 'height': 1, 'channels': 3, 'format': 'rgb'}

    def change_scene(self, scene: str) -> None:
        """
        Dummy method required by parent class.
        """
        pass

    def stop(self) -> None:
        """
        Dummy method required by parent class.
        """
        pass
