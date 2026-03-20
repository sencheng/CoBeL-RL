# basic imports
import os
import abc
import socket
import gymnasium as gym

# typing
from typing import Any, TypedDict
from numpy.typing import NDArray
from ..interface import Observation


class ImageInfo(TypedDict):  # noqa: D101
    width: int
    height: int
    channels: int
    format: str


Pose = tuple[float, float, float, float, float, float]


class Simulator(abc.ABC):
    """
    The abstract simulator class.

    Parameters
    ----------
    scene : str
        The name of the scene that should loaded initially.
    executable : str or None, optional
        The path to the simulator executable.

    Attributes
    ----------
    control_socket : socket.socket
        Socket used for sending commands to the simulator.
    video_socket : socket.socket
        Socket used for retrieving video data from the simulator.
    data_socket : socket.socket
        Socket used for sending and receiving data to and from the simulator.
    agent_pose : cobel.interface.topology.Pose
        The agent's current pose.
    observation_space : gymnasium.spaces.Space
        The observation space.
    eod_string : str
        A string indicating the end of data that was
        sent via a web socket.
    agent_pose : cobel.interface.topology.Pose
        The agent's current pose.

    """

    def __init__(self, scene: str, executable: None | str = None) -> None:
        self.scene = scene
        self.executable = executable
        # ensure that the simulator execu was provided
        if executable is None:
            error_msg = (
                'ERROR: Simulator executable path was neither set or given as argument!'
            )
            assert 'SIMULATOR_EXECUTABLE' in os.environ, error_msg
        # prepare sockets for communication with simulator
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # connect sockets
        self.connect_socket(self.control_socket, 5000)
        print('Control connection has been initiated.')
        self.connect_socket(self.video_socket, 5001)
        print('Video connection has been initiated.')
        self.connect_socket(self.data_socket, 5002)
        print('Data connection has been initiated.')
        self.eod_string = '!simulationeod!'
        # load scene
        self.change_scene(self.scene)
        # define agent pose
        self.agent_pose: Pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.observation_space: gym.spaces.Space

    def connect_socket(self, connection_socket: socket.socket, port: int) -> None:
        """
        Start a connection for a specified socket.

        Parameters
        ----------
        connection_socket : socket.socket
            The socket for which a connection will be started.
        port : int
            The port used for the connection.
        """
        available = False
        while not available:
            try:
                connection_socket.connect(('localhost', port))
                connection_socket.setblocking(True)
                available = True
            except ConnectionError:
                pass
        connection_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def receive(self, socket: socket.socket, data_size: int) -> bytes:
        """
        Read data with a specified size from a specified socket.

        Parameters
        ----------
        socket : socket.socket
            The socket to read the data from.
        data_size : int
            The size of the data to read.

        Returns
        -------
        data : bytes
            The data as a byte string.
        """
        # prepare buffer
        data = b''
        # read data in chunks
        received_bytes = 0
        while received_bytes < data_size:
            data_chunk = socket.recv(data_size - received_bytes)
            data += data_chunk
            received_bytes += len(data_chunk)

        return data

    def receive_in_chunks(self, socket: socket.socket, chunk_size: int) -> bytes:
        """
        Read data in chunks from a specified socket.

        Parameters
        ----------
        socket : socket.socket
            The socket to read the data from.
        chunk_size : int
            The size of the chunk to read.

        Returns
        -------
        data : bytes
            The data as a byte string.
        """
        # prepare buffer
        data = b''
        # read data in chunks
        eod = len(self.eod_string)
        eod_bytes = self.eod_string.encode('utf-8')
        while True:
            data_chunk = socket.recv(chunk_size)
            data += data_chunk
            if data[-eod:] == eod_bytes:
                break

        return data[:-eod]

    def get_observation(self, pose: Pose) -> Observation:
        """
        Return the observation at a given pose.

        Parameters
        ----------
        pose : cobel.interface.topology.Pose
            The global pose that the agent is moved to.

        Returns
        -------
        observation : cobel.interface.interface.Observation
            The observation at the given pose.
        """
        return self.move_agent(pose[0], pose[1], pose[3])[1]

    @abc.abstractmethod
    def move_agent(self, x: float, y: float, yaw: float) -> tuple[NDArray, NDArray]:
        """
        Change the position and orientation of the agent.

        Parameters
        ----------
        x : float
            The global x position that the agent is moved to.
        y : float
            The global y position that the agent is moved to.
        yaw : float
            The global yaw value that the agent is rotated to.

        Returns
        -------
        pose : numpy.ndarray
            The agent's new pose.
        image : numpy.ndarray
            The image data at the agent's new pose.
        """
        pass

    @abc.abstractmethod
    def move_object(self, object_id: str, pose: Pose) -> None:
        """
        Change the pose of a specified object.

        Parameters
        ----------
        object_id : str
            The name/ID of the object.
        pose : cobel.interface.topology.Pose
            The object's new pose.
        """
        pass

    @abc.abstractmethod
    def set_illumination(self, light_source: str, color: NDArray) -> None:
        """
        Set the color of a specified light source.

        Parameters
        ----------
        light_source : str
            The name/ID of the light source.
        color : numpy.ndarray
            The new color values.
        """
        pass

    @abc.abstractmethod
    def get_objects(self) -> dict[str, Any]:
        """
        Retrieve information about all objects present in the simulation.

        Returns
        -------
        object_info : dict
            The dictionary containing information about the simulation objects.
        """
        pass

    @abc.abstractmethod
    def get_illumination(self, light_source: str) -> NDArray:
        """
        Retrieve the color of a specified light source.

        Parameters
        ----------
        light_source : str
            The name/ID of the light source.

        Returns
        -------
        color : numpy.ndarray
            The retrieved color values.
        """
        pass

    @abc.abstractmethod
    def get_image_info(self) -> ImageInfo:
        """
        Change the current scene.

        Returns
        -------
        image_info : cobel.interface.simulator.simulator.ImageInfo
            The dictionary containing information about the images rendered.
        """
        pass

    @abc.abstractmethod
    def change_scene(self, scene: str) -> None:
        """
        Change the current scene.

        Parameters
        ----------
        scene : str
            The name of the scene to be loaded.
        """
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """Shut down the simulator."""
        pass
