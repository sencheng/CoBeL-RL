# basic imports
import sys
import socket
import os
import subprocess
import json
import cv2
import numpy as np
import gymnasium as gym
# framework imports
from .simulator import Simulator, ImageInfo, Pose
# typing
from typing import Any
from numpy.typing import NDArray
from ..interface import Observation


class GodotSimulator(Simulator):
    """
    The Godot interface class. This class connects to the Godot
    environment and controls the flow of commands/data that goes
    to/comes from the Godot environment.

    Parameters
    ----------
    scene : str
        The name of the Godot scene that should loaded initially.
    executable : str of Node or None, optional
        The path to the Godot executable.
    ports : 3-tuple of int, default=(5000, 5001, 5002)
        The ports used for control, video and data connections.
    resize : 2-tuple of int or None, optional
        Optional resize dimensions for image observations.
    running : bool, default=False
        If true the starting of a new process will be skipped.

    Attributes
    ----------
    objects : dict
        A dictionary containing information, e.g., object IDs,
        about available simulation objects.
    actor_id : int
        ID of the object which represents the agent
        in the simulation.
    image_info : ImageInfo
        A dictionary containing information about the image
        size and format.
    resize : 2-tuple of int or None
        Optional resize dimensions for image observations.

    Examples
    --------

    If the appropriate environmental variable was set the
    Godot simulator can easily be started. To properly
    stop simulator calls the stop method.::

        >>> from cobel.interface.simulator.godot import GodotSimulator
        >>> sim = GodotSimulator('room.tscn')
        >>> sim.stop()

    Alternatively, one can provide the executable path directly. ::

        >>> sim = GodotSimulator('room.tscn', 'PATH/TO/GODOT')

    """

    def __init__(
        self,
        scene: str,
        executable: None | str = None,
        ports: tuple[int, int, int] = (5000, 5001, 5002),
        resize: None | tuple[int, int] = None,
        running: bool = False,
    ) -> None:
        assert len(set(ports)) == 3, 'The same port is used more than once!'
        self.scene = scene
        self.executable: str
        # ensure that the simulator executable was provided
        if not running:
            if executable is None:
                error_msg = (
                    'ERROR: Godot executable path was neither set or given as argument!'
                )
                assert 'GODOT_EXECUTABLE' in os.environ, error_msg
                self.executable = os.environ['GODOT_EXECUTABLE']
            else:
                self.executable = executable
            subprocess.Popen([self.executable, '--', '--ports=(%d,%d,%d)' % ports])
        # prepare sockets for communication with simulator
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # connect sockets
        self.connect_socket(self.control_socket, ports[0])
        print('Control connection has been initiated.')
        self.connect_socket(self.video_socket, ports[1])
        print('Video connection has been initiated.')
        self.connect_socket(self.data_socket, ports[2])
        print('Data connection has been initiated.')
        self.eod_string = '!simulationeod!'
        # load scene
        self.change_scene(self.scene)
        # define agent pose
        self.agent_pose: Pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # a dict storing information about objects present in the simulation
        self.get_objects()
        # retrieve object information and determine ID of the agent
        self.actor_id: int
        for _, godot_object in self.objects.items():
            if godot_object['name'] == 'Actor':
                self.actor_id = godot_object['id']
        print('Object information has been retrieved.')
        # retrieve image information
        self.image_info = self.get_image_info()
        print('Image information has been retrieved.')
        self.resize = resize
        self.observation_space: gym.spaces.Space
        if self.resize is None:
            self.observation_space = gym.spaces.Box(
                0,
                255,
                shape=(
                    self.image_info['width'],
                    self.image_info['height'],
                    self.image_info['channels'],
                ),
            )
        else:
            self.observation_space = gym.spaces.Box(
                0,
                255,
                shape=(self.resize[0], self.resize[1], self.image_info['channels']),
            )

    def receive(self, socket: socket.socket, data_size: int) -> bytes:
        """
        This function reads data with a specified size from a socket.

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
        # define buffer
        data = b''
        # read the data in chunks
        received_bytes = 0
        while received_bytes < data_size:
            data_chunk = socket.recv(data_size - received_bytes)
            data += data_chunk
            received_bytes += len(data_chunk)

        return data

    def receive_in_chunks(self, socket: socket.socket, chunk_size: int) -> bytes:
        """
        This function reads data in chunks from a socket.

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
        # define buffer
        data = b''
        # read the data in chunks
        while True:
            data_chunk = socket.recv(chunk_size)
            data += data_chunk
            if data[-16:] == b'!godotservereod!':
                break

        return data[:-16]

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
        return self.move_agent(pose[0], pose[1], pose[3])[1]

    def move_agent(self, x: float, y: float, yaw: float) -> tuple[NDArray, NDArray]:
        """
        This function propels the simulation. It uses teleportation to
        guide the agent/robot directly by means of global x, y, yaw values.

        Parameters
        ----------
        x : float
            The global x position to teleport to.
        y : float
            The global y position to teleport to.
        yaw : float
            The global yaw value to teleport to.

        Returns
        -------
        pose_data : NDArray
            The pose observation received from the simulation.
        image_data : NDArray
            The image observation received from the simulation.
        """
        # send the actuation command to the virtual robot/agent
        send_str = json.dumps(
            {
                'command': 'step_simulation_without_physics',
                'param': [x, -y, np.deg2rad(yaw)],
            }
        )
        self.control_socket.send(send_str.encode('utf-8'))
        self.control_socket.recv(50)
        # retrieve image data from the robot
        self.control_socket.send(
            json.dumps({'command': 'get_image', 'param': []}).encode('utf-8')
        )
        width = self.image_info['width']
        height = self.image_info['height']
        channels = self.image_info['channels']
        image_bytes = self.receive_in_chunks(
            self.video_socket, width * height * channels
        )
        image_data: NDArray = np.frombuffer(image_bytes, dtype=np.uint8)
        image_data = np.reshape(image_data, (height, width, channels))
        if self.resize is not None:
            image_data = cv2.resize(image_data, self.resize)
        self.video_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)
        # retrieve pose data from the robot
        send_str = json.dumps({'command': 'get_pose', 'param': [self.actor_id]})
        self.control_socket.send(send_str.encode('utf-8'))
        pose_data = json.loads(
            self.receive_in_chunks(self.data_socket, 1024).decode('utf-8')
        )['pose']
        pose_data[2] *= -1
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)
        # update robot's/agent's pose
        self.robot_pose = pose_data
        # update environmental information
        self.pose = pose_data

        return pose_data, image_data

    def move_object(self, object_id: int | str, pose: Pose) -> None:
        """
        This function changes the pose of a specified object.

        Parameters
        ----------
        object_id : int or str
            The name/ID of the object.
        pose : Pose
            The object's new pose.
        """
        send_str = json.dumps({'command': 'set_pose', 'param': [object_id, pose]})
        self.control_socket.send(send_str.encode('utf-8'))
        self.control_socket.recv(50)

    def set_illumination(self, light_source: int | str, color: NDArray) -> None:
        """
        This function sets the color of a specified light source.

        Parameters
        ----------
        light_source : int or str
            The name/ID of the light source to change.
        color : NDArray
            The RGB values of the light source (as [red, green, blue]).
        """
        # send the request for illumination change to Godot
        send_str = json.dumps(
            {
                'command': 'set_illumination',
                'param': [light_source, self.hex_color(color)],
            }
        )
        self.control_socket.send(send_str.encode('utf-8'))
        self.control_socket.recv(50)

    def get_illumination(self, light_source: int | str) -> NDArray:
        """
        This function gets the color of a specified light source.

        Parameters
        ----------
        light_source : int or str
            The name/ID of the light source to change.

        Returns
        -------
        color : NDArray
            The RGB values of the light source (as [red, green, blue]).
        """
        # send the request for illumination information to Godot
        send_str = json.dumps({'command': 'get_illumination', 'param': [light_source]})
        self.control_socket.send(send_str.encode('utf-8'))
        color = json.loads(
            self.receive_in_chunks(self.data_socket, 1024).decode('utf-8')
        )['color']
        color = (np.array(color) * 255 + 0.1).astype(int)
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)

        return color

    def get_objects(self) -> dict[str, Any]:
        """
        This function retrieves information about all object present in the simulation.

        Returns
        -------
        objects : dict of Any
            Dictionary containing the object information.
        """
        # clear dictionary
        self.objects = {}
        # send the request for object information to Godot
        self.control_socket.send(
            json.dumps({'command': 'get_objects', 'param': []}).encode('utf-8')
        )
        self.objects = json.loads(
            self.receive_in_chunks(self.data_socket, 1024).decode('utf-8')
        )
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)

        return self.objects

    def get_image_info(self) -> ImageInfo:
        """
        This function retrieves image information from the simulation.

        Returns
        -------
        image_info : ImageInfo
            A dictionary containing information about the images
            rendered by Godot (i.e. dimensions, channels, format).
        """
        # send the request for image information change to Godot
        self.control_socket.send(
            json.dumps({'command': 'get_image_info', 'param': []}).encode('utf-8')
        )
        # retrieve image information
        image_info = json.loads(
            self.receive_in_chunks(self.data_socket, 1024).decode('utf-8')
        )
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)

        return image_info

    def change_scene(self, scene_name: str) -> None:
        """
        This function changes the current scene.

        Parameters
        ----------
        scene_name : str
            The name of the scene to be loaded.
        """
        # send the request for scene change to Godot
        send_str = json.dumps({'command': 'change_scene', 'param': [scene_name]})
        self.control_socket.send(send_str.encode('utf-8'))
        self.control_socket.recv(50)

    def set_texture(self, mesh: int | str, texture: str, surface: int = -1) -> None:
        """
        This function changes the albedo texture of a given mesh object's material.

        Parameters
        ----------
        mesh : int or str
            The name/ID of the mesh object.
        texture : str
            The path to the texture file (relative to the simulator
            executable).
        surface : int, default=-1
            The surface whose texture will be changed. When set to -1
            the texture is applied to all surfaces.
        """
        # send the request for scene change to Godot
        send_str = json.dumps(
            {'command': 'set_texture', 'param': [mesh, texture, surface]}
        )
        self.control_socket.send(send_str.encode('utf-8'))
        self.control_socket.recv(50)

    def set_color(
        self,
        mesh: int | str,
        color: tuple[float, float, float, float],
        surface: int = -1,
    ) -> None:
        """
        This function changes the albedo color of a given
        mesh object's material. Note: This color will be multiplied
        with any image texture already applied.

        Parameters
        ----------
        mesh : int or str
            The name/ID of the mesh object.
        color : 4-tuple of float
            The RGBA color values.
        surface : int, default=-1
            The surface whose color will be changed. When set to
            -1 the color is applied to all surfaces.
        """
        assert np.amin(color) >= 0.0 and np.amax(color) <= 1.0, (
            'Color values must lie within the interval [0, 1]!'
        )
        # send the request for scene change to Godot
        send_str = json.dumps({'command': 'set_color', 'param': [mesh, color, surface]})
        self.control_socket.send(send_str.encode('utf-8'))
        self.control_socket.recv(50)

    def echo(self, message: str) -> None:
        """
        This function sends a message to the godot simulation.

        Parameters
        ----------
        message : str
            The message.
        """
        # send the request for message echo to Godot
        send_str = json.dumps({'command': 'echo', 'param': [message]})
        self.control_socket.send(send_str.encode('utf-8'))
        self.control_socket.recv(50)

    def stop(self) -> None:
        """
        This function shuts down Godot.
        """
        try:
            self.control_socket.send(
                json.dumps({'command': 'stop', 'param': []}).encode('utf-8')
            )
        except (TimeoutError, OSError):
            print(sys.exc_info()[1])

    def hex_color(self, color: NDArray) -> str:
        """
        This function converts RGB to hex.

        Parameters
        ----------
        color_rgb : NDArray
            The color in RGB.

        Returns
        -------
        color_hex : str
            The color in hex.
        """
        hex_color = '#'
        for value in color:
            hex_color += hex(int(value))[2:].zfill(2)

        return hex_color
