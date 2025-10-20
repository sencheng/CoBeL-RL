# basic imports
import os
import math
import json
import socket
import platform
import subprocess
import numpy as np
import gymnasium as gym
import cv2
# framework imports
from cobel.interface.simulator.simulator import Simulator, ImageInfo, Pose
from cobel.interface.interface import Observation
# typing
from typing import Any
from numpy.typing import NDArray


class UnitySimulator(Simulator):
    def __init__(
        self,
        scene_name: str,
        executable: None | str = None,
        resize: None | tuple[int, int] = None,
        running: bool = False,
        ports: None | tuple[int, int, int] = None,
        nb_agents: int = 1,
        batch_mode: bool = True,
        rng: None | np.random.Generator = None,
    ) -> None:
        """
        The Unity interface class. This class connects to the Unity
        environment and controls the flow of commands/data to/from Unity.

        Parameters
        ----------
        scene : str
            The name of the Unity scene that should loaded initially.
        executable : str of Node or None, optional
            The path to the Unity executable.
        resize : 2-tuple of int or None, optional
            Optional resize dimensions for image observations.
        running : bool, default=False
            If true the starting of a new process will be skipped.
        ports : 3-tuple of int or None, optional
            The ports to connect to the Unity simulator. If None, random
            ports will be tried out until a working set was found.
            Must not be None when connecting to a already running process.
        nb_agents : int, default=1
            The number of agents in the Unity scene. This is used to determine
            how many agents are created when the simulator launches.
        batch_mode : bool, default=True
            A flag indicating whether batch mode should be used to prevent
            unnecessary rendering (will result in the Unity window being black).
        rng : numpy.random.Generator or None, optional
            An optional random number generator instance.
            If none is provided a new instance will be created.

        Attributes
        ----------
        image_info : ImageInfo
            A dictionary containing information about the image
            size and format.
        resize : 2-tuple of int or None
            Optional resize dimensions for image observations.

        Examples
        --------

        If the appropriate environmental variable was set the
        Unity simulator can easily be started. To properly
        stop simulator calls the stop method.::

            >>> from cobel.interface.simulator.unity import UnitySimulator
            >>> sim = UnitySimulator('room.')
            >>> sim.stop()

        Alternatively, one can provide the executable path directly. ::

            >>> sim = UnitySimulator('room', 'PATH/TO/UNITY')

        """
        portrange_min = 1024
        portrange_max = 49152  # max - 2 because of the 3 sockets used
        nb_retries = 25
        self.rng = np.random.default_rng() if rng is None else rng
        assert nb_agents > 0, 'The agent must contain at least one agent.'
        if running:
            assert ports is not None, 'Ports must be within the port range.'
        if ports is not None:
            nb_retries = 1
            assert len(set(ports)) == 3, 'Same port used twice.'
            assert (
                portrange_max > min(ports) >= portrange_min
                and portrange_max > max(ports) >= portrange_min
            )
        else:
            port = int(self.rng.integers(portrange_min, portrange_max))
            ports = (port, port + 1, port + 2)
        for i in range(nb_retries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
                    temp_socket.bind(('localhost', ports[0]))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
                    temp_socket.bind(('localhost', ports[1]))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
                    temp_socket.bind(('localhost', ports[2]))
                break
            except OSError:
                if i == nb_retries - 1:
                    raise RuntimeError(
                        f'Unable to find a free port after {nb_retries} attempts.'
                    ) from None
                else:
                    port = int(self.rng.integers(portrange_min, portrange_max))
                    ports = (port, port + 1, port + 2)
                    continue
        # ensure that the simulator executable was provided
        if executable is None:
            error_msg: str = (
                'ERROR: Unity executable path was neither set or given as argument!'
            )
            assert 'UNITY_EXECUTABLE' in os.environ, error_msg
            self.executable = os.environ['UNITY_EXECUTABLE']
        else:
            self.executable = executable
        # start simulator process
        process_cmd = [f'--port={port}', f'--nAgents={nb_agents}']
        if batch_mode:
            process_cmd = ['-batchmode'] + process_cmd
        if 'darwin' in platform.uname():  # Check if platform is macOS
            process_cmd = ['open', '-n', '-a', self.executable, '--args'] + process_cmd
        else:
            process_cmd = [self.executable] + process_cmd
        self.process = subprocess.Popen(
            process_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # connect sockets
        self.control_socket: socket.socket
        self.video_socket: socket.socket
        self.data_socket: socket.socket
        sockets_connected: bool = False
        while not sockets_connected:
            try:
                # init
                self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # connect
                self.connect_socket(self.control_socket, port)
                print(f'Control connection has been initiated. Running on port {port}')
                self.connect_socket(self.video_socket, port + 1)
                print(
                    f'Video connection has been initiated. Running on port {port + 1}'
                )
                self.connect_socket(self.data_socket, port + 2)
                print(f'Data connection has been initiated. Running on port {port + 2}')
                sockets_connected = True
            except (TimeoutError, OSError, ConnectionRefusedError):
                # if socket connects fails (120s), print error, retry
                print("Couldn't connect to unity simulator, retrying...")
        # load scene
        self.change_scene(scene_name)
        # get image info of agent
        self.image_info: ImageInfo = self.get_image_info()
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
        print('Unity simulator initialized!')

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
            if data[-16:] == b'!unityservereod!':
                break

        return data[:-16]

    def echo(self, message: str) -> str:
        """
        This function sends a message to the unity simulator and
        recieves the echo

        Parameters
        ----------
        message : str
            The message to be sent.

        Returns
        -------
        recieved_echo : str
            The echo that came back from the simulator
        """
        # convert message into formatted json command and send it
        self.control_socket.send(
            json.dumps({'command': 'echo', 'param': [message]}).encode('utf-8')
        )
        recieved_echo: str = self.control_socket.recv(1024).decode('utf-8')

        return recieved_echo

    def change_scene(self, scene: str) -> None:
        """
        This function sends the path to a scene to the simulator
        and the command to change to this scene.

        Parameters
        ----------
        scene : str
            The path to the scene to change to.
        """
        # convert scene path into formatted json command and send it
        self.control_socket.send(
            json.dumps({'command': 'change_scene', 'param': [scene]}).encode('utf-8')
        )
        self.control_socket.recv(50)

    def get_image_info(self, agent: int = 0) -> ImageInfo:
        """
        This function retrieves image information from the simulation.

        Parameters
        ----------
        agent : int, default=0
            The agent from which image information should be retrieved.

        Returns
        -------
        image_info : ImageInfo
            A dictionary containing information about the images
            rendered by Unity (i.e. dimensions, channels, format).
        """
        # send the request for image information to unity
        self.control_socket.send(
            json.dumps({'command': 'get_image_info', 'param': [agent]}).encode('utf-8')
        )
        # retrieve image information
        image_info: ImageInfo = json.loads(
            self.receive_in_chunks(self.data_socket, 1024).decode('utf-8')
        )
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)

        return image_info

    def get_observation(self, pose: Pose, agent: int = 0) -> Observation:
        """
        This function returns the observation at a given pose.

        Parameters
        ----------
        pose : Pose
            The global pose that the agent is moved to.
        agent : int, default=0
            The agent that will be moved and from which
            an observation will be retrieved.

        Returns
        -------
        observation : Observation
            The observation at the given pose.
        """
        return self.move_agent(pose[0], pose[1], pose[3], agent=agent)[1]

    def move_agent(
        self, x: float, y: float, yaw: float, agent: int = 0
    ) -> tuple[NDArray, NDArray]:
        """
        This function propels the simulation. It uses teleportation to
        guide the agent directly by means of global x, y, yaw values.

        Parameters
        ----------
        x : float
            The global x position to teleport to.
        y : float
            The global y position to teleport to.
        yaw : float
            The global yaw value to teleport to.
        agent : int, default=0
            The agent that will be moved.

        Returns
        -------
        pose_data : NDArray
            The pose observation received from the simulation.
        image_data : NDArray
            The image observation received from the simulation.
        """
        # send the actuation command to the virtual agent
        self.control_socket.send(
            json.dumps(
                {
                    'command': 'step_simulation_without_physics',
                    'param': [agent, x, y, yaw],
                }
            ).encode('utf-8')
        )
        self.control_socket.recv(50)
        # retrieve image
        self.control_socket.send(
            json.dumps({'command': 'get_image', 'param': [agent]}).encode('utf-8')
        )
        image_bytes: bytes = self.receive_in_chunks(
            self.video_socket,
            self.image_info['width']
            * self.image_info['height']
            * self.image_info['channels'],
        )
        # convert image into right format
        image_data: NDArray = np.frombuffer(image_bytes, dtype=np.uint8)
        image_data = np.reshape(
            image_data,
            (
                self.image_info['height'],
                self.image_info['width'],
                self.image_info['channels'],
            ),
        )
        if self.resize is not None:
            image_data = cv2.resize(image_data, self.resize)
        self.video_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)
        # retrieve agent position
        agent_pose: Pose | None = self.get_pose('agent')

        return np.array(agent_pose), image_data

    def get_illumination(self, light_source: str) -> NDArray:
        """
        This function retrieves color values of a light source

        Parameters
        ----------
        light_source : str
            The name of the light source to change.

        Returns
        -------
        colors : NDArray
            An array containing the RGB values of the light source.
            If no object was found (-1, -1, -1) will be returned.
        """
        self.control_socket.send(
            json.dumps({'command': 'get_illumination', 'param': [light_source]}).encode(
                'utf-8'
            )
        )
        illumination_info_json: bytes = self.receive_in_chunks(self.data_socket, 1024)
        illumination_info: dict = json.loads(illumination_info_json.decode('utf-8'))
        colors: list[int] = illumination_info['colors']
        final_colors: NDArray = (np.array(colors) * 255 + 0.1).astype(int)
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)
        if final_colors.size == 0:
            return np.array({-1, -1, -1})
        return final_colors

    def get_objects(self) -> dict[str, Any]:
        """
        This function retrieves object information from the current unity scene.

        Returns
        -------
        info_dict : dict
            A dictionary containing information about all
            the objects in the current Unity scene
        """
        self.control_socket.send(
            json.dumps({'command': 'get_objects', 'param': []}).encode('utf-8')
        )
        byte_length: bytes = self.data_socket.recv(50)
        self.data_socket.send('AKN'.encode('utf-8'))
        byte_size = int(byte_length[:-16].decode('utf-8'))
        object_info_json: bytes = self.receive_in_chunks(self.data_socket, byte_size)
        object_info: dict = json.loads(object_info_json.decode('utf-8'))
        info_dic: dict[str, Any] = {}
        keys: list[str] = object_info['keys']
        names: list[str] = object_info['names']
        values: list[str] = object_info['values']
        properties: list[np.int16] = object_info['properties']
        for key, name, value, propert in zip(
            keys, names, values, properties, strict=True
        ):
            prop: str = format(propert, '016b')
            info_dic[key] = {
                'name': name,
                'type': value,
                'properties': prop,
                'isLight': prop[15],
                'isCamera': prop[14],
                'isGeometric': prop[13],
            }
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)

        return info_dic

    def move_object(self, object_id: str, pose: Pose) -> None:
        """
        This function propels the simulation. It uses teleportation to
        guide the agent directly by means of global x, y, yaw values.

        Parameters
        ----------
        object_id : str
            The id of the moved object, (currently the name)
        pose : Pose
            The tuple contains the new position and rotation
        """
        # invokes move_object
        send_str_move: str = json.dumps(
            {'command': 'move_object', 'param': [object_id] + list(pose)}
        )
        self.control_socket.send(send_str_move.encode('utf-8'))
        self.control_socket.recv(50)

    def stop(self) -> None:
        """
        This function stops the unity simulator.
        It does not expect a AKN from the simulator.
        """
        self.process.terminate()

    def set_illumination(self, light_source: str, color: NDArray) -> None:
        """
        This function sets the RGB values of a light
        component connected to a game object.

        Parameters
        ----------
        light_source : str
            The name of the object the light source is connected to
        colors : NDArray
            An array of RGB values ranging from 0 to 255
        """
        assert color.size == 3, 'The array should have three elements!'
        assert np.issubdtype(color.dtype, np.integer), 'Array must be of integer type!'
        assert 255 >= np.amin(color) >= 0 and 255 >= np.amax(color) >= 0, (
            'Color values must be in range [0, 255].'
        )
        self.control_socket.send(
            json.dumps(
                {
                    'command': 'set_illumination',
                    'param': [light_source] + [int(c) for c in color],
                }
            ).encode('utf-8')
        )
        self.control_socket.recv(50)

    def get_pose(self, object_id: str) -> Pose | None:
        """
        This function will return the pose of a given object.
        If the object name is not valid it will return None.

        Parameters
        ----------
        object_id : str
            The name of the object.

        Returns
        -------
        pose : Pose or None
            The pose of the object.
        """
        self.control_socket.send(
            json.dumps({'command': 'get_position', 'param': [object_id]}).encode(
                'utf-8'
            )
        )
        pose_info_json: bytes = self.receive_in_chunks(self.data_socket, 1024)
        pose_info: dict = json.loads(pose_info_json.decode('utf-8'))
        pose: Pose = tuple(pose_info['poses'])
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)
        if math.isnan(pose[0]):
            return None

        return pose

    def change_material(self, object_name: str, material_path: str) -> None:
        """
        This function sends the object id and the path to a material to the simulator
        and the command to change to this material for this object.

        Parameters
        ----------
        object_id : str
            The name of the object.
        scene : str
            The path to the scene to change to.
        """
        assert os.path.isfile(material_path), 'File does not exist.'
        # convert material path into formatted json command and send it
        self.control_socket.send(
            json.dumps(
                {'command': 'change_material', 'param': [object_name, material_path]}
            ).encode('utf-8')
        )
        self.control_socket.recv(50)
