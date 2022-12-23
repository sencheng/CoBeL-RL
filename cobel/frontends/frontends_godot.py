# basic imports
import sys
import socket
import os
import subprocess
import json
import numpy as np

 
class FrontendGodotInterface():
    
    def __init__(self, scenario_name: str, godot_executable=None, running=False):
        '''
        The Godot interface class.
        This class connects to the Godot environment and controls the flow of commands/data that goes to/comes from the Blender environment.
        
        Parameters
        ----------
        scenario_name :                     The name of the blender scene.
        godot_executable :                  The path to the blender executable.
        running :                           If true the starting of a new process will be skipped.
        
        Returns
        ----------
        None
        '''
        # determine path to blender executable
        self.GODOT_EXECUTABLE = ''
        # if none is given check environmental variable
        if godot_executable is None:
            try:
                self.GODOT_EXECUTABLE = os.environ['GODOT_EXECUTABLE_PATH']
            except:
                print('ERROR: Godot executable path was neither set or given as parameter!')
                return
        else:
            self.GODOT_EXECUTABLE = godot_executable
        # start blender subprocess
        if not running:
            subprocess.Popen([self.GODOT_EXECUTABLE])
        # prepare sockets for communication with blender
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # wait for BlenderControl to start, this socket take care of command/data(limited amount) transmission
        godot_control_available = False
        while not godot_control_available:
            try:
                self.control_socket.connect(('localhost', 5000))
                self.control_socket.setblocking(1)
                godot_control_available = True
            except:
                pass
        print('Godot control connection has been initiated.')
        self.control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # wait for BlenderVideo to start, this socket transmits video data from Blender
        godot_video_available = False
        while not godot_video_available:
            try:
                self.video_socket.connect(('localhost', 5001))
                self.video_socket.setblocking(1)
                godot_video_available = True
            except:
                pass
        print('Godot video connection has been initiated.')
        self.video_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # wait for BlenderData to start, this socket is currently legacy, can probably be removed in future system versions(?)
        godot_data_available = False
        while not godot_data_available:
            try:
                self.data_socket.connect(('localhost', 5002))
                self.data_socket.setblocking(1)
                godot_data_available = True
            except:
                pass
        print('Godot data connection has been initiated.')
        self.data_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # initial robot pose
        self.robot_pose = np.array([0.0, 0.0, 1.0, 0.0])
        # stores the actual goal position (coordinates of the goal node)
        self.goal_position = None
        # indicator that flags arrival of the agent/robot at the actual goal node (this need NOT be the global goal node!)
        self.goal_reached = False
        # a dict that stores environmental information in each time step
        self.env_data = {'pose': None, 'image': np.zeros((256, 1024, 3))}
        # load scene
        self.change_scene(scenario_name)
        # a dict storing information about objects present in the simulation
        self.get_objects()
        # retrieve object information and determine ID of the agent
        self.actor_id = None
        for object_ID in self.objects:
            if self.objects[object_ID]['name'] == 'Actor':
                self.actor_id = object_ID
        print('Object information has been retrieved.')
        # retrieve image information
        self.image_info = self.get_image_info()
        print('Image information has been retrieved.')
     
    def receive(self, socket: socket.socket, data_size: int) -> str:
        '''
        This function reads data with a specified size from a socket.
        
        Parameters
        ----------
        socket :                            The socket to read the data from.
        data_size :                         The size of the data to read.
        
        Returns
        ----------
        data :                              The data as a byte string.
        '''
        # define buffer
        data = b''
        # read the data in chunks
        received_bytes = 0
        while received_bytes < data_size:
            data_chunk = socket.recv(data_size - received_bytes)
            data += data_chunk
            received_bytes += len(data_chunk)
        
        return data
    
    def receive_in_chunks(self, socket: socket.socket, chunk_size: int) -> str:
        '''
        This function reads data in chunks from a socket.
        
        Parameters
        ----------
        socket :                            The socket to read the data from.
        chunk_size :                        The size of the chunk to read.
        
        Returns
        ----------
        data :                              The data as a byte string.
        '''
        # define buffer
        data = b''
        # read the data in chunks
        while True:
            data_chunk = socket.recv(chunk_size)
            data += data_chunk
            if data[-16:] ==  b'!godotservereod!':
                break
        
        return data[:-16]
       
    def step_simulation_without_physics(self, x: float, y: float, yaw: float) -> (float, np.ndarray, np.ndarray, np.ndarray):
        '''
        This function propels the simulation. It uses teleportation to guide the agent/robot directly by means of global x, y, yaw values.
        
        Parameters
        ----------
        x :                                 The global x position to teleport to.
        y :                                 The global y position to teleport to.
        yaw :                               The global yaw value to teleport to.
        
        Returns
        ----------
        time_data :                         The time observation received from the simulation.
        pose_data :                         The pose observation received from the simulation.
        sensor_data :                       The sensor observation received from the simulation.
        image_data :                        The image observation received from the simulation.
        '''
        # send the actuation command to the virtual robot/agent
        send_str = json.dumps({'command': 'step_simulation_without_physics', 'param': [x, y, np.deg2rad(yaw)]})
        self.control_socket.send(send_str.encode('utf-8'))
        self.control_socket.recv(50)
        # retrieve image data from the robot
        self.control_socket.send(json.dumps({'command': 'get_image', 'param': []}).encode('utf-8'))
        image_data = self.receive_in_chunks(self.video_socket, self.image_info['width'] * self.image_info['height'] * self.image_info['channels'])
        image_data = np.frombuffer(image_data, dtype=np.uint8)
        image_data = np.reshape(image_data, (self.image_info['height'], self.image_info['width'], self.image_info['channels']))
        self.video_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)
        # retrieve pose data from the robot
        send_str = json.dumps({'command': 'get_pose', 'param': [self.actor_id]})
        self.control_socket.send(send_str.encode('utf-8'))
        pose_data = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))['pose']
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)
        # update robot's/agent's pose
        self.robot_pose = pose_data
        # update environmental information
        self.env_data['pose'] = pose_data
        self.env_data['image'] = image_data
        
        return pose_data, image_data
        
    def set_illumination(self, light_source: str, color: np.ndarray):
        '''
        This function sets the color of a specified light source.
        
        Parameters
        ----------
        light_source :                      The name of the light source to change.
        color :                             The RGB values of the light source (as [red, green, blue]).
        
        Returns
        ----------
        None
        '''
        # send the request for illumination change to Godot
        send_str = json.dumps({'command': 'set_illumination', 'param': [light_source, self.hex_color(color)]})
        self.control_socket.send(send_str.encode('utf-8'))
        self.control_socket.recv(50)
        
    def get_illumination(self, light_source: str):
        '''
        This function gets the color of a specified light source.
        
        Parameters
        ----------
        light_source :                      The name of the light source to change.
        
        Returns
        ----------
        color :                             The RGB values of the light source (as [red, green, blue])
        '''
        # send the request for illumination information to Godot
        send_str = json.dumps({'command': 'get_illumination', 'param': [light_source]})
        self.control_socket.send(send_str.encode('utf-8'))
        color = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))['color']
        color = (np.array(color) * 255 + 0.1).astype(int)
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)
        
        return color
        
    def get_objects(self):
        '''
        This function retrieves information about all object present in the simulation.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # clear dictionary
        self.objects = {}
        # send the request for object information to Godot
        self.control_socket.send(json.dumps({'command': 'get_objects', 'param': []}).encode('utf-8'))
        self.objects = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)
        
    def get_image_info(self) -> dict:
        '''
        This function retrieves image information from the simulation.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        image_info :                        A dictionary containing information about the images rendered by Godot (i.e. dimensions, channels, format).
        '''
        # send the request for image information change to Godot
        self.control_socket.send(json.dumps({'command': 'get_image_info', 'param': []}).encode('utf-8'))
        # retrieve image information
        image_info = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50)
        
        return image_info
        
    def change_scene(self, scene_name: str):
        '''
        This function changes the current scene.
        
        Parameters
        ----------
        scene_name :                        The name of the scene to be loaded.
        
        Returns
        ----------
        None
        '''
        # send the request for scene change to Godot
        send_str = json.dumps({'command': 'change_scene', 'param': [scene_name]})
        self.control_socket.send(send_str.encode('utf-8'))
        self.control_socket.recv(50)
        
    def echo(self, message: str):
        '''
        This function sends a message to the godot simulation.
        
        Parameters
        ----------
        message :                           The message.
        
        Returns
        ----------
        None
        '''
        # send the request for message echo to Godot
        send_str = json.dumps({'command': 'echo', 'param': [message]})
        self.control_socket.send(send_str.encode('utf-8'))
        self.control_socket.recv(50)
    
    def stop_godot(self):
        '''
        This function shuts down Blender.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        try:
            self.control_socket.send(json.dumps({'command': 'stop', 'param': []}).encode('utf-8'))
        except:
            print(sys.exc_info()[1])
            
    def hex_color(self, color: np.ndarray) -> str:
        '''
        This function converts RGB to hex.
        
        Parameters
        ----------
        color_rgb :                         The color in RGB.
        
        Returns
        ----------
        color_hex :                         The color in hex.
        '''
        hex_color = '#'
        for value in color:
            hex_color += hex(int(value))[2:].zfill(2)
        
        return hex_color


class FrontendGodotTopology(FrontendGodotInterface):
    
    def __init__(self, scenario_name: str, godot_executable=None, running=False):
        '''
        The Godot interface class for use with the baseline interface.
        
        Parameters
        ----------
        scenario_name :                     The name of the blender scene.
        godot_executable :                  The path to the blender executable.
        running :                           If true the starting of a new process will be skipped.
        
        Returns
        ----------
        None
        '''
        super().__init__(scenario_name, godot_executable, running)
        # initialize world limits
        self.min_x, self.max_x = .0, 1.
        self.min_y, self.max_y = .0, 1.
        # initialize safe zone info
        self.safe_zone_vertices, self.safe_zone_segments = [], []
        # initial robot pose
        self.robot_pose = np.array([0.0, 0.0, 1.0, 0.0])
        # stores the actual goal position (coordinates of the goal node)
        self.goal_position = None
        # indicator that flags arrival of the agent/robot at the actual goal node (this need NOT be the global goal node!)
        self.goal_reached = False
        # a dict that stores environmental information in each time step
        self.env_data = {'time': None, 'pose': None, 'sensor': None, 'image': np.zeros((256, 1024, 3))}
        
    def actuate_robot(self, actuator_command: list) -> (float, np.ndarray, np.ndarray, np.ndarray):
        '''
        This function actually actuates the agent/robot in the virtual environment.
        
        Parameters
        ----------
        actuator_command :                  The command that is used in the actuation process.
        
        Returns
        ----------
        time_data :                         The time observation received from the simulation.
        pose_data :                         The pose observation received from the simulation.
        sensor_data :                       The sensor observation received from the simulation.
        image_data :                        The image observation received from the simulation.
        '''
        time_data, pose_data, sensor_data, image_data = None, None, None, np.zeros((256, 1024, 3))
        # if the actuator command has more than 2 array entries, this is a teleport command, and will cause a teleport jump of the agent/robot (no physics support)
        if actuator_command.shape[0] > 2:
            # call the teleportation routine
            pose_data, image_data = self.step_simulation_without_physics(actuator_command[0], actuator_command[1], actuator_command[2])
            # flag if the robot reached the goal (should always be the case after a teleportation)
            if self.goal_position is not None:
                if np.linalg.norm(pose_data[:2] - self.goal_position) < 0.01:
                    self.goal_reached = True
        
        return time_data, pose_data, sensor_data, image_data
        
    def set_topology(self, topology_module):
        '''
        This function supplies the interface with a valid topology module.
        
        Parameters
        ----------
        topology_module :                   The topologyModule to be supplied.
        
        Returns
        ----------
        None
        '''
        self.topology_module = topology_module
     
    def get_limits(self) -> np.ndarray:
        '''
        This function returns the limits of the environmental perimeter.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        limits :                            The limits of the environmental perimeter.
        '''
        return np.array([[self.min_x, self.max_x], [self.min_y, self.max_y]])
     
    def get_wall_graph(self) -> (list, list):
        '''
        This function returns the environmental perimeter by means of wall vertices/segments.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        walls_limits :                      The wall limits.
        perimeter_nodes :                   The perimeter nodes.
        '''
        return self.safe_zone_vertices, self.safe_zone_segments
