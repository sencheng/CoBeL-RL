# basic imports
import sys
import socket
import os
import json
import subprocess
import numpy as np
# shapely
from shapely.geometry import Polygon
# framework imports
from cobel.spatial_representations.spatial_representation import SpatialRepresentation

 
class FrontendBlenderInterface():
    
    def __init__(self, scenario_name: str, blender_executable: None | str = None):
        '''
        The Blender interface class.
        This class connects to the Blender environment and controls the flow of commands/data that goes to/comes from the Blender environment.
        
        Parameters
        ----------
        scenario_name :                     The name of the blender scene.
        blender_executable :                The path to the blender executable.
        
        Returns
        ----------
        None
        '''
        # determine path to blender executable
        self.BLENDER_EXECUTABLE = ''
        # if none is given check environmental variable
        if blender_executable is None:
            try:
                self.BLENDER_EXECUTABLE = os.environ['BLENDER_EXECUTABLE_PATH']
            except:
                print('ERROR: Blender executable path was neither set or given as parameter!')
                return
        else:
            self.BLENDER_EXECUTABLE = blender_executable
        # start blender subprocess
        subprocess.Popen([self.BLENDER_EXECUTABLE, scenario_name, '--window-border', '--window-geometry', '1320', '480', '600', '600', '--enable-autoexec'])
        # prepare sockets for communication with blender
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # wait for BlenderControl to start, this socket take care of command/data(limited amount) transmission
        blender_control_available = False
        while not blender_control_available:
            try:
                self.control_socket.connect(('localhost', 5000))
                self.control_socket.setblocking(1)
                blender_control_available = True
            except:
                pass
        print('Blender control connection has been initiated.')
        self.control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # wait for BlenderVideo to start, this socket transmits video data from Blender
        blender_video_available = False
        while not blender_video_available:
            try:
                self.video_socket.connect(('localhost', 5001))
                self.video_socket.setblocking(1)
                blender_video_available = True
            except:
                pass
        print('Blender video connection has been initiated.')
        self.video_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # wait for BlenderData to start, this socket is currently legacy, can probably be removed in future system versions(?)
        blender_data_available = False
        while not blender_data_available:
            try:
                self.data_socket.connect(('localhost', 5002))
                self.data_socket.setblocking(1)
                blender_data_available = True
            except:
                pass
        print('Blender data connection has been initiated.')
        self.data_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # retrieve safe zone dimensions
        self.send_command('get_safe_zone_dimensions')
        self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50).decode('utf-8')
        # retrieve safe zone layout
        self.send_command('get_safe_zone_layout')
        vertices, segments = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))
        self.data_socket.send('AKN'.encode('utf-8'))
        self.control_socket.recv(50).decode('utf-8')
        # convert the above lists to numpy arrays
        self.safe_zone_vertices, self.safe_zone_segments = np.array(vertices), np.array(segments)
        # construct the safe polygon from the above lists
        self.safe_zone_polygon = Polygon(vertices)
        #self.control_socket.recv(100).decode('utf-8')
        # initial robot pose
        self.robot_pose = np.array([0.0, 0.0, 1.0, 0.0])
        # stores the actual goal position (coordinates of the goal node)
        self.goal_position = None
        # indicator that flags arrival of the agent/robot at the actual goal node (this need NOT be the global goal node!)
        self.goal_reached = False
        # a dict that stores environmental information in each time step
        self.env_data = {'time': None, 'pose': None, 'sensor': None, 'image': None}
        # retrieve image information
        self.image_info = self.get_image_info()
        print('Image information has been retrieved.')
    
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
            if data[-18:] ==  b'!blenderservereod!':
                break
        
        return data[:-18]
    
    def send_command(self, command: str, param : None | list = None):
        '''
        This function sends a command to Blender.
        
        Parameters
        ----------
        command :                           The command name.
        param :                             The list of command parameters.
        
        Returns
        ----------
        None
        '''
        # send command
        self.control_socket.send(json.dumps({'command': command, 'param': [] if param is None else param}).encode('utf-8') + b'!cobelclienteod!')
    
    def get_manually_defined_topology_nodes(self) -> list:
        '''
        This function reads all manually defined topology nodes from the environment (if such nodes are defined).
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        topology_nodes :                    The list of manually defined topology nodes.
        '''
        # send command
        self.send_command('get_manually_defined_topology_nodes')
        # retrieve data
        manually_defined_topology_nodes = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))
        # send AKN
        self.data_socket.send('AKN'.encode('utf-8'))
        # wait for AKN
        self.control_socket.recv(50).decode('utf-8')
        
        return manually_defined_topology_nodes
    
    def get_manually_defined_topology_edges(self) -> list:
        '''
        This function reads all manually defined topology edges from the environment (if such edges are defined).
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        topology_edges :                    The list of manually defined topology edges.
        '''
        # send command
        self.send_command('get_manually_defined_topology_edges')
        # retrieve data
        manually_defined_topology_edges = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))
        # send AKN
        self.data_socket.send('AKN'.encode('utf-8'))
        # wait for AKN
        self.control_socket.recv(50).decode('utf-8')

        return manually_defined_topology_edges
    
    def get_image_info(self) -> dict:
        '''
        This function retrieves image information from the simulation.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        image_info :                        A dictionary containing information about the images rendered by Blender (i.e. dimensions, channels, format).
        '''
        # send command
        self.send_command('get_image_info')
        # retrieve data
        image_info = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))
        # send AKN
        self.data_socket.send('AKN'.encode('utf-8'))
        # wait for AKN
        self.control_socket.recv(50).decode('utf-8')

        return image_info
    
    def set_camera_resolution(self, camera_width: int, camera_height: int):
        '''
        This function updates the camera resolution.
        
        Parameters
        ----------
        camera_width :                      The new camera width.
        camera_height :                     The new camera height.
        
        Returns
        ----------
        None
        '''
        # send command
        self.send_command('set_camera_resolution', [camera_width, camera_height])
        # wait for acknowledge from Blender
        self.control_socket.recv(50)
        # update image information
        self.image_info['width'] = 4 * camera_width
        self.image_info['height'] = camera_height

    def step_simulation(self, velocity_linear: float, omega: float) -> (float, np.ndarray, np.ndarray, np.ndarray):
        '''
        This function propels the simulation. It uses physics to guide the agent/robot with linear and angular velocities (currently not working!).
        
        Parameters
        ----------
        velocity_linear :                   The requested linear (translational) velocity of the agent/robot.
        omega :                             The requested rotational (angular) velocity of the agent/robot.
        
        Returns
        ----------
        time_data :                         The time observation received from the simulation.
        pose_data :                         The pose observation received from the simulation.
        sensor_data :                       The sensor observation received from the simulation.
        image_data :                        The image observation received from the simulation.
        '''
        # from the linear/angular velocities, compute the left and right wheel velocities
        velocity_left, velocity_right = 0., 0.#self.set_robot_velocities(velocity_linear, omega)
        # send command
        self.send_command('step_simulation', [velocity_left, velocity_right])
        # retrieve video data
        image_data = self.receive_in_chunks(self.video_socket, 1024)
        image_data = np.frombuffer(image_data, dtype=np.uint8)
        image_data = np.reshape(image_data, (self.image_info['height'], self.image_info['width'], self.image_info['channels']))
        # send AKN
        self.video_socket.send('AKN'.encode('utf-8'))
        # wait for AKN
        self.control_socket.recv(50)
        # center (?) image
        image_data = np.roll(image_data, int(self.image_info['width'] * 3/8), axis=1)
        # send command
        self.send_command('get_telemetry_data')
        # retrieve data
        telemetry_data = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))
        # send AKN
        self.data_socket.send('AKN'.encode('utf-8'))
        # wait for AKN
        self.control_socket.recv(50)
        time_data, pose_data, sensor_data = telemetry_data['time'], telemetry_data['pose'], telemetry_data['sensor']
        # update robot's/agent's pose
        self.robot_pose = np.array(pose_data)
        # update environmental information
        self.env_data['time'] = time_data
        self.env_data['pose'] = np.array(pose_data)
        self.env_data['sensor'] = np.array(sensor_data)
        self.env_data['image'] = image_data
        
        return time_data, pose_data, sensor_data, image_data
       
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
        # send command
        self.send_command('step_simulation_without_physics', [x, y, yaw])
        # retrieve video data
        image_data = self.receive_in_chunks(self.video_socket, 1024)
        image_data = np.frombuffer(image_data, dtype=np.uint8)
        image_data = np.reshape(image_data, (self.image_info['height'], self.image_info['width'], self.image_info['channels']))
        # send AKN
        self.video_socket.send('AKN'.encode('utf-8'))
        # wait for AKN
        self.control_socket.recv(50)
        # center (?) image
        image_data = np.roll(image_data, int(self.image_info['width'] * 3/8), axis=1)
        # send command
        self.send_command('get_telemetry_data')
        # retrieve data
        telemetry_data = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))
        # send AKN
        self.data_socket.send('AKN'.encode('utf-8'))
        # wait for AKN
        self.control_socket.recv(50)
        time_data, pose_data, sensor_data = telemetry_data['time'], telemetry_data['pose'], telemetry_data['sensor']
        # update robot's/agent's pose
        self.robot_pose = np.array(pose_data)
        # update environmental information
        self.env_data['time'] = time_data
        self.env_data['pose'] = np.array(pose_data)
        self.env_data['sensor'] = np.array(sensor_data)
        self.env_data['image'] = image_data
        
        return time_data, pose_data, sensor_data, image_data
    
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
        # if the actuator command has more than 2 array entries, this is a teleport command, and will cause a teleport jump of the agent/robot (no physics support)
        if actuator_command.shape[0] > 2:
            # call the teleportation routine
            time_data, pose_data, sensor_data, image_data = self.step_simulation_without_physics(actuator_command[0], actuator_command[1], actuator_command[2])
            # flag if the robot reached the goal (should always be the case after a teleportation)
            if self.goal_position is not None:
                if np.linalg.norm(pose_data[:2] - self.goal_position) < 0.01:
                    self.goal_reached = True
            return time_data, pose_data, sensor_data, image_data
        else:
            # otherwise, this is a standard motion command with physics support (the agent/robot approaches the goal by actuating the robot's/agent's wheels)
            time_data, pose_data, sensor_data, image_data = self.step_simulation(actuator_command[0], actuator_command[1])
            # flag if the robot/agent reached the goal already
            if self.goal_position is not None:
                if np.linalg.norm(pose_data[:2] - self.goal_position) < 0.01:
                    self.goal_reached = True
            return time_data, pose_data, sensor_data, image_data
 
    def set_XYYaw(self, object_name: str, pose: list):
        '''
        This function teleports the robot to a novel pose.
        
        Parameters
        ----------
        object_name :                       The object name (unused/irrelevant).
        pose :                              The pose to teleport to.
        
        Returns
        ----------
        None
        '''
        # send command
        self.send_command('set_XYYaw', [object_name, pose])
        # wait for acknowledge from Blender
        self.control_socket.recv(50)
        
    def set_illumination(self, light_source: str, color: list):
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
        # send command
        self.send_command('set_illumination', [light_source, color])
        # wait for acknowledge from Blender
        self.control_socket.recv(50)
        
    def set_topology(self, topology_module: SpatialRepresentation):
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
    
    def stop_blender(self):
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
            self.send_command('stop_simulation')
        except:
            print(sys.exc_info()[1])


class FrontendBlenderDynamic(FrontendBlenderInterface):
    
    def __init__(self, scenario_name: str, blender_executable: None | str = None):
        '''
        The Blender interface class for use with the dynamic barrier environment.
        
        Parameters
        ----------
        scenario_name :                     The name of the blender scene.
        blender_executable :                The path to the blender executable.
        
        Returns
        ----------
        None
        '''
        super().__init__(scenario_name, blender_executable)
        
    def set_render_state(self, barrier_ID: str, render_state: bool):
        '''
        This function sets the render state of a given barrier on the topology graph to true/false.
        
        Parameters
        ----------
        barrier_ID :                        The ID of the barrier whose render state is to be set.
        render_state :                      The render state (True/False).
        
        Returns
        ----------
        None
        '''
        # send command
        self.send_command('set_render_state', [barrier_ID, render_state])
        # wait for acknowledge from Blender
        self.control_socket.recv(50)

    def set_rotation(self, barrier_ID: str, rotation: float):
        '''
        This function sets the rotation of a given barrier.
        
        Parameters
        ----------
        barrier_ID :                        The ID of the barrier to be rotated.
        rotation :                          The rotation in degrees (0-360).
        
        Returns
        ----------
        None
        '''
        # send command
        self.send_command('set_rotation', [barrier_ID, rotation])
        # wait for acknowledge from Blender
        self.control_socket.recv(50)

    def set_texture(self, barrier_ID: str, texture: str):
        '''
        This function sets the texture of the barrier.
        
        Parameters
        ----------
        barrier_ID :                        The ID of the barrier whose texture is to be changed.
        texture :                           Filepath to the chosen texture.
        
        Returns
        ----------
        None
        '''
        # send command
        self.send_command('set_texture', [barrier_ID, texture])
        # wait for acknowledge from Blender
        self.control_socket.recv(50)

    def set_barrier(self, barrier_ID: str, render_state: bool, rotation: float, texture: str):
        '''
        This function calls all the barrier defining functions for the given barrier.
        
        Parameters
        ----------
        barrier_ID :                        The ID of the barrier whose texture is to be changed.
        render_state :                      Boolean for setting wether the given barrier should be rendered.
        rotation :                          The rotation in degrees.
        texture :                           Filepath to the chosen texture.
        
        Returns
        ----------
        None
        '''
        self.set_render_state(barrier_ID, render_state)
        self.set_rotation(barrier_ID, rotation)
        self.set_texture(barrier_ID, texture)

    def get_barrier_info(self, barrier_ID: str) -> dict:
        '''
        This function returns a dictionary containing render_state, rotation and texture of a given barrier.
        
        Parameters
        ----------
        barrier_ID :                        The ID of the barrier whose information is to be retrieved.
        
        Returns
        ----------
        barrier_info :                      The barrier information.
        '''
        # send command
        self.send_command('get_barrier_info', [barrier_ID])
        # retrieve data
        render_state, rotation, texture = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))
        # send AKN
        self.data_socket.send('AKN'.encode('utf-8'))
        # wait for AKN
        self.control_socket.recv(50).decode('utf-8')
        
        return {'render_state': render_state, 'rotation': rotation, 'texture': texture}

    def get_barrier_IDs(self) -> list:
        '''
        This function returns a list of all barrier objects.
        BarrierIDs are in the from "barrierxxx-yyy", where xxx and yyy are the
        numbers of the nodes the barrier is standing between. This may be subject to change.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        barrier_IDs :                       The list of barrier IDs.
        '''
        # send command
        self.send_command('get_barrier_IDs')
        # retrieve data
        barrier_IDs = json.loads(self.receive_in_chunks(self.data_socket, 1024).decode('utf-8'))
        # send AKN
        self.data_socket.send('AKN'.encode('utf-8'))
        # wait for AKN
        self.control_socket.recv(50).decode('utf-8')
        
        return barrier_IDs

    def set_spotlight(self, spotlight_ID: str, render_state: bool):
        '''
        This function sets the render state of a given spotlight object.
        A spotlight lights up the area around a topology graph node.
        
        Parameters
        ----------
        spotlight_ID :                      The ID of the spotlight to be toggled.
        render_state :                      Boolean for setting wether the given spotlight should be on.
        
        Returns
        ----------
        None
        '''
        # send command
        self.send_command('set_spotlight', [spotlight_ID, render_state])
        # wait for acknowledge from Blender
        self.control_socket.recv(50)


class FrontendBlenderMultipleContexts(FrontendBlenderInterface):
    
    def __init__(self, scenario_name: str, blender_executable: None | str = None):
        '''
        The Blender interface class for use with the multiple contexts environment.
        
        Parameters
        ----------
        scenario_name :                     The name of the blender scene.
        blender_executable :                The path to the blender executable.
        
        Returns
        ----------
        None
        '''
        super().__init__(scenario_name, blender_executable)
        
    def set_wall_textures(self, left_wall_texture: str, front_wall_texture: str, right_wall_texture: str, back_wall_texture: str):
        '''
        This function updates the wall textures of the box.
        
        Parameters
        ----------
        left_wall_texture :                 The texture that will be applied to the left wall.
        front_wall_texture :                The texture that will be applied to the front wall.
        right_wall_texture :                The texture that will be applied to the right wall.
        back_wall_texture :                 The texture that will be applied to the back wall.
        
        Returns
        ----------
        None
        '''
        # send command
        self.send_command('set_wall_textures', [left_wall_texture, front_wall_texture, right_wall_texture, back_wall_texture])
        # wait for acknowledge from Blender
        self.control_socket.recv(50)


class ImageInterface():
    
    def __init__(self, image_set: str, safe_zone_dimensions: str, safe_zone_vertices: str, safe_zone_segments: str):
        '''
        A very basic class for performing ABA renewal experiments from static image input.
        It is assumed that the agent sees in every node of a topology graph a static 360deg image of the environment.
        Herein, rotation of the agent is not enabled. This setup accellerates standard ABA renewal experiments,
        since the Blender rendering 'overhead' is not reqired. It is necessary that Blender is run prior to the application of this class,
        since the interface assumes a worldStruct data block in the 'world' directory of the experiment folder.
        This data is generated by a initial run of the getWorldInformation.py script in the experiment folder.
        
        Parameters
        ----------
        image_set :                         The file containing the set of prerendered images for contexts A and B.
        safe_zone_dimensions :              The file containing the recovered zone dimensions.
        safe_zone_vertices :                The file containing the set of recovered safe zone vertices.
        safe_zone_segments :                The file containing the set of recovered safe zone segments.
        
        Returns
        ----------
        None
        '''
        # retrieve all image information from the 'worldInfo' directory
        self.images = np.load(image_set)
        # get the maximum safe zone dimensions for the robot to move in 
        self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z = np.load(safe_zone_dimensions)
        self.safe_zone_vertices = np.load(safe_zone_vertices)
        self.safe_zone_segments = np.load(safe_zone_segments)
        # initial robot pose
        self.robot_pose = np.array([0.0, 0.0, 1.0, 0.0])
        # stores the actual goal position (coordinates of the goal node)
        self.goal_position = None
        # indicator that flags arrival of the agent/robot at the actual goal node (this need NOT be the global goal node!)
        self.goal_reached = False
        # a dict that stores environmental information in each time step
        self.env_data = {'time': 0.0, 'pose': None, 'sensor': None, 'image': None}
        # this interface class requires a topologyModule
        self.topology_module = None
     
    def set_topology(self, topology_module: SpatialRepresentation):
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
 
    def get_manually_defined_topology_nodes(self) -> np.ndarray:
        '''
        This function reads all manually defined topology nodes from the environment (if such nodes are defined).
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        topology_nodes :                    The list of manually defined topology nodes.
        '''
        return np.load('worldInfo/topologyNodes.npy')
        
    def get_manually_defined_topology_edges(self) -> np.ndarray:
        '''
        This function reads all manually defined topology nodes from the environment (if such nodes are defined)
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        topology_edges :                    The list of manually defined topology edges.
        '''
        return np.load('worldInfo/topologyEdges.npy')
         
    def step_simulation_without_physics(self, x: float, y: float, yaw: float) -> (float, np.ndarray, np.ndarray, np.ndarray):
        '''
        This function propels the simulation. It uses teleportation to guide the agent/robot directly by means of global x, z, yaw values.
        
        Parameters
        ----------
        x :                                 The global x position to teleport to.
        z :                                 The global z position to teleport to.
        yaw :                               The global yaw value to teleport to.
        
        Returns
        ----------
        time_data :                         The time observation received from the simulation.
        pose_data :                         The pose observation received from the simulation.
        sensor_data :                       The sensor observation received from the simulation.
        image_data :                        The image observation received from the simulation.
        '''
        # update robot's/agent's pose
        self.robot_pose = np.array([x, y, yaw])
        # update environmental information
        # propel the simulation time by 1/100 of a second (standard time step)
        self.env_data['time'] += 0.01
        # the position can be updated instantaneously
        self.env_data['pose'] = np.array([x, y, np.cos(yaw/180.0*np.pi), np.sin(yaw/180.0*np.pi)])
        # there will be no need for sensor data in this interface class
        self.env_data['sensor'] = np.zeros(8)
        # the image data is read from the 'worldInfo' directory
        self.env_data['image'] = self.images[self.topology_module.next_node]
        
        return self.env_data['time'], self.env_data['pose'], self.env_data['sensor'], self.env_data['image']
        
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
        # if the actuator command has more than 2 array entries, this is a teleport command, and will cause a teleport jump of the agent/robot (no physics support)
        if actuator_command.shape[0] > 2:
            # call the teleportation routine
            time_data, pose_data, sensor_data, image_data = self.step_simulation_without_physics(actuator_command[0], actuator_command[1], actuator_command[2])
            # flag if the robot reached the goal (should always be the case after a teleportation)
            if self.goal_position is not None:
                if np.linalg.norm(pose_data[:2] - self.goal_position) < 0.01:
                    self.goal_reached = True
            return time_data, pose_data, sensor_data, image_data
        else:
            # otherwise, this is a standard motion command with physics support (the agent/robot approaches the goal by actuating the robot's/agent's wheels)
            time_data, pose_data, sensor_data, image_data = self.step_simulation(actuator_command[0], actuator_command[1])
            # flag if the robot/agent reached the goal already
            if self.goal_position is not None:
                if np.linalg.norm(pose_data[:2] - self.goal_position) < 0.01:
                    self.goal_reached = True
            return time_data, pose_data, sensor_data, image_data
    
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
