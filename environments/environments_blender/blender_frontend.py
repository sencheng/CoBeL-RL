# basic imports
import math
import json
import socket
import numpy as np
# blender imports
import bge
import bpy
import mathutils
from bge import texture


class BlenderFrontend():
    
    def __init__(self, control_buffer_size: int = 1024):
        '''
        The basic Blender frontend.
        
        Parameters
        ----------
        control_buffer_size :               The buffer size of the control connection when receiving commands from the framework.
        
        Returns
        ----------
        None
        '''
        # port/address for controlling the simulation
        self.CONTROL_IP_ADDRESS = '127.0.0.1'
        self.CONTROL_PORT = 5000
        # port/address for the transfer of captured images
        self.VIDEO_IP_ADDRESS = '127.0.0.1'
        self.VIDEO_PORT = 5001
        # port/address for the transfer of sensor data, etc.
        self.DATA_IP_ADDRESS = '127.0.0.1'
        self.DATA_PORT = 5002
        # define camera resolution
        self.camera_width, self.camera_height = 64, 64
        self.channels = 4
        # buffer size of the control connection
        self.control_buffer_size = control_buffer_size
        # the simulation time counter
        self.simulation_time = 0.0
        # the timestep used in the simulation
        self.delta = 0.01
        # get main scene
        self.scene = bge.logic.getCurrentScene()
        # instantiate access to the single components of the scene
        self.controller = self.scene.objects['simulationBaseline']
        self.robot_support = self.scene.objects['robotSupport']
        self.left_wheel = self.scene.objects['leftWheel']
        self.right_wheel = self.scene.objects['rightWheel']
        # instantiate sensors
        self.sensor_forward = self.robot_support.sensors['sensorForward']
        self.sensor_left = self.robot_support.sensors['sensorLeft']
        self.sensor_right = self.robot_support.sensors['sensorRight']
        self.sensor_backward = self.robot_support.sensors['sensorBackward']
        self.sensor_array = np.zeros(8, dtype='float')
        # canvasses
        self.canvas_front = self.scene.objects['canvasFront']
        self.canvas_left = self.scene.objects['canvasLeft']
        self.canvas_right = self.scene.objects['canvasRight']
        self.canvas_back = self.scene.objects['canvasBack']
        # cameras
        self.camera_front = self.scene.objects['camRobotFront']
        self.camera_left = self.scene.objects['camRobotLeft']
        self.camera_right = self.scene.objects['camRobotRight']
        self.camera_back = self.scene.objects['camRobotBack']
        # prepare canvases for image transfer    
        self.ID_front = texture.materialID(self.canvas_front, 'MAscreenFront')
        self.ID_left = texture.materialID(self.canvas_left, 'MAscreenLeft')
        self.ID_right = texture.materialID(self.canvas_right, 'MAscreenRight')
        self.ID_back = texture.materialID(self.canvas_back, 'MAscreenBack')
        # front canvas
        self.canvas_front['canvasTextureFront'] = texture.Texture(self.canvas_front, self.ID_front)
        self.canvas_front['canvasTextureFront'].source = texture.ImageRender(self.scene, self.camera_front)
        self.canvas_front['canvasTextureFront'].source.capsize = [self.camera_width, self.camera_height]
        # left canvas
        self.canvas_left['canvasTextureLeft'] = texture.Texture(self.canvas_left, self.ID_left)
        self.canvas_left['canvasTextureLeft'].source = texture.ImageRender(self.scene, self.camera_left)
        self.canvas_left['canvasTextureLeft'].source.capsize = [self.camera_width, self.camera_height]
        # right canvas
        self.canvas_right['canvasTextureRight'] = texture.Texture(self.canvas_right, self.ID_left)
        self.canvas_right['canvasTextureRight'].source = texture.ImageRender(self.scene, self.camera_right)
        self.canvas_right['canvasTextureRight'].source.capsize = [self.camera_width, self.camera_height]
        # back canvas
        self.canvas_back['canvasTextureBack'] = texture.Texture(self.canvas_back, self.ID_back)
        self.canvas_back['canvasTextureBack'].source = texture.ImageRender(self.scene, self.camera_back)
        self.canvas_back['canvasTextureBack'].source.capsize = [self.camera_width, self.camera_height]
        # prepare the buffers that store the single images
        self.buffer_front = np.zeros(self.camera_width * self.camera_height * self.channels, dtype = 'uint8')
        self.buffer_left = np.zeros(self.camera_width * self.camera_height * self.channels, dtype = 'uint8')
        self.buffer_right = np.zeros(self.camera_width * self.camera_height * self.channels, dtype = 'uint8')
        self.buffer_back = np.zeros(self.camera_width * self.camera_height * self.channels, dtype = 'uint8')
        # engage control method
        self.controller['control_socket'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.controller['control_socket'].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.controller['control_socket'].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)    
        self.controller['control_socket'].bind((self.CONTROL_IP_ADDRESS, self.CONTROL_PORT))
        self.controller['control_socket'].listen(1)
        self.controller['control_connection'], address = self.controller['control_socket'].accept()
        print('Accepted control client from: ', address)
        # engage video transfer
        self.controller['video_socket'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.controller['video_socket'].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.controller['video_socket'].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.controller['video_socket'].bind((self.VIDEO_IP_ADDRESS, self.VIDEO_PORT))
        self.controller['video_socket'].listen(1)
        self.controller['video_connection'], address = self.controller['video_socket'].accept()
        print('Accepted video client from: ', address)
        # engage data transfer
        self.controller['data_socket'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.controller['data_socket'].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.controller['data_socket'].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.controller['data_socket'].bind((self.DATA_IP_ADDRESS, self.DATA_PORT))
        self.controller['data_socket'].listen(1)
        self.controller['data_connection'], address = self.controller['data_socket'].accept()
        print('Accepted data client from: ', address)
        # signal presence of a network connection
        self.controller['network_up'] = True
        print('network is up')
        # we will control the simulation through an external clock!
        bge.logic.setUseExternalClock(True)
        # instantiate the time system
        bge.logic.setClockTime(self.simulation_time)
        # set time scale
        bge.logic.setTimeScale(1.0)
        # define functions accessible by the main loop
        self.functions = {}
        self.functions['step_simulation'] = self.step_simulation
        self.functions['step_simulation_without_physics'] = self.step_simulation_without_physics
        self.functions['get_safe_zone_dimensions'] = self.get_safe_zone_dimensions
        self.functions['get_safe_zone_layout'] = self.get_safe_zone_layout
        self.functions['stop_simulation'] = self.stop_simulation
        self.functions['set_XYYaw'] = self.set_XYYaw
        self.functions['get_manually_defined_topology_nodes'] = self.get_manually_defined_topology_nodes
        self.functions['get_manually_defined_topology_edges'] = self.get_manually_defined_topology_edges
        self.functions['set_illumination'] = self.set_illumination
        self.functions['get_telemetry_data'] = self.get_telemetry_data
        self.functions['get_image_info'] = self.get_image_info
        self.functions['set_camera_resolution'] = self.set_camera_resolution
    
    def refresh_canvases(self):
        '''
        This function refreshes the canvases (i.e., the current camera input).
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # refresh canvases
        self.canvas_front['canvasTextureFront'].source.refresh(self.buffer_front, 'BGRA')
        self.canvas_left['canvasTextureLeft'].source.refresh(self.buffer_left, 'BGRA')
        self.canvas_right['canvasTextureRight'].source.refresh(self.buffer_right, 'BGRA')
        self.canvas_back['canvasTextureBack'].source.refresh(self.buffer_back, 'BGRA')
    
    def main_loop(self):
        '''
        This is the blender frontend's main loop.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # start blender frontend main loop
        while not bge.logic.NextFrame():
            # refresh canvases
            self.refresh_canvases()
            # if the network is up start listening
            if self.controller['network_up']:
                # retrieve data string from port
                data = json.loads(self.receive_in_chunks(self.controller['control_connection'], 1024).decode('utf-8'))
                # execute command if it exists
                if data['command'] in self.functions:
                    self.functions[data['command']](*data['param'])
                # send AKN
                self.controller['control_connection'].send('AKN.'.encode('utf-8'))
                
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
            if data[-16:] ==  b'!cobelclienteod!':
                break
        
        return data[:-16]
                
    def send_data(self, socket: socket.socket, data, serialize: bool = True):
        '''
        This function sends data via a specified web socket.
        
        Parameters
        ----------
        socket :                            The socket that data will be send over.
        data :                              The data that will be send.
        serialize :                         If true, the data will be serialized with JSON.
        
        Returns
        ----------
        None
        '''
        # serialize data if necessary
        if serialize:
            data = json.dumps(data).encode('utf-8')
        # send data
        socket.send(data + b'!blenderservereod!')
        # wait for AKN
        socket.recv(50).decode('utf-8')
        
    def get_safe_zone_dimensions(self):
        '''
        This function sends back the dimensions of the safe area.
        Note: currently, the safeZoneLayout MUST have a scale of [1., 1., 1.]!
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # retrieve layout from scene
        layout = self.scene.objects['safeZoneLayout']
        # define min and max values
        max_x, max_y, max_z = -1000.0, -1000.0, -1000.0
        min_x, min_y, min_z = 1000.0, 1000.0, 1000.0
        # loop over all vertices
        for mesh in layout.meshes:
            for mi in range(len(mesh.materials)):
                for vi in range(mesh.getVertexArrayLength(mi)):
                    vertex = mesh.getVertex(mi, vi)
                    # find extremal values
                    min_x, min_y, min_z = min(min_x, vertex.x), min(min_y, vertex.y), min(min_z, vertex.z)
                    max_x, max_y, max_z = max(max_x, vertex.x), max(max_y, vertex.y), max(max_z, vertex.z)
        # send data
        self.send_data(self.controller['data_connection'], [min_x, min_y, min_z, max_x, max_y, max_z])
        
    def get_safe_zone_layout(self):
        '''
        This function sends back the safe area's layout.
        Note: currently, the safeZoneLayout MUST have a scale of [1., 1., 1.]!
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # retrieve vertices
        mesh = bpy.data.objects['safeZoneLayout'].data
        # loop over mesh
        vertices, segments = [], []
        current_vertex = 0
        for p in mesh.polygons:
            for li in range(p.loop_start, p.loop_start + p.loop_total):
                index = mesh.loops[li].vertex_index
                vertices.append([mesh.vertices[index].co[0], mesh.vertices[index].co[1]])
                segments.append([current_vertex, (current_vertex + 1) % len(mesh.vertices)])
                current_vertex += 1
        # send data
        self.send_data(self.controller['data_connection'], [vertices, segments])
        
    def get_manually_defined_topology_nodes(self):
        '''
        This function sends back the manually defined topology nodes.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # retrieve nodes
        nodes = []
        for obj in self.scene.objects:
            if 'graphNode' in obj.name:
                node_type = 'standardNode'
                if 'startNode' in obj:
                    node_type = 'startNode'
                elif 'goalNode' in obj:
                    node_type = 'goalNode'
                nodes.append([obj.name, obj.worldPosition.x, obj.worldPosition.y, node_type])
        # send data
        self.send_data(self.controller['data_connection'], nodes)
        
    def get_manually_defined_topology_edges(self):
        '''
        This function sends back the manually defined topology edges.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # retrieve edges
        edges = []
        for obj in self.scene.objects:
            if 'graphEdge' in obj.name:
                edges.append([obj.name, obj['first'], obj['second']])
        # send data
        self.send_data(self.controller['data_connection'], edges)
        
    def step_simulation(self, velocity_left: float, velocity_right: float):
        '''
        This function updates the robot's velocity and propels the simulation by one time step.
        
        Parameters
        ----------
        velocity_left :                     The desired left wheel velocity.
        velocity_right :                    The desired right wheel velocity.
        
        Returns
        ----------
        None
        '''
        # update simulation time
        self.simulation_time += self.delta
        # set wheel velocities
        self.left_wheel.setLinearVelocity([velocity_left, 0.0, 0.0], True)
        self.right_wheel.setLinearVelocity([velocity_right, 0.0, 0.0], True)
        # update BGE clock
        bge.logic.setClockTime(self.simulation_time)
        # refresh canvases
        self.refresh_canvases()
        # send video data
        video_data = np.concatenate((np.reshape(self.buffer_front, (self.camera_height, self.camera_width, self.channels)),
                                     np.reshape(self.buffer_right, (self.camera_height, self.camera_width, self.channels)),
                                     np.reshape(self.buffer_back, (self.camera_height, self.camera_width, self.channels)),
                                     np.reshape(self.buffer_left, (self.camera_height, self.camera_width, self.channels))), axis=1)[:, :, :3].tobytes()
        self.send_data(self.controller['video_connection'], video_data, False)
        
    def step_simulation_without_physics(self, x: float, y: float, yaw: float):
        '''
        This function teleports the robot and propels the simulation by one time step.
        
        Parameters
        ----------
        x :                                 The robot's new x position.
        y :                                 The robot's new y position.
        yaw :                               The robot's new orientation in degrees.
        
        Returns
        ----------
        None
        '''
        # update simulation time
        self.simulation_time += self.delta
        # switch off physics for object in teleport
        self.robot_support.setLinearVelocity([0.0, 0.0, 0.0], False)
        self.robot_support.setAngularVelocity([0.0, 0.0, 0.0], False)
        # tie wheels to the robot's support
        self.left_wheel.setParent(self.robot_support)
        self.right_wheel.setParent(self.robot_support)
        # update the robot's position
        self.robot_support.worldPosition.x = x
        self.robot_support.worldPosition.y = y
        # update the robot's orientation
        euler = mathutils.Euler((0.0, 0.0, yaw / 180 * math.pi), 'XYZ')
        self.robot_support.worldOrientation = euler.to_matrix()
        # untie the wheels from the robot's support
        self.left_wheel.removeParent()
        self.right_wheel.removeParent()
        # update BGE clock
        bge.logic.setClockTime(self.simulation_time)
        # refresh canvases
        self.refresh_canvases()
        # send video data
        video_data = np.concatenate((np.reshape(self.buffer_front, (self.camera_height, self.camera_width, self.channels)),
                                     np.reshape(self.buffer_right, (self.camera_height, self.camera_width, self.channels)),
                                     np.reshape(self.buffer_back, (self.camera_height, self.camera_width, self.channels)),
                                     np.reshape(self.buffer_left, (self.camera_height, self.camera_width, self.channels))), axis=1)[:, :, :3].tobytes()
        self.send_data(self.controller['video_connection'], video_data, False)
        
    def get_telemetry_data(self):
        '''
        This function sends back the telemetry data.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # retrieve telemetry data
        telemetry = {'time': self.simulation_time, 'sensor': list(self.sensor_array),
                     'pose': [self.robot_support.worldPosition[0], self.robot_support.worldPosition[1], self.robot_support.worldOrientation[0][0], self.robot_support.worldOrientation[1][0]]}
        # send data
        self.send_data(self.controller['data_connection'], telemetry)
        
    def get_image_info(self):
        '''
        This function sends back the image information.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # retrieve image info
        image_info = {'width': self.camera_width * 4, 'height': self.camera_height, 'channels': self.channels - 1, 'format': 'bgr'}
        # send data
        self.send_data(self.controller['data_connection'], image_info)
        
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
        # update camera resolution
        self.camera_width, self.camera_height = camera_width, camera_height
        # update canvases
        self.canvas_front['canvasTextureFront'].source.capsize = [self.camera_width, self.camera_height]
        self.canvas_left['canvasTextureLeft'].source.capsize = [self.camera_width, self.camera_height]
        self.canvas_right['canvasTextureRight'].source.capsize = [self.camera_width, self.camera_height]
        self.canvas_back['canvasTextureBack'].source.capsize = [self.camera_width, self.camera_height]
        # update the buffers
        self.buffer_front = np.zeros(self.camera_width * self.camera_height * self.channels, dtype = 'uint8')
        self.buffer_left = np.zeros(self.camera_width * self.camera_height * self.channels, dtype = 'uint8')
        self.buffer_right = np.zeros(self.camera_width * self.camera_height * self.channels, dtype = 'uint8')
        self.buffer_back = np.zeros(self.camera_width * self.camera_height * self.channels, dtype = 'uint8')
        # refresh canvases
        self.refresh_canvases()
        
    def set_XYYaw(self, object_name: str, pose: list):
        '''
        This function teleports an object to a desired new position and yaw.
        
        Parameters
        ----------
        object :                            The object name.
        pose :                              The object's new pose (x position, y position, orientation in degrees).
        
        Returns
        ----------
        None
        '''
        x, y, yaw = pose
        roll, pitch = 0., 0.
        # retrieve object from scene
        obj = self.scene.objects[object_name]
        # switch off physics for object in teleport
        obj.setLinearVelocity([0.0, 0.0, 0.0], False)
        obj.setAngularVelocity([0.0, 0.0, 0.0], False)
        # tie wheels to object (?)
        self.left_wheel.setParent(obj)
        self.right_wheel.setParent(obj)
        # update the object's position
        obj.worldPosition.x = x
        obj.worldPosition.y = y
        # update the object's orientation
        euler = mathutils.Euler((roll / 180 * math.pi, pitch / 180 * math.pi, yaw / 180 * math.pi), 'XYZ')
        obj.worldOrientation = euler.to_matrix()
        # untie wheels to object (?)
        self.left_wheel.removeParent()
        self.right_wheel.removeParent()
        
    def set_illumination(self, light_source: str, color: list):
        '''
        This function sets desired RGB values for a given light source.
        
        Parameters
        ----------
        light_source :                      The light source's name.
        color :                             The light source's new color.
        
        Returns
        ----------
        None
        '''
        # retrieve light source and update its color
        obj = self.scene.objects[light_source]
        obj.color = color
        
    def stop_simulation(self):
        '''
        This function ends the simulation.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # end the simulation
        bge.logic.endGame() 
        # close connections
        self.controller['control_connection'].close()
        self.controller['video_connection'].close()
        self.controller['network_up'] = False
        