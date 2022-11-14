# basic imports
import os
import time
import subprocess
import json
import signal
import numpy as np
import asyncio
import threading
from typing import Dict, Callable, Optional
from socket import socket, AF_INET, SOCK_STREAM, timeout


class FrontendGodotInterface:

    def __init__(self, scenario_name: str, godot_executable=None):
        '''
        The Godot interface class.
        This class connects to the Godot environment and controls the flow of commands/data that goes to/comes from the Blender environment.
        
        Parameters
        ----------
        scenario_name :                     The name of the blender scene.
        blender_executable :                The path to the blender executable.
        
        Returns
        ----------
        None
        '''
        # determine path to Godot executable
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
        subprocess.Popen([self.GODOT_EXECUTABLE])
        # define connectors
        self.data_connector = GodotConnector(65320)
        self.image_connector = GodotConnector(65444, True)
        self.control_connector = GodotConnector(65433)
        self.control_connector.start()
        print ('Godot control connection has been initiated.')
        self.data_connector.start()
        print ('Godot data connection has been initiated.')
        self.image_connector.start()
        print ('Godot video connection has been initiated.')
        # sometimes events are fired before has finished registering the event-listeners
        time.sleep(0.5)
        # for save closing use custom signal interrupt handler
        signal.signal(signal.SIGINT, self.on_exit)
        # store env data
        self.env_data = {'image': None}
        # load scene
        self.change_scene(scenario_name)

    def receive_image(self) -> np.ndarray:
        '''
        This function retrieves the current image observation from Godot.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        image :                             The retrieved image.
        '''
        self.control_connector.send('RecvImage')
        data = self.image_connector.receive_image()
        data = np.frombuffer(data, dtype=np.uint8).reshape((256, 1024, 3))
        #image = Image.frombuffer('RGB', (1024, 256), data, 'raw')

        return data
    
    def step_simulation_without_physics(self, x: float, y: float, yaw: float) -> None:
        '''
        This function propels the simulation. It uses teleportation to guide the agent/robot directly by means of global x, z, yaw values.
        
        Parameters
        ----------
        x :                                 The global x position to teleport to.
        z :                                 The global z position to teleport to.
        yaw :                               The global yaw value to teleport to.
        
        Returns
        ----------
        None
        '''
        self.control_connector.send('StepSimulationWithoutPhysics', f'{x},{y},{yaw}')
        self.env_data['image'] = self.receive_image()

    def get_illumination(self, light_source: str) -> np.ndarray:
        '''
        This function retrieves the color of a specified light source.
        
        Parameters
        ----------
        light_source :                      The name of a light source (needs to inherit Light Class).
        
        Returns
        ----------
        color :                             The retrieved color (RGBA).
        '''
        self.control_connector.send('GetIllumination', light_source)
        data = self.data_connector.receive()
        
        return np.array([float(value) for value in data.split(',')])

    def set_illumination(self, light_source: str, color: np.ndarray) -> None:
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
        hex_color = '#'
        for value in color:
            hex_color += hex(int(value))[2:].zfill(2)
        self.control_connector.send('SetIllumination', f'{light_source},{hex_color}')

    def set_move_actor(self, x: float, y: float, omega: float, node_name='Actor') -> None:
        '''
        This function moves the Actor Node given x, y and rotational velocity.
        
        Parameters
        ----------
        x :                                 The velocity in x-axis direction in m/s.
        y :                                 The velocity in y-axis direction in m/s.
        omega :                             The rotational velocity in radians/s.
        node_name :                         The name of the node that gets moved.
        
        Returns
        ----------
        None
        '''
        self.control_connector.send('SetMoveActor', f'{node_name}, {x}, {y}, {omega}')

    def get_pose(self, object_id: int) -> np.ndarray:
        '''
        This function retrieves the pose of a specified object.
        
        Parameters
        ----------
        object_ID :                         The name of the object.
        
        Returns
        ----------
        pose :                              The retrieved pose (x, y, z, pitch, roll, yaw).
        '''
        self.control_connector.send('GetPose', object_id)
        data = self.data_connector.receive()
        pos, rot = data.split(' ')
        # parse all values to floats
        return np.array([float(i) for i in pos.split(',')] + [float(i) for i in rot.split(',')])

    def set_pose(self, object_id: int, x: float, y: float, z: float, pitch: float, roll: float, yaw: float) -> None:
        '''
        This function sets the pose of a specified object.
        
        Parameters
        ----------
        object_ID :                         The name of the object.
        
        Returns
        ----------
        None
        '''
        self.control_connector.send('SetPose', f'{object_id} {x},{y},{z},{pitch},{roll},{yaw}')

    def set_material(self, object_id: int, object_material: str) -> None:
        '''
        This function sets the material of a GeometryInstance.
        
        Parameters
        ----------
        object_ID :                         The name of the object.
        object_material :                   The file name of a material (e.g. "wall_01.material").
        
        Returns
        ----------
        None
        '''
        self.control_connector.send('SetMaterial', f'{object_id} {object_material}')
        
    def get_object_ids(self):
        '''
        This function retrieves the IDs of all scene objects.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        object_IDs :                        The object IDs.
        '''
        self.control_connector.send('GetObjectIds')
        data = self.data_connector.receive()
        parsed = json.loads(str(data))
        
        return parsed

    def change_scene(self, scene: str) -> None:
        '''
        This function loads a given scene.
        
        Parameters
        ----------
        scene :                             The name of the scene (not the path nor the file name, e.g. 'room').
        
        Returns
        ----------
        None
        '''
        self.control_connector.send('ChangeScene', scene)
        time.sleep(1)   # wait for the scene to be loaded

    def stop_godot(self):
        '''
        This function closes Godot.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        self.control_connector.send('CloseGodot')
        self.control_connector.close()
        self.data_connector.close()
        self.image_connector.close()
        
    def on_exit(self):
        '''
        This function closes all connections properly.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        self.stop_godot()
        os._exit(-1)


class GodotConnector:

    def __init__(self, port: int, image: bool = False):
        '''
        Connector object managing the communication of specific data between CoBeL-RL and Godot.
        
        Parameters
        ----------
        port :                              The port that will be used.
        image :                             If true, then the connector can handle image data.
        
        Returns
        ----------
        None
        '''
        self.server: Server = Server('localhost', port, image)
        self.events: Dict[str, Callable[[str, Event], None]] = dict()

    def close(self):
        '''
        This function closes the connection.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        self.server.close()

    def start(self):
        '''
        This function starts the connection.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        asyncio.run(self.server.start())

    def register(self, name: str, callback) -> None:
        '''
        Register an event.
        
        Parameters
        ----------
        name :                              The event name.
        callback :                          A callback method that is called once the event is triggered.
        
        Returns
        ----------
        None
        '''
        self.events[name] = callback

    def receive_image(self) -> bytes:
        '''
        Receives an image from the image socket. This method is blocking.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        image_data :                        The image data as bytes.
        '''
        return asyncio.run(self.server.poll_image_async())

    def receive(self) -> Optional[str]:
        '''
        Receives a string from the socket.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        data :                              The data as a string.
        '''
        return asyncio.run(self.server.receive_data())

    def send(self, name: str, data=None) -> None:
        '''
        Sends an event with given data to godot.
        
        Parameters
        ----------
        name :                              The event name.
        data :                              The data to be send.
        
        Returns
        ----------
        None
        '''
        event = Event(name, data)
        asyncio.run(self.server.send_data([event]))
        

class Server:

    def __init__(self, host: str, port: int, raw: bool = False) -> None:
        '''
        Wrapper class to sum up all the networking.
        Create socket and binds it to host and port.
        
        Parameters
        ----------
        host :                              The hostname.
        port :                              The port that will be used.
        image :                             If true, data will be received as bytes instead of events.
        
        Returns
        ----------
        None
        '''
        self.conn: Optional[socket] = None
        self.raw = raw
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.bind((host, port))
        self.lastImage: Optional[bytes] = None
        self.stop = False
        if raw:
            self.poll_thread = threading.Thread(target=self.poll_image)
            self.poll_thread.start()
        else:
            self.poll_thread = None

    def close(self) -> None:
        '''
        Closes the connection.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        if self.poll_thread:
            self.stop = True
        if self.conn:
            self.conn.close()
        self.socket.close()

    async def start(self):
        '''
        Blocks thread until a connection could be established.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        self.socket.listen()
        self.conn, address = self.socket.accept()
        print(f"connected to {address}")

    def poll_image(self) -> None:
        '''
        Continuously reads new image data from the image socket and stores completed images as a member variable.
        This method should always run in separate thread.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        while not self.conn:
            continue
        magic = bytes([0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00])
        buffer = b""
        while True:
            if not self.conn or self.stop:
                break
            buffer += self.conn.recv(1024)
            if not buffer.endswith(magic) or self.stop:
                continue
            data = _remove_buffer_suffix(buffer, magic)
            self.lastImage = data
            buffer = b""

    async def poll_image_async(self) -> bytes:
        '''
        Continuously waits until a full image was received from the image socket.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        image :                             Image data as bytes if there is one.
        '''
        while self.lastImage is None:
            await asyncio.sleep(0.01)
            continue
        data = self.lastImage
        self.lastImage = None

        return data

    async def receive_data(self) -> Optional[str]:
        '''
        Continuously waits until data was received from the socket.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        data :                              String containing data or empty string in the case of failure. When not connected None is returned.
        '''
        if not self.conn:
            return None
        self.conn.settimeout(0.2)
        data = ""
        while True:
            try:
                buffer = self.conn.recv(1024)
            except timeout:
                break
            data += buffer.decode("utf-8")
        index = data.find(":") + 1

        return data[index::]

    async def send_data(self, events) -> None:
        '''
        Sends an arbitrary amount of events.
        Since it is blocking code it was declared as async.
        
        Parameters
        ----------
        events :                            A collection of events.
        
        Returns
        ----------
        None
        '''
        '''
            sends an arbitrary amount of events
            this is blocking code therefore (as stated previously) it's declared async
            :param events:
        '''
        data = dict()
        data["events"] = []
        for event in events:
            data["events"].append(event.encode())
        self.conn.sendall(json.dumps(data).encode(encoding="utf-8") + '\u2000'.encode(encoding="utf-8"))
        
        
class Event:

    def __init__(self, name: str, data='') -> None:
        '''
        Wrapper class for events.
        
        Parameters
        ----------
        name :                              The event name.
        data :                              The event data.
        
        Returns
        ----------
        None
        '''
        self.name = name
        self.data = data

    def encode(self) -> Dict:
        '''
        Converts the event to a dictionary.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        data :                              The encoded event data.
        '''
        return {'name': self.name, 'data': self.data}


def _remove_buffer_suffix(input_string: bytes, suffix: bytes) -> bytes:
    '''
    Removes the string of magic bytes specified in the suffix from the input_string.
    
    Parameters
    ----------
    input_string :                      The input data.
    suffix :                            The suffix to be removed.
    
    Returns
    ----------
    input_string :                      The input data with its suffix removed.
    '''
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string
