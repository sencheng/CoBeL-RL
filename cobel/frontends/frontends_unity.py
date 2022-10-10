# basic imports
import os
import numpy as np
import time
import pickle
# ML-agents
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from mlagents_envs.base_env import ActionTuple
# framework imports
from cobel.frontends.env_sidechannel import EnvSideChannel


def get_env_path() -> str:
    '''
    Returns the unity environment path.
    TODO move this to some kind of utility class?
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    env_path :                          The unity environment path.
    '''
    if 'UNITY_ENVIRONMENT_EXECUTABLE' in os.environ.keys():
        return os.environ['UNITY_ENVIRONMENT_EXECUTABLE']
    else:
        return None


class FrontendUnityInterface():
    
    def __init__(self, scenario_name=None, side_channels=None, seed=42, timeout_wait=60, time_scale=3., worker_id=0):
        '''
        The Unity interface class. This class connects to the Unity environment and controls the flow of commands/data that goes to/comes from the Unity environment.
        
        Parameters
        ----------
        scenario_name :                     The name of the scenario that should be processed.
        side_channels :                     The ML-agents side channels.
        seed :                              The random seed used.
        timeout_wait :                      The timeout length.
        time_scale :                        The time scale.
        worker_id :                         The worker ID.
        
        Returns
        ----------
        None
        '''
        self.env_exec_path = None
        # setup side channels
        if side_channels is None:
            side_channels = []
        self.offline = False
        # Create the channel for sending world information of the Unity env
        self.env_channel = EnvSideChannel()
        side_channels.append(self.env_channel)
        # setup engine channel
        self.engine_configuration_channel = EngineConfigurationChannel()
        # add engine config channel
        side_channels.append(self.engine_configuration_channel)
        # step parameters
        self.env_configuration_channel = FloatPropertiesChannel()
        self.env_configuration_channel.set_property("max_step", 100000)
        # add env config channel
        side_channels.append(self.env_configuration_channel)
        # command line args
        args = []
        # when env is an executable mlagents can load a given scene.
        if scenario_name is not None:
            args = ["--mlagents-scene-name", scenario_name]
        self.scenario_name = scenario_name
        # select random worker id.
        # There is an issue in Linux where the worker id becomes available only after some time has passed since the
        # last usage. In order to mitigate the issue, a (hopefully) new worker id is automatically selected unless
        # specifically instructed not to.
        # Implementation note: two consecutive samples between 0 and 1200 have an immediate 1/1200 chance of
        # being the same. By using the modulo of unix time we arrive to that likelihood only after an hour, by when
        # the port has hopefully been released.
        # Additional notes: The ML-agents framework adds 5004 to the worker_id internally, so no need to worry about
        # port collision with the OS.
        if self.scenario_name is not None:
            # get the path to the unity environment executable
            self.env_exec_path = get_env_path()
            worker_id = round(time.time()) % 1200
        # try to start the communicator
        try:
            # connect python to executable environment
            env = UnityEnvironment(file_name=self.env_exec_path, worker_id=worker_id, seed=seed, base_port=5004,
                                   timeout_wait=timeout_wait, side_channels=side_channels, no_graphics=False,
                                   additional_args=args)
            # reset the environment
            env.reset()
            ## the world information has been sent from Unity to Python, env_info contains three strings corresponding
            ## to: (1) min and max XY values of the world (2) The wall information (3) The polygon that defines the perimeter
            ## of the maze
            env_info = self.env_channel.received_info
            # clear the received info, very important to do so; otherwise mess up the received info
            self.env_channel.clear_received_info()
            # [minx, minz, maxx, maxz]
            self.world_limits = [float(x) for x in env_info[0].split()]
            # extract the [minx, minz, maxx, maxz] for each wall
            self.walls_limits = []
            for item in env_info[1].split(","):
                wl = [float(x) for x in item.split()]
                self.walls_limits.append(wl)
            # extract the polygon nodes coordinates
            self.perimeter_nodes = []
            for item in env_info[2].split(","):
                p_node = [float(x) for x in item.split()]
                self.perimeter_nodes.append(p_node)
            self.perimeter_nodes.append(self.perimeter_nodes[0]) #repeat the first point to create a 'closed loop'
            self.world_limits = np.asarray(self.world_limits)
            self.walls_limits = np.asarray(self.walls_limits)
            self.perimeter_nodes = np.asarray(self.perimeter_nodes)
            # receive environment name from environment
            behavior_name = list(env.behavior_specs)[0]
            print(f"Name of the behavior : {behavior_name}")
            # save environment variables
            self.env = env
            self.behavior_name = behavior_name
        except UnityWorkerInUseException as e:
            print("the desired port is still in use. please retry after a few seconds.")
        # a dict that stores environmental information in each time step
        self.env_data = {'pose': None, 'image': None}
        # initial robot pose of [X, Z, Yaw]; it is Z coordinate here because the agent is moving on the XZ plane
        self.robot_pose = np.array([0.0, 0.0, 0.0])

    def step_simulation_without_physics(self, x: float, z: float, yaw: float) -> (np.ndarray, np.ndarray):
        '''
        This function propels the simulation. It uses teleportation to guide the agent/robot directly by means of global x, z, yaw values.
        
        Parameters
        ----------
        x :                                 The global x position to teleport to.
        z :                                 The global z position to teleport to.
        yaw :                               The global yaw value to teleport to.
        
        Returns
        ----------
        image_data :                        The image observation received from the simulation.
        pose_data :                         The pose observation received from the simulation.
        '''
        # send teleported positions to Unity and retrieve the observation
        # step the env with the provided action.
        # the orientation of the coordinate sys in Unity and here is different such that the sum of two is phase 90 degree
        action = ActionTuple(np.asarray([[x, z, 90 - yaw]]))
        self.env.set_actions(self.behavior_name, action)
        # forward the simulation by a tick (and execute action)
        self.env.step()
        # get the camera observation, time of the simulation and pose of the robot
        image_data, pose_data = self.__get_step_results()
        # update robot's/agent's pose
        self.robot_pose = np.array(pose_data)
        # update environmental information
        self.env_data['image'] = image_data
        self.env_data['pose'] = pose_data
        
        return image_data, pose_data
    
    def step_simulation(self, velocityLinear: float, omega: float):
        '''
        This function propels the simulation. It uses physics to guide the agent/robot with linear and angular velocities.
        
        Parameters
        ----------
        velocityLinear :                    The requested linear (translational) velocity of the agent/robot.
        omega :                             The requested rotational (angular) velocity of the agent/robot.
        
        Returns
        ----------
        None
        '''
        pass

    def __get_step_results(self) -> (np.ndarray, np.ndarray):
        '''
        This function returns image and pose data.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        image_data :                        The image observation received from the simulation.
        pose_data :                         The pose observation received from the simulation.
        '''
        # get the step result for our agent
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        # get the camera observations
        image_data = decision_steps.obs[0][0, :, :, :]
        # receive the robot pose data via the string channel
        addi_info = self.env_channel.received_info
        # clear the received info, very important to do so; otherwise mess up the received info
        self.env_channel.clear_received_info()
        pose_data = [float(x) for x in addi_info[-1].split()]
        # the orientation of the coordinate sys in Unity and here is different such that the sum of two is phase 90 degree
        pose_data[-1] = 90 - pose_data[-1]

        return image_data, pose_data

    def actuate_robot(self, actuator_command: list) -> (np.ndarray, np.ndarray):
        '''
        This function actually actuates the agent/robot in the virtual environment.
        
        Parameters
        ----------
        actuator_command :                  The command that is used in the actuation process.
        
        Returns
        ----------
        image_data :                        The image observation received from the simulation.
        pose_data :                         The pose observation received from the simulation.
        '''
        # if the actuator command has more than 2 array entries, this is a teleport command, and will cause a teleport jump of the agent/robot (no physics support)
        if actuator_command.shape[0] > 2:
            # call the teleportation routine
            image_data, pose_data = self.step_simulation_without_physics(actuator_command[0], actuator_command[1], actuator_command[2])
            # return the data acquired from the robot/agent/environment
            return image_data, pose_data
        else:
            # otherwise, this is a standard motion command with physics support (the agent/robot approaches the goal by actuating the robot's/agent's wheels)
            image_data, pose_data = self.step_simulation(actuator_command[0], actuator_command[1])
            # return the data acquired from the robot/agent/environment
            return image_data, pose_data

    def set_XYYaw(self, pose: list):
        '''
        This function teleports the robot to a novel pose.
        
        Parameters
        ----------
        pose :                              The pose to teleport to, format: [x, z, yaw].
        
        Returns
        ----------
        None
        '''
        action = ActionTuple(pose)
        self.env.set_actions(self.behavior_name, action)
        # forward the simulation by a tick (and execute action)
        self.env.step()
        
    def set_topology(self, topology_module):
        '''
        This function supplies the interface with a valid topology module.
        
        Parameters
        ----------
        topology_module :                   The topology module to be supplied.
        
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
        return np.asarray([[self.world_limits[0], self.world_limits[2]], [self.world_limits[1], self.world_limits[3]]])

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
        return self.walls_limits, self.perimeter_nodes

    def stop_unity(self):
        '''
        This function shuts down Unity.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        self.env.close()


class FrontendUnityOfflineInterface(FrontendUnityInterface):
    
    def __init__(self, scenario_name=None):
        '''
        The Unity interface class. This is tailored for offline simulation.
        
        Parameters
        ----------
        scenario_name :                     The name of the scenario that should be processed.
        
        Returns
        ----------
        None
        '''
        # load the saved images for the environment
        data = None
        try:
            with open(scenario_name, 'rb') as handle:
                data = pickle.load(handle)
        except FileNotFoundError:
            raise Exception('Offline environment for %s, step_size=%s does not exist, please use script environments/generate_off_unity.py to generate it.' % scenario_name)
        # Here env is a dict containing the topology idx as keys and corresponding images as values
        self.env = data
        self.world_limits = data['world_limits']
        self.walls_limits = data['walls_limits']
        self.perimeter_nodes = data['perimeter_nodes']
        # a dict that stores environmental information in each time step
        self.env_data = {'pose': None, 'image': None}
        # flag to tell other modules that it is now offline mode
        self.offline = True
        self.scenario_name = scenario_name

    def step_simulation_without_physics(self, node: int, yaw: float):
        '''
        This function propels the simulation. It uses teleportation to guide the agent/robot directly by means of global x, z, yaw values.
        
        Parameters
        ----------
        node :                              The node to teleport to.
        yaw :                               The global yaw value to teleport to.
        
        Returns
        ----------
        image_data :                        The image observation received from the simulation.
        pose_data :                         The pose observation received from the simulation.
        '''
        # get the camera observation, time of the simulation and pose of the robot
        image_data = self.env[str([node, yaw])]
        pose_data = np.zeros(3) # Here 3 dimension is only for being compatible with other modules
        pose_data[2] = yaw
        # update environmental information
        self.env_data['image'] = image_data
        self.env_data['pose'] = pose_data
        
        return image_data, pose_data

    def actuate_robot(self, actuator_command):
        '''
        This function actually actuates the agent/robot in the offline virtual environment.
        
        Parameters
        ----------
        actuator_command :                  The command that is used in the actuation process.
        
        Returns
        ----------
        image_data :                        The image observation received from the simulation.
        pose_data :                         The pose observation received from the simulation.
        '''
        return self.step_simulation_without_physics(actuator_command[0], actuator_command[1])
    
    def stop_unity(self):
        '''
        This function clears the offline environment.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        del self.env
