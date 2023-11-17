# basic imports
import numpy as np
# Qt
import PyQt5 as qt
import pyqtgraph as qg
import pyqtgraph.functions
# OpenAI Gym
import gymnasium as gym
# framework imports
from .misc.topology_node import TopologyNode
from .misc.cog_arrow import CogArrow
from cobel.spatial_representations.spatial_representation import SpatialRepresentation


class ManualTopologyGraphWithRotation(SpatialRepresentation):
    
    def __init__(self, modules: dict, graph_info: dict):
        '''
        Manually defined topology graph module with rotation.
        
        Parameters
        ----------
        modules :                           Dictionary containing the framework modules.
        graph_info :                        Dictionary containing topology graph relevant information.
        
        Returns
        ----------
        None
        '''
        # call the base class init
        super(ManualTopologyGraphWithRotation,self).__init__()
        # normally, the topology graph is not shown in gui_parent
        self.visual_output = False
        self.gui_parent = None
        #store the graph parameters
        self.graph_info = graph_info
        # extract the world module
        self.modules = modules
        # the world module is required here
        #TODO : if world_module is not None : 
        world_module = modules['world']
        world_module.set_topology(self)
        # get the limits of the given environment
        self.world_limits = world_module.get_limits()
        # retrieve all boundary information from the environment
        self.world_nodes, self.world_edges = world_module.get_wall_graph()
        # inherent definitions for the topology graph
        # this is the node corresponding to the robot's actual position
        self.current_node = -1
        # this is the node corresponding to the robot's next position
        self.next_node = -1
        # this list of topologyNode[s] stores all nodes of the graph
        self.nodes = []
        # this list of [int,int]-entries defines all edges (unique) which make up the graph's connectivity
        self.edges = []
        self.clique_size = graph_info['clique_size']
        # set up a manually constructed topology graph
        # read topology structure from world module
        nodes = np.array(world_module.get_manually_defined_topology_nodes())
        nodes = nodes[nodes[:, 0].argsort()]
        edges = np.array(world_module.get_manually_defined_topology_edges())
        edges = edges[edges[:, 0].argsort()]
        # transfer the node points into the self.nodes list
        for idx, n in enumerate(nodes):
            # create the corresponding node, where i is the running index of the mesh_points/corresponding nodes
            node = TopologyNode(idx, float(n[1]), float(n[2]))
            self.nodes.append(node)
        # fill in the self.edges list from the edges information
        for e in edges:
            self.edges.append([int(e[1]), int(e[2])])
        # define a dedicated 'noneNode' that acts as a placeholder for neighborhood construction
        none_node = TopologyNode(-1, 0.0, 0.0)
        # comstruct the neighborhoods of each node in the graph
        for edge in self.edges:
            # first edge node
            a = self.nodes[int(edge[0])]
            # second edge node
            b = self.nodes[int(edge[1])]
            # add node a to the neighborhood of node b, and vice versa
            a.neighbors.append(b)
            b.neighbors.append(a)
        # it is possible that a node does not have the maximum possible number of neighbors, to stay consistent in RL, fill up the neighborhood
        # with noneNode[s]:
        for node in self.nodes:
            while len(node.neighbors) < self.clique_size:
                node.neighbors.append(none_node)
        # assign start nodes
        for node_index in self.graph_info['start_nodes']:
            self.nodes[node_index].start_node = True
        # assign goal nodes
        for node_index in self.graph_info['goal_nodes']:
            self.nodes[node_index].goal_node = True
        #TODO : Test : Remove from class definition if it is only being used for visualization
        self.sample_state_space()
        
    def set_visual_debugging(self, visual_output: bool, gui_parent: qg.GraphicsLayoutWidget):
        '''
        This function sets visualization flags.
        
        Parameters
        ----------
        visual_output :                     If true, the topology graph will be visualized.
        gui_parent :                        The main window used during visualization.
        
        Returns
        ----------
        None
        '''
        self.gui_parent = gui_parent
        self.visual_output = visual_output
        self.init_visual_elements()

    def init_visual_elements(self):
        '''
        This function initializes the visual elements.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        if self.visual_output:
            # add the graph plot to the GUI widget
            self.plot = self.gui_parent.addPlot(title='Topology graph')
            # set extension of the plot, lock aspect ratio
            self.plot.setXRange( self.world_limits[0, 0], self.world_limits[0, 1] )
            self.plot.setYRange( self.world_limits[1, 0], self.world_limits[1, 1] )
            self.plot.setAspectLocked()
            # set up indicator arrows for each node, except the goal node, and all nodes in active shock zones iff shock zones exist
            for node in self.nodes:
                if not node.goal_node:
                    node.q_indicator = CogArrow(angle=0.0, headLen=20.0, tipAngle=25.0, tailLen=0.0, brush=(255, 255, 0))
                    self.plot.addItem(node.q_indicator)
            # overlay the world's perimeter
            self.perimeter_graph = qg.GraphItem()
            self.plot.addItem(self.perimeter_graph)
            self.perimeter_graph.setData(pos=np.array(self.world_nodes), adj=np.array(self.world_edges), brush=(128, 128, 128))
            # overlay the topology graph
            self.topology_graph = qg.GraphItem()
            self.plot.addItem(self.topology_graph)
            # set up a brushes array for visualization of the nodes
            # normal nodes are grey
            symbol_brushes=[qg.mkBrush(color=(128, 128, 128))] * len(self.nodes)
            # set colors of normal and goal nodes
            for node in self.nodes:    
                # start nodes are green
                if node.start_node:
                    symbol_brushes[node.index] = qg.mkBrush(color=(0, 255, 0))
                # goal node is red
                if node.goal_node:
                    symbol_brushes[node.index] = qg.mkBrush(color=(255, 0, 0))
            # construct appropriate arrays from the self.nodes and the self.edges information
            temp_nodes, temp_edges = [], []
            for node in self.nodes:
                temp_nodes.append([node.x, node.y])
            for edge in self.edges:
                temp_edges.append([edge[0], edge[1]])
            self.topology_graph.setData(pos=np.array(temp_nodes), adj=np.array(temp_edges), symbolBrush=symbol_brushes)
            # eventually, overlay robot marker
            self.pos_marker = CogArrow(angle=0.0, headLen=20.0, tipAngle=25.0, tailLen=0.0, brush=(255, 0, 0))
            self.plot.addItem(self.pos_marker)
            # initial position to center, this has to be worked over later!
            self.pos_marker.set_data(0.0, 0.0, 0.0)
        
    def update_visual_elements(self):
        '''
        This function updates the visual elements.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # not currently used
        pass
    
    def update_robot_pose(self, pose: np.ndarray):
        '''
        This function updates the visual depiction of the agent(robot).
        
        Parameters
        ----------
        pose :                              The agent's pose.
        
        Returns
        ----------
        None
        '''
        if self.visual_output:
            self.pos_marker.set_data(pose[0],pose[1],np.rad2deg(np.arctan2(pose[3],pose[2])))

    def sample_state_space(self):
        '''
        This function samples observations at all topology nodes.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # not currently used
        pass

    def generate_behavior_from_action(self, action: int) -> dict:
        '''
        This function executes the action selected by the agent.
        
        Parameters
        ----------
        action :                            The action to be executed.
        
        Returns
        ----------
        callback_value :                    Dictionary containing information about the new environmental state.
        '''
        # the world module is required here
        world_module = self.modules['world']
        next_node_pos = np.array([0.0,0.0])
        callback_value = dict()
        # if a standard action is performed, NOT a reset
        if action != 'reset':
            # get current heading
            heading = np.array([world_module.env_data['pose'][2], world_module.env_data['pose'][3]])
            heading = heading/np.linalg.norm(heading)
            # get directions of all edges
            actual_node = self.nodes[self.current_node]
            neighbors = self.nodes[self.current_node].neighbors
            # lists for edges
            left_edges, right_edges, forward_edge = [], [], []
            # find possible movement directions. Note: when a left edge is found, it is simultaneously stored as a right edge with huge turning angle, and vice versa. That way,
            # the agent does not get stuck in situations where there is only a forward edge, and say, a left edge, and the action is 'right'. In such a situation, the agent will just turn
            # right using the huge 'right' turning angle.
            for n in neighbors:
                if n.index != -1:
                    actual_node_position = np.array([actual_node.x, actual_node.y])
                    neighbor_position = np.array([n.x, n.y])
                    vec_edge = neighbor_position - actual_node_position
                    vec_edge = vec_edge/np.linalg.norm(vec_edge)
                    angle = np.arctan2(heading[0]*vec_edge[1]-heading[1]*vec_edge[0], heading[0]*vec_edge[0]+heading[1]*vec_edge[1])
                    angle = angle/np.pi*180.0
                    if angle < -1e-5:
                        right_edges.append([n.index, vec_edge, angle])
                        left_edges.append([n.index, vec_edge, (360.0+angle)])
                    if angle > 1e-5:
                        right_edges.append([n.index, vec_edge, angle])
                        left_edges.append([n.index, vec_edge, -(360.0-angle)])
                    if angle < 1e-5 and angle >- 1e-5:
                        forward_edge = [n.index, vec_edge, angle]
            # sort left and right edges in such a way that the smallest angular difference is placed in front
            left_edges = sorted(left_edges, key=lambda element: element[2], reverse=False)
            right_edges = sorted(right_edges, key=lambda element: element[2], reverse=True)
            # store the current node as previous node
            previous_node = self.current_node
            # with action given, the next node can be computed
            if action == 0:
                # this is a forward movement
                angle = 180.0/np.pi*np.arctan2(heading[1], heading[0])
                if len(forward_edge) != 0:
                    # there is a forward edge that the agent can use
                    self.next_node = forward_edge[0]
                    next_node_pos = np.array([self.nodes[self.next_node].x, self.nodes[self.next_node].y])
                else:
                    # no forward edge found, the agent has to wait for a rotation action
                    self.next_node = self.current_node
                    next_node_pos = np.array([self.nodes[self.next_node].x, self.nodes[self.next_node].y])
                self.update_robot_pose([next_node_pos[0], next_node_pos[1], heading[0], heading[1]])
                self.modules['world'].actuate_robot(np.array([next_node_pos[0], next_node_pos[1], angle])) 
            elif action == 1:
                # this is a left turn movement
                self.next_node = self.current_node
                next_node_pos = np.array([self.nodes[self.next_node].x, self.nodes[self.next_node].y])
                angle = 180.0/np.pi*np.arctan2(left_edges[0][1][1], left_edges[0][1][0])
                self.update_robot_pose([next_node_pos[0], next_node_pos[1], left_edges[0][1][0], left_edges[0][1][1]])
                self.modules['world'].actuate_robot(np.array([next_node_pos[0], next_node_pos[1], angle])) 
            elif action == 2:
                # this is a right turn movement
                self.next_node = self.current_node
                next_node_pos = np.array([self.nodes[self.next_node].x, self.nodes[self.next_node].y])
                angle = 180.0/np.pi*np.arctan2(right_edges[0][1][1], right_edges[0][1][0])
                self.update_robot_pose([next_node_pos[0], next_node_pos[1], right_edges[0][1][0], right_edges[0][1][1]])
                self.modules['world'].actuate_robot(np.array([next_node_pos[0], next_node_pos[1], angle])) 
            # update the observation
            self.modules['observation'].update()
            # make the current node the one the agent travelled to
            self.current_node = self.next_node
            # here, next node is already set and the current node is set to this next node.
            callback_value['current_node'] = self.nodes[self.next_node]
        # if a reset is performed
        else:
            # a random node is chosen to place the agent at (this node MUST NOT be the global goal node!)
            nodes = self.nodes
            nodes_selection = [n.index for n in nodes if n.start_node == True]
            # store the current node as previous node
            previous_node = self.current_node
            self.next_node = np.random.choice(nodes_selection)
            next_node_pos = np.array([self.nodes[self.next_node].x, self.nodes[self.next_node].y])
            # from all heading directions available at the chosen node, select one randomly
            self.current_node = self.next_node
            neighbors = self.nodes[self.next_node].neighbors
            # list for available neighbor directions
            directions = []
            for n in neighbors:
                if n.index != -1:
                    # only parse valid neighbors
                    next_node_position = np.array([self.nodes[self.next_node].x, self.nodes[self.next_node].y])
                    neighbor_position = np.array([n.x, n.y])
                    vec_edge = neighbor_position - next_node_position
                    vec_edge = vec_edge/np.linalg.norm(vec_edge)
                    world_angle = np.arctan2(vec_edge[1],vec_edge[0])
                    directions.append([n.index, vec_edge, world_angle])                    
            # select new heading randomly
            new_heading_selection = directions[np.random.randint(len(directions))]
            new_heading_angle = new_heading_selection[2]
            new_heading_vector = new_heading_selection[1]
            # update the agents position and orientation (heading)
            self.modules['world'].actuate_robot(np.array([next_node_pos[0], next_node_pos[1], new_heading_angle])) 
            self.update_robot_pose([next_node_pos[0], next_node_pos[1], new_heading_vector[0], new_heading_vector[1]])
            # update the observation
            self.modules['observation'].update()
        if hasattr(qt.QtGui, 'QApplication'):
            if qt.QtGui.QApplication.instance() is not None:
                qt.QtGui.QApplication.instance().processEvents()
        else:
            if qt.QtWidgets.QApplication.instance() is not None:
                qt.QtWidgets.QApplication.instance().processEvents()

        return callback_value

    def get_action_space(self) -> gym.spaces.Discrete:
        '''
        This function returns the action space.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        action_space :                      The (gym) action space.
        '''
        return gym.spaces.Discrete(3)
