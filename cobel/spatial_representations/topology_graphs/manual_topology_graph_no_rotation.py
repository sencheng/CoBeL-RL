# basic imports
import numpy as np
# Qt
import PyQt5 as qt
import pyqtgraph as qg
import pyqtgraph.functions
# OpenAI Gym
import gym
# framework imports
from .misc.topology_node import TopologyNode
from .misc.cog_arrow import CogArrow
from cobel.spatial_representations.spatial_representation import SpatialRepresentation


class ManualTopologyGraphNoRotation(SpatialRepresentation):
    
    #TODO : should not take modules as input, specific inputs
    def __init__(self, modules: dict, graph_info: dict):
        '''
        Manually defined topology graph module without rotation.
        
        Parameters
        ----------
        modules :                           Dictionary containing the framework modules.
        graph_info :                        Dictionary containing topology graph relevant information.
        
        Returns
        ----------
        None
        '''
        # call the base class init
        super(ManualTopologyGraphNoRotation, self).__init__()
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
    
    def set_visual_debugging(self, visual_output: bool, gui_parent):
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
        #TODO : make different parts of visualization optional
        # overlay the policy arrows
        if self.visual_output:
            # for all nodes in the topology graph
            #TODO : sample state space here
            for node in self.nodes:
                # query the model at each node's position
                # only for valid nodes!
                if node.index != -1:
                    observation = self.state_space[node.index]
                    data = np.array([observation])
                    # get the q-values at the queried node's position
                    q_values = self.rl_agent.predict_on_batch(data)[0]
                    # find all neighbors that are actually valid (index != -1)
                    valid_index = 0
                    for n_index in range(len(node.neighbors)):
                        if node.neighbors[n_index].index != -1:
                            valid_index = n_index  
                    # find the index of the neighboring node that is 'pointed to' by the highest q-value, AND is valid!
                    max_neigh_node = node.neighbors[np.argmax(q_values[:valid_index+1])]
                    # find the direction of the selected neighboring node
                    # to node: maxNeighNode
                    to_node = np.array([max_neigh_node.x, max_neigh_node.y])
                    # from node: node
                    from_node = np.array([node.x, node.y])
                    # the difference vector between to and from
                    vec = to_node - from_node
                    # normalize the direction vector
                    l = np.linalg.norm(vec)
                    vec = vec/l
                    # make the corresponding indicator point in the direction of the difference vector
                    node.q_indicator.set_data(node.x, node.y, np.rad2deg(np.arctan2(vec[1], vec[0])))  

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
            self.pos_marker.set_data(pose[0],pose[1],np.rad2deg(np.arctan2(pose[3], pose[2])))

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
        # TODO : test what this does
        # the world module is required here
        world_module = self.modules['world']
        # the observation module is required here
        observation_module = self.modules['observation']
        # In this specific topology graph, a state is an image sampled from a specific node of the graph. There
        # is no rotation, so one image per node is sufficient.
        self.state_space = []
        for node_index in range(len(self.nodes)):
            node = self.nodes[node_index]
            # set agent to x/y-position of 'node'
            world_module.step_simulation_without_physics(node.x, node.y, 90.0)
            observation_module.update()
            observation = observation_module.observation
            self.state_space += [observation]

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
        next_node_pos = np.array([0.0, 0.0])
        callback_value = dict()
        # if a standard action is performed
        if action != 'reset':
            previous_node = self.current_node
            # with action given, the next node can be computed
            # TODO :remove dependence on same module
            self.next_node = self.nodes[self.current_node].neighbors[action].index
            # array to store the next node's coordinates
            if self.next_node != -1:
                # compute the next node's coordinates
                next_node_pos = np.array([self.nodes[self.next_node].x,
                                          self.nodes[self.next_node].y])
            else:
                # if the next node corresponds to an invalid node, the agent stays in place
                self.next_node = self.current_node
                # prevent the agent from starting any motion pattern
                self.modules['world'].goal_reached = True
                next_node_pos = np.array([self.nodes[self.current_node].x,
                                          self.nodes[self.current_node].y])
            # here, next node is already set and the current node is set to this next node.
            # TODO : make callbacks not mandatory
            callback_value['current_node'] = self.nodes[self.next_node]
        # if a reset is performed
        else:
            # a random node is chosen to place the agent at (this node MUST NOT be the global goal node!)
            next_node = -1
            while True:
                next_node = np.random.randint(len(self.nodes))
                if self.nodes[next_node].start_node:
                    break
            next_node_pos = np.array([self.nodes[next_node].x, self.nodes[next_node].y])
            self.next_node = next_node
        # actually move the robot to the node
        self.modules['world'].actuate_robot(np.array([next_node_pos[0], next_node_pos[1], 90.])) 
        # make the current node the one the agent travelled to
        self.current_node = self.next_node
        self.modules['observation'].update()
        self.update_robot_pose([next_node_pos[0], next_node_pos[1], 0., 1.])
        self.update_visual_elements()
        # if possible try to update the visual debugging display
        if hasattr(qt.QtGui, 'QApplication'):
            if qt.QtGui.QApplication.instance() is not None:
                qt.QtGui.QApplication.instance().processEvents()
        else:
            if qt.QtWidgets.QApplication.instance() is not None:
                qt.QtWidgets.QApplication.instance().processEvents()
        
        return callback_value

    def get_action_space(self) -> gym.spaces.Discrete:
        '''
        This function returns the action space (i.e. clique size).
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        action_space :                      The (gym) action space.
        '''
        return gym.spaces.Discrete(self.clique_size)


class ManualTopologyGraphNoRotationDynamic(ManualTopologyGraphNoRotation):
    
    def __init__(self, modules: dict, graph_info: dict):
        '''
        Manually defined topology graph module with dynamically changing barriers.
        
        Parameters
        ----------
        modules :                           Dictionary containing the framework modules.
        graph_info :                        Dictionary containing topology graph relevant information.
        
        Returns
        ----------
        None
        '''
        # call the base class init
        super().__init__(modules, graph_info)
        
    def reload(self):
        '''
        This funtion reloads the topology graph based on the changes in the environment,
        without the need to initialize a new object after each change.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # the world module is required here
        world_module = self.modules['world']
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
        self.clique_size = self.graph_info['clique_size']
        # set up a manually constructed topology graph
        # read topology structure from world module
        nodes = np.array(world_module.get_manually_defined_topology_nodes())
        nodes = nodes[nodes[:, 0].argsort()]
        edges = np.array(world_module.get_manually_defined_topology_edges())
        edges = edges[edges[:, 0].argsort()]
        # transfer the node points into the self.nodes list
        for i, n in enumerate(nodes):
            # create the corresponding node, where i is the running index of the mesh_points/corresponding nodes
            self.nodes.append(TopologyNode(i, float(n[1]), float(n[2])))
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
        # resample state space and reload visual elements
        self.sample_state_space()
        self.reload_visual_elements()
        
    def reload_visual_elements(self):
        '''
        This function reloads the visual elements. It is called after the topology graph has been changed.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        if self.visual_output:
            #self.plot = plot
            self.plot.clear()
            # set extension of the plot, lock aspect ratio
            self.plot.setXRange(self.world_limits[0, 0], self.world_limits[0, 1])
            self.plot.setYRange(self.world_limits[1, 0], self.world_limits[1, 1])
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
            symbol_brushes = [qg.mkBrush(color=(128, 128, 128))] * len(self.nodes)
            # set colors of normal and goal nodes
            for node in self.nodes:
                # start nodes are green
                if node.start_node:
                    symbol_brushes[node.index] = qg.mkBrush(color=(0, 255, 0))
                # goal node is red
                if node.goal_node:
                    symbol_brushes[node.index] = qg.mkBrush(color=(255, 0, 0))
            # construct appropriate arrays from the self.nodes and the self.edges information
            temp_nodes, temp_edges =[], []
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
            
    def is_traversable(self) -> bool:
        '''
        This function checks if the graph is traversable from start to goal.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        is_traversable :                    Flag indicating whether the graph is traverable or not.
        '''
        traversed_list = np.zeros(len(self.nodes))
        to_traverse = []
        for node in self.nodes:
            if node.start_node == True:
                start_node = node
            elif node.goal_node == True:
                end_node = node
        to_traverse.append(start_node)
        traversed_list[start_node.index] = 1
        while len(to_traverse) > 0:
            current_node = to_traverse.pop(0)
            if current_node.goal_node == True:
                return True
            for node in current_node.neighbors:
                if node.index != -1:
                    if traversed_list[node.index] == 0:
                        to_traverse.append(node)
                        traversed_list[node.index] = 1

        return False
