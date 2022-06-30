# basic imports
import numpy as np
import gym
import random
from scipy.spatial import Delaunay
import PyQt5 as qt
import pyqtgraph as qg
import pyqtgraph.functions
# framework imports
from cobel.spatial_representations.spatial_representation import SpatialRepresentation
from cobel.spatial_representations.topology_graphs.misc.topology_node import TopologyNode
from cobel.spatial_representations.topology_graphs.misc.cog_arrow import CogArrow


class AbstractTopologyGraph(SpatialRepresentation) : 
    '''
    Abstract topology graph class.
    
    | **Args**
    | n_nodes_x:                    The number of nodes along the x axis.
    | n_nodes_y:                    The number of nodes along the y axis.
    | n_neighbors:                  The number of neighbors of each node.
    | visual_output:                If true, then the topology graph will be visualized.
    '''
    def __init__(self, n_nodes_x=5, n_nodes_y=5, n_neighbors=4, 
                 visual_output=None, **kwargs) : 
        # node parameters
        self.n_nodes_x = n_nodes_x
        self.n_nodes_y = n_nodes_y
        self.n_neighbors = n_neighbors
        # the nodes
        self.nodes = []
        self.edges = []
        # RL relevant info
        self.current_node = None
        self.next_node = None
        # visualization flag
        self.visual_output = None
    
    def get_nodes(self):
        '''
        This function returns a list of nodes
        '''
        return self.nodes
    
    def get_edges(self):
        '''
        This function returns a list of edges.
        '''
        return self.edges
    
    def set_visual_debugging(self):
        '''
        Sets visualization flag.
        '''
        pass
    
    def get_action_space(self):
        '''
        Returns the action space.
        '''
        pass
    
    def generate_behavior_from_action(self, action):
        '''
        Moves the agent on the topology graph according the action selected by the agent.
        
        | **Args**
        | action:                       The action that was selected by the agent.
        '''
        pass
    
    def sample_state_space(self):
        '''
        Samples observations for all nodes from the environment.
        '''
        pass
    
    
class GridGraph(AbstractTopologyGraph): 
    '''
    Square/Rectangular grid.
    
     **Args**
    | n_nodes_x:                    The number of nodes along the x axis.
    | n_nodes_y:                    The number of nodes along the y axis.
    | n_neighbors:                  The number of neighbors of each node.
    | x_limits:                     The range of x coordinates.
    | y_limits:                     The range of y coordinates.
    | start_nodes:                  The list of starting nodes.
    | goal_nodes:                   The list of goal nodes.
    | visual_output:                If true, then the topology graph will be visualized.
    | rotation:                     If true, the agent can also change its orientation.
    | world_module:                 The world module.
    | use_world_limits:             If true, limits on x/y coordinates are taken into account.
    | observation_module:           The observation module.
    '''
    def __init__(self, n_nodes_x=5, n_nodes_y=5, n_neighbors=4,
                 x_limits=[-1,1], y_limits=[-1,1], start_nodes=None,
                 goal_nodes=None, visual_output=None, rotation=False, 
                 world_module=None, use_world_limits=False, 
                 observation_module=None):
        super().__init__(n_nodes_x=5, n_nodes_y=5, n_neighbors=4, 
                         visual_output=None)
        # action space
        self.rotation = rotation
        # world info
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.use_world_limits = use_world_limits
        # module references
        self.world_module = world_module
        self.observation_module = observation_module
        # retrieve info from world module
        if self.world_module is not None : 
            self.world_limits = self.world_module.getLimits()
            self.world_nodes, self.world_edges = world_module.getWallGraph()
            if self.use_world_limits:
                self.x_limits = [self.world_limits[0,0], self.world_limits[0,1]]
                self.y_limits = [self.world_limits[1,0], self.world_limits[1,1]]
            #TODO : catch exceptions!
        #Build the graph
        grid_points = self.find_grid_points(self.n_nodes_x, self.n_nodes_y, 
                                       self.x_limits, self.y_limits)
        self.build_graph(grid_points)
        self.n_nodes = len(self.nodes)    
        # RL relevant info
        self.current_node = -1
        self.next_node    = -1
        # starting and goal nodes
        self.start_nodes = start_nodes
        self.goal_nodes  = goal_nodes
        # set starting nodes
        if self.start_nodes is not None:
            for node_idx in self.start_nodes:
                self.nodes[node_idx].startNode = True
        # if none were defined consider all nodes as starting nodes
        else:
            for node in self.nodes : 
                node.startNode = True
        # set goal nodes
        if self.goal_nodes is not None:
            for node_idx in self.goal_nodes:
                self.nodes[node_idx].goalNode = True
    
    @staticmethod
    def find_grid_points(n_nodes_x, n_nodes_y, x_limits, y_limits):
        '''
        This function finds the coordinates for all nodes.
        
         **Args**
        | n_nodes_x:                    The number of nodes along the x axis.
        | n_nodes_y:                    The number of nodes along the y axis.
        | x_limits:                     The range of x coordinates.
        | y_limits:                     The range of y coordinates.
        '''
        nodes_x = np.linspace(x_limits[0], x_limits[1], n_nodes_x)
        nodes_y = np.linspace(y_limits[0], y_limits[1], n_nodes_y)
        grid_points = [[x, y] for y in nodes_y for x in nodes_x]
        
        return grid_points

    def build_graph(self, grid_points):
        """
        
        Parameters
        ----------
        grid_points : LIST 
            list of x,y coordinates of nodes in the grid
            
            Builds the actual nodes (type : topologynode object) and edges and 
            fills in the neighborhood corresponding to the following scheme : 
                0 - RIGHT
                1 - TOP
                2 - LEFT
                3 - BOTTOM
        Returns
        -------
        None.

        """
        for idx, point in enumerate(grid_points):
            self.nodes.append(TopologyNode(idx, point[0], point[1]))
        
        none_node = TopologyNode(-1,0.0,0.0)
        
        for n in self.nodes : 
            n.neighbors = np.full(4, none_node)
        
        bottom_nodes = self.nodes[:self.n_nodes_x]
        top_nodes    = self.nodes[-self.n_nodes_x:]
        left_nodes   = self.nodes[0::self.n_nodes_x]
        right_nodes  = self.nodes[self.n_nodes_x-1::self.n_nodes_x]
        
        for node in self.nodes : 
            if node not in right_nodes : 
                node.neighbors[0] = self.nodes[node.index + 1]
            if node not in top_nodes : 
                node.neighbors[1] = self.nodes[node.index + self.n_nodes_x]
            if node not in left_nodes : 
                node.neighbors[2] = self.nodes[node.index - 1]
            if node not in bottom_nodes : 
                node.neighbors[3] = self.nodes[node.index - self.n_nodes_x]
       
        edges = []
        for n in self.nodes : 
            [edges.append([n.index,neighbor.index]) for neighbor in n.neighbors]
        
        [e.sort() for e in edges]
        edges = [e for e in edges if e[0]!=-1]
        
        unique_edges = [list(edge) for edge in set(tuple(edge) for edge in edges)]           
        unique_edges.sort()
        self.edges = unique_edges

    def set_rotation(self, rotation):
        '''
        This function enables/disables rotation.
        '''
        self.rotation = rotation    

    def update_position_marker(self, pose):
        '''
        This function updates the position marker.
        
         **Args**
        | pose:                         The agent's current pose.
        '''
        self.position_marker.setData(pose[0],pose[1],
                                     np.rad2deg(np.arctan2(pose[3], pose[2])))
    
    def generate_behavior_from_action(self, action):
        '''
        Moves the agent on the topology graph according the action selected by the agent.
        
        | **Args**
        | action:                       The action that was selected by the agent.
        '''
        next_node_pos  = np.array([0.0,0.0])
        
        callback_value = dict()
        
        if not self.rotation :
            
            if action!='reset' : 
                node_id = self.nodes[self.current_node].neighbors[action].index
                if node_id != -1 : 
                    self.move_to_node(node_id)
                else  :
                    self.next_node = self.current_node
                    self.world_module.goalReached = True
                                                    
            else : 
                node_id = np.random.choice(self.start_nodes)
                self.move_to_node(node_id)
                            
            self.current_node = self.next_node
            callback_value['currentNode'] = self.nodes[self.current_node]
        else : #ROTATION IS ENABLED
            
            if action!='reset':    
                heading = np.array([self.world_module.envData['poseData'][2],
                                    self.world_module.envData['poseData'][3]])
                
                heading = heading/np.linalg.norm(heading)
                node = self.nodes[self.current_node]

                forward_edge, left_edges, right_edges = self.calculate_angles_edges(node,heading,threshold=5)[1:]
                
                if action == 0 : 
                    # MOVE FORWARD
                    angle = 180.0 / np.pi * np.arctan2(heading[1], heading[0])
                    if len(forward_edge) != 0 :
                        next_node_id = forward_edge[0]
                        self.move_to_node(next_node_id, angle)
                    else : 
                        self.next_node = self.current_node
                        
                if action == 1 : 
                    # TURN LEFT
                    angle = 180.0 / np.pi * np.arctan2(left_edges[0][1][1],
                                                     left_edges[0][1][0])
                    next_node_id = self.current_node
                    self.move_to_node(next_node_id, angle)
                    
                if action == 2 : 
                    #TURN RIGHT
                    angle = 180.0 / np.pi * np.arctan2(right_edges[0][1][1],
                                                     right_edges[0][1][0])
                    next_node_id = self.current_node
                    self.move_to_node(next_node_id, angle)     
            else :                 
                node_id = random.choice(self.start_nodes)
                directions = self.calculate_angles_edges(self.nodes[node_id],[0,1])[0]
                random_direction = random.choice(directions)
                self.move_to_node(node_id, random_direction[2])
                
            self.current_node = self.next_node
            callback_value['currentNode'] = self.nodes[self.current_node]
            
        return callback_value
                
    def calculate_angles_edges(self, node, heading, threshold=5):
        '''
        This function computes the angles between a given  node and its neighbors.
        
        | **Args**
        | node:                         The node.
        | heading:                      The heading.
        | threshold:                    The threshold.
        '''
        left_edges   = []
        right_edges  = []
        forward_edge = []
        directions   = []
        
        for n in node.neighbors : 
            if n.index != -1 : 
                current_node_pos = np.array([node.x,
                                             node.y])
                neighbor_pos     = np.array([n.x, n.y])
                edge_vector      = neighbor_pos - current_node_pos
                edge_vector      = edge_vector / np.linalg.norm(edge_vector)
                
                world_angle = np.arctan2(edge_vector[1],edge_vector[0])
                directions += [[n.index,edge_vector,world_angle]]
                
                angle = np.arctan2(heading[0]*edge_vector[1] - heading[1]*edge_vector[0],
                                   heading[0]*edge_vector[0] + heading[1]*edge_vector[1])
                angle = angle / np.pi*180.0
                
                if angle < -threshold:
                    right_edges+=[[n.index,edge_vector,angle]]
                    left_edges+=[[n.index,edge_vector,(360.0+angle)]]
             
                if angle > threshold:
                     left_edges  += [[n.index,edge_vector,angle]]
                     right_edges += [[n.index,edge_vector,-(360.0-angle)]]
                 
                if angle < threshold and angle > -threshold:
                    forward_edge = [n.index,edge_vector,angle]
                    
        left_edges  = sorted(left_edges, key=lambda element: element[2],
                                     reverse=False)
        right_edges = sorted(right_edges, key=lambda element: element[2],
                                     reverse=True)
        
        return directions, forward_edge, left_edges, right_edges
            
    def get_action_space(self): 
        '''
        Returns the action space.
        '''
        if self.rotation:
            return gym.spaces.Discrete(3)
        else:
            return gym.spaces.Discrete(self.n_neighbors)
         
    def set_visual_debugging(self, gui_parent):
        '''
        Sets visualization.
        '''
        self.gui_parent = gui_parent
        self.init_visual_elements()
    
    def init_visual_elements(self):
        '''
        Initializes visualization.
        '''
        if self.visual_output:

            self.plot = self.gui_parent.addPlot(title='Topology graph')

            if self.use_world_limits :
                self.plot.setXRange(self.x_limits[0], self.x_limits[1])
                self.plot.setYRange(self.y_limits[0], self.y_limits[1])
                self.plot.setAspectLocked()

            # set up indicator arrows for each node, except the goal node, and all nodes in active shock zones iff shock zones exist
            for node in self.nodes:
                if not node.goalNode:
                    node.qIndicator = CogArrow(
                        angle=0.0, headLen=20.0, tipAngle=25.0, tailLen=0.0, brush=(255, 255, 0))
                    self.plot.addItem(node.qIndicator)

            # overlay the world's perimeter
            self.perimeterGraph = qg.GraphItem()
            self.plot.addItem(self.perimeterGraph)

            self.perimeterGraph.setData(pos=np.array(self.world_nodes), adj=np.array(
                self.world_edges), brush=(128, 128, 128))

            # overlay the topology graph
            self.topologyGraph = qg.GraphItem()
            self.plot.addItem(self.topologyGraph)

            # set up a brushes array for visualization of the nodes
            # normal nodes are grey
            symbolBrushes = [qg.mkBrush(color=(128, 128, 128))]*len(self.nodes)

            # set colors of normal and goal nodes
            for node in self.nodes:

                # start nodes are green
                if node.startNode:
                    symbolBrushes[node.index] = qg.mkBrush(color=(0, 255, 0))

                # goal node is red
                if node.goalNode:
                    symbolBrushes[node.index] = qg.mkBrush(color=(255, 0, 0))

            # construct appropriate arrays from the self.nodes and the self.edges information
            tempNodes = []
            tempEdges = []
            for node in self.nodes:
                tempNodes = tempNodes+[[node.x, node.y]]

            for edge in self.edges:
                tempEdges = tempEdges+[[edge[0], edge[1]]]

            self.topologyGraph.setData(pos=np.array(tempNodes), adj=np.array(
                tempEdges), symbolBrush=symbolBrushes)

            # eventually, overlay robot marker
            self.position_marker = CogArrow(
                angle=0.0, headLen=20.0, tipAngle=25.0, tailLen=0.0, brush=(255, 0, 0))
            self.plot.addItem(self.position_marker)
            # initial position to center, this has to be worked over later!
            self.position_marker.setData(0.0, 0.0, 0.0)
    
    def update_visual_elements(self):
        '''
        This function is only used for visualization. It generates arrows which points towards the highest q values. 
        
        Parameters
        ----------
        observation_module : It contains the state of the agent for all the nodes in Topology graph
        
        Returns
        --------
        none
        '''

        if not self.rotation : 
            self.sample_state_space() 
                
            if self.visual_output:
                # for all nodes in the topology graph
                #TODO : sample state space here
                for node in self.nodes:
                    # query the model at each node's position
                    # only for valid nodes!
                    if node.index!=-1:
                        observation_module=self.state_space[node.index]
                        data=np.array([[observation_module]])
                        # get the q-values at the queried node's position# 
                        #TODO remove dependency from rlAgent
                        q_values = self.rlAgent.agent.model.predict_on_batch(data)[0]                    
                        # find all neighbors that are actually valid (index != -1)
                        validIndex=0
                        for n_index in range(len(node.neighbors)):
                            if node.neighbors[n_index].index!=-1:
                                validIndex=n_index
                        
                        # find the index of the neighboring node that is 'pointed to' by the highest q-value, AND is valid!
                        maxNeighNode=node.neighbors[np.argmax(q_values[0:validIndex+1])]
                        # find the direction of the selected neighboring node
                        # to node: maxNeighNode
                        toNode=np.array([maxNeighNode.x,maxNeighNode.y])
                        # from node: node
                        fromNode=np.array([node.x,node.y])
                        # the difference vector between to and from
                        vec=toNode-fromNode
                    
                        l=np.linalg.norm(vec)
                        vec=vec/l
                        # make the corresponding indicator point in the direction of the difference vector
                        node.qIndicator.setData(node.x,node.y,np.rad2deg(np.arctan2(vec[1],vec[0])))  
    

    def sample_state_space(self):
        '''
        This function fetches the image data from world module for each node 
        and updates observation
        
        Parameters
        ----------
        module['world'] : returns [timeData,poseData,sensorData,imageData]
        module['observation'] : image data is updated in Observation module
        
        Returns
        --------
        state of the agent i.e. list of images from each node
        '''

        if self.world_module is not None and self.observation_module is not None : 
            self.state_space = []
            for node_index in range(len(self.nodes)):
                #update each node
                node = self.nodes[node_index]
                #retrive image data for each node from the camera
                self.world_module.step_simulation_without_physics(node.x,node.y,90.0)
                #udate observation for every node with the imageData
                self.observation_module.update()
                observation=self.observation_module.observation
                #create a list of imageData for each node
                self.state_space+=[observation]
            #return list of state space
            return
        
    def move_to_node(self, node_id, angle=90.0):
        '''
        This function moves the agent to a specified node.
        
         **Args**
        | node_id:                      The ID of the node that the agent will be moved to.
        | angle:                        The agent's orientation.
        '''
        self.next_node = node_id
        next_node_pos = np.array([self.nodes[self.next_node].x, 
                          self.nodes[self.next_node].y])
        self.world_module.actuateRobot(np.array([next_node_pos[0],
                                         next_node_pos[1],
                                         angle]))
        self.current_node = self.next_node
        self.observation_module.update()
        
        if self.visual_output : 
            self.update_position_marker([next_node_pos[0], next_node_pos[1],
                                         np.cos(np.deg2rad(angle)),  
                                         np.sin(np.deg2rad(angle))])
            
            if hasattr(qt.QtGui, 'QApplication'):
                if qt.QtGui.QApplication.instance() is not None:
                    qt.QtGui.QApplication.instance().processEvents()
            else:
                if qt.QtWidgets.QApplication.instance() is not None:
                    qt.QtWidgets.QApplication.instance().processEvents()
        
class HexagonalGraph(GridGraph):
    '''
    Hexagonal grid.
    
     **Args**
    | n_nodes_x:                    The number of nodes along the x axis.
    | n_nodes_y:                    The number of nodes along the y axis.
    | n_neighbors:                  The number of neighbors of each node.
    | x_limits:                     The range of x coordinates.
    | y_limits:                     The range of y coordinates.
    | start_nodes:                  The list of starting nodes.
    | goal_nodes:                   The list of goal nodes.
    | visual_output:                If true, then the topology graph will be visualized.
    | rotation:                     If true, the agent can also change its orientation.
    | world_module:                 The world module.
    | use_world_limits:             If true, limits on x/y coordinates are taken into account.
    | observation_module:           The observation module.
    '''
    
    def calculate_angle(node_info, neighbor_info) : 
        '''
        Calculates angle.
        
         **Args**
        | node_info:                    The node's info.
        | neighbor_info:                The neighbor's info.
        '''
        ref = np.array([node_info.x,node_info.y-1])
        node = np.array([node_info.x,node_info.y])
        neighbor = np.array([neighbor_info.x,neighbor_info.y])
        ref_vector = ref - node
        vector = neighbor - node
    
        cos_ang = np.dot(ref_vector, vector) / (np.linalg.norm(ref_vector) * np.linalg.norm(vector))
        det = ref_vector[0] * vector[1] - ref_vector[1] * vector[0]
        angle = np.arccos(cos_ang)
        if det>0 :
            return 360 - np.degrees(angle)
        else : 
            return np.degrees(angle)
        
    def sort_graph(self, node_info):
        '''
        This function sorts the graph.
        
         **Args**
        | node_info:                    The node's info.
        '''
        none_node = TopologyNode(-1,0.0,0.0)
        n_sorted_index = np.full(6,none_node)
        
        for n in node_info.neighbors : 
            a = self.calculate_angle(node_info,n)
            if 0 <= a < 89 :
                n_sorted_index[0] = n
            if 89 < a < 149 :
                n_sorted_index[1] = n
            if 149 < a < 209 :
                n_sorted_index[2] = n
            if 209 < a < 269 :
                n_sorted_index[3] = n
            if 269 < a < 329 :
                n_sorted_index[4] = n
            if 329 < a < 360 :
                n_sorted_index[5] = n
                
        node_info.neighbors = n_sorted_index
        
    @staticmethod
    def find_grid_points(n_nodes_x, n_nodes_y, x_limits, y_limits):
        '''
        This function finds the coordinates for all nodes.
        
         **Args**
        | n_nodes_x:                    The number of nodes along the x axis.
        | n_nodes_y:                    The number of nodes along the y axis.
        | x_limits:                     The range of x coordinates.
        | y_limits:                     The range of y coordinates.
        '''
        nodes_x = np.linspace(x_limits[0], x_limits[1], n_nodes_x)
        nodes_y = np.linspace(y_limits[0], y_limits[1], n_nodes_y)
        
        period = nodes_x[1] - nodes_x[0]
        grid_points = np.array([[x, y] for y in nodes_y for x in nodes_x])
        shift_indices =  []
        
        for y in nodes_y[1::2] :
            shift_indices.append(np.where(grid_points[:,1]==y)[0])

        s = np.ravel(shift_indices)
            
        for idx in s:
            grid_points[idx][0] += period/2 

        del_indices = np.where(grid_points[:,0] > x_limits[1])[0]
        grid_points = np.delete(grid_points,del_indices,0)
        
        return grid_points
    
    def build_graph(self, grid_points): 
        """
        
        Parameters
        ----------
        grid_points : LIST 
            list of x,y coordinates of nodes in the grid
            
            Builds the actual nodes (type : topologynode object) and edges and 
            fills in the neighborhood corresponding to the following scheme : 
                0 - RIGHT
                1 - TOP
                2 - LEFT
                3 - BOTTOM
        Returns
        -------
        None.

        """
        
        mesh = Delaunay(grid_points, qhull_options='Qt Qbb Qc')        
        ne = mesh.simplices.shape[0]
        edges = np.array([mesh.simplices[:,0], mesh.simplices[:,1], 
                            mesh.simplices[:,1], mesh.simplices[:,2],
                            mesh.simplices[:,2], mesh.simplices[:,0]]).T.reshape(3*ne,2)
        edges = np.sort(edges)
        
        edges = np.unique(edges,axis=0)

        # transfer the node points into the self.nodes list
        indexCounter=0
        for p in mesh.points:
            # create the corresponding node, where i is the running index of the mesh_points/corresponding nodes
            node=TopologyNode(indexCounter,p[0],p[1])
            self.nodes=self.nodes+[node]
            indexCounter+=1
        # fill in the self.edges list from the edges information
        for e in edges:
            self.edges=self.edges+[[int(e[0]),int(e[1])]]
        # construct the neighborhoods of each node in the graph
        for edge in self.edges:
            # first edge node
            a=self.nodes[int(edge[0])]
            # second edge node
            b=self.nodes[int(edge[1])]
            # add node a to the neighborhood of node b, and vice versa
            a.neighbors=a.neighbors+[b]
            b.neighbors=b.neighbors+[a]
        for node in self.nodes:
            self.sort_graph(node)
