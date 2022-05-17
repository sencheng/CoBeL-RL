import numpy as np
from spatial_representations.spatial_representation import SpatialRepresentation
from .misc.topology_node import TopologyNode
from .misc.cog_arrow import CogArrow

import PyQt5 as qt
import pyqtgraph as qg
import pyqtgraph.functions

class AbstractTopologyGraph(SpatialRepresentation) : 
    
    def __init__(self, n_nodes_x=5, n_nodes_y=5, n_neighbors=4, 
                 visual_output=None, **kwargs) : 
        
        self.n_nodes_x   = n_nodes_x
        self.n_nodes_y   = n_nodes_y
        self.n_neighbors = n_neighbors
        
        self.nodes = []
        self.edges = []
        
        self.current_node = None
        self.next_node    = None
        
        self.visual_output = None
    
    def get_nodes(self) : 
        return self.nodes
    
    def get_edges(self) : 
        return self.edges
    
    def set_visual_debugging(self) : 
        pass
    
    def get_action_space(self) : 
        pass
    
    def generate_behavior_from_action(self, action) : 
        pass
    
    def sample_state_space(self) : 
        pass
    
    
class GridGraph(AbstractTopologyGraph) : 
    '''
    Square/Rectangular grid
    '''
    def __init__(self, n_nodes_x=5, n_nodes_y=5, n_neighbors=4,
                 x_limits=[-1,1], y_limits=[-1,1], start_nodes=None,
                 goal_nodes=None, visual_output=None, rotation=False, 
                 world_module=None, use_world_limits=False, 
                 observation_module=None) : 
        
        super().__init__(n_nodes_x=5, n_nodes_y=5, n_neighbors=4, 
                         visual_output=None)
        
        self.n_nodes_x = n_nodes_x
        self.n_nodes_y = n_nodes_y
        self.n_neighbors = n_neighbors
        
        self.rotation = rotation
        self.visual_output = visual_output

        self.world_module = world_module
        self.x_limits = x_limits
        self.y_limits = y_limits
        
        self.observation_module = observation_module
        #store the graph parameters
        self.cliqueSize=self.cliqueSize
        
        if self.world_module is not None : 
            self.world_limits = self.world_module.getLimits()
            if self.use_world_limits : 
                self.x_limits = [self.world_limits[0,0], self.world_limits[0,1]]
                self.y_limits = [self.world_limits[1,0], self.world_limits[1,1]]
            #TODO : catch exceptions!
        
        #Build the graph
        self.nodes = []
        self.edges = []
        grid_points = self.find_grid_points(self.n_nodes_x, self.n_nodes_y, 
                                       self.x_limits, self.y_limits)
        self.build_graph(grid_points)
        self.n_nodes = len(self.nodes)    
        
        self.current_node = -1
        self.next_node    = -1
        
        self.start_nodes = start_nodes
        self.goal_nodes  = goal_nodes
        
        if self.start_nodes is not None : 
            for node_idx in self.start_nodes : 
                self.nodes[node_idx].startNode = True
        else : 
            for node in self.nodes : 
                node.startNode = True
        
        if self.goal_nodes is not None : 
            for node_idx in self.goal_nodes :
                self.nodes[node_idx].goalNode = True
    
        
    @staticmethod
    def find_grid_points(n_nodes_x, n_nodes_y, x_limits, y_limits) :
        
        nodes_x = np.linspace(x_limits[0], x_limits[1], n_nodes_x)
        nodes_y = np.linspace(y_limits[0], y_limits[1], n_nodes_y)
        
        grid_points = [[x, y] for y in nodes_y for x in nodes_x]
        
        return grid_points

    def build_graph(self, grid_points) : 
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
        for idx, point in enumerate(grid_points) : 
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

    
    def set_rotation(self, rotation) : 
        self.rotation = rotation    

    
    def update_position_marker(self, pose) : 
        self.position_marker.setData(pose[0],pose[1],
                                     np.rad2deg(np.arctan2(pose[3], pose[2])))
    
    
    def generate_behavior_from_action(self, action) :
        next_node_pos  = np.array([0.0,0.0])
        #TODO : is there a more intuitive way to accomplish all of this 
        #instead of using callback_value?
        callback_value = dict()
        
        if not self.rotation :
            
            if action!='reset':
                previous_node = self.current_node 
                self.next_node = self.nodes[self.current_node].neighbors[action].index
                
                if self.next_node != -1 :
                    next_node_pos = np.array([self.nodes[self.next_node].x, 
                                              self.nodes[self.next_node].y])
                    
                else : 
                    self.next_node = self.current_node
                    if self.world_module is not None : 
                        self.world_module.goalReached = True
                    next_node_pos = np.array([self.nodes[self.next_node].x, 
                                              self.nodes[self.next_node].y])
                
        
                callback_value['currentNode'] = self.nodes[self.next_node]
                
            else : 
                self.next_node = np.random.choice(self.start_nodes)
                next_node_pos = np.array([self.nodes[self.next_node].x, 
                                          self.nodes[self.next_node].y])
                
            if self.world_module is not None : 
                #TODO : check why this is called twice! If it's necessary,
                #add a comment explaining why
                #TODO : maybe change "actuate robot" to a more general name
                #that can be implemented by different world modules
                self.world_module.actuateRobot(np.array([next_node_pos[0],
                                                         next_node_pos[1],
                                                         90.0]))
                self.world_module.actuateRobot(np.array([next_node_pos[0],
                                         next_node_pos[1],
                                         90.0]))
            
            self.current_node = self.next_node
            
            if self.observation_module is not None : 
                self.observation_module.update()
                
            #TODO : else, write in a mechanism to navigate on the same graph
            #only based on indices
            if self.visual_output : 
                self.update_position_marker([next_node_pos[0], next_node_pos[1],
                                             0.0, 1.0])
                if qt.QtGui.QApplication.instance() is not None:
                    qt.QtGui.QApplication.instance().processEvents()
        else : 
            #TODO : with rotation
            pass
        
        return callback_value

            
    def get_action_space(self) : 
        #if rotation is True : 
            #left, right, forward
        if self.rotation:
            return gym.spaces.Discrete(3)
        #if rotation is False : 
            #pick neighbor
        return gym.spaces.Discrete(self.n_neighbors)

        
        
            
    def set_visual_debugging(self, gui_parent) :
        self.gui_parent=gui_parent
        self.visual_output=visual_output
        self.init_visual_elements()
    
    def init_visual_elements(self) :
        # do basic visualization
        # iff visualOutput is set to True!
        if self.visual_output:

            # add the graph plot to the GUI widget
            self.plot = self.gui_parent.addPlot(title='Topology graph')
            # set extension of the plot, lock aspect ratio
            if self.use_world_limits :
                self.plot.setXRange(self.x_limits)
                self.plot.setYRange(self.y_limits)
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
            self.posMarker = CogArrow(
                angle=0.0, headLen=20.0, tipAngle=25.0, tailLen=0.0, brush=(255, 0, 0))
            self.plot.addItem(self.posMarker)
            # initial position to center, this has to be worked over later!
            self.posMarker.setData(0.0, 0.0, 0.0)
    
    def update_visual_elements(self) :
        '''
        This function is only used for visualization. It generates arrows which points towards the highest q values. 
        
        Parameters
        ----------
        observation_module : It contains the state of the agent for all the nodes in Topology graph
        
        Returns
        --------
        none
        '''
       
        #somehow remove direct dependence on rlAgent
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
                    #q_values = self.rlAgent.agent.model.predict_on_batch(data)[0]
                    
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
            self.posMarker.setData(pose[0], pose[1], np.rad2deg(np.arctan2(pose[3], pose[2])))

        
    
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
        pass
        # TODO : test what this does
        # the world module is required here
        #returns [timeData,poseData,sensorData,imageData]
        #only imageData required
        world_module=self.modules['world']
        
        #Observation module uses imageData from world module to update the 
        #observation
        observation_module=self.modules['observation']
        
        # In this specific topology graph, a state is an image sampled from a 
        #specific node of the graph. There
        # is no rotation, so one image per node is sufficient.
        self.state_space = []
        for node_index in range(len(self.nodes)):
            #update each node
            node = self.nodes[node_index]
            #retrive image data for each node from the camera
            world_module.step_simulation_without_physics(node.x,node.y,90.0)
            #udate observation for every node with the imageData
            observation_module.update()
            observation=observation_module.observation
            #create a list of imageData for each node
            self.state_space+=[observation]
        #return list of state space
        return
        
        
        
        
        
        
        
