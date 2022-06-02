import numpy as np
from spatial_representations.spatial_representation import SpatialRepresentation
from .misc.topology_node import TopologyNode
from .misc.cog_arrow import CogArrow

import PyQt5 as qt
import pyqtgraph as qg
import pyqtgraph.functions

import gym
import random

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
        self.use_world_limits = use_world_limits
        
        self.observation_module = observation_module
        
        if self.world_module is not None : 
            self.world_limits = self.world_module.getLimits()
            self.world_nodes, self.world_edges = world_module.getWallGraph()
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
                if hasattr(qt.QtGui, 'QApplication'):
                    if qt.QtGui.QApplication.instance() is not None:
                        qt.QtGui.QApplication.instance().processEvents()
                else:
                    if qt.QtWidgets.QApplication.instance() is not None:
                        qt.QtWidgets.QApplication.instance().processEvents()
        
        else : #ROTATION IS ENABLED
            
            if action!='reset':                   
                if self.world_module is not None : 
                    heading = np.array([self.world_module.envData['poseData'][2],
                                      self.world_module.envData['poseData'][3]])
                    heading = heading/np.linalg.norm(heading)
                # get directions of all edges
                actual_node = self.nodes[self.current_node]
                neighbors = self.nodes[self.current_node].neighbors
                
                # lists for edges
                left_edges   = []
                right_edges  = []
                forward_edge = []
                
                
                # find possible movement directions. Note: when a left edge is found, it is simultaneously stored as a right edge with huge turning angle, and vice versa. That way,
                # the agent does not get stuck in situations where there is only a forward edge, and say, a left edge, and the action is 'right'. In such a situation, the agent will just turn
                # right using the huge 'right' turning angle.
                for n in neighbors:
                    if n.index!=-1:
                        actual_node_position = np.array([actual_node.x,actual_node.y])
                        
                        neighbor_position = np.array([n.x,n.y])
                        vec_edge = neighbor_position-actual_node_position
                        vec_edge = vec_edge/np.linalg.norm(vec_edge)
                        
                        angle = np.arctan2(heading[0]*vec_edge[1]-heading[1]*vec_edge[0],
                                           heading[0]*vec_edge[0]+heading[1]*vec_edge[1])
                        angle = angle/np.pi*180.0
                        
                        
                        if angle<-1e-5:
                            right_edges+=[[n.index,vec_edge,angle]]
                            left_edges+=[[n.index,vec_edge,(360.0+angle)]]
                    
                        if angle>1e-5:
                            left_edges+=[[n.index,vec_edge,angle]]
                            right_edges+=[[n.index,vec_edge,-(360.0-angle)]]
                        
                        if angle<1e-5 and angle>-1e-5:
                            forward_edge=[n.index,vec_edge,angle]
                            

                left_edges=sorted(left_edges,key=lambda element: element[2],reverse=False)
                right_edges=sorted(right_edges,key=lambda element: element[2],reverse=True)
                
                # store the current node as previous node
                previous_node=self.current_node
                        
            # with action given, the next node can be computed
                if action==0:
                    
                    # this is a forward movement
                    angle=180.0/np.pi*np.arctan2(heading[1],heading[0])
                        
                    if len(forward_edge)!=0:
                        # there is a forward edge that the agent can use
                        self.next_node=forward_edge[0]
                        
                        nextNodePos=np.array([self.nodes[self.next_node].x,self.nodes[self.next_node].y])
                                       
                    else:
                        # no forward edge found, the agent has to wait for a rotation action
                        self.next_node=self.current_node
                        nextNodePos=np.array([self.nodes[self.next_node].x,self.nodes[self.next_node].y])
                        
                    self.update_position_marker([nextNodePos[0],nextNodePos[1],heading[0],heading[1]])
                    self.world_module.actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle])) 
                    self.world_module.actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle]))
                                
                if action==1:
                    # this is a left turn movement
                    self.next_node=self.current_node
                    nextNodePos=np.array([self.nodes[self.next_node].x,self.nodes[self.next_node].y])
                        
                    angle=180.0/np.pi*np.arctan2(left_edges[0][1][1],left_edges[0][1][0])
                    self.update_position_marker([nextNodePos[0],nextNodePos[1],left_edges[0][1][0],left_edges[0][1][1]])
                    self.world_module.actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle])) 
                    self.world_module.actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle]))
                
                if action==2:
                    # this is a right turn movement
                    self.next_node=self.current_node
                    nextNodePos=np.array([self.nodes[self.next_node].x,self.nodes[self.next_node].y])
                    angle=180.0/np.pi*np.arctan2(right_edges[0][1][1],right_edges[0][1][0])
                    self.update_position_marker([nextNodePos[0],nextNodePos[1],right_edges[0][1][0],right_edges[0][1][1]])
                    self.world_module.actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle])) 
                    self.world_module.actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle]))
            
                if self.observation_module is not None : 
                    self.observation_module.update()
            
            # make the current node the one the agent travelled to
                self.current_node=self.next_node
                            
                # here, next node is already set and the current node is set to this next node.
                callback_value['currentNode']=self.nodes[self.next_node]
                
            else : 
                            # a random node is chosen to place the agent at (this node MUST NOT be the global goal node!)
            
                nodes=self.nodes
                nodes_selection=[n for n in nodes if n.startNode==True]
                
                # store the current node as previous node
                previousNode=self.current_node
                
                self.next_node=random.choice(nodes_selection)
                
                nextNodePos=np.array([self.next_node.x,self.next_node.y])
                
                # from all heading directions available at the chosen node, select one randomly
                
                self.current_node=self.next_node.index
                neighbors=self.next_node.neighbors
                
                # list for available neighbor directions
                directions=[]
                
                for n in neighbors:
                    if n.index!=-1:
                        # only parse valid neighbors
                        next_node_position=np.array([self.next_node.x,self.next_node.y])
                        neighbor_position=np.array([n.x,n.y])
                        vec_edge=neighbor_position-next_node_position
                        vec_edge=vec_edge/np.linalg.norm(vec_edge)
                        world_angle=np.arctan2(vec_edge[1],vec_edge[0])
                        directions+=[[n.index,vec_edge,world_angle]]
                        
                # select new heading randomly
                new_heading_selection=random.choice(directions)
                new_heading_angle=new_heading_selection[2]
                new_heading_vector=new_heading_selection[1]
                
                # update the agents position and orientation (heading)
                self.world_module.actuateRobot(np.array([nextNodePos[0],nextNodePos[1],new_heading_angle])) 
                self.world_module.actuateRobot(np.array([nextNodePos[0],nextNodePos[1],new_heading_angle]))
                self.update_position_marker([nextNodePos[0],nextNodePos[1],new_heading_vector[0],new_heading_vector[1]])
                
                # update the observation
                self.observation_module.update()
            
            
            
            # if possible try to update the visual debugging display
            # TODO: previous version had a double call to processEvents(). Intended?
            # was:
            #if qt.QtGui.QApplication.instance() is not None:
            #    qt.QtGui.QApplication.instance().processEvents()
            #    qt.QtGui.QApplication.instance().processEvents()
    
            if hasattr(qt.QtGui, 'QApplication'):
                if qt.QtGui.QApplication.instance() is not None:
                    qt.QtGui.QApplication.instance().processEvents()
            else:
                if qt.QtWidgets.QApplication.instance() is not None:
                    qt.QtWidgets.QApplication.instance().processEvents()

        
        return callback_value

            
    def get_action_space(self) : 

        if self.rotation:
            return gym.spaces.Discrete(3)

        else :
            return gym.spaces.Discrete(self.n_neighbors)

                
    def set_visual_debugging(self, gui_parent) :
        self.gui_parent = gui_parent
        self.init_visual_elements()
    
    def init_visual_elements(self) :

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
        # TODO : test what this does
        # the world module is required here
        #returns [timeData,poseData,sensorData,imageData]

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
        
        
        
        
        
        
        
