




import PyQt5 as qt
import pyqtgraph as qg
import pyqtgraph.functions
import numpy as np
import gym

from PyQt5 import QtGui
from .misc.topology_node import TopologyNode
from .misc.cog_arrow import CogArrow
from math import atan2
from numpy.linalg import norm

from cobel.spatial_representations.spatial_representation import SpatialRepresentation

import time
import random



class ManualTopologyGraphWithRotation(SpatialRepresentation):
    
    def __init__(self, modules, graph_info):
        
        # call the base class init
        super(ManualTopologyGraphWithRotation,self).__init__()
        
        
        # normally, the topology graph is not shown in gui_parent
        self.visual_output=False
        self.gui_parent=None
        
        
        #store the graph parameters
        self.graph_info=graph_info
        
        # extract the world module
        self.modules=modules
        
        # the world module is required here
        world_module=modules['world']
        
        # get the limits of the given environment
        self.world_limits = world_module.getLimits()
        
        # retrieve all boundary information from the environment
        self.world_nodes, self.world_edges = world_module.getWallGraph()
        
            
        
        # inherent definitions for the topology graph
        
        # this is the node corresponding to the robot's actual position
        self.currentNode=-1
        # this is the node corresponding to the robot's next position
        self.nextNode=-1
        # this list of topologyNode[s] stores all nodes of the graph
        self.nodes=[]
        # this list of [int,int]-entries defines all edges (unique) which make up the graph's connectivity
        self.edges=[]
        
        
        
        
        self.cliqueSize=graph_info['cliqueSize']
        # set up a manually constructed topology graph
        
        # read topology structure from world module
        nodes=np.array(world_module.getManuallyDefinedTopologyNodes())
        nodes=nodes[nodes[:,0].argsort()]
        edges=np.array(world_module.getManuallyDefinedTopologyEdges())
        edges=edges[edges[:,0].argsort()]
        
        # transfer the node points into the self.nodes list
        indexCounter=0
        for n in nodes:
            # create the corresponding node, where i is the running index of the mesh_points/corresponding nodes
            node=TopologyNode(indexCounter,float(n[1]),float(n[2]))
            self.nodes=self.nodes+[node]
            indexCounter+=1
        
        
        # fill in the self.edges list from the edges information
        for e in edges:
            self.edges=self.edges+[[int(e[1]),int(e[2])]]
            
            
            
        # define a dedicated 'noneNode' that acts as a placeholder for neighborhood construction
        noneNode=TopologyNode(-1,0.0,0.0)
        
        # comstruct the neighborhoods of each node in the graph
        for edge in self.edges:
            # first edge node
            a=self.nodes[int(edge[0])]
            # second edge node
            b=self.nodes[int(edge[1])]
            # add node a to the neighborhood of node b, and vice versa
            a.neighbors=a.neighbors+[b]
            b.neighbors=b.neighbors+[a]
        
        
        # it is possible that a node does not have the maximum possible number of neighbors, to stay consistent in RL, fill up the neighborhood
        # with noneNode[s]:
        for node in self.nodes:
            while len(node.neighbors)<self.cliqueSize:
                node.neighbors=node.neighbors+[noneNode]
        
        # assign start nodes
        for nodeIndex in self.graph_info['startNodes']:
            self.nodes[nodeIndex].startNode=True
            
        # assign goal nodes
        for nodeIndex in self.graph_info['goalNodes']:
            self.nodes[nodeIndex].goalNode=True
        
        
        self.sample_state_space()
        



    def set_visual_debugging(self,visual_output,gui_parent):
        
        self.gui_parent=gui_parent
        self.visual_output=visual_output
        self.initVisualElements()
        



    def initVisualElements(self):
        # do basic visualization
        # iff visualOutput is set to True!
        if self.visual_output:

            # add the graph plot to the GUI widget
            self.plot = self.gui_parent.addPlot(title='Topology graph')
            # set extension of the plot, lock aspect ratio
            self.plot.setXRange( self.world_limits[0,0], self.world_limits[0,1] )
            self.plot.setYRange( self.world_limits[1,0], self.world_limits[1,1] )
            self.plot.setAspectLocked()
            
            
                
            # overlay the world's perimeter
            self.perimeterGraph=qg.GraphItem()
            self.plot.addItem(self.perimeterGraph)
            
            self.perimeterGraph.setData(pos=np.array(self.world_nodes),adj=np.array(self.world_edges),brush=(128,128,128))
            
            
            
            # overlay the topology graph
            self.topologyGraph=qg.GraphItem()
            self.plot.addItem(self.topologyGraph)
            
            # set up a brushes array for visualization of the nodes
            # normal nodes are grey
            symbolBrushes=[qg.mkBrush(color=(128,128,128))]*len(self.nodes)
            
            
            # set colors of normal and goal nodes
            for node in self.nodes:
                
                # start nodes are green
                if node.startNode:
                    symbolBrushes[node.index]=qg.mkBrush(color=(0,255,0))
            
                # goal node is red
                if node.goalNode:
                    symbolBrushes[node.index]=qg.mkBrush(color=(255,0,0))
        
            # construct appropriate arrays from the self.nodes and the self.edges information
            tempNodes=[]
            tempEdges=[]
            for node in self.nodes:
                tempNodes=tempNodes+[[node.x,node.y]]
                
            for edge in self.edges:
                tempEdges=tempEdges+[[edge[0],edge[1]]]

            self.topologyGraph.setData(pos=np.array(tempNodes),adj=np.array(tempEdges),symbolBrush=symbolBrushes)
            
            
            # eventually, overlay robot marker
            self.posMarker=CogArrow(angle=0.0,headLen=20.0,tipAngle=25.0,tailLen=0.0,brush=(255,0,0))
            self.plot.addItem(self.posMarker)
            # initial position to center, this has to be worked over later!
            self.posMarker.setData(0.0,0.0,0.0)
            
        
    def updateVisualElements(self):
        # not currently used
        pass
    
    
    

    # This function updates the visual depiction of the agent(robot).
    # 
    # pose: the agent's pose to visualize
    def updateRobotPose(self,pose):
        if self.visual_output:
            self.posMarker.setData(pose[0],pose[1],np.rad2deg(np.arctan2(pose[3],pose[2])))



    def sample_state_space(self):
        # not currently used
        pass





    def generate_behavior_from_action(self,action):
       
        
        # the world module is required here
        world_module=self.modules['world']
        nextNodePos=np.array([0.0,0.0])
        callback_value=dict()
        # if a standard action is performed, NOT a reset
        if action!='reset':
            
            # get current heading
            heading=np.array([world_module.envData['poseData'][2],world_module.envData['poseData'][3]])
            heading=heading/norm(heading)
            # get directions of all edges
            actual_node=self.nodes[self.currentNode]
            neighbors=self.nodes[self.currentNode].neighbors
            
            # lists for edges
            left_edges=[]
            right_edges=[]
            forward_edge=[]
            
            
            # find possible movement directions. Note: when a left edge is found, it is simultaneously stored as a right edge with huge turning angle, and vice versa. That way,
            # the agent does not get stuck in situations where there is only a forward edge, and say, a left edge, and the action is 'right'. In such a situation, the agent will just turn
            # right using the huge 'right' turning angle.
            for n in neighbors:
                if n.index!=-1:
                    actual_node_position=np.array([actual_node.x,actual_node.y])
                    
                    neighbor_position=np.array([n.x,n.y])
                    vec_edge=neighbor_position-actual_node_position
                    vec_edge=vec_edge/norm(vec_edge)
                    
                    angle=np.arctan2(heading[0]*vec_edge[1]-heading[1]*vec_edge[0],heading[0]*vec_edge[0]+heading[1]*vec_edge[1])
                    angle=angle/np.pi*180.0
                    
                    
                    if angle<-1e-5:
                        right_edges+=[[n.index,vec_edge,angle]]
                        left_edges+=[[n.index,vec_edge,(360.0+angle)]]
                
                    if angle>1e-5:
                        left_edges+=[[n.index,vec_edge,angle]]
                        right_edges+=[[n.index,vec_edge,-(360.0-angle)]]
                    
                    if angle<1e-5 and angle>-1e-5:
                        forward_edge=[n.index,vec_edge,angle]
                        
                 
            
            # sort left and right edges in such a way that the smallest angular difference is placed in front
            left_edges=sorted(left_edges,key=lambda element: element[2],reverse=False)
            right_edges=sorted(right_edges,key=lambda element: element[2],reverse=True)
            
            # store the current node as previous node
            previousNode=self.currentNode
            
            
            # with action given, the next node can be computed
            if action==0:
                
                # this is a forward movement
                angle=180.0/np.pi*np.arctan2(heading[1],heading[0])
                    
                if len(forward_edge)!=0:
                    # there is a forward edge that the agent can use
                    self.nextNode=forward_edge[0]
                    
                    nextNodePos=np.array([self.nodes[self.nextNode].x,self.nodes[self.nextNode].y])
                    
            
                else:
                    # no forward edge found, the agent has to wait for a rotation action
                    self.nextNode=self.currentNode
                    nextNodePos=np.array([self.nodes[self.nextNode].x,self.nodes[self.nextNode].y])
                    
                self.updateRobotPose([nextNodePos[0],nextNodePos[1],heading[0],heading[1]])
                self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle])) 
                self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle]))
                
                
            if action==1:
                # this is a left turn movement
                self.nextNode=self.currentNode
                nextNodePos=np.array([self.nodes[self.nextNode].x,self.nodes[self.nextNode].y])
                    
                angle=180.0/np.pi*np.arctan2(left_edges[0][1][1],left_edges[0][1][0])
                self.updateRobotPose([nextNodePos[0],nextNodePos[1],left_edges[0][1][0],left_edges[0][1][1]])
                self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle])) 
                self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle]))
                
            if action==2:
                # this is a right turn movement
                self.nextNode=self.currentNode
                nextNodePos=np.array([self.nodes[self.nextNode].x,self.nodes[self.nextNode].y])
                angle=180.0/np.pi*np.arctan2(right_edges[0][1][1],right_edges[0][1][0])
                self.updateRobotPose([nextNodePos[0],nextNodePos[1],right_edges[0][1][0],right_edges[0][1][1]])
                self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle])) 
                self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle]))
            
            # update the observation
            self.modules['observation'].update()
            
            # make the current node the one the agent travelled to
            self.currentNode=self.nextNode
        
            
            # here, next node is already set and the current node is set to this next node.
            callback_value['currentNode']=self.nodes[self.nextNode]
            
            
        # if a reset is performed
        else:
            # a random node is chosen to place the agent at (this node MUST NOT be the global goal node!)
            
            nodes=self.nodes
            nodes_selection=[n for n in nodes if n.startNode==True]
            
            # store the current node as previous node
            previousNode=self.currentNode
            
            self.nextNode=random.choice(nodes_selection)
            
            nextNodePos=np.array([self.nodes[self.nextNode.index].x,self.nodes[self.nextNode.index].y])
            
            # from all heading directions available at the chosen node, select one randomly
            
            self.currentNode=self.nextNode.index
            neighbors=self.nextNode.neighbors
            
            # list for available neighbor directions
            directions=[]
            
            for n in neighbors:
                if n.index!=-1:
                    # only parse valid neighbors
                    next_node_position=np.array([self.nextNode.x,self.nextNode.y])
                    neighbor_position=np.array([n.x,n.y])
                    vec_edge=neighbor_position-next_node_position
                    vec_edge=vec_edge/norm(vec_edge)
                    world_angle=np.arctan2(vec_edge[1],vec_edge[0])
                    directions+=[[n.index,vec_edge,world_angle]]
                    
            # select new heading randomly
            new_heading_selection=random.choice(directions)
            new_heading_angle=new_heading_selection[2]
            new_heading_vector=new_heading_selection[1]
            
            # update the agents position and orientation (heading)
            self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],new_heading_angle])) 
            self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],new_heading_angle]))
            self.updateRobotPose([nextNodePos[0],nextNodePos[1],new_heading_vector[0],new_heading_vector[1]])
            
        
            # update the observation
            self.modules['observation'].update()
        
        
        
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



    def get_action_space(self):
        # for this spatial representation type, there are three possible actions: forward, left, right
        return gym.spaces.Discrete(3)
