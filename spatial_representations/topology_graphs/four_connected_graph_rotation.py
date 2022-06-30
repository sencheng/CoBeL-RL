# basic imports
import PyQt5 as qt
import pyqtgraph as pg
import pyqtgraph.functions
import numpy as np
import gym
from numpy.linalg import norm
from math import sin, cos
import random
# framework imports
from .misc.topology_node import TopologyNode
from .misc.cog_arrow import CogArrow
from .misc.utils import lineseg_dists, point_in_polygon
from cobel.spatial_representations.spatial_representation import SpatialRepresentation


class Four_Connected_Graph_Rotation(SpatialRepresentation):

    def __init__(self, modules, graph_info, step_size=1.0):

        # call the base class init
        super(Four_Connected_Graph_Rotation,self).__init__()

        # normally, the topology graph is not shown in graphicsWindow
        self.visual_output=False
        self.graphicsWindow=None
        self.layout=None
        #store the graph parameters
        self.graph_info=graph_info

        # extract the world module
        self.modules=modules

        # the world module is required here
        world_module=modules['world']
        self.scenarioName = world_module.scenarioName
        self.offline = world_module.offline

        # get the limits of the given environment
        self.world_limits = world_module.getLimits()
        # retrieve all the wall info from the environment, here self.wall_limits store the min and max XY of each wall,
        # including the surrounding walls and those inside as obstacles; self.perimeter_nodes are a list of points
        # defining the polygon of the surrounding walls. These info is needed to build a basic topology graph of env.
        self.wall_limits, self.perimeter_nodes = world_module.getWallGraph()
        # constructed for visulization purpose
        self.world_nodes = self.perimeter_nodes[0:-1]
        self.world_edges = []
        for i in range(len(self.world_nodes)):
            self.world_edges.append([i, (i+1)%len(self.world_nodes)])

        # this is the node corresponding to the robot's actual position
        self.currentNode=-1
        # this is the node corresponding to the robot's next position
        self.nextNode=-1
        # this list of topologyNode[s] stores all nodes of the graph
        self.nodes=[]
        # this list of [int,int]-entries defines all edges (unique) which make up the graph's connectivity
        self.edges=[]
        # the starting orientation, default not specified
        self.startOri = graph_info['start_ori']

        self.cliqueSize=graph_info['cliqueSize']

        # the list for storing the trajectory of the agent where each entry is the agent's pos as [nodeIdx, orientation]
        self.trajectories = []

        # read topology structure from world module
        # here each node is a XY coordinate and each edge is a list containing 2 integers, indicating a connectivity
        # between 2 nodes
        nodes, edges = self.generate_topology_from_worldInfo(step_size)

        # store the nodes and edges of the env
        # topo_info = {'nodes': nodes, 'edges': edges, 'walls': self.wall_limits}
        # path = '/home/joshua/phd_projects/em_rl/data/%s_Top.pickle' % self.scenarioName
        # with open(path, 'wb') as handle:
        #     pickle.dump(topo_info, handle)

        # transfer the node points into the self.nodes list
        indexCounter=0
        for n in nodes:
            # create the corresponding node, where i is the running index of the mesh_points/corresponding nodes
            node=TopologyNode(indexCounter,float(n[0]),float(n[1]))
            self.nodes=self.nodes+[node]
            indexCounter+=1

        # fill in the self.edges list from the edges information
        for e in edges:
            self.edges=self.edges+[[int(e[0]),int(e[1])]]

        # define a dedicated 'noneNode' that acts as a placeholder for neighborhood construction
        noneNode=TopologyNode(-1,0.0,0.0)

        # construct the neighborhoods of each node in the graph
        for edge in self.edges:
            # first edge node
            a=self.nodes[int(edge[0])]
            # second edge node
            b=self.nodes[int(edge[1])]
            # add node a to the neighborhood of node b, we don't do the other way around here, since both edges [a, b] and
            # [b, a] exist in self.edges.
            a.neighbors=a.neighbors+[b]

        # it is possible that a node does not have the maximum possible number of neighbors, to stay consistent in RL, fill up the neighborhood
        # with noneNode[s]:
        for node in self.nodes:
            while len(node.neighbors)<self.cliqueSize:
                node.neighbors=node.neighbors+[noneNode]

        # assign start nodes. there are two options: 1. randomly chose one node out of all nodes as the starting
        # one every trial, or 2. give a specific group of nodes as the starting ones
        if isinstance(self.graph_info['startNodes'][0], str):
            try:
                self.graph_info['startNodes'][0] == 'random'
            except ValueError:
                print("Please either enter 'random' or a series of integer numbers as the starting nodes")
            i = np.random.randint(0, len(self.nodes))
            self.nodes[i].startNode=True

        else:
            for nodeIndex in self.graph_info['startNodes']:
                self.nodes[nodeIndex].startNode=True

        # assign goal nodes
        for nodeIndex in self.graph_info['goalNodes']:
            self.nodes[nodeIndex].goalNode=True


    def set_visual_debugging(self,visual_output,graphicsWindow):
        self.graphicsWindow=graphicsWindow
        self.visual_output=visual_output
        if visual_output:
            self.layout=graphicsWindow.centralWidget
        self.initVisualElements()

    def sample_state_space(self):
        # not currently used
        pass

    def initVisualElements(self):
        # do basic visualization
        # iff visualOutput is set to True!
        if self.visual_output:

            # add the graph plot to the GUI widget
            self.topology_plot_viewbox = pg.ViewBox(parent=self.layout, enableMouse=False, enableMenu=False)
            # set extension of the plot, lock aspect ratio
            self.topology_plot_viewbox.setXRange( self.world_limits[0,0], self.world_limits[0,1] )
            self.topology_plot_viewbox.setYRange( self.world_limits[1,0], self.world_limits[1,1] )
            self.topology_plot_viewbox.setAspectLocked(lock=True)
            # add the viewbox to the window layout
            self.layout.addItem(self.topology_plot_viewbox, colspan=2, rowspan=1, row=2, col=0)

            # overlay the world's perimeter
            self.perimeterGraph=pg.GraphItem()
            self.topology_plot_viewbox.addItem(self.perimeterGraph)

            self.perimeterGraph.setData(pos=np.array(self.world_nodes),adj=np.array(self.world_edges),brush=(128,128,128))

            # overlay the topology graph
            self.topologyGraph=pg.GraphItem()
            self.topology_plot_viewbox.addItem(self.topologyGraph)

            # set up a brushes array for visualization of the nodes
            # normal nodes are grey
            symbolBrushes=[pg.mkBrush(color=(128,128,128))]*len(self.nodes)

            # set colors of normal and goal nodes
            for node in self.nodes:

                # start nodes are green
                if node.startNode:
                    symbolBrushes[node.index]=pg.mkBrush(color=(0,255,0))

                # goal node is red
                if node.goalNode:
                    symbolBrushes[node.index]=pg.mkBrush(color=(255,0,0))

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
            self.topology_plot_viewbox.addItem(self.posMarker)
            # initial position to center, this has to be worked over later!
            self.posMarker.setData(0.0,0.0,90.0)

    # This function updates the visual depiction of the agent(robot).
    #
    # pose: the agent's pose to visualize
    def updateRobotPose(self, pose):
        if self.visual_output:
            self.posMarker.setData(pose[0],pose[1],np.rad2deg(np.arctan2(pose[3],pose[2])))


    def reset_start_nodes(self, startNodes, ori):
        self.startOri = None
        for node in self.nodes:
            node.startNode=False
        for nodeIndex in startNodes:
            self.nodes[nodeIndex].startNode=True
        self.startOri = ori

    def get_node_number(self):
        return len(self.nodes)

    def generate_topology_from_worldInfo(self, step_size=1.0):
        '''
        step_size: the minimal distance between nodes
        '''
        # for each wall limit, create the correponding four edges, since each wall is a square on 2D plane
        wall_edges = []
        for item in self.wall_limits:
            # edge: (minx, minz), (minx, maxz)
            wall_edges.append([[item[0], item[1]], [item[0], item[3]]])
            # edge: (minx, maxz), (maxx, maxz)
            wall_edges.append([[item[0], item[3]], [item[2], item[3]]])
            # edge: (maxx, maxz), (maxx, minz)
            wall_edges.append([[item[2], item[3]], [item[2], item[1]]])
            # edge: (maxx, minz), (minx, minz)
            wall_edges.append([[item[2], item[1]], [item[0], item[1]]])

        wall_edges = np.asarray(wall_edges)

        # generate 2d grid based on the world limits
        x = np.arange(self.world_limits[0, 0]+step_size, self.world_limits[0, 1], step_size)
        y = np.arange(self.world_limits[1, 0]+step_size, self.world_limits[1, 1], step_size)
        xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')

        # store each nodes which is not too close to the wall
        nodes = []
        edges = []
        nodes_index = []
        index = 0
        # from min x to max x; from min y to max y
        for xi in xv[0,:]:
            for yi in yv[:,0]:
                distances = lineseg_dists(np.array([xi, yi]), wall_edges[:, 0], wall_edges[:, 1])
                # if there is a node too close to any of the wall, its index is None
                distances = distances < (step_size/2)
                if distances.nonzero()[0].size > 0:
                    nodes_index.append(None)
                else:
                    # if a node is outside the perimeter polygon, its index is None
                    if not point_in_polygon(self.perimeter_nodes, np.array([xi, yi])):
                        nodes_index.append(None)
                    else:
                        nodes_index.append(index)
                        nodes.append(np.array([xi, yi]))
                        index += 1

        # Construct the 4-coneected edges
        num_xgrid = len(xv[0,:])
        num_ygrid = len(yv[:,0])
        # the first node is at the left bottom corner, then fill the space in y direction
        for i in range(len(nodes_index)):
            row = int(i % num_ygrid)
            col = int(i / num_ygrid)
            if nodes_index[i] is None:
                continue
            # add the left connectivity
            if col-1 < 0:
                pass
            else:
                if nodes_index[(col-1)*num_ygrid + row] is not None:
                    edges.append([nodes_index[i], nodes_index[(col-1)*num_ygrid + row]])
            # add the right connectivity
            if col+1 > num_xgrid-1:
                pass
            else:
                if nodes_index[(col+1)*num_ygrid + row] is not None:
                    edges.append([nodes_index[i], nodes_index[(col+1)*num_ygrid + row]])
            # add the down connectivity
            if row-1 < 0:
                pass
            else:
                if nodes_index[col*num_ygrid + row-1] is not None:
                    edges.append([nodes_index[i], nodes_index[col*num_ygrid + row-1]])
            # add the up connectivity
            if row+1 > num_ygrid-1:
                pass
            else:
                if nodes_index[col*num_ygrid + row+1] is not None:
                    edges.append([nodes_index[i], nodes_index[col*num_ygrid + row+1]])

        nodes = np.asarray(nodes)
        edges = np.asarray(edges)

        return nodes, edges

    def normalize_angle(self, angle):
        '''
        In this topology, there are only 3 options for the absolute value of the agent's angle
        [0, 90, 180]. Here we force the real angle of the robot to be one of them because there was
        always a numerical error caused by computation inaccuracy..
        '''
        possible_angles = np.array([0, 90, 180])
        angle = np.rad2deg(np.arctan2(np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))))
        deviations = np.abs(possible_angles - np.abs(angle))
        ideal_angle = possible_angles[np.argmin(deviations)]

        if ideal_angle == 90:
            if angle < 0:
                angle = -ideal_angle
            else:
                angle = ideal_angle
        elif ideal_angle == 0:
            angle = ideal_angle
        else:
            angle = -ideal_angle # 180 and -180 are the same, here we always take -180

        return angle

    def generate_behavior_from_action(self, action):
        # the world module is required here
        world_module=self.modules['world']
        nextNodePos=np.array([0.0,0.0])
        callback_value=dict()
        # if a standard action is performed, NOT a reset
        if action!='reset':
            # get current heading
            headingX = cos((world_module.envData['poseData'][2])/180*np.pi)
            headingY = sin((world_module.envData['poseData'][2])/180*np.pi)
            heading=np.array([headingX, headingY])
            heading=heading/norm(heading)
            # get directions of all edges
            # here, self.modules['spatial_representation'] is this class itself
            actual_node=self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].currentNode]
            neighbors=self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].currentNode].neighbors

            # lists for forward edges
            forward_edge = []
            right_edge = []
            left_edge = []
            backward_edge = []

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
                    angle=self.normalize_angle(angle)

                    if angle == 0:
                        forward_edge=[n.index,vec_edge,angle]
                    elif angle == 90:
                        left_edge=[n.index,vec_edge,angle]
                    elif angle == -90:
                        right_edge=[n.index,vec_edge,angle]
                    elif angle == 180 or angle == -180:
                        backward_edge=[n.index,vec_edge,angle]

            # store the current node as previous node
            previousNode=self.modules['spatial_representation'].currentNode
            angle=world_module.envData['poseData'][2]
            # angle=self.normalize_angle(angle)

            if action==0:
                # this is a forward movement
                if len(forward_edge)!=0:
                    # there is a forward edge that the agent can use
                    self.modules['spatial_representation'].nextNode=forward_edge[0]

                    nextNodePos=np.array([self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].x,self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].y])

                else:
                    # no forward edge found, the agent has to wait for a rotation action
                    self.modules['spatial_representation'].nextNode=self.modules['spatial_representation'].currentNode
                    nextNodePos=np.array([self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].x,self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].y])

                self.modules['spatial_representation'].updateRobotPose([nextNodePos[0],nextNodePos[1],heading[0],heading[1]])

            elif action==1:
                # this is a left turn movement
                self.modules['spatial_representation'].nextNode=self.modules['spatial_representation'].currentNode
                nextNodePos=np.array([self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].x,self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].y])
                angle += 90  # turn 90 degrees to the left

                self.modules['spatial_representation'].updateRobotPose([nextNodePos[0],nextNodePos[1],cos(angle/180*np.pi),sin(angle/180*np.pi)])

            elif action==2:
                # this is a right turn movement
                self.modules['spatial_representation'].nextNode=self.modules['spatial_representation'].currentNode
                nextNodePos=np.array([self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].x,self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].y])
                angle -= 90

                self.modules['spatial_representation'].updateRobotPose([nextNodePos[0],nextNodePos[1],cos(angle/180*np.pi),sin(angle/180*np.pi)])

            elif action==3:
                # this is a left movement
                if len(left_edge)!=0:
                    # there is a left edge that the agent can use
                    self.modules['spatial_representation'].nextNode=left_edge[0]

                    nextNodePos=np.array([self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].x,self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].y])

                else:
                    # no left edge found, the agent has to wait for a rotation action
                    self.modules['spatial_representation'].nextNode=self.modules['spatial_representation'].currentNode
                    nextNodePos=np.array([self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].x,self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].y])

                self.modules['spatial_representation'].updateRobotPose([nextNodePos[0],nextNodePos[1],heading[0],heading[1]])

            elif action==4:
                # this is a right movement
                if len(right_edge)!=0:
                    # there is a right edge that the agent can use
                    self.modules['spatial_representation'].nextNode=right_edge[0]

                    nextNodePos=np.array([self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].x,self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].y])

                else:
                    # no right edge found, the agent has to wait for a rotation action
                    self.modules['spatial_representation'].nextNode=self.modules['spatial_representation'].currentNode
                    nextNodePos=np.array([self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].x,self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].y])

                self.modules['spatial_representation'].updateRobotPose([nextNodePos[0],nextNodePos[1],heading[0],heading[1]])

            elif action==5:
                # this is a backward movement
                if len(backward_edge)!=0:
                    # there is a backward edge that the agent can use
                    self.modules['spatial_representation'].nextNode=backward_edge[0]

                    nextNodePos=np.array([self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].x,self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].y])

                else:
                    # no backward edge found, the agent has to wait for a rotation action
                    self.modules['spatial_representation'].nextNode=self.modules['spatial_representation'].currentNode
                    nextNodePos=np.array([self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].x,self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode].y])

                self.modules['spatial_representation'].updateRobotPose([nextNodePos[0],nextNodePos[1],heading[0],heading[1]])

            angle = self.normalize_angle(angle)

            # For online simulation, the control commands are next node's XY postion and angle
            # For offline simulation, the control commands are next node's index and angle
            if not self.offline:
                self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],angle]))
            else:
                self.modules['world'].actuateRobot(np.array([self.modules['spatial_representation'].nextNode, angle]))
            # update the observation
            self.modules['observation'].update()
            # make the current node the one the agent travelled to
            self.modules['spatial_representation'].currentNode=self.modules['spatial_representation'].nextNode

            # here, next node is already set and the current node is set to this next node.
            callback_value['currentNode']=self.nodes[self.nextNode]

            # record the current node idx and angle into the last trajectory history
            self.trajectories[-1].append([self.modules['spatial_representation'].currentNode, angle])
            # put the trajectories up till now of this episode in the call back value
            callback_value['episode_traj'] = self.trajectories[-1]

        # if a reset is performed
        else:
            # a random node is chosen to place the agent at (this node MUST NOT be the global goal node!)
            nodes=self.modules['spatial_representation'].nodes
            nodes_selection=[n for n in nodes if n.startNode==True]

            # store the current node as previous node
            previousNode=self.modules['spatial_representation'].currentNode

            self.modules['spatial_representation'].nextNode=random.choice(nodes_selection)

            nextNodePos=np.array([self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode.index].x,self.modules['spatial_representation'].nodes[self.modules['spatial_representation'].nextNode.index].y])

            self.modules['spatial_representation'].currentNode=self.modules['spatial_representation'].nextNode.index
            neighbors=self.modules['spatial_representation'].nextNode.neighbors

            # list for available neighbor directions
            directions=[]

            for n in neighbors:
                if n.index!=-1:
                    # only parse valid neighbors
                    next_node_position=np.array([self.modules['spatial_representation'].nextNode.x,self.modules['spatial_representation'].nextNode.y])
                    neighbor_position=np.array([n.x,n.y])
                    vec_edge=neighbor_position-next_node_position
                    vec_edge=vec_edge/norm(vec_edge)
                    world_angle=np.arctan2(vec_edge[1],vec_edge[0])
                    directions+=[[n.index,vec_edge,world_angle]]

            # select new heading randomly
            new_heading_selection=random.choice(directions)
            new_heading_angle=new_heading_selection[2]
            new_heading_angle = np.rad2deg(new_heading_angle)
            new_heading_vector=new_heading_selection[1]

            new_heading_angle = self.normalize_angle(new_heading_angle)
            if self.startOri is not None:
                new_heading_angle = self.startOri
                new_heading_vector[0] = np.cos(np.deg2rad(new_heading_angle))
                new_heading_vector[1] = np.sin(np.deg2rad(new_heading_angle))

            # update the agents position and orientation (heading)
            if not self.offline:
                self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],new_heading_angle]))
            else:
                self.modules['world'].actuateRobot(np.array([self.modules['spatial_representation'].nextNode.index, new_heading_angle]))
            self.modules['spatial_representation'].updateRobotPose([nextNodePos[0],nextNodePos[1],new_heading_vector[0],new_heading_vector[1]])

            # update the observation
            self.modules['observation'].update()

            # whenever there is a reset, start recording a new trajectory
            self.trajectories.append([[self.modules['spatial_representation'].currentNode, new_heading_angle]])

        # if possible try to update the visual debugging display
        if qt.QtGui.QApplication.instance() is not None:
            qt.QtGui.QApplication.instance().processEvents()
            # qt.QtGui.QThread.currentThread().processEvents()


        return callback_value

    def updateVisualElements(self):
        # not currently used
        pass


    def get_action_space(self):
        # for this spatial representation type, there are three possible actions: forward, left, right
        return gym.spaces.Discrete(6)

    def clear_trajectories(self):
        # clear the trajectory history if needed
        self.trajectories = []
