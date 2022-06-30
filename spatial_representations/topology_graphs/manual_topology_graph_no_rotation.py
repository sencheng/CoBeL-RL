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
    '''
    Manually defined topology graph module.
    
    | **Args**
    | modules:                      Framework modules.
    | graph_info:                   Topology graph relevant information.
    '''
    #TODO : should not take modules as input, specific inputs
    def __init__(self, modules, graph_info):
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
        world_module.setTopology(self)
        # get the limits of the given environment
        self.world_limits = world_module.getLimits()
        # retrieve all boundary information from the environment
        self.world_nodes, self.world_edges = world_module.getWallGraph()
        # inherent definitions for the topology graph
        # this is the node corresponding to the robot's actual position
        self.currentNode = -1
        # this is the node corresponding to the robot's next position
        self.nextNode = -1
        # this list of topologyNode[s] stores all nodes of the graph
        self.nodes = []
        # this list of [int,int]-entries defines all edges (unique) which make up the graph's connectivity
        self.edges = []
        self.cliqueSize = graph_info['cliqueSize']
        # set up a manually constructed topology graph
        # read topology structure from world module
        nodes = np.array(world_module.getManuallyDefinedTopologyNodes())
        nodes = nodes[nodes[:, 0].argsort()]
        edges = np.array(world_module.getManuallyDefinedTopologyEdges())
        edges = edges[edges[:, 0].argsort()]
        # transfer the node points into the self.nodes list
        indexCounter = 0
        for n in nodes:
            # create the corresponding node, where i is the running index of the mesh_points/corresponding nodes
            node=TopologyNode(indexCounter, float(n[1]), float(n[2]))
            self.nodes = self.nodes + [node]
            indexCounter += 1
        # fill in the self.edges list from the edges information
        for e in edges:
            self.edges = self.edges + [[int(e[1]), int(e[2])]]
        # define a dedicated 'noneNode' that acts as a placeholder for neighborhood construction
        noneNode = TopologyNode(-1, 0.0, 0.0)
        # comstruct the neighborhoods of each node in the graph
        for edge in self.edges:
            # first edge node
            a = self.nodes[int(edge[0])]
            # second edge node
            b = self.nodes[int(edge[1])]
            # add node a to the neighborhood of node b, and vice versa
            a.neighbors = a.neighbors + [b]
            b.neighbors = b.neighbors + [a]
        # it is possible that a node does not have the maximum possible number of neighbors, to stay consistent in RL, fill up the neighborhood
        # with noneNode[s]:
        for node in self.nodes:
            while len(node.neighbors) < self.cliqueSize:
                node.neighbors = node.neighbors + [noneNode]
        # assign start nodes
        for nodeIndex in self.graph_info['startNodes']:
            self.nodes[nodeIndex].startNode = True
        # assign goal nodes
        for nodeIndex in self.graph_info['goalNodes']:
            self.nodes[nodeIndex].goalNode = True
        #TODO : Test : Remove from class definition if it is only being used for visualization
        self.sample_state_space()
    
    def set_visual_debugging(self, visual_output, gui_parent):
        '''
        This function sets visualization flags.
        
        | **Args**
        | visual_output:                If true, the topology graph will be visualized.
        | gui_parent:                   The main window used during visualization.
        '''
        self.gui_parent = gui_parent
        self.visual_output = visual_output
        self.init_visual_elements()
    
    def init_visual_elements(self):
        '''
        This function initializes the visual elements.
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
                if not node.goalNode:
                    node.qIndicator = CogArrow(angle=0.0, headLen=20.0, tipAngle=25.0, tailLen=0.0, brush=(255, 255, 0))
                    self.plot.addItem(node.qIndicator)
            # overlay the world's perimeter
            self.perimeterGraph = qg.GraphItem()
            self.plot.addItem(self.perimeterGraph)
            self.perimeterGraph.setData(pos=np.array(self.world_nodes), adj=np.array(self.world_edges), brush=(128, 128, 128))
            # overlay the topology graph
            self.topologyGraph = qg.GraphItem()
            self.plot.addItem(self.topologyGraph)
            # set up a brushes array for visualization of the nodes
            # normal nodes are grey
            symbolBrushes=[qg.mkBrush(color=(128, 128, 128))] * len(self.nodes)
            # set colors of normal and goal nodes
            for node in self.nodes:    
                # start nodes are green
                if node.startNode:
                    symbolBrushes[node.index] = qg.mkBrush(color=(0, 255, 0))
                # goal node is red
                if node.goalNode:
                    symbolBrushes[node.index] = qg.mkBrush(color=(255, 0, 0))
            # construct appropriate arrays from the self.nodes and the self.edges information
            tempNodes, tempEdges = [], []
            for node in self.nodes:
                tempNodes = tempNodes + [[node.x, node.y]]
            for edge in self.edges:
                tempEdges = tempEdges + [[edge[0], edge[1]]]
            self.topologyGraph.setData(pos=np.array(tempNodes), adj=np.array(tempEdges), symbolBrush=symbolBrushes)
            # eventually, overlay robot marker
            self.posMarker = CogArrow(angle=0.0, headLen=20.0, tipAngle=25.0, tailLen=0.0, brush=(255, 0, 0))
            self.plot.addItem(self.posMarker)
            # initial position to center, this has to be worked over later!
            self.posMarker.setData(0.0, 0.0, 0.0)
            
    def update_visual_elements(self):
        '''
        This function updates the visual elements.
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
                    data = np.array([[observation]])
                    # get the q-values at the queried node's position
                    q_values = self.rl_agent.predict_on_batch(data)[0]
                    # find all neighbors that are actually valid (index != -1)
                    validIndex = 0
                    for n_index in range(len(node.neighbors)):
                        if node.neighbors[n_index].index != -1:
                            validIndex = n_index  
                    # find the index of the neighboring node that is 'pointed to' by the highest q-value, AND is valid!
                    maxNeighNode = node.neighbors[np.argmax(q_values[:validIndex+1])]
                    # find the direction of the selected neighboring node
                    # to node: maxNeighNode
                    toNode = np.array([maxNeighNode.x, maxNeighNode.y])
                    # from node: node
                    fromNode = np.array([node.x, node.y])
                    # the difference vector between to and from
                    vec = toNode - fromNode
                    # normalize the direction vector
                    l = np.linalg.norm(vec)
                    vec = vec/l
                    # make the corresponding indicator point in the direction of the difference vector
                    node.qIndicator.setData(node.x, node.y, np.rad2deg(np.arctan2(vec[1], vec[0])))  

    def updateRobotPose(self, pose):
        '''
        This function updates the visual depiction of the agent(robot).
        
        | **Args**
        | pose:                         The agent's pose.
        '''
        if self.visual_output:
            self.posMarker.setData(pose[0],pose[1],np.rad2deg(np.arctan2(pose[3],pose[2])))



    def sample_state_space(self):
        '''
        This function samples observations at all topology nodes.
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

    def generate_behavior_from_action(self, action):
        '''
        This function executes the action selected by the agent.
        
        | **Args**
        | action:                       The action to be executed.
        '''
        nextNodePos = np.array([0.0, 0.0])
        callback_value = dict()
        # if a standard action is performed
        if action != 'reset':
            previousNode = self.currentNode
            # with action given, the next node can be computed
            # TODO :remove dependence on same module
            self.nextNode = self.nodes[self.currentNode].neighbors[action].index
            # array to store the next node's coordinates
            if self.nextNode != -1:
                # compute the next node's coordinates
                nextNodePos = np.array([self.nodes[self.nextNode].x,
                                        self.nodes[self.nextNode].y])
            else:
                # if the next node corresponds to an invalid node, the agent stays in place
                self.nextNode = self.currentNode
                # prevent the agent from starting any motion pattern
                self.modules['world'].goalReached = True
                nextNodePos = np.array([self.nodes[self.currentNode].x,
                                        self.nodes[self.currentNode].y])
            # here, next node is already set and the current node is set to this next node.
            # TODO : make callbacks not mandatory
            callback_value['currentNode'] = self.nodes[self.nextNode]
        # if a reset is performed
        else:
            # a random node is chosen to place the agent at (this node MUST NOT be the global goal node!)
            nextNode = -1
            while True:
                nrNodes = len(self.nodes)
                nextNode = np.random.random_integers(0, nrNodes-1)
                if self.nodes[nextNode].startNode:
                    break
            nextNodePos = np.array([self.nodes[nextNode].x, self.nodes[nextNode].y])
            self.nextNode = nextNode
        # actually move the robot to the node
        self.modules['world'].actuateRobot(np.array([nextNodePos[0], nextNodePos[1], 90.0])) 
        # make the current node the one the agent travelled to
        self.currentNode = self.nextNode
        self.modules['observation'].update()
        self.updateRobotPose([nextNodePos[0], nextNodePos[1], 0.0, 1.0])
        self.update_visual_elements()
        # if possible try to update the visual debugging display
        if hasattr(qt.QtGui, 'QApplication'):
            if qt.QtGui.QApplication.instance() is not None:
                qt.QtGui.QApplication.instance().processEvents()
        else:
            if qt.QtWidgets.QApplication.instance() is not None:
                qt.QtWidgets.QApplication.instance().processEvents()
        
        return callback_value

    def get_action_space(self):
        '''
        This function returns the clique size.
        '''
        return gym.spaces.Discrete(self.cliqueSize)


class ManualTopologyGraphNoRotationDynamic(ManualTopologyGraphNoRotation):
    '''
    Manually defined topology graph module with dynamically changing barriers.
    
    | **Args**
    | modules:                      Framework modules.
    | graph_info:                   Topology graph relevant information.
    '''
    
    def __init__(self, modules, graph_info):
        # call the base class init
        super().__init__(modules, graph_info)
        
    def reload(self):
        '''
        This funtion reloads the topology graph based on the changes in the environment,
        without the need to initialize a new object after each change.
        '''
        # the world module is required here
        world_module = self.modules['world']
        # get the limits of the given environment
        self.world_limits = world_module.getLimits()
        # retrieve all boundary information from the environment
        self.world_nodes, self.world_edges = world_module.getWallGraph()
        # inherent definitions for the topology graph
        # this is the node corresponding to the robot's actual position
        self.currentNode = -1
        # this is the node corresponding to the robot's next position
        self.nextNode = -1
        # this list of topologyNode[s] stores all nodes of the graph
        self.nodes = []
        # this list of [int,int]-entries defines all edges (unique) which make up the graph's connectivity
        self.edges = []
        self.cliqueSize = self.graph_info['cliqueSize']
        # set up a manually constructed topology graph
        # read topology structure from world module
        nodes = np.array(world_module.getManuallyDefinedTopologyNodes())
        nodes = nodes[nodes[:, 0].argsort()]
        edges = np.array(world_module.getManuallyDefinedTopologyEdges())
        edges = edges[edges[:, 0].argsort()]
        # transfer the node points into the self.nodes list
        indexCounter = 0
        for n in nodes:
            # create the corresponding node, where i is the running index of the mesh_points/corresponding nodes
            node = TopologyNode(indexCounter, float(n[1]), float(n[2]))
            self.nodes = self.nodes + [node]
            indexCounter += 1
        # fill in the self.edges list from the edges information
        for e in edges:
            self.edges = self.edges + [[int(e[1]), int(e[2])]]
        # define a dedicated 'noneNode' that acts as a placeholder for neighborhood construction
        noneNode = TopologyNode(-1, 0.0, 0.0)
        # comstruct the neighborhoods of each node in the graph
        for edge in self.edges:
            # first edge node
            a = self.nodes[int(edge[0])]
            # second edge node
            b = self.nodes[int(edge[1])]
            # add node a to the neighborhood of node b, and vice versa
            a.neighbors = a.neighbors + [b]
            b.neighbors = b.neighbors + [a]
        # it is possible that a node does not have the maximum possible number of neighbors, to stay consistent in RL, fill up the neighborhood
        # with noneNode[s]:
        for node in self.nodes:
            while len(node.neighbors) < self.cliqueSize:
                node.neighbors = node.neighbors + [noneNode]
        # assign start nodes
        for nodeIndex in self.graph_info['startNodes']:
            self.nodes[nodeIndex].startNode = True            
        # assign goal nodes
        for nodeIndex in self.graph_info['goalNodes']:
            self.nodes[nodeIndex].goalNode = True
        # resample state space and reload visual elements
        self.sample_state_space()
        self.reload_visual_elements()
        
    def reload_visual_elements(self):
        '''
        This function reloads the visual elements. It is called after the topology graph has been changed.
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
                if not node.goalNode:
                    node.qIndicator = CogArrow(angle=0.0, headLen=20.0, tipAngle=25.0, tailLen=0.0, brush=(255, 255, 0))
                    self.plot.addItem(node.qIndicator)
            # overlay the world's perimeter
            self.perimeterGraph = qg.GraphItem()
            self.plot.addItem(self.perimeterGraph)
            self.perimeterGraph.setData(pos=np.array(self.world_nodes), adj=np.array(self.world_edges), brush=(128, 128, 128))
            # overlay the topology graph
            self.topologyGraph = qg.GraphItem()
            self.plot.addItem(self.topologyGraph)
            # set up a brushes array for visualization of the nodes
            # normal nodes are grey
            symbolBrushes = [qg.mkBrush(color=(128, 128, 128))] * len(self.nodes)
            # set colors of normal and goal nodes
            for node in self.nodes:
                # start nodes are green
                if node.startNode:
                    symbolBrushes[node.index] = qg.mkBrush(color=(0, 255, 0))
                # goal node is red
                if node.goalNode:
                    symbolBrushes[node.index] = qg.mkBrush(color=(255, 0, 0))
            # construct appropriate arrays from the self.nodes and the self.edges information
            tempNodes, tempEdges =[], []
            for node in self.nodes:
                tempNodes = tempNodes + [[node.x, node.y]]
            for edge in self.edges:
                tempEdges = tempEdges + [[edge[0], edge[1]]]
            self.topologyGraph.setData(pos=np.array(tempNodes), adj=np.array(tempEdges), symbolBrush=symbolBrushes)
            # eventually, overlay robot marker
            self.posMarker = CogArrow(angle=0.0, headLen=20.0, tipAngle=25.0, tailLen=0.0, brush=(255, 0, 0))
            self.plot.addItem(self.posMarker)
            # initial position to center, this has to be worked over later!
            self.posMarker.setData(0.0, 0.0, 0.0)
            
    def is_traversable(self):
        '''
        This function checks if the graph is traversable from start to goal.
        '''
        traversedList = np.zeros(len(self.nodes))
        toTraverse = []
        for node in self.nodes:
            if node.startNode == True:
                startNode = node
            elif node.goalNode == True:
                endNode = node
        toTraverse = toTraverse + [startNode]
        traversedList[startNode.index] = 1
        while len(toTraverse) > 0:
            currentNode = toTraverse.pop(0)
            if currentNode.goalNode == True:
                return True
            for node in currentNode.neighbors:
                if node.index != -1:
                    if traversedList[node.index] == 0:
                        toTraverse = toTraverse + [node]
                        traversedList[node.index] = 1

        return False