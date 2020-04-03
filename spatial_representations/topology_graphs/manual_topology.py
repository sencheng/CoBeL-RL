




import PyQt5 as qt
import pyqtgraph as qg
import pyqtgraph.functions
import numpy as np

from PyQt5 import QtGui




### Helper class for the visualization of the topology graph
### Constructs a centered arrow pointing in a dedicated direction, inherits from 'ArrowItem'
class CogArrow(qg.ArrowItem):
    # set the position and direction of the arrow
    # x: x position of the arrow's center
    # y: y position of the arrow's center
    # angle: the orientation of the arrow
    
    def setData(self,x,y,angle):
        
        # the angle has to be modified to suit the demands of the environment(?)
        angle=-angle/np.pi*180.0+180.0
        
        # assemble a new temporary dict that is used for path construction
        tempOpts=dict()
        tempOpts['headLen']=self.opts['headLen']
        tempOpts['tipAngle']=self.opts['tipAngle']
        tempOpts['baseAngle']=self.opts['baseAngle']
        tempOpts['tailLen']=self.opts['tailLen']
        tempOpts['tailWidth']=self.opts['tailWidth']
        
        
        # create the path
        arrowPath=qg.functions.makeArrowPath(**tempOpts)
        # identify boundaries of the arrows, required to shif the arrow
        bounds=arrowPath.boundingRect()
        # prepare a transform
        transform=QtGui.QTransform()
        # shift and rotate the path (arrow)
        transform.rotate(angle)
        transform.translate(int(-float(bounds.x())-float(bounds.width())/10.0*7.0),int(float(-bounds.y())-float(bounds.height())/2.0))
        # 'remap' the path
        self.path=transform.map(arrowPath)
        self.setPath(self.path)
        # set position of the arrow
        self.setPos(x,y)
            
            
            

### This class defines a single node of the topology graph.
class TopologyNode():
    def __init__(self,index,x,y):
        # the node's global index
        self.index=index
        
        # the node's global position
        self.x=x
        self.y=y
        
        # is this node the requested goal node?
        self.goalNode=False
        
        # the clique of the node's neighboring nodes
        self.neighbors=[]
        
        # an indicator arrow that points in the direction of the most probable next neighbor (as planned by the RL system)
        self.qIndicator=CogArrow()

        # if not otherwise defined or inhibited, each node is also a starting node
        self.startNode=False
        
        # this reward bias is assigned to the node as standard (0.0), and can be changed dynamically to reflect environmental reconfigurations of rewards
        self.nodeRewardBias=0.0
        





class ManualTopologyGraph():
    
    def __init__(self, world, guiParent, graphInfo,visualOutput=True):
    
        
        
        # extract the world module and the observation module
        self.world=world
        
        
        # memorize the output window
        self.guiParent=guiParent
        
        # store the graphInfo structure
        self.graphInfo=graphInfo
        
        # store the reinforcement learning agent
        self.rlAgent=None
        # shall the module produce visual output?
        self.visualOutput=visualOutput
        
        # get the limits of the given environment
        self.world_limits = self.world.getLimits()
        
        # retrieve all boundary information from the environment
        self.world_nodes, self.world_edges = self.world.getWallGraph()
        
            
        
        # inherent definitions for the topology graph
        
        # this is the node corresponding to the robot's actual position
        self.currentNode=-1
        # this is the node corresponding to the robot's next position
        self.nextNode=-1
        # this list of topologyNode[s] stores all nodes of the graph
        self.nodes=[]
        # this list of [int,int]-entries defines all edges (unique) which make up the graph's connectivity
        self.edges=[]
        
        
        
        
        self.cliqueSize=graphInfo['cliqueSize']
        # set up a manually constructed topology graph
        
        # read topology structure from world module
        nodes=np.array(self.world.getManuallyDefinedTopologyNodes())
        nodes=nodes[nodes[:,0].argsort()]
        edges=np.array(self.world.getManuallyDefinedTopologyEdges())
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
        for nodeIndex in self.graphInfo['startNodes']:
            self.nodes[nodeIndex].startNode=True
            
        # assign goal nodes
        for nodeIndex in self.graphInfo['goalNodes']:
            self.nodes[nodeIndex].goalNode=True
        
        
        
        self.initVisualElements()


    def initVisualElements(self):
        # do basic visualization
        # iff visualOutput is set to True!
        if self.visualOutput:

            # add the graph plot to the GUI widget
            self.plot = self.guiParent.addPlot(title='Topology graph')
            # set extension of the plot, lock aspect ratio
            self.plot.setXRange( self.world_limits[0,0], self.world_limits[0,1] )
            self.plot.setYRange( self.world_limits[1,0], self.world_limits[1,1] )
            self.plot.setAspectLocked()

            
           
                
            # set up indicator arrows for each node, except the goal node, and all nodes in active shock zones iff shock zones exist
            for node in self.nodes:
                if not node.goalNode:
                    node.qIndicator=CogArrow(angle=0.0,headLen=20.0,tipAngle=25.0,tailLen=0.0,brush=(255,255,0))
                    self.plot.addItem(node.qIndicator)
            
            
            
            
                
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
        
                            
            
            
            ## overlay the policy arrows
            
            #if self.visualOutput:
                ## for all nodes in the topology graph
                #for node in self.nodes:
                    
                    
                    ## query the model at each node's position
                    ## only for valid nodes!
                    #if node.index!=-1:
                        #observation=self.observationModule.observationFromNodeIndex(node.index)
                        #data=np.array([[observation]])
                        ## get the q-values at the queried node's position
                        #q_values = self.rlAgent.agent.model.predict_on_batch(data)[0]
                        
                        ## find all neighbors that are actually valid (index != -1)
                        #validIndex=0
                        #for n_index in range(len(node.neighbors)):
                            #if node.neighbors[n_index].index!=-1:
                                #validIndex=n_index
                        
                        ## find the index of the neighboring node that is 'pointed to' by the highest q-value, AND is valid!
                        #maxNeighNode=node.neighbors[np.argmax(q_values[0:validIndex+1])]
                        ## find the direction of the selected neighboring node
                        ## to node: maxNeighNode
                        #toNode=np.array([maxNeighNode.x,maxNeighNode.y])
                        ## from node: node
                        #fromNode=np.array([node.x,node.y])
                        ## the difference vector between to and from
                        #vec=toNode-fromNode
                        ## normalize the direction vector
                        #l=np.linalg.norm(vec)
                        #vec=vec/l
                        ## make the corresponding indicator point in the direction of the difference vector
                        #node.qIndicator.setData(node.x,node.y,np.arctan2(vec[1],vec[0]))  
            pass

    # This function updates the visual depiction of the agent(robot).
    # 
    # pose: the agent's pose to visualize
    def updateRobotPose(self,pose):
        if self.visualOutput:
            self.posMarker.setData(pose[0],pose[1],np.arctan2(pose[3],pose[2]))
